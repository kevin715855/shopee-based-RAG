"""
rag_pipeline/qdrant_store.py

LangChain-compatible Qdrant vector store với hybrid search.
- Dense search  : bge-m3 dense vectors
- Sparse search : BM25 (rank_bm25) trên corpus cục bộ
- Hybrid fusion : Reciprocal Rank Fusion (RRF)

Metadata payload cho mỗi point:
  text, product_id, shopee_product_id, rating, sentiment,
  author, reviewed_at, chunk_index
"""

from __future__ import annotations
import uuid
from typing import List, Optional
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
    ScrollRequest,
)
from rank_bm25 import BM25Okapi

# ✅ Fix: import từ rag_pipeline (flat), không phải rag.embeddings
from rag_pipeline.bge_embedder import get_embedder
from rag_pipeline.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    EMBEDDING_DIM,
    RETRIEVAL_TOP_K,
)


class QdrantReviewStore:
    """
    Vector store cho reviews Shopee.

    Dùng Qdrant làm dense store + BM25 in-memory cho sparse.
    Hybrid fusion bằng Reciprocal Rank Fusion (RRF).
    """

    def __init__(self):
        self.client   = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.embedder = get_embedder()
        self._bm25    = None           # lazy build
        self._bm25_corpus: list[dict] = []
        self._ensure_collection()

    # ─── Setup ──────────────────────────────────────────────────────────────

    def _ensure_collection(self):
        names = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in names:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            # Payload indexes cho filter nhanh
            for field, schema in [
                ("product_id",        "integer"),
                ("shopee_product_id", "keyword"),
                ("rating",            "integer"),
                ("sentiment",         "keyword"),
            ]:
                self.client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field,
                    field_schema=schema,
                )
            logger.info(f"Tạo Qdrant collection '{COLLECTION_NAME}' ✓")

    # ─── Indexing ───────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[dict]) -> List[str]:
        """
        Embed và upsert chunks vào Qdrant.

        Args:
            chunks: List[dict] với keys:
                text, product_id, shopee_product_id,
                rating, sentiment, author, reviewed_at,
                review_id, chunk_index
        Returns:
            List of Qdrant point IDs
        """
        if not chunks:
            return []

        texts   = [c["text"] for c in chunks]
        vectors = self.embedder.embed_documents(texts)

        points = []
        ids    = []
        for chunk, vec in zip(chunks, vectors):
            pid = str(uuid.uuid4())
            ids.append(pid)
            points.append(PointStruct(
                id      = pid,
                vector  = vec,
                payload = {
                    "text":              chunk["text"],
                    "review_id":         chunk.get("review_id", ""),
                    "product_id":        chunk.get("product_id", 0),
                    "shopee_product_id": chunk.get("shopee_product_id", ""),
                    "rating":            chunk.get("rating", 0),
                    "sentiment":         chunk.get("sentiment", "neutral"),
                    "author":            chunk.get("author", ""),
                    "reviewed_at":       chunk.get("reviewed_at", ""),
                    "chunk_index":       chunk.get("chunk_index", 0),
                },
            ))

        # Upsert theo batch 100
        for i in range(0, len(points), 100):
            self.client.upsert(collection_name=COLLECTION_NAME, points=points[i:i+100])

        # Cập nhật BM25 corpus
        self._bm25_corpus.extend(chunks)
        self._bm25 = None  # reset, sẽ rebuild khi cần

        logger.debug(f"Upserted {len(points)} chunks vào Qdrant")
        return ids

    # ─── Retrieval ──────────────────────────────────────────────────────────

    def dense_search(
        self,
        query: str,
        product_id:        Optional[int] = None,
        shopee_product_id: Optional[str] = None,
        rating_min:        Optional[int] = None,
        rating_max:        Optional[int] = None,
        sentiment:         Optional[str] = None,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> List[dict]:
        """Dense vector search với optional metadata filter."""
        query_vec = self.embedder.embed_query(query)

        must = []
        if product_id is not None:
            must.append(FieldCondition(key="product_id",        match=MatchValue(value=product_id)))
        if shopee_product_id:
            must.append(FieldCondition(key="shopee_product_id", match=MatchValue(value=shopee_product_id)))
        if sentiment:
            must.append(FieldCondition(key="sentiment",         match=MatchValue(value=sentiment)))
        if rating_min is not None or rating_max is not None:
            must.append(FieldCondition(key="rating",            range=Range(gte=rating_min or 1, lte=rating_max or 5)))

        # ✅ Fix: qdrant-client >= 1.10 dùng query_points() thay search() (đã bị xóa)
        response = self.client.query_points(
            collection_name = COLLECTION_NAME,
            query           = query_vec,
            query_filter    = Filter(must=must) if must else None,
            limit           = top_k,
            with_payload    = True,
        )
        return [self._to_doc(r.payload, r.score, "dense") for r in response.points]

    def _rebuild_bm25_from_qdrant(self) -> None:
        """
        Rebuild BM25 corpus từ Qdrant.

        ✅ Fix: Được gọi khi BM25 corpus rỗng – thường xảy ra sau khi app restart
        hoặc khi data được index trực tiếp vào Qdrant qua client khác.
        Dùng scroll() để fetch toàn bộ payload mà không cần vectors.
        """
        logger.info("[BM25] Corpus rỗng, đang rebuild từ Qdrant ...")
        corpus: list[dict] = []
        offset = None

        while True:
            try:
                points, next_offset = self.client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as e:
                # Collection có thể chưa tồn tại hoặc Qdrant chưa sẵn sàng
                logger.warning(f"[BM25] Scroll thất bại, corpus rỗng: {e}")
                break

            for r in points:
                payload = r.payload or {}
                if payload.get("text"):
                    # ✅ Fix #4: Lưu đầy đủ metadata thay vì chỉ text + product_id
                    corpus.append({
                        "text":              payload["text"],
                        "product_id":        payload.get("product_id", 0),
                        "shopee_product_id": payload.get("shopee_product_id", ""),
                        "rating":            payload.get("rating", 0),
                        "sentiment":         payload.get("sentiment", "neutral"),
                        "author":            payload.get("author", ""),
                        "reviewed_at":       payload.get("reviewed_at", ""),
                        "review_id":         payload.get("review_id", ""),
                        "chunk_index":       payload.get("chunk_index", 0),
                    })

            if next_offset is None:
                break
            offset = next_offset

        self._bm25_corpus = corpus
        logger.info(f"[BM25] Rebuild xong: {len(corpus)} docs")

    def sparse_search(
        self,
        query: str,
        product_id:        Optional[int] = None,
        shopee_product_id: Optional[str] = None,
        rating_min:        Optional[int] = None,
        rating_max:        Optional[int] = None,
        sentiment:         Optional[str] = None,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> List[dict]:
        """BM25 sparse search (in-memory, filter theo product_id và các metadata filter)."""
        # ✅ Fix: Auto-rebuild BM25 corpus từ Qdrant nếu corpus chưa được populate
        if not self._bm25_corpus:
            self._rebuild_bm25_from_qdrant()

        corpus = self._bm25_corpus
        # ✅ Fix #5: Áp dụng đầy đủ filter params (trước chỉ filter product_id)
        if product_id is not None:
            corpus = [c for c in corpus if c.get("product_id") == product_id]
        if shopee_product_id:
            corpus = [c for c in corpus if c.get("shopee_product_id") == shopee_product_id]
        if sentiment:
            corpus = [c for c in corpus if c.get("sentiment") == sentiment]
        if rating_min is not None:
            corpus = [c for c in corpus if c.get("rating", 0) >= rating_min]
        if rating_max is not None:
            corpus = [c for c in corpus if c.get("rating", 0) <= rating_max]
        if not corpus:
            return []

        tokenized    = [c["text"].lower().split() for c in corpus]
        bm25         = BM25Okapi(tokenized)
        query_tokens = query.lower().split()
        scores       = bm25.get_scores(query_tokens)

        ranked = sorted(
            [(score, chunk) for score, chunk in zip(scores, corpus)],
            key=lambda x: x[0], reverse=True,
        )[:top_k]

        return [self._to_doc(chunk, score, "sparse") for score, chunk in ranked if score > 0]

    def hybrid_search(
        self,
        query: str,
        product_id:        Optional[int] = None,
        shopee_product_id: Optional[str] = None,
        rating_min:        Optional[int] = None,
        rating_max:        Optional[int] = None,
        sentiment:         Optional[str] = None,
        top_k: int = RETRIEVAL_TOP_K,
        alpha: float = 0.7,   # 0=pure sparse, 1=pure dense
    ) -> List[dict]:
        """
        Hybrid search: Dense + Sparse → Reciprocal Rank Fusion.

        Args:
            alpha: weight cho dense score (0.7 = 70% dense, 30% sparse)
        """
        dense_results  = self.dense_search(
            query, product_id, shopee_product_id,
            rating_min, rating_max, sentiment, top_k,
        )
        # ✅ Fix #5: Truyền đầy đủ filter params sang sparse_search
        sparse_results = self.sparse_search(
            query, product_id, shopee_product_id,
            rating_min, rating_max, sentiment, top_k,
        )

        # Reciprocal Rank Fusion
        scores: dict[str, float] = {}
        docs:   dict[str, dict]  = {}
        k = 60  # RRF constant

        for rank, doc in enumerate(dense_results):
            # ✅ Fix #1: Dùng review_id+chunk_index làm key thay vì text[:100]
            # text[:100] dễ bị collision khi review ngắn hoặc trùng đầu câu
            key = f"{doc.get('review_id', '')}_{doc.get('chunk_index', rank)}"
            scores[key] = scores.get(key, 0) + alpha * (1 / (k + rank + 1))
            docs[key]   = doc

        for rank, doc in enumerate(sparse_results):
            key = f"{doc.get('review_id', '')}_{doc.get('chunk_index', rank)}"
            scores[key] = scores.get(key, 0) + (1 - alpha) * (1 / (k + rank + 1))
            if key not in docs:
                docs[key] = doc

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        result = []
        for key, rrf_score in top:
            doc = docs[key].copy()
            doc["score"]       = rrf_score
            doc["search_type"] = "hybrid"
            result.append(doc)

        return result

    # ─── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _to_doc(payload: dict, score: float, search_type: str) -> dict:
        return {
            "text":              payload.get("text", ""),
            "score":             float(score),
            "search_type":       search_type,
            "review_id":         payload.get("review_id", ""),
            "product_id":        payload.get("product_id", 0),
            "shopee_product_id": payload.get("shopee_product_id", ""),
            "rating":            payload.get("rating", 0),
            "sentiment":         payload.get("sentiment", "neutral"),
            "author":            payload.get("author", ""),
            "reviewed_at":       payload.get("reviewed_at", ""),
        }

    def collection_info(self) -> dict:
        info = self.client.get_collection(COLLECTION_NAME)
        return {
            "total_points": info.points_count,
            "collection":   COLLECTION_NAME,
        }


# Singleton
_store_instance: QdrantReviewStore | None = None


def get_store() -> QdrantReviewStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = QdrantReviewStore()
    return _store_instance
