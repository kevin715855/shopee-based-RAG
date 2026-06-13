"""
rag_pipeline/bge_reranker.py

Reranker dùng BAAI/bge-reranker-v2-m3 chạy FP16 để tối ưu VRAM.

Cross-encoder: nhận cặp (query, document) → cho score relevance.
Pipeline: vector search top-20 → reranker → top-5 → LLM.
"""

from __future__ import annotations
from typing import List, Optional
from loguru import logger

from rag_pipeline.config import (
    RERANKER_MODEL,
    RERANKER_USE_FP16,
    RERANKER_TOP_K,
)


class BGEReranker:
    """
    FlagEmbedding FlagReranker wrapper.
    Lazy-load, singleton pattern.
    """

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        logger.info(f"Loading reranker: {RERANKER_MODEL} | fp16={RERANKER_USE_FP16}")
        from FlagEmbedding import FlagReranker
        self._model = FlagReranker(RERANKER_MODEL, use_fp16=RERANKER_USE_FP16)
        logger.success("Reranker loaded ✓")

    def rerank(
        self,
        query: str,
        documents: List[dict],
        top_k: Optional[int] = None,
    ) -> List[dict]:
        """
        Rerank documents theo relevance với query.

        Args:
            query     : câu query
            documents : output từ hybrid_search, mỗi item có key 'text'
            top_k     : số docs trả về sau rerank (default: RERANKER_TOP_K từ config)

        Returns:
            List docs đã sort giảm dần theo rerank_score, chỉ lấy top_k
        """
        if not documents:
            return []

        top_k = top_k or RERANKER_TOP_K
        self._load()

        pairs  = [[query, d["text"]] for d in documents]
        scores = self._model.compute_score(pairs, normalize=True)

        # normalize=True → scores trong [0,1]
        if not isinstance(scores, list):
            scores = scores.tolist()

        # ✅ Fix #2: Tạo copy dict thay vì mutate trực tiếp dict gốc
        # Mutate in-place sẽ làm dirty list candidates gốc → side effect khó debug
        scored_docs = [
            {**doc, "rerank_score": float(score)}
            for doc, score in zip(documents, scores)
        ]

        reranked = sorted(scored_docs, key=lambda x: x["rerank_score"], reverse=True)
        logger.debug(
            f"Rerank: {len(documents)} → top-{top_k} "
            f"| best={reranked[0]['rerank_score']:.3f}"
        )
        return reranked[:top_k]


_reranker_instance: BGEReranker | None = None


def get_reranker() -> BGEReranker:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = BGEReranker()
    return _reranker_instance
