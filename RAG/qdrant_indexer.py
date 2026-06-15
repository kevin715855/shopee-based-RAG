"""
qdrant_indexer.py
-----------------
Script chạy một lần để index reviews vào Qdrant:
  1. Đọc reviews từ file JSON
  2. Chunk reviews bằng ReviewChunker (tối ưu retrieval quality)
  3. Tạo dense embeddings bằng BAAI/bge-m3 (qua rag_pipeline)
  4. Upsert vectors vào Qdrant và đồng bộ BM25 corpus

Chạy:
    python qdrant_indexer.py
    python qdrant_indexer.py --file data/reviews.json

✅ Fix: Dùng store.add_chunks() thay vì direct QdrantClient
       → BM25 corpus được đồng bộ tự động
✅ Fix: Map user_name → author, date → reviewed_at nhất quán với qdrant_store
✅ Fix: Tích hợp ReviewChunker để chunk review dài
"""

import json
import uuid
import argparse
from pathlib import Path

from qdrant_client.models import PointStruct

from rag_pipeline.config import COLLECTION_NAME
from rag_pipeline import get_embedder, get_store
from rag_pipeline.chunker import ReviewChunker

DEFAULT_REVIEWS_FILE = "shopee/reviews"

# def load_reviews(filepath: str) -> list[dict]:
#     """Đọc danh sách reviews từ file JSON."""
#     path = Path(filepath)
#     if not path.exists():
#         raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)

# def load_reviews(filepath: str) -> list[dict]:
#     """Đọc danh sách reviews từ file JSON."""
#     path = Path(filepath)
#     if not path.exists():
#         raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)

def load_reviews(filepath: str) -> list[dict]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {filepath}")

    reviews = []

    if path.is_dir():
        files = list(path.glob("*.json")) + list(path.glob("*.jsonl"))
    else:
        files = [path]

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            if file.suffix == ".jsonl":
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        reviews.append(_normalize_review(r))
            else:
                data = json.load(f)
                if isinstance(data, list):
                    reviews.extend(_normalize_review(r) for r in data)
                else:
                    reviews.append(_normalize_review(data))

    return reviews

#xử lý file jsonl -> json
def _get_username(r: dict) -> str:
    """Trích xuất tên hiển thị từ review record.

    Ưu tiên:
      1. Field 'user_name' trực tiếp (mock data / old format)
      2. raw_payload.anonymous → "Ẩn danh"
      3. raw_payload.userid    → "Người dùng #<id>"
      4. Fallback              → "Ẩn danh"
    """
    if r.get("user_name"):
        return r["user_name"]
    raw = r.get("raw_payload", {})
    if isinstance(raw, dict):
        if raw.get("anonymous"):
            return "Ẩn danh"
        uid = raw.get("userid")
        if uid:
            return f"Người dùng #{uid}"
    return "Ẩn danh"


def _normalize_review(r: dict) -> dict:
    # Thử lấy content: field trực tiếp → raw_payload.comment (Shopee API)
    content = (
        r.get("content")
        or r.get("text", "")
        or (r.get("raw_payload") or {}).get("comment", "")   # ✅ Added fallback
    )
    return {
        "review_id":     str(r.get("review_id", "")),
        "product_id":    str(r.get("product_id", "")),   # Shopee string ID
        "user_name":     _get_username(r),                # ✅ Fixed
        "rating":        r.get("rating", 0),
        "content":       content,
        "date":          r.get("date") or r.get("created_at", ""),
        "helpful_count": r.get("helpful_count", 0),
        "category_id":   r.get("category_id", ""),
        "category":      r.get("category", ""),
        "review_url":    r.get("review_url", ""),
    }


def index_reviews(reviews: list[dict]) -> int:
    """
    Chunk, embed và upsert toàn bộ reviews vào Qdrant.

    ✅ Fix BM25: Dùng store.add_chunks() để đồng bộ BM25 corpus trong QdrantReviewStore.
    ✅ Fix field mismatch: Chuẩn hóa user_name → author, date → reviewed_at.
    ✅ Fix chunking: Dùng ReviewChunker để chia nhỏ review dài.

    Args:
        reviews: List review dicts từ mock_reviews.json / Shopee crawler

    Returns:
        Số lượng chunks đã được index thành công.
    """
    # 1. Load embedder (lazy load, dùng singleton từ rag_pipeline)
    embedder = get_embedder()

    # Bỏ qua reviews không có nội dung (rating-only, không có giá trị cho RAG)
    valid_reviews = [r for r in reviews if r.get("content", "").strip()]
    skipped = len(reviews) - len(valid_reviews)
    if skipped:
        print(f"⚠️  Bỏ qua {skipped} reviews không có nội dung (rating-only).")

    texts = [r["content"] for r in valid_reviews]
    print(f"Đang tạo embeddings cho {len(texts)} reviews ...")
    vectors = embedder.embed_documents(texts)

    # ── Chunk ───────────────────────────────────────────────────────────────
    normalized = valid_reviews
    print(f"Đang chunk {len(normalized)} reviews ...")
    chunker = ReviewChunker()
    # product_id=0 → tìm toàn bộ (không filter theo internal product_id)
    chunks = chunker.chunk_reviews(normalized, product_id=0)
    chunk_dicts = [c.to_dict() for c in chunks]
    print(f"  → {len(chunk_dicts)} chunks sau khi split")

    # ── Embed + Upsert vào Qdrant (đồng thời cập nhật BM25 corpus) ──────────
    print(f"Đang embed và upsert {len(chunk_dicts)} chunks vào Qdrant ...")
    store = get_store()
    client = store.client
    ids = store.add_chunks(chunk_dicts)

    # 4. Tạo PointStruct và upsert
    points = []
    for review, vector in zip(valid_reviews, vectors):
        # Dùng uuid5 deterministc từ review_id để tránh trùng lặp khi re-index
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, review["review_id"]))
        points.append(
            PointStruct(
                id      = point_id,
                vector  = vector,
                payload = {
                    "text":               review["content"],
                    "review_id":          review["review_id"],
                    # product_id (int) = 0 → internal placeholder, KHÔNG dùng để filter
                    # Dùng shopee_product_id (str) để filter theo sản phẩm
                    "product_id":         0,
                    "shopee_product_id":  review.get("product_id", ""),  # Shopee string ID
                    "author":             review.get("user_name", "Ẩn danh"),  # ✅ Fixed: dùng key 'author'
                    "rating":             review.get("rating", 0),
                    "sentiment":          _rating_to_sentiment(review.get("rating", 0)),
                    "reviewed_at":        review.get("date", ""),
                    "helpful_count":      review.get("helpful_count", 0),
                    "chunk_index":        0,
                    "category_id":        review.get("category_id", ""),
                    "category":           review.get("category", ""),
                    "review_url":         review.get("review_url", ""),
                },
            )
        )

    # Upsert theo batch 100
    for i in range(0, len(points), 100):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i:i+100])

    print(f"✅ Đã index thành công {len(points)} reviews vào '{COLLECTION_NAME}'.")
    return len(points)


def _rating_to_sentiment(rating: int) -> str:
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"


def main():
    parser = argparse.ArgumentParser(description="Index reviews vào Qdrant")
    parser.add_argument("--file", default=DEFAULT_REVIEWS_FILE,
                        help=f"Đường dẫn file JSON (default: {DEFAULT_REVIEWS_FILE})")
    args = parser.parse_args()

    reviews = load_reviews(args.file)
    print(f"Đọc được {len(reviews)} reviews từ '{args.file}'.")
    index_reviews(reviews)


if __name__ == "__main__":
    main()
