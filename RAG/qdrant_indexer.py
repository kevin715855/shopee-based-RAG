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
import argparse
from pathlib import Path

from rag_pipeline.config import COLLECTION_NAME

DEFAULT_REVIEWS_FILE = "shopee/reviews"


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
def _normalize_review(r: dict) -> dict:
    return {
        "review_id": str(r.get("review_id", "")),
        "product_id": str(r.get("product_id", "")),
        "user_name": r.get("user_name") or r.get("raw_payload", {}).get("author_username", ""),
        "rating": r.get("rating", 0),
        "content": r.get("content") or r.get("text", ""),
        "date": r.get("date") or r.get("created_at", ""),
        "helpful_count": r.get("helpful_count", 0),
        "category_id": r.get("category_id", ""),
        "category": r.get("category", ""),
        "review_url": r.get("review_url", ""),
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
    # Import lazy để tránh load model khi chỉ chạy --help
    from rag_pipeline.qdrant_store import get_store
    from rag_pipeline.chunker import ReviewChunker

    # ── Chuẩn hóa field names ───────────────────────────────────────────────
    # mock_reviews.json dùng: user_name, date, product_id (là Shopee ID)
    # ReviewChunker / qdrant_store dùng: author, reviewed_at, shopee_product_id
    normalized = []
    for r in reviews:
        normalized.append({
            "review_id":         r["review_id"],
            "content":           r["content"],
            "shopee_product_id": r.get("product_id", ""),   # Shopee product ID
            "rating":            r.get("rating", 0),
            "sentiment":         r.get("sentiment", ""),    # ✅ Fix #3: giữ lại sentiment gốc
            "author":            r.get("user_name", ""),    # map user_name → author
            "reviewed_at":       r.get("date", ""),         # map date → reviewed_at
        })

    # ── Chunk ───────────────────────────────────────────────────────────────
    print(f"Đang chunk {len(normalized)} reviews ...")
    chunker = ReviewChunker()
    # product_id=0 → tìm toàn bộ (không filter theo internal product_id)
    chunks = chunker.chunk_reviews(normalized, product_id=0)
    chunk_dicts = [c.to_dict() for c in chunks]
    print(f"  → {len(chunk_dicts)} chunks sau khi split")

    # ── Embed + Upsert vào Qdrant (đồng thời cập nhật BM25 corpus) ──────────
    print(f"Đang embed và upsert {len(chunk_dicts)} chunks vào Qdrant ...")
    store = get_store()
    ids = store.add_chunks(chunk_dicts)

    print(f"✅ Đã index thành công {len(ids)} chunks từ {len(reviews)} reviews vào '{COLLECTION_NAME}'.")
    return len(ids)


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
