"""
qdrant_indexer.py
-----------------
Script chạy một lần để index reviews vào Qdrant:
  1. Đọc reviews từ file JSON
  2. Tạo dense embeddings bằng BAAI/bge-m3 (qua rag_pipeline)
  3. Upsert vectors vào Qdrant collection

Chạy:
    python qdrant_indexer.py
    python qdrant_indexer.py --file data/reviews.json
"""

import json
import uuid
import argparse
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# ✅ Dùng embedder từ rag_pipeline để nhất quán
from rag_pipeline.bge_embedder import get_embedder
from rag_pipeline.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,   # ✅ Fix: "shopee_reviews" – khớp qdrant_store.py
    EMBEDDING_DIM,
)

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
    Tạo embeddings và upsert toàn bộ reviews vào Qdrant.

    Returns:
        Số lượng điểm đã được index thành công.
    """
    # 1. Load embedder (lazy load, dùng singleton từ rag_pipeline)
    embedder = get_embedder()
    texts    = [r["content"] for r in reviews]

    print(f"Đang tạo embeddings cho {len(texts)} reviews ...")
    vectors = embedder.embed_documents(texts)

    # 2. Kết nối Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # 3. Tạo collection nếu chưa có
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        print(f"Tạo mới collection '{COLLECTION_NAME}' ...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' đã tồn tại, tiến hành upsert ...")

    # 4. Tạo PointStruct và upsert
    points = []
    for review, vector in zip(reviews, vectors):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, review["review_id"]))
        points.append(
            PointStruct(
                id      = point_id,
                vector  = vector,
                payload = {
                    "text":          review["content"],
                    "review_id":     review["review_id"],
                    "product_id":    0,
                    "shopee_product_id": review.get("product_id", ""),
                    "user_name":     review.get("user_name", ""),
                    "rating":        review.get("rating", 0),
                    "sentiment":     _rating_to_sentiment(review.get("rating", 0)),
                    "date":          review.get("date", ""),
                    "helpful_count": review.get("helpful_count", 0),
                    "chunk_index":   0,
                    "category_id": review.get("category_id", ""),
                    "category": review.get("category", ""),
                    "review_url": review.get("review_url", ""),
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
