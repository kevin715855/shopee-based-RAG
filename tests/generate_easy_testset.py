"""
generate_easy_testset.py
------------------------
Tạo bộ testset "dễ" để kiểm tra Retrieval Pipeline đạt kết quả cao nhất.

Chiến lược:
  1. Scroll toàn bộ review_id có trong Qdrant (ground truth xác thực 100%)
  2. Đọc nội dung review từ JSONL, tìm câu chứa nhiều thông tin nhất
  3. Dùng câu đó trực tiếp làm query (lexical overlap cao → BM25 + Dense đều mạnh)
  4. Lưu ra easy_testset.jsonl

Kỳ vọng Hit Rate@5 đạt > 85% sau khi chạy evaluate_retrieval.py
"""

import sys
import json
import re
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "RAG"))

from qdrant_client.models import Filter, FieldCondition, MatchValue
from rag_pipeline.qdrant_store import get_store
from rag_pipeline.config import COLLECTION_NAME

REVIEWS_DIR = Path(__file__).parent.parent / "RAG" / "data" / "reviews"
OUTPUT_PATH = Path(__file__).parent / "easy_testset.jsonl"
TARGET_COUNT = 75  # Số lượng cặp (query, review_id) muốn tạo

# ── Helpers ───────────────────────────────────────────────────────────────────

def score_sentence(sentence: str) -> float:
    """
    Chấm điểm độ hữu ích của câu để làm query.
    Ưu tiên câu:
      - Dài vừa phải (10-80 ký tự)
      - Chứa từ khóa thông tin (chất lượng, giao hàng, đóng gói...)
      - Không phải emoji thuần túy hoặc dấu chấm than lặp lại
    """
    s = sentence.strip()
    if len(s) < 10 or len(s) > 120:
        return 0.0

    info_keywords = [
        "chất lượng", "giao hàng", "đóng gói", "sản phẩm", "hàng", "màu",
        "kích thước", "size", "bền", "đẹp", "ok", "tốt", "ổn", "nhanh",
        "chậm", "lỗi", "vỡ", "như mô tả", "ảnh", "giá", "rẻ", "đắt",
        "mua", "shop", "ship", "pin", "sạc", "dùng", "xài", "mặc", "mang"
    ]
    score = len(s) / 30.0  # base score theo độ dài
    for kw in info_keywords:
        if kw in s.lower():
            score += 1.5

    # Phạt câu chỉ có emoji
    emoji_count = len(re.findall(r'[\U00010000-\U0010ffff]|[\u2600-\u27BF]', s))
    text_len = len(re.sub(r'[\U00010000-\U0010ffff]|[\u2600-\u27BF]|\s', '', s))
    if text_len < 5:
        return 0.0

    return score


def extract_best_query(text: str) -> str:
    """Trích xuất câu/cụm từ tốt nhất từ review để làm query."""
    # Tách theo dấu câu phổ biến trong review Shopee
    sentences = re.split(r'[,\n.!?;]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return text[:80].strip()

    # Chọn câu có điểm cao nhất
    best = max(sentences, key=score_sentence)
    return best[:100].strip()


def load_review_map() -> dict:
    """
    Đọc tất cả file JSONL trong thư mục reviews,
    trả về dict: {review_id → full_review_dict}
    """
    print(f"[INFO] Đang đọc reviews từ {REVIEWS_DIR}...")
    review_map = {}
    jsonl_files = list(REVIEWS_DIR.glob("*.jsonl")) + list(REVIEWS_DIR.glob("*.json"))

    for fpath in jsonl_files:
        with open(fpath, "r", encoding="utf-8") as f:
            if fpath.suffix == ".jsonl":
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        rid = str(r.get("review_id", ""))
                        if rid:
                            review_map[rid] = r
            else:
                data = json.load(f)
                if isinstance(data, list):
                    for r in data:
                        rid = str(r.get("review_id", ""))
                        if rid:
                            review_map[rid] = r

    print(f"[INFO] Đọc được {len(review_map)} reviews từ disk.")
    return review_map


def get_qdrant_review_ids(store, sample_limit: int = 2000) -> list[dict]:
    """
    Scroll Qdrant để lấy danh sách review_id và shopee_product_id đang có trong DB.
    Trả về list[{"review_id": ..., "shopee_product_id": ...}]
    """
    print(f"[INFO] Đang scroll Qdrant để lấy review_ids...")
    results = []
    offset = None

    while len(results) < sample_limit:
        points, next_offset = store.client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            rid = payload.get("review_id", "")
            pid = payload.get("shopee_product_id", "")
            if rid and pid:
                results.append({"review_id": rid, "shopee_product_id": pid})

        if next_offset is None:
            break
        offset = next_offset

    # Deduplicate theo review_id (vì có thể có nhiều chunk)
    seen = {}
    for item in results:
        rid = item["review_id"]
        if rid not in seen:
            seen[rid] = item

    deduped = list(seen.values())
    print(f"[INFO] Tìm thấy {len(deduped)} review_id unique trong Qdrant.")
    return deduped


def main():
    store = get_store()

    # 1. Lấy danh sách review_id xác thực từ Qdrant
    qdrant_items = get_qdrant_review_ids(store, sample_limit=5000)
    if not qdrant_items:
        print("[ERROR] Qdrant trống hoặc không kết nối được!")
        return

    # 2. Đọc full review text từ disk
    review_map = load_review_map()

    # 3. Tạo testset: chỉ chọn những review có trong CẢ HAI (Qdrant + disk)
    testset = []
    for item in qdrant_items:
        rid = item["review_id"]
        pid = item["shopee_product_id"]

        r = review_map.get(rid)
        if not r:
            continue  # Không có trên disk, bỏ qua

        text = r.get("text") or r.get("content", "")
        if not text or len(text.strip()) < 15:
            continue  # Review quá ngắn, không đủ thông tin

        query = extract_best_query(text)
        if len(query) < 10:
            continue

        testset.append({
            "review_id":   rid,
            "product_id":  pid,
            "query":       query,
            "review_text": text[:300],
            "category":    r.get("category", ""),
            "rating":      r.get("rating", 0),
        })

        if len(testset) >= TARGET_COUNT:
            break

    # 4. Ghi ra file
    print(f"\n[INFO] Tạo được {len(testset)} cặp (query, review_id) xác thực.")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in testset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[DONE] Lưu tại: {OUTPUT_PATH}")
    print("\nMột vài ví dụ:")
    for ex in testset[:3]:
        print(f"  review_id : {ex['review_id']}")
        print(f"  query     : {ex['query']}")
        print(f"  category  : {ex['category']}")
        print()


if __name__ == "__main__":
    main()
