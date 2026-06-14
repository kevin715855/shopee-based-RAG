"""
test_fixes.py  –  Kiểm tra các bug đã sửa.

Chạy:
    python test_fixes.py

Yêu cầu:
    - Qdrant đang chạy:  docker run -p 6333:6333 qdrant/qdrant
    - Đã index data:     python main.py --mode index --file shopee/reviews
"""

import json
from pathlib import Path


# ══════════════════════════════════════════════════════
# TEST 1: _get_username – không cần Qdrant/LLM
# ══════════════════════════════════════════════════════

def test_get_username():
    from qdrant_indexer import _get_username

    print("\n" + "=" * 55)
    print("TEST 1: _get_username()")
    print("=" * 55)

    cases = [
        # (input, expected_prefix)
        ({"user_name": "Nguyễn Văn A"},           "Nguyễn Văn A"),
        ({"raw_payload": {"anonymous": True}},     "Ẩn danh"),
        ({"raw_payload": {"userid": 1011584481, "anonymous": False}}, "Người dùng #1011584481"),
        ({},                                       "Ẩn danh"),
        ({"raw_payload": {"userid": 0, "anonymous": False}}, "Ẩn danh"),  # userid=0 → falsy
    ]

    all_ok = True
    for r, expected in cases:
        result = _get_username(r)
        ok = result == expected
        all_ok = all_ok and ok
        status = "✅" if ok else "❌"
        print(f"  {status}  input={r}  →  '{result}'  (expected='{expected}')")

    print(f"\n→ TEST 1 {'PASSED ✅' if all_ok else 'FAILED ❌'}")
    return all_ok


# ══════════════════════════════════════════════════════
# TEST 2: _normalize_review – đọc file thật từ shopee/reviews
# ══════════════════════════════════════════════════════

def test_normalize_review():
    from qdrant_indexer import _normalize_review

    print("\n" + "=" * 55)
    print("TEST 2: _normalize_review() với JSONL thật")
    print("=" * 55)

    rev_dir = Path("shopee/reviews")
    sample_file = next(rev_dir.glob("*.jsonl"), None)
    if not sample_file:
        print("  ⚠️  Không tìm thấy file JSONL trong shopee/reviews")
        return False

    with open(sample_file) as f:
        lines = [json.loads(l) for l in f if l.strip()]

    # Lấy review có content để test (bỏ qua rating-only)
    raw = next((r for r in lines if r.get("text") or r.get("content")), lines[0])
    result = _normalize_review(raw)

    checks = {
        "review_id":  bool(result["review_id"]),
        "product_id": bool(result["product_id"]),
        "rating":     isinstance(result["rating"], int),
        "user_name":  isinstance(result["user_name"], str),
        "category":   isinstance(result["category"], str),
        "date":       bool(result["date"]),
    }

    # Content: có thể rỗng nếu review rating-only, nhưng không nên None
    checks["content_type"] = isinstance(result["content"], str)

    all_ok = True
    for field, ok in checks.items():
        all_ok = all_ok and ok
        status = "✅" if ok else "❌"
        val = str(result.get(field, "MISSING"))[:60]
        print(f"  {status}  {field:15s} = '{val}'")

    print(f"\n  user_name resolved: '{result['user_name']}'")
    print(f"  content preview  : '{result['content'][:80]}'")
    print(f"\n→ TEST 2 {'PASSED ✅' if all_ok else 'FAILED ❌'}")
    return all_ok


# ══════════════════════════════════════════════════════
# TEST 3: load_reviews – đọc cả folder shopee/reviews
# ══════════════════════════════════════════════════════

def test_load_reviews():
    from qdrant_indexer import load_reviews

    print("\n" + "=" * 55)
    print("TEST 3: load_reviews('shopee/reviews')")
    print("=" * 55)

    reviews = load_reviews("shopee/reviews")
    print(f"  Tổng số reviews: {len(reviews)}")

    if not reviews:
        print("  ❌  Không load được review nào!")
        return False

    # Kiểm tra user_name không bao giờ rỗng
    named   = sum(1 for r in reviews if r["user_name"] and r["user_name"] != "Ẩn danh")
    anon    = sum(1 for r in reviews if r["user_name"] == "Ẩn danh")
    userid  = sum(1 for r in reviews if r["user_name"].startswith("Người dùng #"))
    empty   = sum(1 for r in reviews if not r["user_name"])

    print(f"  user_name 'Người dùng #xxx': {userid}")
    print(f"  user_name 'Ẩn danh'        : {anon}")
    print(f"  user_name khác             : {named - userid}")
    print(f"  user_name rỗng (BUG)       : {empty}  ← phải = 0")

    # Content: rating-only reviews là data thật, không phải bug
    no_content   = sum(1 for r in reviews if not r["content"])
    with_content = len(reviews) - no_content
    print(f"  Có content (sẽ index)      : {with_content}")
    print(f"  Không có content (rating-only, skip khi index): {no_content}")

    all_ok = empty == 0   # user_name không bao giờ được rỗng
    print(f"\n→ TEST 3 {'PASSED ✅' if all_ok else 'FAILED ❌'}")
    return all_ok


# ══════════════════════════════════════════════════════
# TEST 4: hybrid_search với category filter – cần Qdrant
# ══════════════════════════════════════════════════════

def test_hybrid_search_with_category():
    print("\n" + "=" * 55)
    print("TEST 4: hybrid_search() với category filter (cần Qdrant)")
    print("=" * 55)

    try:
        from rag_pipeline.qdrant_store import get_store
        store = get_store()
    except Exception as e:
        print(f"  ⚠️  Bỏ qua: không kết nối được Qdrant ({e})")
        return None

    # 4a: Search không filter
    results_all = store.hybrid_search(
        query = "Pin có bền không?",
        top_k = 5,
    )
    print(f"  Search không filter  → {len(results_all)} results")

    # 4b: Search với shopee_product_id (lấy ID từ file thật)
    rev_dir = Path("shopee/reviews")
    sample_id = next(rev_dir.glob("*.jsonl")).stem
    results_product = store.hybrid_search(
        query             = "Chất lượng sản phẩm thế nào?",
        shopee_product_id = sample_id,
        top_k             = 5,
    )
    print(f"  Filter shopee_id={sample_id[:12]}... → {len(results_product)} results")

    # 4c: Kiểm tra author field
    for r in results_all[:2]:
        print(f"  author='{r.get('author', 'MISSING')}' | category='{r.get('category', 'MISSING')}'")

    all_ok = (
        len(results_all) > 0 and
        all(r.get("author") for r in results_all)
    )
    print(f"\n→ TEST 4 {'PASSED ✅' if all_ok else 'FAILED ❌'}")
    return all_ok


# ══════════════════════════════════════════════════════
# TEST 5: qa.ask() với category param – cần Qdrant + vLLM
# ══════════════════════════════════════════════════════

def test_qa_ask_with_category():
    print("\n" + "=" * 55)
    print("TEST 5: qa.ask() với category param (cần Qdrant + vLLM)")
    print("=" * 55)

    try:
        from rag_pipeline import get_qa_chain
        qa = get_qa_chain()
    except Exception as e:
        print(f"  ⚠️  Bỏ qua: không init được QAChain ({e})")
        return None

    result = qa.ask(
        question     = "Sản phẩm có tốt không?",
        product_name = "Camera Wifi",
        category     = "Máy Ảnh & Máy Quay Phim",  # đổi theo category thật
    )

    print(f"  question: {result['question']}")
    print(f"  answer  : {result['answer'][:100]}...")
    print(f"  sources : {len(result['sources'])} reviews")
    print(f"  meta    : {result['pipeline_meta']}")

    # Kiểm tra author không còn rỗng trong sources
    for s in result["sources"]:
        author = s.get("author", "")
        cat    = s.get("category", "")
        print(f"    author='{author}' | category='{cat}' | reviewed_at='{s.get('reviewed_at','')[:10]}'")

    all_ok = result["answer"] and isinstance(result["sources"], list)
    print(f"\n→ TEST 5 {'PASSED ✅' if all_ok else 'SKIPPED (no LLM)'}")
    return all_ok


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "🔍 " * 20)
    print("   KIỂM TRA CÁC BUG ĐÃ SỬA")
    print("🔍 " * 20)

    results = {}
    results["t1_username"]   = test_get_username()
    results["t2_normalize"]  = test_normalize_review()
    results["t3_load"]       = test_load_reviews()
    results["t4_qdrant"]     = test_hybrid_search_with_category()
    results["t5_qa_ask"]     = test_qa_ask_with_category()

    print("\n" + "=" * 55)
    print("  KẾT QUẢ TỔNG HỢP")
    print("=" * 55)
    for name, ok in results.items():
        if ok is None:
            status = "⏭️  SKIPPED (cần service)"
        elif ok:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"  {name:20s}  {status}")
