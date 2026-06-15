import sys
import json
from pathlib import Path
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent / "RAG"))

from rag_pipeline.qdrant_store import get_store
from rag_pipeline.bge_reranker import get_reranker
TESTSET_FILE = "easy_llm_testset.jsonl"

def compute_metrics(results_per_query: list[dict], top_ks: list[int] = [1, 5, 10]) -> dict:
    """
    Tính Hit Rate@k và MRR từ list kết quả.

    Args:
        results_per_query: list of {"expected": str, "ranked_ids": list[str]}
        top_ks: các mức k cần tính (ví dụ: [1, 5, 10])
    Returns:
        dict với hits_at_k cho từng k, mrr, failed
    """
    hits = {k: 0 for k in top_ks}
    mrr_sum = 0.0
    failed = []
    max_k = max(top_ks)

    for r in results_per_query:
        expected = r["expected"]
        ranked   = r["ranked_ids"][:max_k]
        query    = r["query"]

        hit_rank = 0
        for rank, rid in enumerate(ranked, start=1):
            if rid == expected:
                hit_rank = rank
                break

        for k in top_ks:
            if 0 < hit_rank <= k:
                hits[k] += 1

        if hit_rank > 0:
            mrr_sum += 1.0 / hit_rank
        else:
            failed.append(query)

    total = len(results_per_query)
    return {
        "total":   total,
        "hits":    hits,       # {1: count, 5: count, 10: count}
        "top_ks":  top_ks,
        "mrr":     mrr_sum / total if total else 0,
        "failed":  failed,
    }


def print_report(title: str, m: dict):
    total  = m["total"]
    top_ks = m["top_ks"]
    hits   = m["hits"]
    print("\n" + "=" * 58)
    print(f"  📊 {title}")
    print("=" * 58)
    print(f"  Tổng số truy vấn      : {total}")
    for k in top_ks:
        h = hits[k]
        print(f"  Hit Rate @ {k:<2}         : {h / total:.2%}  ({h}/{total})")
    print(f"  MRR                   : {m['mrr']:.4f}")
    print("=" * 58)
    if m["failed"]:
        max_k = max(top_ks)
        print(f"  ⚠  {len(m['failed'])} câu không tìm được trong top {max_k}:")
        for q in m["failed"][:3]:
            print(f"     - {q}")


def evaluate():
    logger.info("Khởi tạo Qdrant Store...")
    store = get_store()

    logger.info("Khởi tạo Reranker...")
    reranker = get_reranker()

    logger.info("Rebuild BM25 corpus từ Qdrant...")
    store._rebuild_bm25_from_qdrant()

    testset_path = Path(__file__).parent / TESTSET_FILE
    if not testset_path.exists():
        logger.error(f"Không tìm thấy file testset: {testset_path}")
        return

    queries = []
    with open(testset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    total = len(queries)
    if total == 0:
        logger.error("Test set trống!")
        return

    logger.info(f"Bắt đầu đánh giá {total} câu hỏi — Ablation: Dense / Sparse / Hybrid / +Reranker...")

    dense_records    = []  # Dense-only
    sparse_records   = []  # Sparse/BM25-only
    hybrid_records   = []  # Hybrid (Dense + BM25) top 10
    reranked_records = []  # Hybrid top 20 → Reranker top 10

    for idx, item in enumerate(queries):
        query    = item["query"]
        expected = str(item["review_id"])
        pid      = str(item["product_id"])

        # ── Dense-only ───────────────────────────────────────────────────────
        dense_docs = store.dense_search(query=query, top_k=10, shopee_product_id=pid)
        dense_records.append({
            "query": query, "expected": expected,
            "ranked_ids": [str(d.get("review_id", "")) for d in dense_docs],
        })

        # ── Sparse-only (BM25) ───────────────────────────────────────────────
        sparse_docs = store.sparse_search(query=query, top_k=10, shopee_product_id=pid)
        sparse_records.append({
            "query": query, "expected": expected,
            "ranked_ids": [str(d.get("review_id", "")) for d in sparse_docs],
        })

        # ── Hybrid (Dense + BM25 via RRF), top 20 ───────────────────────────
        candidates = store.hybrid_search(
            query=query, 
            top_k=20, 
            shopee_product_id=pid,
            alpha=0.5  # Thử nghiệm 0.5 Dense / 0.5 Sparse
        )
        hybrid_records.append({
            "query": query, "expected": expected,
            "ranked_ids": [str(d.get("review_id", "")) for d in candidates[:10]],
        })

        # ── Hybrid top 20 → Reranker top 10 ─────────────────────────────────
        reranked = reranker.rerank(query=query, documents=candidates, top_k=10)
        reranked_records.append({
            "query": query, "expected": expected,
            "ranked_ids": [str(d.get("review_id", "")) for d in reranked],
        })

        if (idx + 1) % 10 == 0:
            logger.info(f"  [{idx + 1}/{total}] done...")

    # ── In kết quả so sánh ────────────────────────────────────────────────────
    TOP_KS = [1, 5, 10]
    m_dense    = compute_metrics(dense_records,    top_ks=TOP_KS)
    m_sparse   = compute_metrics(sparse_records,   top_ks=TOP_KS)
    m_hybrid   = compute_metrics(hybrid_records,   top_ks=TOP_KS)
    m_reranked = compute_metrics(reranked_records, top_ks=TOP_KS)

    print_report("ABLATION — Dense Search Only (BAAI/bge-m3)", m_dense)
    print_report("ABLATION — Sparse Search Only (BM25)", m_sparse)
    print_report("STAGE 1 — Hybrid Search (Dense + Sparse RRF)", m_hybrid)
    print_report("STAGE 2 — Hybrid Search + Reranker (Top-20 → Rerank → Top-10)", m_reranked)

    # ── Delta (mức cải thiện) ─────────────────────────────────────────────────
    n = m_hybrid["total"]
    sign = lambda x: "+" if x >= 0 else ""
    print("\n" + "─" * 58)
    print("  📈 MỨC CẢI THIỆN KHI THÊM RERANKER")
    print("─" * 58)
    for k in TOP_KS:
        delta = (m_reranked["hits"][k] - m_hybrid["hits"][k]) / n
        print(f"  ΔHit Rate @ {k:<2}: {sign(delta)}{delta:.2%}")
    delta_mrr = m_reranked["mrr"] - m_hybrid["mrr"]
    print(f"  ΔMRR        : {sign(delta_mrr)}{delta_mrr:.4f}")
    print("─" * 58)


if __name__ == "__main__":
    evaluate()
