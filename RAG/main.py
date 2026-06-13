"""
main.py – Entry point cho hệ thống hỏi-đáp reviews Shopee (RAG + LLM).

Modes:
    python main.py --mode index                         # Index mock_reviews.json vào Qdrant
    python main.py --mode index --file data/reviews.json
    python main.py --mode ask --question "Pin có bền không?"
    python main.py --mode ask --question "Giao hàng có nhanh không?" --product "Anker 10000mAh"
    python main.py --mode ask --question "Giao hàng có nhanh không?" --product "Anker 10000mAh" --category "Bách Hóa Online"

    python main.py --mode demo                          # Chạy 5 câu hỏi mẫu tự động

Yêu cầu:
    - VLLM Docker đang chạy:  docker compose up -d
    - Qdrant đang chạy:       docker run -p 6333:6333 qdrant/qdrant
    - Copy .env.example → .env và chỉnh sửa nếu cần
"""

import argparse

from loguru import logger

from qdrant_indexer import load_reviews, index_reviews

# ─── 5 câu hỏi mẫu người dùng hay hỏi khi xem reviews sản phẩm ──────────────

SAMPLE_QUESTIONS = [
    "Pin có bền không, dùng được bao lâu?",
    "Sản phẩm có hỗ trợ sạc nhanh 20W không?",
    "Shop có uy tín không, giao hàng có nhanh không?",
    "Hay bị lỗi không, nếu lỗi thì bảo hành có hỗ trợ không?",
    "So với giá tiền thì sản phẩm có đáng mua không?",
]


# ─── Modes ───────────────────────────────────────────────────────────────────

def run_index(reviews_file: str):
    """Bước 1: Index reviews vào Qdrant."""
    logger.info(f"[INDEX] Đọc reviews từ '{reviews_file}' ...")
    reviews = load_reviews(reviews_file)
    logger.info(f"[INDEX] {len(reviews)} reviews tìm thấy.")
    n = index_reviews(reviews)
    logger.success(f"[INDEX] Hoàn tất: {n} reviews đã index vào Qdrant.")


def run_ask(
    question:     str,
    product_name: str = "Sản phẩm",
    product_id:   int = 0,
    category:     str = "",
) -> dict:
    """Hỏi một câu hỏi tự do về sản phẩm dựa trên reviews."""
    from rag_pipeline import get_qa_chain

    logger.info("[Q&A] Khởi tạo ReviewQAChain ...")
    qa = get_qa_chain()

    logger.info(f"[Q&A] Câu hỏi: '{question}'")
    result = qa.ask(
        question     = question,
        product_id   = product_id,
        product_name = product_name,
        category = category,
    )

    _print_qa_result(result)
    return result


def run_demo(product_name: str = "Sạc dự phòng Anker 10000mAh"):
    """Demo: chạy 5 câu hỏi mẫu tự động để test pipeline Q&A."""
    from rag_pipeline import get_qa_chain

    logger.info("[DEMO] Khởi tạo ReviewQAChain ...")
    qa = get_qa_chain()

    print("\n" + "=" * 65)
    print(f"  DEMO Q&A – {product_name}")
    print(f"  {len(SAMPLE_QUESTIONS)} câu hỏi mẫu sẽ được hỏi tự động")
    print("=" * 65)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"\n❓ Câu hỏi {i}/{len(SAMPLE_QUESTIONS)}: {question}")
        print("-" * 65)

        result = qa.ask(question=question, product_name=product_name)

        print(f"💬 Trả lời:\n{result['answer']}")

        meta = result.get("pipeline_meta", {})
        print(
            f"\n📊 Pipeline: retrieved={meta.get('candidates_retrieved', 0)}, "
            f"reranked={meta.get('docs_reranked', 0)}, "
            f"to_llm={meta.get('docs_to_llm', 0)}"
        )

        if result.get("sources"):
            print("📎 Reviews sử dụng:")
            for s in result["sources"][:2]:  # Chỉ in 2 nguồn đầu
                score_str = f"  [rel={s['rerank_score']:.2f}]" if s.get("rerank_score") else ""
                print(f"   • {s['rating']}/5⭐ — {s['text'][:80]}...{score_str}")

        print("=" * 65)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _print_qa_result(result: dict):
    """In kết quả Q&A ra terminal theo format đẹp."""
    meta     = result.get("pipeline_meta", {})
    rejected = meta.get("rejected_reason")

    print("\n" + "=" * 65)
    print(f"  ❓ CÂU HỎI: {result.get('question', '')}")
    print("=" * 65)

    if rejected == "irrelevant_question":
        print(f"\n⚠️  KHÔNG LIÊN QUAN: Câu hỏi bị từ chối trước khi gọi LLM.")
        print(f"   (best rerank score = {meta.get('best_rerank_score', 0):.4f} < threshold)\n")

    print(f"\n💬 TRẢ LỜI:\n{result.get('answer', 'Không có câu trả lời.')}\n")

    sources = result.get("sources", [])
    if sources:
        print(f"📎 DỰA TRÊN {len(sources)} REVIEWS:")
        for i, s in enumerate(sources, 1):
            score_str = f" [rel={s['rerank_score']:.2f}]" if s.get("rerank_score") else ""
            print(f"  [{i}] {s['rating']}/5⭐ — {s['author']}{score_str}")
            print(f"       {s['text'][:100]}...")

    if not rejected:
        print(
            f"\n📊 Pipeline: retrieved={meta.get('candidates_retrieved', 0)}, "
            f"reranked={meta.get('docs_reranked', 0)}, "
            f"to_llm={meta.get('docs_to_llm', 0)}"
        )
    print("=" * 65)



# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RAG Q&A System – Hỏi đáp dựa trên reviews Shopee",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python main.py --mode index
  python main.py --mode ask --question "Pin có bền không?"
  python main.py --mode ask --question "Giao hàng nhanh không?" --product "Anker 10000mAh"
  python main.py --mode demo

5 câu hỏi mẫu (dùng trong demo mode):
  1. Pin có bền không, dùng được bao lâu?
  2. Sản phẩm có hỗ trợ sạc nhanh 20W không?
  3. Shop có uy tín không, giao hàng có nhanh không?
  4. Hay bị lỗi không, nếu lỗi thì bảo hành có hỗ trợ không?
  5. So với giá tiền thì sản phẩm có đáng mua không?
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["index", "ask", "demo"],
        default="demo",
        help="index: index data | ask: hỏi 1 câu | demo: chạy 5 câu hỏi mẫu",
    )
    parser.add_argument(
        "--file",
        default="mock_reviews.json",
        help="File reviews JSON để index (default: mock_reviews.json)",
    )
    parser.add_argument(
        "--question", "-q",
        default=SAMPLE_QUESTIONS[0],
        help=f'Câu hỏi cần hỏi (default: "{SAMPLE_QUESTIONS[0]}")',
    )
    parser.add_argument(
        "--product",
        default="Sản phẩm",
        help="Tên sản phẩm (dùng trong prompt, default: Sản phẩm)",
    )

    parser.add_argument(
        "--category",
        default="",
        help="Tên category để lọc ngữ cảnh khi hỏi, ví dụ: Bách Hóa Online",
    )

    parser.add_argument(
        "--product-id",
        type=int,
        default=0,
        help="Product ID nội bộ để filter trong Qdrant (default: 0 = tất cả)",
    )

    args = parser.parse_args()

    if args.mode == "index":
        run_index(args.file)
    elif args.mode == "ask":
        run_ask(
            question     = args.question,
            product_name = args.product,
            product_id   = args.product_id,
            category     = args.category,
        )
    elif args.mode == "demo":
        run_demo(product_name=args.product)


if __name__ == "__main__":
    main()
