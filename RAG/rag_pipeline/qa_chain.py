"""
rag_pipeline/qa_chain.py

ReviewQAChain – hệ thống hỏi-đáp tự do dựa trên reviews Shopee.

Luồng xử lý:
    Câu hỏi của người dùng
         │
    [QdrantReviewStore.hybrid_search]  ← top-20 reviews liên quan nhất
         │
    [BGEReranker.rerank]               ← top-5 reviews chính xác nhất
         │
    [qa_prompt | llm]                  ← LLM trả lời tự nhiên
         │
    Câu trả lời (plain text) + metadata

Điểm khác biệt so với ReviewSummaryChain cũ:
- Input: câu hỏi tự do (bất kỳ) thay vì product_id cố định
- Output: câu trả lời ngôn ngữ tự nhiên thay vì JSON format cứng
- Linh hoạt với mọi loại câu hỏi: pin, ship, lỗi, giá, v.v.
"""

from __future__ import annotations
from typing import Optional, AsyncGenerator

from loguru import logger
from langchain_core.output_parsers import StrOutputParser

from rag_pipeline.qdrant_store import get_store
from rag_pipeline.bge_reranker import get_reranker
from rag_pipeline.llm_client import get_llm, get_streaming_llm
from rag_pipeline.qa_prompts import qa_prompt, format_reviews_for_qa
from rag_pipeline.config import RETRIEVAL_TOP_K, RERANKER_TOP_K

# Số reviews tối đa đưa vào LLM
# Tăng lên 5 (cũ là 3) vì Q&A cần nhiều góc nhìn hơn tóm tắt
QA_MAX_REVIEWS = 5

# Ngưỡng relevance score tối thiểu từ reranker.
# Nếu review liên quan nhất có score < threshold → câu hỏi không liên quan đến sản phẩm.
# Dựa trên quan sát thực tế:
#   - Câu hỏi liên quan ("Sạc nhanh 20W?"):      best ≈ 0.976
#   - Câu hỏi liên quan ("Hay bị lỗi không?"):   best ≈ 0.018
#   - Câu hỏi KHÔNG liên quan ("Thành có gay?"): best ≈ 0.000
RELEVANCE_THRESHOLD = 0.005


class ReviewQAChain:
    """
    RAG Q&A chain cho hệ thống hỏi-đáp reviews Shopee.

    Attributes:
        store    : QdrantReviewStore – vector DB
        reranker : BGEReranker – cross-encoder reranker
        llm      : ChatOpenAI trỏ vào vLLM Docker server
    """

    def __init__(self):
        self.store    = get_store()
        self.reranker = get_reranker()
        self.llm      = get_llm()
        self._chain   = qa_prompt | self.llm | StrOutputParser()

    # ─── Public API ──────────────────────────────────────────────────────────

    def ask(
        self,
        question:    str,
        product_id:  int              = 0,
        product_name: str             = "Sản phẩm",
        shopee_product_id: Optional[str] = None,
        rating_min:  Optional[int]    = None,
        rating_max:  Optional[int]    = None,
        sentiment:   Optional[str]    = None,
    ) -> dict:
        """
        Trả lời câu hỏi tự do dựa trên reviews đã index.

        Args:
            question         : câu hỏi của người dùng (bất kỳ)
            product_id       : ID nội bộ của sản phẩm (0 = tìm toàn bộ)
            product_name     : tên sản phẩm (dùng trong prompt)
            shopee_product_id: ID Shopee để filter
            rating_min/max   : lọc theo rating
            sentiment        : lọc theo sentiment ("positive"/"negative"/"neutral")

        Returns:
            dict với keys:
                answer         : câu trả lời của LLM (plain text)
                question       : câu hỏi gốc
                sources        : list reviews đã dùng để trả lời
                pipeline_meta  : thông tin pipeline (candidates, reranked, ...)
        """
        logger.info(f"[Q&A] Câu hỏi: '{question}'")

        # Step 1: Hybrid search – dùng câu hỏi làm query
        candidates = self.store.hybrid_search(
            query             = question,
            product_id        = product_id if product_id else None,
            shopee_product_id = shopee_product_id,
            rating_min        = rating_min,
            rating_max        = rating_max,
            sentiment         = sentiment,
            top_k             = RETRIEVAL_TOP_K,
        )
        logger.info(f"[Q&A] Hybrid search → {len(candidates)} candidates")

        if not candidates:
            return self._no_data_response(question, product_name)

        # Step 2: Rerank – tìm reviews liên quan nhất với câu hỏi
        top_docs = self.reranker.rerank(
            query     = question,
            documents = candidates,
            top_k     = RERANKER_TOP_K,
        )
        logger.info(f"[Q&A] Reranked → top-{len(top_docs)}")

        # Kiểm tra relevance: nếu score cao nhất vẫn dưới ngưỡng →
        # câu hỏi không liên quan đến sản phẩm, từ chối trước khi gọi LLM
        best_score = top_docs[0].get("rerank_score", 0) if top_docs else 0
        logger.debug(f"[Q&A] Best rerank score = {best_score:.4f} (threshold={RELEVANCE_THRESHOLD})")

        if best_score < RELEVANCE_THRESHOLD:
            logger.warning(f"[Q&A] Câu hỏi không liên quan (score={best_score:.4f}): '{question}'")
            return self._irrelevant_question_response(question, product_name)

        # Step 3: LLM trả lời
        chain_input = {
            "product_name": product_name,
            "question":     question,
            "reviews_text": format_reviews_for_qa(top_docs, QA_MAX_REVIEWS),
        }

        answer = self._chain.invoke(chain_input)
        logger.success(f"[Q&A] Trả lời xong ({len(answer)} ký tự)")

        return {
            "answer":   answer.strip(),
            "question": question,
            "sources":  [
                {
                    "text":         d["text"][:150] + "..." if len(d["text"]) > 150 else d["text"],
                    "rating":       d.get("rating", 0),
                    "sentiment":    d.get("sentiment", "neutral"),
                    "rerank_score": round(d.get("rerank_score", 0), 3),
                    "author":       d.get("author") or d.get("user_name", "Ẩn danh"),
                }
                for d in top_docs[:QA_MAX_REVIEWS]
            ],
            "pipeline_meta": {
                "candidates_retrieved": len(candidates),
                "docs_reranked":        len(top_docs),
                "docs_to_llm":          min(len(top_docs), QA_MAX_REVIEWS),
                "query_used":           question,
                "product_id":           product_id,
            },
        }

    async def aask(
        self,
        question:     str,
        product_id:   int = 0,
        product_name: str = "Sản phẩm",
    ) -> AsyncGenerator[str, None]:
        """Async streaming – dùng cho SSE / real-time UI."""
        candidates = self.store.hybrid_search(
            query      = question,
            product_id = product_id if product_id else None,
            top_k      = RETRIEVAL_TOP_K,
        )
        if not candidates:
            yield "Xin lỗi, không tìm thấy reviews liên quan đến câu hỏi của bạn."
            return

        top_docs = self.reranker.rerank(question, candidates, RERANKER_TOP_K)

        chain_input = {
            "product_name": product_name,
            "question":     question,
            "reviews_text": format_reviews_for_qa(top_docs, QA_MAX_REVIEWS),
        }

        streaming_chain = qa_prompt | get_streaming_llm() | StrOutputParser()
        async for chunk in streaming_chain.astream(chain_input):
            yield chunk

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _irrelevant_question_response(question: str, product_name: str) -> dict:
        """Trả về khi câu hỏi không liên quan đến reviews sản phẩm."""
        return {
            "answer":   (
                f"Xin lỗi, câu hỏi '{question}' không liên quan đến các reviews của sản phẩm '{product_name}'. "
                f"Tôi chỉ có thể trả lời các câu hỏi về chất lượng, tính năng, giao hàng, "
                f"bảo hành hoặc trải nghiệm mua sắm dựa trên reviews thực tế của khách hàng."
            ),
            "question":      question,
            "sources":       [],
            "pipeline_meta": {
                "candidates_retrieved": 0,
                "rejected_reason":      "irrelevant_question",
                "best_rerank_score":    0.0,
            },
        }

    @staticmethod
    def _no_data_response(question: str, product_name: str) -> dict:
        return {
            "answer":   (
                f"Xin lỗi, tôi không tìm thấy reviews nào liên quan đến câu hỏi "
                f"'{question}' cho sản phẩm '{product_name}'. "
                f"Vui lòng thử lại với từ khóa khác."
            ),
            "question":      question,
            "sources":       [],
            "pipeline_meta": {"candidates_retrieved": 0},
        }


# ─── Singleton ────────────────────────────────────────────────────────────────

_qa_instance: ReviewQAChain | None = None


def get_qa_chain() -> ReviewQAChain:
    global _qa_instance
    if _qa_instance is None:
        _qa_instance = ReviewQAChain()
    return _qa_instance
