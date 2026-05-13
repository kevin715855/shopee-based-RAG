"""
rag_pipeline – Hệ thống hỏi-đáp (Q&A) dựa trên reviews Shopee.

Stack:
  - Embed   : BAAI/bge-m3 (dense, FP16)
  - Rerank  : BAAI/bge-reranker-v2-m3 (cross-encoder, FP16)
  - Vector DB : Qdrant (hybrid: dense + BM25 + RRF)
  - LLM     : phamhai/Llama-3.2-3B-Instruct-Frog qua vLLM Docker
  - Chain   : LangChain LCEL

Usage:
    from rag_pipeline import get_qa_chain

    qa = get_qa_chain()

    # Hỏi bất kỳ câu hỏi nào
    result = qa.ask(
        question="Pin có bền không, dùng được bao lâu?",
        product_id=0,
        product_name="Sạc dự phòng Anker 10000mAh",
    )
    print(result["answer"])
    print(result["sources"])  # reviews đã dùng để trả lời
"""

from rag_pipeline.bge_embedder import get_embedder, BGEM3Embeddings
from rag_pipeline.bge_reranker import get_reranker, BGEReranker
from rag_pipeline.qdrant_store import get_store, QdrantReviewStore
from rag_pipeline.qa_chain import get_qa_chain, ReviewQAChain

__all__ = [
    "get_embedder",
    "get_reranker",
    "get_store",
    "get_qa_chain",
    "BGEM3Embeddings",
    "BGEReranker",
    "QdrantReviewStore",
    "ReviewQAChain",
]
