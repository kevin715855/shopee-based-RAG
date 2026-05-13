"""
rag_pipeline/bge_embedder.py

LangChain-compatible embedding wrapper cho BAAI/bge-m3.
bge-m3 hỗ trợ 3 loại retrieval: dense, sparse, colbert.
Ở đây dùng dense (Qdrant compatible) + sparse (BM25-style).

FP16 để tối ưu VRAM khi dùng GPU.
"""

from __future__ import annotations
from typing import List
from loguru import logger
from langchain_core.embeddings import Embeddings

from rag_pipeline.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    EMBEDDING_USE_FP16,
    EMBEDDING_BATCH_SIZE,
)


class BGEM3Embeddings(Embeddings):
    """
    LangChain Embeddings adapter cho bge-m3.
    Implement embed_documents() và embed_query() theo interface của LangChain.

    Lazy-load model khi lần đầu gọi (tránh load khi import).
    """

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        logger.info(
            f"Loading bge-m3: {EMBEDDING_MODEL} | "
            f"device={EMBEDDING_DEVICE} | fp16={EMBEDDING_USE_FP16}"
        )
        from FlagEmbedding import BGEM3FlagModel
        self._model = BGEM3FlagModel(
            EMBEDDING_MODEL,
            use_fp16=EMBEDDING_USE_FP16,
            device=EMBEDDING_DEVICE,
        )
        logger.success("bge-m3 loaded ✓")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed list of documents (reviews/chunks). Không dùng instruction prefix."""
        self._load()
        output = self._model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return output["dense_vecs"].tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed query. bge-m3 khuyến nghị instruction prefix cho query
        để tăng retrieval accuracy.
        """
        self._load()
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        output = self._model.encode(
            [prefixed],
            batch_size=1,
            max_length=256,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return output["dense_vecs"][0].tolist()

    def embed_sparse(self, texts: List[str]) -> List[dict]:
        """
        Sparse embedding (lexical weights) từ bge-m3.
        Dùng để kết hợp với dense trong hybrid search.
        Returns list of {token_id: weight} dicts.
        """
        self._load()
        output = self._model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return output["lexical_weights"]


# Singleton để tái sử dụng
_embedder_instance: BGEM3Embeddings | None = None


def get_embedder() -> BGEM3Embeddings:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = BGEM3Embeddings()
    return _embedder_instance
