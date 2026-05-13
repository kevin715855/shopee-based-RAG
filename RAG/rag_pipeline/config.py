"""
rag_pipeline/config.py
----------------------
Nguồn duy nhất cho tất cả config của pipeline.
Đọc từ biến môi trường, fallback về giá trị mặc định khớp với docker-compose.yml.

Tạo file .env.example (hoặc copy sang .env) để ghi đè:
    cp .env.example .env
"""

from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

# ── VLLM Server ────────────────────────────────────────────────────────────────
# Khớp với docker-compose.yml: port 8036, model llama-frog, api-key 1234
# Nếu dùng Cloudflare tunnel, set VLLM_BASE_URL=https://<random>.trycloudflare.com/v1
VLLM_BASE_URL   : str   = os.getenv("VLLM_BASE_URL",   "http://localhost:8036/v1")
VLLM_MODEL_NAME : str   = os.getenv("VLLM_MODEL_NAME", "llama-frog")
VLLM_API_KEY    : str   = os.getenv("VLLM_API_KEY",    "1234")
VLLM_MAX_TOKENS : int   = int(os.getenv("VLLM_MAX_TOKENS",  "512"))
VLLM_TEMPERATURE: float = float(os.getenv("VLLM_TEMPERATURE", "0.3"))

# ── Qdrant Vector DB ───────────────────────────────────────────────────────────
QDRANT_HOST     : str = os.getenv("QDRANT_HOST",       "localhost")
QDRANT_PORT     : int = int(os.getenv("QDRANT_PORT",   "6333"))
COLLECTION_NAME : str = os.getenv("QDRANT_COLLECTION", "shopee_reviews")
EMBEDDING_DIM   : int = int(os.getenv("EMBEDDING_DIM", "1024"))
RETRIEVAL_TOP_K : int = int(os.getenv("RETRIEVAL_TOP_K", "20"))

# ── BGE Embedder (bge-m3) ──────────────────────────────────────────────────────
EMBEDDING_MODEL  : str  = os.getenv("EMBEDDING_MODEL",      "BAAI/bge-m3")
EMBEDDING_DEVICE : str  = os.getenv("EMBEDDING_DEVICE",     "cuda")
EMBEDDING_USE_FP16: bool = os.getenv("EMBEDDING_USE_FP16", "true").lower() == "true"
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# ── BGE Reranker ───────────────────────────────────────────────────────────────
RERANKER_MODEL : str  = os.getenv("RERANKER_MODEL",    "BAAI/bge-reranker-v2-m3")
RERANKER_USE_FP16: bool = os.getenv("RERANKER_USE_FP16", "true").lower() == "true"
RERANKER_TOP_K : int  = int(os.getenv("RERANKER_TOP_K", "5"))
