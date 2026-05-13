"""
rag_pipeline/llm_client.py

LangChain ChatOpenAI client trỏ vào vLLM Docker server.
Model name và API key khớp với docker-compose.yml:
    served-model-name: llama-frog
    api-key: 1234

Khi dùng Cloudflare tunnel, set env:
    VLLM_BASE_URL=https://<random>.trycloudflare.com/v1
"""

from __future__ import annotations
from functools import lru_cache
# ✅ Fix: dùng langchain_openai thay langchain_community
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from rag_pipeline.config import (
    VLLM_BASE_URL,
    VLLM_MODEL_NAME,
    VLLM_API_KEY,
    VLLM_MAX_TOKENS,
    VLLM_TEMPERATURE,
)


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Singleton ChatOpenAI client – kết nối vLLM Docker server."""
    return ChatOpenAI(
        model       = VLLM_MODEL_NAME,  # "llama-frog" – khớp docker-compose
        api_key     = VLLM_API_KEY,     # "1234"
        base_url    = VLLM_BASE_URL,    # ✅ base_url thay openai_api_base (deprecated)
        max_tokens  = VLLM_MAX_TOKENS,
        temperature = VLLM_TEMPERATURE,
        streaming   = False,
    )


@lru_cache(maxsize=1)
def get_streaming_llm() -> BaseChatModel:
    """LLM client với streaming=True cho SSE endpoint."""
    return ChatOpenAI(
        model       = VLLM_MODEL_NAME,
        api_key     = VLLM_API_KEY,
        base_url    = VLLM_BASE_URL,
        max_tokens  = VLLM_MAX_TOKENS,
        temperature = VLLM_TEMPERATURE,
        streaming   = True,
    )
