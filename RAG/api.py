"""
api.py  –  FastAPI server cho hệ thống RAG Q&A Reviews Shopee.

Endpoints:
    GET  /health                   – kiểm tra server + Qdrant + vLLM
    POST /ask                      – hỏi đáp (sync, trả JSON)
    POST /ask/stream               – hỏi đáp (async streaming, SSE)
    POST /index                    – index reviews mới vào Qdrant (background)
    GET  /index/status             – xem trạng thái job index đang chạy
    GET  /collection/info          – thống kê Qdrant collection

Chạy:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Hoặc qua docker compose (xem docker-compose.yml).
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field
import pandas as pd
from pathlib import Path

# ── Lifespan: khởi tạo singleton khi app start ─────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model 1 lần khi startup, giải phóng khi shutdown."""
    logger.info("🚀 Khởi tạo RAG pipeline ...")
    try:
        from rag_pipeline import get_qa_chain, get_store
        app.state.qa   = get_qa_chain()   # load embedder + reranker + llm client
        app.state.store = get_store()
        logger.success("✅ RAG pipeline sẵn sàng.")
    except Exception as e:
        logger.error(f"❌ Lỗi khởi tạo pipeline: {e}")
        raise
    yield
    logger.info("🛑 Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Shopee RAG Q&A API",
    description = "Hệ thống hỏi-đáp dựa trên reviews Shopee (RAG + vLLM)",
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # Đổi thành domain cụ thể khi production
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Pydantic schemas ───────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question:          str   = Field(...,          description="Câu hỏi của người dùng")
    product_name:      str   = Field("Sản phẩm",  description="Tên sản phẩm (dùng trong prompt)")
    shopee_product_id: Optional[str] = Field(None, description="Shopee Product ID để filter")
    category:          Optional[str] = Field(None, description="Danh mục sản phẩm để filter")
    rating_min:        Optional[int] = Field(None, ge=1, le=5, description="Rating tối thiểu (1-5)")
    rating_max:        Optional[int] = Field(None, ge=1, le=5, description="Rating tối đa (1-5)")
    sentiment:         Optional[str] = Field(None, pattern="^(positive|negative|neutral)$",
                                             description="Lọc theo cảm xúc: positive/negative/neutral")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question":          "Pin có bền không, dùng được bao lâu?",
                "product_name":      "Sạc dự phòng Anker 10000mAh",
                "shopee_product_id": "10299503780",
                "category":          "Thiết Bị Điện Tử",
            }
        }
    }


class SourceItem(BaseModel):
    text:         str
    rating:       int
    sentiment:    str
    rerank_score: float
    author:       str
    reviewed_at:  str
    review_url:   str
    category:     str


class AskResponse(BaseModel):
    answer:        str
    question:      str
    sources:       list[SourceItem]
    pipeline_meta: dict
    latency_ms:    float


class IndexRequest(BaseModel):
    file_path: str = Field(
        "shopee/reviews",
        description="Đường dẫn file/folder JSON hoặc JSONL để index"
    )

    model_config = {
        "json_schema_extra": {
            "example": {"file_path": "shopee/reviews"}
        }
    }


class IndexResponse(BaseModel):
    job_id:  str
    status:  str
    message: str


# ── In-memory job tracker ──────────────────────────────────────────────────

_index_jobs: dict[str, dict] = {}   # job_id → {status, started_at, result, error}


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/metadata", tags=["System"])
async def get_metadata():
    """
    Trả về danh sách Category và Product từ file CSV đã xử lý.
    Phục vụ cho Frontend tạo Dropdown menu cho user chọn trước khi chat.
    """
    products_path = Path("data/processed/products.csv")
    
    if not products_path.exists():
        return {"categories": [], "products": []}
        
    try:
        # Đọc dữ liệu sản phẩm
        df = pd.read_csv(products_path)
        
        # Lấy danh sách các danh mục (Categories) duy nhất
        categories = df['category'].dropna().unique().tolist()
        
        products = df[['product_id', 'name', 'category', 'image_url', 'avg_rating']].fillna("").to_dict(orient="records")
        
        return {
            "categories": categories,
            "products": products
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đọc dữ liệu sản phẩm: {e}")

@app.get("/health", tags=["System"])
async def health_check():
    """Kiểm tra trạng thái server, Qdrant và vLLM."""
    results: dict = {"api": "ok", "timestamp": datetime.now().isoformat()}

    # Qdrant
    try:
        info = app.state.store.collection_info()
        results["qdrant"] = {"status": "ok", "total_points": info["total_points"]}
    except Exception as e:
        results["qdrant"] = {"status": "error", "detail": str(e)}

    # vLLM (thử gọi models endpoint)
    try:
        import httpx
        from rag_pipeline.config import VLLM_BASE_URL, VLLM_API_KEY
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(
                f"{VLLM_BASE_URL.rstrip('/v1')}/v1/models",
                headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
            )
        results["vllm"] = {"status": "ok", "models": [m["id"] for m in r.json().get("data", [])]}
    except Exception as e:
        results["vllm"] = {"status": "error", "detail": str(e)}

    overall = "ok" if all(v.get("status") == "ok" for v in results.values() if isinstance(v, dict)) else "degraded"
    results["overall"] = overall
    return results


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
async def ask(req: AskRequest):
    """
    Hỏi đáp dựa trên reviews Shopee.

    Trả về câu trả lời từ LLM cùng danh sách reviews đã dùng.
    """
    import time
    t0 = time.perf_counter()

    try:
        result = await asyncio.to_thread(
            app.state.qa.ask,
            question          = req.question,
            product_name      = req.product_name,
            shopee_product_id = req.shopee_product_id,
            category          = req.category,
            rating_min        = req.rating_min,
            rating_max        = req.rating_max,
            sentiment         = req.sentiment,
        )
    except Exception as e:
        logger.exception(f"Lỗi ask(): {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.perf_counter() - t0) * 1000

    # Normalize sources để khớp SourceItem schema
    sources = [
        SourceItem(
            text         = s.get("text", ""),
            rating       = s.get("rating", 0),
            sentiment    = s.get("sentiment", "neutral"),
            rerank_score = s.get("rerank_score", 0.0),
            author       = s.get("author", "Ẩn danh"),
            reviewed_at  = s.get("reviewed_at", ""),
            review_url   = s.get("review_url", ""),
            category     = s.get("category", ""),
        )
        for s in result.get("sources", [])
    ]

    return AskResponse(
        answer        = result["answer"],
        question      = result["question"],
        sources       = sources,
        pipeline_meta = result.get("pipeline_meta", {}),
        latency_ms    = round(latency_ms, 1),
    )


@app.post("/ask/stream", tags=["Q&A"])
async def ask_stream(req: AskRequest):
    """
    Hỏi đáp với streaming (Server-Sent Events).

    Dùng cho UI real-time, chatbot. Token trả về từng phần.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for chunk in app.state.qa.aask(
                question          = req.question,
                product_name      = req.product_name,
                shopee_product_id = req.shopee_product_id,
                category          = req.category,
            ):
                # SSE format
                yield f"data: {chunk}\n\n"
        except Exception as e:
            logger.exception(f"Lỗi aask(): {e}")
            yield f"data: [ERROR] {e}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",       # tắt Nginx buffering
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.post("/index", response_model=IndexResponse, tags=["Index"])
async def index_reviews(req: IndexRequest, background_tasks: BackgroundTasks):
    """
    Index reviews mới vào Qdrant (chạy background, không block).

    - Nếu collection chưa có → tạo mới rồi index.
    - Nếu đã có → upsert (deterministic UUID → không trùng lặp).
    - Dùng GET /index/status/{job_id} để theo dõi tiến độ.
    """
    job_id = str(uuid.uuid4())
    _index_jobs[job_id] = {
        "status":     "queued",
        "started_at": datetime.now().isoformat(),
        "file_path":  req.file_path,
        "result":     None,
        "error":      None,
    }

    async def _run_index():
        _index_jobs[job_id]["status"] = "running"
        try:
            from qdrant_indexer import load_reviews, index_reviews as _index
            reviews = await asyncio.to_thread(load_reviews, req.file_path)
            n       = await asyncio.to_thread(_index, reviews)
            _index_jobs[job_id].update({
                "status":      "done",
                "result":      {"indexed": n, "total_loaded": len(reviews)},
                "finished_at": datetime.now().isoformat(),
            })
            logger.success(f"[INDEX job={job_id}] Indexed {n} reviews from '{req.file_path}'")
        except Exception as e:
            _index_jobs[job_id].update({"status": "error", "error": str(e)})
            logger.error(f"[INDEX job={job_id}] Error: {e}")

    background_tasks.add_task(_run_index)

    return IndexResponse(
        job_id  = job_id,
        status  = "queued",
        message = f"Job {job_id} đã được khởi động. Theo dõi tại GET /index/status/{job_id}",
    )


@app.get("/index/status/{job_id}", tags=["Index"])
async def index_status(job_id: str):
    """Xem trạng thái job index đang chạy."""
    job = _index_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' không tồn tại.")
    return {"job_id": job_id, **job}


@app.get("/index/jobs", tags=["Index"])
async def list_index_jobs():
    """Liệt kê tất cả jobs index (max 20 gần nhất)."""
    jobs = list(_index_jobs.items())[-20:]
    return {"jobs": [{"job_id": k, **v} for k, v in jobs]}


@app.get("/collection/info", tags=["System"])
async def collection_info():
    """Thống kê Qdrant collection (số points đã index, tên collection...)."""
    try:
        return app.state.store.collection_info()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Không kết nối được Qdrant: {e}")
