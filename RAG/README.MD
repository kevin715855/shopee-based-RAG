# RAG Pipeline – Shopee Reviews Analyzer

Pipeline phân tích đánh giá sản phẩm Shopee dùng VLLM + RAG.

---

## Stack

| Component | Model / Tool |
|-----------|-------------|
| LLM | `phamhai/Llama-3.2-3B-Instruct-Frog` qua **vLLM Docker** |
| Embedder | `BAAI/bge-m3` (dense, FP16) |
| Reranker | `BAAI/bge-reranker-v2-m3` (cross-encoder, FP16) |
| Vector DB | **Qdrant** (hybrid: Dense + BM25 + RRF) |
| Chain | **LangChain LCEL** |

---

## Cài đặt

```bash
pip install -r requirements.txt
cp .env.example .env   # chỉnh sửa nếu cần
```

---

## Khởi động server

### 1. Start VLLM (Docker)
```bash
docker compose up -d
```

### 2. Xem log – chờ "Application startup complete"
```bash
docker logs -f vllm_sopi
```

### 3. Kiểm tra server
```bash
curl http://localhost:8036/v1/models
```

### 4. Start Qdrant
```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

---

## Chạy pipeline

### Index data vào Qdrant
```bash
python qdrant_indexer.py                        # dùng mock_reviews.json
python qdrant_indexer.py --file reviews.json    # dùng file khác
```

### Demo full pipeline (index + query)
```bash
python main.py --mode demo
```

### Chỉ query (data đã index sẵn)
```bash
python main.py --mode query --product "Sạc dự phòng Anker 10000mAh"
```

### Test nhanh với LLM
```bash
python test.py
```

---

## Kết nối từ máy ngoài (Cloudflare Tunnel)

### Cài cloudflared (mỗi phiên chạy 1 lần)
```bash
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O ~/cloudflared
chmod +x ~/cloudflared
```

### Tạo tunnel
```bash
~/cloudflared tunnel --url http://localhost:8036
```
> Terminal sẽ in ra URL dạng `https://random-name.trycloudflare.com`

### Set URL cho client ngoài server
```bash
export VLLM_BASE_URL=https://random-name.trycloudflare.com/v1
python request.py
```

---

## Cấu trúc project

```
VLLM_shoppe/
├── docker-compose.yml        # VLLM server (port 8036, model llama-frog)
├── .env.example              # Template env vars
├── requirements.txt          # Python dependencies
├── main.py                   # Entry point: --mode index/query/demo
├── qdrant_indexer.py         # Index reviews vào Qdrant
├── request.py                # Test client (local hoặc Cloudflare)
├── test.py                   # Test nhanh gọi LLM trực tiếp
├── mock_reviews.json         # Dữ liệu mẫu
└── rag_pipeline/
    ├── __init__.py           # Public API: get_chain, get_store, ...
    ├── config.py             # Tất cả env vars tập trung tại đây
    ├── bge_embedder.py       # BGEM3Embeddings (LangChain-compatible)
    ├── bge_reranker.py       # BGEReranker (cross-encoder)
    ├── chunker.py            # Review preprocessor + sentence chunker
    ├── qdrant_store.py       # Vector store (Dense + BM25 hybrid)
    ├── llm_client.py         # ChatOpenAI → vLLM Docker
    ├── summary_prompts.py    # LangChain prompt templates
    ├── review_chain.py       # LCEL RAG chain (invoke + astream)
    └── vllm_server.py        # [Tham khảo] launch vLLM không qua Docker
```

---

## Endpoints vLLM

| Endpoint | Mô tả |
|----------|-------|
| `GET  /v1/models` | Xem model đang chạy |
| `POST /v1/chat/completions` | Chat/inference |
| `GET  /health` | Kiểm tra server còn sống |