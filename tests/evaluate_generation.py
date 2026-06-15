import sys
import json
import traceback
from pathlib import Path
from loguru import logger
from openai import OpenAI

# Đảm bảo có thể import module từ thư mục cha (RAG)
sys.path.append(str(Path(__file__).parent.parent / "RAG"))

from rag_pipeline.qa_chain import get_qa_chain
from rag_pipeline.qdrant_store import get_store

# Khởi tạo client OpenAI kết nối với Qwen2.5 (11435) làm Giám khảo
QWEN_CLIENT = OpenAI(
    api_key="1234",
    base_url="http://localhost:11435/v1"
)
QWEN_MODEL = "qwen-research"

PROMPT_TEMPLATE = """Bạn là một giám khảo chuyên nghiệp đánh giá chất lượng hệ thống RAG.
Dựa trên [Câu hỏi], [Ngữ cảnh (Context)] và [Câu trả lời], hãy chấm điểm từ 1 đến 5 cho 3 tiêu chí:
1. Faithfulness: Câu trả lời có trung thực với Ngữ cảnh không? (Bịa đặt = 1 điểm, Dựa hoàn toàn vào ngữ cảnh = 5 điểm)
2. Relevance: Câu trả lời có đi thẳng vào trọng tâm Câu hỏi không? (Lan man = 1 điểm, Trực tiếp = 5 điểm)
3. Completeness: Câu trả lời có cung cấp đủ chi tiết từ Ngữ cảnh không? (Thiếu sót = 1 điểm, Đầy đủ = 5 điểm)

[Câu hỏi]
{query}

[Ngữ cảnh (Context)]
{context}

[Câu trả lời]
{answer}

Bạn BẮT BUỘC phải trả về định dạng JSON hợp lệ với cấu trúc sau:
{{
    "Faithfulness": int,
    "Relevance": int,
    "Completeness": int,
    "reasoning": "giải thích ngắn gọn tiếng Việt"
}}
Chỉ trả về chuỗi JSON, không giải thích gì thêm bên ngoài JSON.
"""

def get_judge_scores(query: str, context: str, answer: str) -> dict:
    """Gọi Qwen2.5 để chấm điểm"""
    prompt = PROMPT_TEMPLATE.format(query=query, context=context, answer=answer)
    
    try:
        response = QWEN_CLIENT.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful JSON-only assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Nhiệt độ thấp để output ổn định
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        logger.error(f"Lỗi khi gọi Qwen2.5: {e}")
        return {"Faithfulness": 1, "Relevance": 1, "Completeness": 1, "reasoning": "Lỗi API Giám khảo"}

def evaluate_generation():
    logger.info("Khởi tạo QA Chain (Llama-3B)...")
    qa_chain = get_qa_chain()
    
    # Rebuild BM25 if needed
    store = get_store()
    store._rebuild_bm25_from_qdrant()

    base_dir = Path(__file__).parent
    testset_path = base_dir / "synthetic_testset.jsonl"
    progress_path = base_dir / "synthetic_generation_progress.jsonl"

    if not testset_path.exists():
        logger.error(f"Không tìm thấy file testset: {testset_path}")
        return

    # 1. Đọc test set
    queries = []
    with open(testset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    
    total = len(queries)
    if total == 0:
        return

    # 2. Đọc checkpoints
    evaluated = {}
    if progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    evaluated[item["review_id"]] = item
                    
    logger.info(f"Đã load {len(evaluated)} kết quả từ checkpoint.")

    # 3. Tiến hành đánh giá
    new_results = 0
    with open(progress_path, "a", encoding="utf-8") as out_file:
        for idx, item in enumerate(queries):
            review_id = str(item["review_id"])
            if review_id in evaluated:
                continue

            query = item["query"]
            shopee_product_id = str(item["product_id"])
            
            logger.info(f"[{idx+1}/{total}] Đang xử lý: {query}")
            
            # BƯỚC 1: Gọi RAG để lấy câu trả lời (Dùng Llama-3B mặc định)
            rag_result = qa_chain.ask(
                question=query, 
                shopee_product_id=shopee_product_id
            )
            
            answer = rag_result["answer"]
            sources = rag_result.get("sources", [])
            
            # Xây dựng Context từ các sources
            context_text = "\n\n".join([f"Review {i+1}: {s['text']}" for i, s in enumerate(sources)])
            if not context_text.strip():
                context_text = "Không tìm thấy ngữ cảnh."
                
            # Nếu hệ thống từ chối trả lời do Relevance Score thấp
            if not sources:
                logger.warning("Hệ thống RAG không tìm thấy thông tin phù hợp, chấm điểm 1.")
                scores = {"Faithfulness": 1, "Relevance": 1, "Completeness": 1, "reasoning": "RAG không tìm ra context"}
            else:
                # BƯỚC 2: Gọi Giám khảo chấm điểm (Dùng Qwen2.5)
                scores = get_judge_scores(query, context_text, answer)
                
            # BƯỚC 3: Lưu lại
            result_item = {
                "review_id": review_id,
                "query": query,
                "context": context_text,
                "answer": answer,
                "scores": scores,
                "timing": rag_result.get("pipeline_meta", {})
            }
            
            out_file.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            out_file.flush()  # Ép ghi đĩa ngay lập tức
            
            evaluated[review_id] = result_item
            new_results += 1
            
            # Log nhanh kết quả
            logger.info(f"  -> Score: F:{scores.get('Faithfulness')} | R:{scores.get('Relevance')} | C:{scores.get('Completeness')}")

    # 4. In báo cáo tổng hợp
    f_sum, r_sum, c_sum = 0, 0, 0
    t_hybrid_sum, t_rerank_sum, t_llm_sum = 0, 0, 0
    valid_count = 0
    
    for item in evaluated.values():
        s = item["scores"]
        f_sum += s.get("Faithfulness", 1)
        r_sum += s.get("Relevance", 1)
        c_sum += s.get("Completeness", 1)
        
        timing = item.get("timing", {})
        t_hybrid_sum += timing.get("time_hybrid_s", 0)
        t_rerank_sum += timing.get("time_rerank_s", 0)
        t_llm_sum += timing.get("time_llm_s", 0)
        
        valid_count += 1
        
    print("\n" + "=" * 50)
    print(" 📊 KẾT QUẢ ĐÁNH GIÁ SINH VĂN BẢN (GENERATION)")
    print("=" * 50)
    print(f"Tổng số truy vấn đã chấm : {valid_count}/{total}")
    if valid_count > 0:
        print(f"Điểm trung bình (Thang điểm 5):")
        print(f" - Faithfulness (Độ trung thực) : {f_sum / valid_count:.2f} / 5.0")
        print(f" - Relevance (Độ liên quan)     : {r_sum / valid_count:.2f} / 5.0")
        print(f" - Completeness (Độ đầy đủ)     : {c_sum / valid_count:.2f} / 5.0")
        
        print("\n⏳ Thời gian chạy trung bình của Pipeline RAG:")
        print(f" - Hybrid Search (Retrieval) : {t_hybrid_sum / valid_count:.3f}s")
        print(f" - BGE-Reranker (Reranking)  : {t_rerank_sum / valid_count:.3f}s")
        print(f" - Llama-3B (LLM Generation) : {t_llm_sum / valid_count:.3f}s")
        print(f" - Tổng cộng (RAG Pipeline)  : {(t_hybrid_sum + t_rerank_sum + t_llm_sum) / valid_count:.3f}s")
    print("=" * 50)

if __name__ == "__main__":
    evaluate_generation()
