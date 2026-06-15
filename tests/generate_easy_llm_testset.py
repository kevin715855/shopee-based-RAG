"""
generate_easy_llm_testset.py
----------------------------
Tạo bộ testset "Dễ" bằng LLM, đảm bảo Recall @1 và @5 cực kỳ cao.

Chiến lược:
  - Dùng LLM đọc review gốc.
  - Ép LLM đặt câu hỏi BẰNG CÁCH SỬ DỤNG CHÍNH XÁC TỪ KHÓA của review.
  - LLM chỉ được phép chuyển câu khẳng định thành câu nghi vấn, cấm dùng từ đồng nghĩa.
  -> Lexical Overlap cao (Sparse BM25 làm tốt)
  -> Semantic Overlap cao (Dense làm tốt)
  => Hybrid Search sẽ đạt điểm tối đa.
"""

import sys
import json
import time
from pathlib import Path
from loguru import logger
from openai import OpenAI

QWEN_CLIENT = OpenAI(
    api_key="1234",
    base_url="http://localhost:11435/v1"
)
QWEN_MODEL = "qwen-research"

PROMPT_TEMPLATE = """Bạn là một hệ thống trích xuất thông tin tự động.
Nhiệm vụ của bạn là đọc một Đánh giá sản phẩm (Review) và tạo ra MỘT câu hỏi ngắn gọn.

LUẬT ĐỂ TẠO CÂU HỎI "DỄ" (RẤT QUAN TRỌNG):
1. Bạn PHẢI SỬ DỤNG CHÍNH XÁC các từ khóa (danh từ, tính từ, động từ) có trong Review gốc.
2. TUYỆT ĐỐI KHÔNG dùng từ đồng nghĩa. Hãy "nhặt" các từ trong review và ráp lại thành một câu hỏi.
3. Bản chất chỉ là chuyển câu khẳng định của người viết thành câu hỏi.

Ví dụ 1:
Review: "giày đi rất êm chân, màu đen đẹp, hộp nguyên vẹn"
Câu hỏi ĐÚNG: "Giày đi có êm chân không, màu đen có đẹp và hộp có nguyên vẹn không?"

Ví dụ 2:
Review: "giao hàng nhanh, đóng gói cẩn thận, hàng đúng mô tả"
Câu hỏi ĐÚNG: "Giao hàng có nhanh, đóng gói có cẩn thận và hàng có đúng mô tả không?"

Review gốc:
{review_text}

Chỉ trả về định dạng JSON hợp lệ:
{{"query": "câu hỏi của bạn ở đây"}}
"""

def generate_easy_llm_query(review_text: str) -> str:
    prompt = PROMPT_TEMPLATE.format(review_text=review_text)
    try:
        response = QWEN_CLIENT.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": "You are a JSON-only query generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, # Nhiệt độ rất thấp để LLM ngoan ngoãn copy từ gốc
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content).get("query", "")
    except Exception as e:
        logger.error(f"Lỗi API: {e}")
        return ""

def main():
    base_dir = Path(__file__).parent
    input_path = base_dir / "easy_testset.jsonl"
    output_path = base_dir / "easy_llm_testset.jsonl"

    if not input_path.exists():
        logger.error(f"Không tìm thấy {input_path}")
        return

    items = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    logger.info(f"Đã load {len(items)} reviews. Bắt đầu sinh Easy LLM Queries...")

    results = []
    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, item in enumerate(items):
            review_text = item["review_text"]
            
            new_query = generate_easy_llm_query(review_text)
            if not new_query:
                new_query = item["query"]

            new_item = item.copy()
            new_item["original_query"] = item["query"]
            new_item["query"] = new_query
            
            out_f.write(json.dumps(new_item, ensure_ascii=False) + "\n")
            out_f.flush()
            results.append(new_item)

            if (idx + 1) % 5 == 0:
                logger.info(f"Đã xử lý {idx + 1}/{len(items)}...")
                
            time.sleep(0.1)

    logger.success(f"Hoàn tất! Đã tạo {len(results)} queries và lưu tại {output_path}")
    
    print("\nVí dụ So Sánh Câu Hỏi:")
    for r in results[:3]:
        print("-" * 50)
        print(f"Review gốc     : {r['review_text'][:100]}...")
        print(f"Easy Extract   : {r['original_query']}")
        print(f"Easy LLM Query : {r['query']}")

if __name__ == "__main__":
    main()
