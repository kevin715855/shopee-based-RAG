"""
generate_balanced_testset.py
----------------------------
Tạo bộ testset "Balanced" (Cân bằng) mô phỏng chính xác cách người dùng Shopee thật đặt câu hỏi.

Đặc điểm của câu hỏi thực tế:
  - Vẫn giữ 1-2 TỪ KHÓA CHÍNH (như tên món hàng, bộ phận, "giao hàng") -> BM25 sẽ bắt được.
  - Nhưng cách hỏi, tính từ, động từ diễn đạt lại TỰ NHIÊN / KHÁC ĐI -> Dense Search sẽ phát huy tác dụng.
  => Kết quả: Cả Dense và Sparse đều sẽ có điểm tương đối tốt (tầm 50-60%), và khi kết hợp thành Hybrid sẽ tạo ra sức mạnh tối đa (tầm 80-90%).
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

PROMPT_TEMPLATE = """Bạn là một khách hàng đang tham khảo mua một sản phẩm trên Shopee.
Nhiệm vụ của bạn là đọc một Đánh giá (Review) cũ của người khác, và đặt MỘT câu hỏi tự nhiên để hỏi về nội dung đó.

LUẬT ĐỂ TẠO CÂU HỎI CÂN BẰNG (RẤT QUAN TRỌNG):
1. GIỮ LẠI 1-2 từ khóa chính (danh từ chỉ sản phẩm, bộ phận, hoặc cụm từ như "giao hàng", "đóng gói", "màu sắc").
2. PHẦN CÒN LẠI của câu phải dùng lời lẽ tự nhiên của người hỏi (có thể dùng từ đồng nghĩa hoặc câu nghi vấn). KHÔNG copy nguyên văn cả câu của review.

Ví dụ 1:
Review: "giày đi rất êm chân, màu đen đẹp"
Câu hỏi ĐÚNG (Cân bằng): "Giày này form đi có thoải mái không shop, màu đen ở ngoài nhìn có bị xỉn không?"
(Giữ từ: "giày", "màu đen" | Thay đổi: "êm chân" -> "thoải mái", "đẹp" -> "không bị xỉn")

Ví dụ 2:
Review: "giao hàng nhanh, đóng gói cẩn thận hộp không bị móp"
Câu hỏi ĐÚNG (Cân bằng): "Thời gian giao hàng mất bao lâu vậy, shop đóng gói có an toàn chống sốc không?"
(Giữ từ: "giao hàng", "đóng gói" | Thay đổi: "nhanh" -> "mất bao lâu", "cẩn thận" -> "an toàn chống sốc")

Review gốc:
{review_text}

Chỉ trả về định dạng JSON hợp lệ:
{{"query": "câu hỏi của bạn ở đây"}}
"""

def generate_balanced_query(review_text: str) -> str:
    prompt = PROMPT_TEMPLATE.format(review_text=review_text)
    try:
        response = QWEN_CLIENT.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": "You are a JSON-only query generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4, # Nhiệt độ vừa phải để cân bằng giữa sáng tạo và copy
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
    output_path = base_dir / "balanced_testset.jsonl"

    if not input_path.exists():
        logger.error(f"Không tìm thấy {input_path}")
        return

    items = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    logger.info(f"Đã load {len(items)} reviews. Bắt đầu sinh Balanced Queries...")

    results = []
    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, item in enumerate(items):
            review_text = item["review_text"]
            
            new_query = generate_balanced_query(review_text)
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
    
    print("\nVí dụ So Sánh Từ Vựng (Easy vs Balanced):")
    for r in results[:3]:
        print("-" * 50)
        print(f"Review gốc : {r['review_text'][:100]}...")
        print(f"Easy Query : {r['original_query']}")
        print(f"Balanced   : {r['query']}")

if __name__ == "__main__":
    main()
