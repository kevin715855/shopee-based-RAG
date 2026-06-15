"""
generate_synthetic_testset.py
------------------------------
Step 1: Sample 75 reviews evenly across all product files in RAG/data/reviews/
Step 2: For each review, call Qwen2.5 (vLLM OpenAI-compatible API, port 11435)
        to generate 1 question that the review can answer.
Step 3: Save ground truth as JSONL: {query, review_id, product_id, category, review_text}

Usage:
    python tests/generate_synthetic_testset.py
"""

import json
import os
import random
import time
import sys
from pathlib import Path
from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────────
REVIEWS_DIR = Path(__file__).parent.parent / "RAG" / "data" / "reviews"
OUTPUT_FILE = Path(__file__).parent / "synthetic_testset.jsonl"
PROGRESS_FILE = Path(__file__).parent / "synthetic_testset_progress.jsonl"

LLM_BASE_URL = "http://localhost:11435/v1"
LLM_MODEL    = "qwen-research"
LLM_TIMEOUT  = 120          # seconds per request

TARGET_TOTAL      = 75      # total reviews to sample
REVIEWS_PER_FILE  = 1       # reviews to sample per product file (may take 2 for large files)
MIN_TEXT_LEN      = 60      # minimum review text length (chars) to be useful
MAX_TEXT_LEN      = 2000    # truncate very long reviews before sending to LLM

SEED = 42
random.seed(SEED)
MAX_RETRIES = 3          # retries if Chinese characters are detected

# ─── Prompt ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Bạn là trợ lý tạo dữ liệu đánh giá hệ thống RAG (Retrieval-Augmented Generation).
Nhiệm vụ của bạn: đọc một review sản phẩm và sinh ra MỘT câu hỏi tự nhiên mà người dùng thực sự có thể hỏi,
và review đó có thể trả lời được câu hỏi này.

QUY TẮC BẮT BUỘC:
- Câu hỏi PHẢI viết hoàn toàn bằng TIẾNG VIỆT, không dùng bất kỳ ngôn ngữ nào khác
- TUYỆT ĐỐI KHÔNG dùng tiếng Trung, tiếng Anh, hay bất kỳ ngôn ngữ nào khác ngoài tiếng Việt
- Câu hỏi phải tự nhiên, như người dùng thật sự hỏi trước khi mua hàng
- Câu hỏi phải cụ thể (không quá chung chung như "Sản phẩm này tốt không?")
- CHỈ trả về DUY NHẤT câu hỏi, không giải thích, không ghi thêm gì khác"""

USER_TEMPLATE = """Review sản phẩm (tiếng Việt):
\"\"\"{review_text}\"\"\"

Sản phẩm: {product_name}
Danh mục: {category}

Hãy viết ĐÚNG 1 câu hỏi bằng TIẾNG VIỆT mà người dùng có thể hỏi và review trên có thể trả lời được.
Không được dùng tiếng Trung hay tiếng Anh. Chỉ trả lời bằng câu hỏi, không thêm gì khác.
Câu hỏi:"""

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_reviews_from_file(filepath: Path, min_len: int = MIN_TEXT_LEN) -> list[dict]:
    """Load all valid reviews from a JSONL file."""
    reviews = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                text = rec.get("text", "").strip()
                if len(text) >= min_len:
                    reviews.append(rec)
            except json.JSONDecodeError:
                continue
    return reviews


def extract_product_name(variant_str: str) -> str:
    """Try to extract a readable product name from the variant JSON-like string."""
    try:
        # variant is a Python-repr list string; try to find 'name' field
        import ast
        variants = ast.literal_eval(variant_str)
        if variants and isinstance(variants, list):
            return variants[0].get("name", "")
    except Exception:
        pass
    return ""


def sample_reviews(reviews_dir: Path, target: int) -> list[dict]:
    """
    Sample `target` reviews evenly across all product files.
    Strategy: for each file, sample ceiling(target / num_files) reviews,
    then shuffle & trim to target.
    """
    files = sorted([f for f in reviews_dir.glob("*.jsonl") if f.name != ".gitkeep"])
    print(f"[INFO] Found {len(files)} product files in {reviews_dir}")

    per_file = max(1, -(-target // len(files)))  # ceiling division
    print(f"[INFO] Sampling up to {per_file} review(s) per product file")

    sampled = []
    skipped_files = 0

    for fpath in files:
        reviews = load_reviews_from_file(fpath)
        if not reviews:
            skipped_files += 1
            continue
        k = min(per_file, len(reviews))
        chosen = random.sample(reviews, k)
        sampled.extend(chosen)

    print(f"[INFO] Skipped {skipped_files} files with no valid reviews")
    print(f"[INFO] Collected {len(sampled)} candidates before trimming")

    random.shuffle(sampled)
    sampled = sampled[:target]
    print(f"[INFO] Final sample size: {len(sampled)} reviews")
    return sampled


CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs (core Chinese/Japanese/Korean)
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x20000, 0x2A6DF), # CJK Extension B
    (0x2A700, 0x2B73F), # CJK Extension C
    (0x3000, 0x303F),   # CJK Symbols and Punctuation
]


def has_chinese(text: str) -> bool:
    """Return True if text contains CJK (Chinese/Japanese/Korean) ideographic characters."""
    return any(
        any(lo <= ord(ch) <= hi for lo, hi in CJK_RANGES)
        for ch in text
    )


def clean_question(raw: str) -> str:
    """
    Strip common LLM preambles like 'Câu hỏi:' or leading/trailing quotes.
    Also remove anything after a newline (model sometimes appends explanation).
    """
    text = raw.strip()
    # Take only the first non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        text = lines[0]
    # Remove leading label like "Câu hỏi:", "Question:", etc.
    for prefix in ("Câu hỏi:", "Question:", "Q:", "Hỏi:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    # Strip surrounding quotes
    text = text.strip('"\"\'')
    return text


def call_qwen(client: OpenAI, review: dict) -> str | None:
    """Call Qwen2.5 to generate a question for the given review. Returns the question string."""
    """Retries up to MAX_RETRIES times if the output contains Chinese characters."""
    text = review.get("text", "").strip()
    text = text[:MAX_TEXT_LEN]  # truncate if too long

    variant_str = review.get("variant", "")
    product_name = extract_product_name(variant_str) or review.get("product_id", "")
    category = review.get("category", "Không rõ")

    user_msg = USER_TEMPLATE.format(
        review_text=text,
        product_name=product_name,
        category=category,
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.5 + (attempt - 1) * 0.1,  # increase temp on retry
                max_tokens=120,
                timeout=LLM_TIMEOUT,
            )
            raw = response.choices[0].message.content.strip()
            question = clean_question(raw)

            if has_chinese(question):
                print(f"  [WARN] Attempt {attempt}: Chinese detected in output, retrying...")
                print(f"         Raw: {raw[:120]!r}")
                continue

            if not question.endswith("?"):
                # Tolerate if it ends with Vietnamese question words
                pass  # still accept — some valid questions don't end with ?

            return question

        except Exception as e:
            print(f"  [ERROR] Attempt {attempt}: LLM call failed: {e}")
            if attempt == MAX_RETRIES:
                return None

    print(f"  [FAIL] All {MAX_RETRIES} attempts returned Chinese. Skipping review.")
    return None


def load_progress(progress_file: Path) -> dict:
    """Load already-processed review_ids to allow resume."""
    done = {}
    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        done[rec["review_id"]] = rec
                    except Exception:
                        pass
    return done


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Synthetic Test Set Generator — Shopee Reviews + Qwen2.5")
    print("=" * 60)

    # 1. Sample reviews
    sampled = sample_reviews(REVIEWS_DIR, TARGET_TOTAL)

    # 2. Load existing progress (resume support)
    done_map = load_progress(PROGRESS_FILE)
    print(f"\n[INFO] Already processed: {len(done_map)} review(s)")

    # 3. Setup OpenAI-compatible client
    client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")

    # 4. Process each review
    results = list(done_map.values())  # start from existing
    failed = 0

    todo = [r for r in sampled if r["review_id"] not in done_map]
    print(f"[INFO] Need to generate questions for {len(todo)} review(s)\n")

    for i, review in enumerate(todo, start=1):
        rid = review["review_id"]
        pid = review["product_id"]
        cat = review.get("category", "")
        text = review.get("text", "")[:120].replace("\n", " ")
        print(f"[{i:03d}/{len(todo)}] product={pid} | review={rid}")
        print(f"        preview: {text!r}")

        question = call_qwen(client, review)

        if question:
            print(f"        question: {question}")
            record = {
                "query":        question,
                "review_id":    rid,
                "product_id":   pid,
                "category":     cat,
                "rating":       review.get("rating"),
                "review_text":  review.get("text", ""),
                "product_name": extract_product_name(review.get("variant", "")),
                "review_url":   review.get("review_url", ""),
            }
            results.append(record)
            # Append to progress file immediately (crash-safe)
            with open(PROGRESS_FILE, "a", encoding="utf-8") as pf:
                pf.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            print(f"        [SKIP] No question generated")
            failed += 1

        print()
        # Small courtesy delay to avoid hammering the server
        time.sleep(0.2)

    # 5. Write final output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for rec in results:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(f"  Done! {len(results)} QA pairs saved → {OUTPUT_FILE}")
    print(f"  Failed / skipped: {failed}")
    print("=" * 60)

    # 6. Print summary stats
    from collections import Counter
    cat_counts = Counter(r["category"] for r in results)
    print("\nCategory distribution:")
    for cat, cnt in cat_counts.most_common():
        print(f"  {cnt:3d}  {cat}")


if __name__ == "__main__":
    main()
