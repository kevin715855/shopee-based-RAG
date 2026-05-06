from __future__ import annotations

from pathlib import Path

from .category import CategoryBatch
from .schemas import ProductRecord, ReviewRecord
from .utils import append_jsonl, ensure_dir, write_json


def create_sample_data(raw_dir: str | Path = "data/raw/shopee") -> dict[str, int]:
    raw_dir = Path(raw_dir)
    product_dir = ensure_dir(raw_dir / "products")
    review_dir = ensure_dir(raw_dir / "reviews")
    category_dir = ensure_dir(raw_dir / "categories")
    for file in list(product_dir.glob("*.json")) + list(review_dir.glob("*.jsonl")):
        file.unlink()
    for file in list(category_dir.glob("*.json")):
        file.unlink()

    categories = [
        CategoryBatch(
            category_id="cat-electronics",
            category_name="Đồ điện tử",
            category_url="https://shopee.vn/electronics",
            batch_id="sample-batch",
            product_urls=[
                "https://shopee.vn/airbass-pro-i.123456.789012",
                "https://shopee.vn/powermax-i.123456.789013",
            ],
        ),
    ]

    for index, category in enumerate(categories, start=1):
        write_json(
            category_dir / f"{index:03d}_{category.category_id}.json",
            {
                "record": {
                    "category_id": category.category_id,
                    "category_name": category.category_name,
                    "category_url": category.category_url,
                    "batch_id": category.batch_id,
                    "product_count": len(category.product_urls),
                },
                "raw_payload": {"product_urls": category.product_urls},
            },
        )

    products = [
        ProductRecord(
            product_id="SP-EL-1024",
            name="Tai nghe Bluetooth chống ồn AirBass Pro",
            category_id="cat-electronics",
            category="Đồ điện tử",
            category_url="https://shopee.vn/electronics",
            crawl_batch_id="sample-batch",
            description="Tai nghe không dây chống ồn, pin dài, hỗ trợ gọi điện.",
            price=590000,
            avg_rating=4.6,
            review_count=8421,
            image_url="https://example.com/airbass.jpg",
            product_url="https://shopee.vn/airbass-pro-i.123456.789012",
            source="shopee",
        ),
        ProductRecord(
            product_id="SP-EL-4110",
            name="Pin sạc dự phòng PowerMax 20000mAh",
            category_id="cat-electronics",
            category="Phụ kiện điện tử",
            category_url="https://shopee.vn/electronics",
            crawl_batch_id="sample-batch",
            description="Pin dự phòng dung lượng cao, hỗ trợ sạc nhanh.",
            price=420000,
            avg_rating=4.7,
            review_count=9734,
            image_url="https://example.com/powermax.jpg",
            product_url="https://shopee.vn/powermax-i.123456.789013",
            source="shopee",
        ),
    ]

    for product in products:
        product.raw_file = str(product_dir / f"{product.product_id}.json")
        write_json(product.raw_file, {"record": product.model_dump(), "raw_payload": {}})

    reviews = [
        ReviewRecord(
            review_id="RV-8012",
            product_id="SP-EL-1024",
            text="Pin dùng được khoảng 7 tiếng nếu bật chống ồn, âm thanh rõ và bass vừa.",
            rating=4,
            category_id="cat-electronics",
            category="Đồ điện tử",
            category_url="https://shopee.vn/electronics",
            crawl_batch_id="sample-batch",
            created_at="2026-04-12T00:00:00+00:00",
            variant="Đen",
            review_url=products[0].product_url,
        ).with_dedupe_key(),
        ReviewRecord(
            review_id="RV-8018",
            product_id="SP-EL-1024",
            text="Mua làm quà khá ổn vì hộp đẹp, shop giao nhanh nhưng thiếu một bộ nút tai phụ.",
            rating=4,
            category_id="cat-electronics",
            category="Đồ điện tử",
            category_url="https://shopee.vn/electronics",
            crawl_batch_id="sample-batch",
            created_at="2026-03-28T00:00:00+00:00",
            variant="Trắng",
            review_url=products[0].product_url,
        ).with_dedupe_key(),
        ReviewRecord(
            review_id="RV-9301",
            product_id="SP-EL-4110",
            text="Sạc iPhone và tai nghe cùng lúc vẫn ổn, máy không nóng nhiều.",
            rating=5,
            category_id="cat-electronics",
            category="Phụ kiện điện tử",
            category_url="https://shopee.vn/electronics",
            crawl_batch_id="sample-batch",
            created_at="2026-04-21T00:00:00+00:00",
            variant="20000mAh",
            review_url=products[1].product_url,
        ).with_dedupe_key(),
        ReviewRecord(
            review_id="RV-9320",
            product_id="SP-EL-4110",
            text="Pin chắc tay nhưng hơi nặng, muốn sạc nhanh laptop thì phải mua thêm cáp C-C.",
            rating=4,
            category_id="cat-electronics",
            category="Phụ kiện điện tử",
            category_url="https://shopee.vn/electronics",
            crawl_batch_id="sample-batch",
            created_at="2026-03-06T00:00:00+00:00",
            variant="20000mAh",
            review_url=products[1].product_url,
        ).with_dedupe_key(),
        ReviewRecord(
            review_id="RV-DROP",
            product_id="SP-EL-4110",
            text="ok",
            rating=5,
            category_id="cat-electronics",
            category="Phụ kiện điện tử",
            category_url="https://shopee.vn/electronics",
            crawl_batch_id="sample-batch",
            created_at="2026-03-06T00:00:00+00:00",
            variant="20000mAh",
            review_url=products[1].product_url,
        ).with_dedupe_key(),
    ]

    grouped: dict[str, list[dict[str, object]]] = {}
    for review in reviews:
        review.raw_file = str(review_dir / f"{review.product_id}.jsonl")
        grouped.setdefault(review.product_id, []).append(review.model_dump())
    for product_id, rows in grouped.items():
        append_jsonl(review_dir / f"{product_id}.jsonl", rows)

    return {"products": len(products), "reviews": len(reviews)}
