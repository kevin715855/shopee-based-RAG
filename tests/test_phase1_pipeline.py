from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from crawler.preprocess import preprocess
from crawler.validate import validate_dataset


class Phase1PipelineTest(unittest.TestCase):
    def test_preprocess_and_validation_with_minimal_raw_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw" / "shopee"
            processed_dir = root / "processed"
            _write_minimal_raw_fixture(raw_dir)

            stats = preprocess(raw_dir, processed_dir)
            validation = validate_dataset(processed_dir)

            self.assertEqual(stats["categories"], 1)
            self.assertEqual(stats["products"], 2)
            self.assertEqual(stats["reviews_raw"], 5)
            self.assertEqual(stats["reviews_clean"], 4)
            self.assertTrue(validation["ok"], validation["errors"])
            self.assertEqual(validation["sentiment_distribution"]["positive"], 3)
            self.assertEqual(validation["sentiment_distribution"]["neutral"], 1)
            self.assertEqual(validation["category_count"], 1)


def _write_minimal_raw_fixture(raw_dir: Path) -> None:
    category_dir = raw_dir / "categories"
    product_dir = raw_dir / "products"
    review_dir = raw_dir / "reviews"
    category_dir.mkdir(parents=True)
    product_dir.mkdir(parents=True)
    review_dir.mkdir(parents=True)

    _write_json(
        category_dir / "electronics.json",
        {
            "record": {
                "category_id": "electronics",
                "category_name": "Thiết Bị Điện Tử",
                "category_url": "",
                "batch_id": "test-batch",
                "product_count": 2,
            },
        },
    )

    for product_id, name in (("p1", "Tai nghe Bluetooth"), ("p2", "Sạc nhanh USB-C")):
        _write_json(
            product_dir / f"{product_id}.json",
            {
                "record": {
                    "product_id": product_id,
                    "name": name,
                    "category_id": "electronics",
                    "category": "Thiết Bị Điện Tử",
                    "category_url": "",
                    "crawl_batch_id": "test-batch",
                    "description": "",
                    "price": 120000,
                    "avg_rating": 4.5,
                    "review_count": 3,
                    "image_url": "",
                    "product_url": f"https://shopee.vn/product/1/{product_id}",
                    "source": "shopee",
                    "crawl_time": "2026-01-01T00:00:00+00:00",
                },
            },
        )

    reviews = [
        ("r1", "p1", "Pin tốt, dùng ổn trong nhiều giờ.", 5),
        ("r2", "p1", "Đóng gói chắc chắn, giao hàng nhanh.", 5),
        ("r3", "p2", "Sạc nhanh nhưng hơi nóng khi dùng lâu.", 3),
        ("r4", "p2", "Chất lượng tốt so với giá.", 4),
        ("r5", "p2", "ok", 5),
    ]
    with (review_dir / "reviews.jsonl").open("w", encoding="utf-8") as handle:
        for review_id, product_id, text, rating in reviews:
            handle.write(
                json.dumps(
                    {
                        "review_id": review_id,
                        "product_id": product_id,
                        "text": text,
                        "rating": rating,
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "source": "shopee",
                        "variant": "",
                        "helpful_count": 0,
                        "media_count": 0,
                        "review_url": f"https://shopee.vn/product/1/{product_id}",
                        "category_id": "electronics",
                        "category": "Thiết Bị Điện Tử",
                        "category_url": "",
                        "crawl_batch_id": "test-batch",
                        "crawl_time": "2026-01-01T00:00:00+00:00",
                        "clean_status": "raw",
                        "dedupe_key": f"shopee:{product_id}:{review_id}",
                    },
                    ensure_ascii=False,
                )
                + "\n",
            )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
