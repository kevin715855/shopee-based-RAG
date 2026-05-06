from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crawler.preprocess import preprocess
from crawler.browser import extract_product_urls_from_payload
from crawler.sample_data import create_sample_data
from crawler.validate import validate_dataset


class Phase1PipelineTest(unittest.TestCase):
    def test_sample_data_preprocess_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw" / "shopee"
            processed_dir = root / "processed"

            create_sample_data(raw_dir)
            stats = preprocess(raw_dir, processed_dir)
            validation = validate_dataset(processed_dir)

            self.assertEqual(stats["categories"], 1)
            self.assertEqual(stats["products"], 2)
            self.assertEqual(stats["reviews_raw"], 5)
            self.assertEqual(stats["reviews_clean"], 4)
            self.assertTrue(validation["ok"], validation["errors"])
            self.assertEqual(validation["sentiment_distribution"]["positive"], 4)
            self.assertEqual(validation["category_count"], 1)

    def test_extract_product_urls_from_common_shopee_payload_shapes(self) -> None:
        payload = {
            "items": [
                {"item_basic": {"shopid": 111, "itemid": 222}},
                {"shopid": 333, "itemid": 444},
                {"nested": [{"item_basic": {"shop_id": "555", "item_id": "666"}}]},
            ],
        }

        self.assertEqual(
            extract_product_urls_from_payload(payload),
            [
                "https://shopee.vn/product/111/222",
                "https://shopee.vn/product/333/444",
                "https://shopee.vn/product/555/666",
            ],
        )


if __name__ == "__main__":
    unittest.main()
