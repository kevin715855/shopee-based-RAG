from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .browser import discover_category_products, extract_product_urls_from_html, parse_title_from_html
from .config import CrawlConfig
from .utils import ensure_dir, sleep_random, stable_hash, write_json


@dataclass(slots=True)
class CategoryBatch:
    category_id: str
    category_name: str
    category_url: str
    batch_id: str
    product_urls: list[str] = field(default_factory=list)


class ShopeeCategoryCrawler:
    def __init__(self, config: CrawlConfig, output_dir: str | Path = "data/raw/shopee") -> None:
        self.config = config
        self.output_dir = Path(output_dir)

    def crawl_categories(self) -> list[CategoryBatch]:
        categories: list[CategoryBatch] = []
        category_dir = ensure_dir(self.output_dir / "categories")
        state_dir = ensure_dir(Path("data/state"))
        for index, category_url in enumerate(self.config.category_urls[: self.config.max_categories], start=1):
            category_id = stable_hash(category_url)
            try:
                discovery = discover_category_products(
                    category_url,
                    timeout_ms=self.config.timeout_seconds * 1000,
                    max_scrolls=self.config.max_pages_per_category,
                    max_products=self.config.max_products_per_category,
                    auth_state_path=self.config.auth_state_path,
                    user_data_dir=self.config.user_data_dir,
                )
            except Exception as exc:  # noqa: BLE001 - keep batch moving and log the failing URL
                write_json(
                    category_dir / f"{index:03d}_{category_id}.error.json",
                    {
                        "record": {
                            "category_id": category_id,
                            "category_name": self.config.category,
                            "category_url": category_url,
                            "batch_id": self.config.batch_id,
                            "product_count": 0,
                            "error": str(exc),
                        },
                        "raw_payload": {},
                    },
                )
                continue
            product_urls = discovery.network_product_urls
            if not product_urls:
                product_urls = discovery.html_product_urls
            if not product_urls:
                product_urls = self._discover_from_hints(discovery.html)
            category_name = discovery.title or self.config.category
            batch = CategoryBatch(
                category_id=category_id,
                category_name=category_name,
                category_url=category_url,
                batch_id=self.config.batch_id,
                product_urls=product_urls[: self.config.max_products_per_category],
            )
            write_json(
                category_dir / f"{index:03d}_{category_id}.json",
                {
                    "record": {
                        "category_id": batch.category_id,
                        "category_name": batch.category_name,
                        "category_url": batch.category_url,
                        "batch_id": batch.batch_id,
                        "product_count": len(batch.product_urls),
                    },
                    "raw_payload": {
                        "html_title": category_name,
                        "product_urls": batch.product_urls,
                        "network_product_urls": discovery.network_product_urls,
                        "html_product_urls": discovery.html_product_urls,
                        "api_statuses": discovery.api_statuses,
                        "blocked_responses": discovery.blocked_responses,
                        "html_length": len(discovery.html),
                    },
                },
            )
            self._write_checkpoint(state_dir, batch)
            categories.append(batch)
            sleep_random(self.config.delay_seconds.min, self.config.delay_seconds.max)
        return categories

    def _discover_from_hints(self, html: str) -> list[str]:
        urls = extract_product_urls_from_html(html)
        if urls:
            return urls
        return []

    def _write_checkpoint(self, state_dir: Path, batch: CategoryBatch) -> None:
        write_json(
            state_dir / f"{batch.batch_id}.json",
            {
                "batch_id": batch.batch_id,
                "category_id": batch.category_id,
                "category_name": batch.category_name,
                "category_url": batch.category_url,
                "product_count": len(batch.product_urls),
            },
        )


def category_metadata_from_url(url: str, fallback_name: str = "") -> tuple[str, str]:
    category_id = stable_hash(url)
    parsed = urlparse(url)
    slug = parsed.path.rstrip("/").split("/")[-1]
    return category_id, fallback_name or slug or category_id
