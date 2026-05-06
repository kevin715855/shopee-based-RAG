from __future__ import annotations

from .config import CrawlConfig


class TikiCrawler:
    """Placeholder with the same construction pattern as ShopeeCrawler for Phase 1 extension."""

    def __init__(self, config: CrawlConfig, output_dir: str = "data/raw/tiki") -> None:
        self.config = config
        self.output_dir = output_dir

    def crawl_products(self) -> None:
        raise NotImplementedError("Tiki crawler is planned after the Shopee v1 pipeline is stable.")

    def crawl_reviews(self) -> None:
        raise NotImplementedError("Tiki crawler is planned after the Shopee v1 pipeline is stable.")
