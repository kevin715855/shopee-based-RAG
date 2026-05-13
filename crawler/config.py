from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DelayConfig(BaseModel):
    min: float = 2
    max: float = 6


class SeedCategoryConfig(BaseModel):
    category_id: str = ""
    category_name: str
    category_url: str = ""
    product_urls: list[str] = Field(default_factory=list)


class CrawlConfig(BaseModel):
    source: str = "shopee"
    category: str = "Đồ điện tử / Phụ kiện điện tử"
    categories: list[SeedCategoryConfig] = Field(default_factory=list)
    max_products: int = 5
    max_categories: int = 1
    max_products_per_category: int = 10
    max_pages_per_category: int = 8
    max_reviews_per_product: int = 50
    dedupe_across_categories: bool = True
    delay_seconds: DelayConfig = Field(default_factory=DelayConfig)
    timeout_seconds: int = 20
    retry_attempts: int = 3
    user_agents: list[str] = Field(default_factory=list)
    product_urls: list[str] = Field(default_factory=list)
    batch_id: str = "phase2-default"
    auth_state_path: str = "data/session/shopee_auth_state.json"
    user_data_dir: str = "data/browser/shopee-profile"


def load_config(path: str | Path) -> CrawlConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        data: dict[str, Any] = yaml.safe_load(file) or {}
    return CrawlConfig(**data)
