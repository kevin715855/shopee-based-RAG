from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


SourceName = Literal["shopee", "tiki"]
Sentiment = Literal["negative", "neutral", "positive"]


class ProductRecord(BaseModel):
    product_id: str
    name: str
    category_id: str = ""
    category: str = ""
    category_url: str = ""
    crawl_batch_id: str = ""
    description: str = ""
    price: float | None = None
    avg_rating: float | None = None
    review_count: int | None = None
    image_url: str | None = None
    product_url: str
    source: SourceName = "shopee"
    crawl_time: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    raw_file: str | None = None

    @field_validator("name", "product_id", "product_url")
    @classmethod
    def must_not_be_empty(cls, value: str) -> str:
        if not str(value).strip():
            raise ValueError("field must not be empty")
        return str(value).strip()


class ReviewRecord(BaseModel):
    review_id: str
    product_id: str
    text: str
    rating: int
    created_at: str | None = None
    source: SourceName = "shopee"
    variant: str = ""
    helpful_count: int = 0
    media_count: int = 0
    review_url: str | None = None
    category_id: str = ""
    category: str = ""
    category_url: str = ""
    crawl_batch_id: str = ""
    crawl_time: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    raw_file: str | None = None
    clean_status: str = "raw"
    dedupe_key: str = ""
    raw_payload: dict[str, Any] | None = None

    @field_validator("rating")
    @classmethod
    def rating_between_one_and_five(cls, value: int) -> int:
        if value < 1 or value > 5:
            raise ValueError("rating must be between 1 and 5")
        return value

    def with_dedupe_key(self) -> "ReviewRecord":
        self.dedupe_key = f"{self.source}:{self.product_id}:{self.review_id}"
        return self
