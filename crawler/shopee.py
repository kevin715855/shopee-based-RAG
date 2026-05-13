from __future__ import annotations

import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .browser import fetch_json_from_product_page, fetch_json_with_browser_context, fetch_product_api_by_rendering_page
from .config import CrawlConfig
from .schemas import ProductRecord, ReviewRecord
from .utils import append_jsonl, ensure_dir, sleep_random, write_json


SHOPEE_PRODUCT_RE = re.compile(r"(?:-i\.|/product/)(?P<shop_id>\d+)[./](?P<item_id>\d+)")


def parse_shopee_ids(product_url: str) -> tuple[str, str]:
    match = SHOPEE_PRODUCT_RE.search(product_url)
    if not match:
        raise ValueError(f"Cannot parse Shopee shop_id/item_id from URL: {product_url}")
    return match.group("shop_id"), match.group("item_id")


class ShopeeCrawler:
    def __init__(self, config: CrawlConfig, output_dir: str | Path = "data/raw/shopee") -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.session = requests.Session()

    def headers(self) -> dict[str, str]:
        user_agent = random.choice(self.config.user_agents) if self.config.user_agents else "Mozilla/5.0"
        return {
            "accept": "application/json",
            "accept-language": "vi-VN,vi;q=0.9,en;q=0.8",
            "referer": "https://shopee.vn/",
            "user-agent": user_agent,
        }

    def get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=self.headers(),
                    timeout=self.config.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                if payload:
                    return payload
            except Exception as exc:  # noqa: BLE001 - retry boundary
                last_error = exc
            if attempt < self.config.retry_attempts:
                sleep_random(self.config.delay_seconds.min, self.config.delay_seconds.max)
        try:
            return fetch_json_with_browser_context(
                url,
                params,
                timeout_ms=self.config.timeout_seconds * 1000,
                auth_state_path=self.config.auth_state_path,
                user_data_dir=self.config.user_data_dir,
            )
        except Exception as browser_error:  # noqa: BLE001 - keep both failure paths visible
            if "pdp/get_pc" in url:
                try:
                    product_url = f"https://shopee.vn/product/{params['shop_id']}/{params['item_id']}"
                    return fetch_product_api_by_rendering_page(
                        product_url,
                        "api/v4/pdp/get_pc",
                        timeout_ms=self.config.timeout_seconds * 1000,
                        auth_state_path=self.config.auth_state_path,
                        user_data_dir=self.config.user_data_dir,
                    )
                except Exception as render_error:  # noqa: BLE001
                    raise RuntimeError(
                        f"Shopee request failed after retries: {last_error}; "
                        f"browser fallback failed: {browser_error}; "
                        f"render fallback failed: {render_error}",
                    ) from render_error
            if "item/get_ratings" in url:
                try:
                    product_url = f"https://shopee.vn/product/{params['shopid']}/{params['itemid']}"
                    return fetch_json_from_product_page(
                        product_url,
                        url,
                        params,
                        timeout_ms=self.config.timeout_seconds * 1000,
                        auth_state_path=self.config.auth_state_path,
                        user_data_dir=self.config.user_data_dir,
                    )
                except Exception as page_fetch_error:  # noqa: BLE001
                    raise RuntimeError(
                        f"Shopee request failed after retries: {last_error}; "
                        f"browser fallback failed: {browser_error}; "
                        f"page fetch fallback failed: {page_fetch_error}",
                    ) from page_fetch_error
            raise RuntimeError(
                f"Shopee request failed after retries: {last_error}; "
                f"browser fallback failed: {browser_error}",
            ) from browser_error

    def crawl_products(self) -> list[ProductRecord]:
        products: list[ProductRecord] = []
        product_dir = ensure_dir(self.output_dir / "products")
        for product_url in self.config.product_urls[: self.config.max_products]:
            products.append(self.crawl_product(product_url, raw_dir=product_dir))
            sleep_random(self.config.delay_seconds.min, self.config.delay_seconds.max)
        return products

    def crawl_product(self, product_url: str, raw_dir: str | Path | None = None, category_metadata: dict[str, str] | None = None) -> ProductRecord:
        shop_id, item_id = parse_shopee_ids(product_url)
        payload = self.get_json(
            "https://shopee.vn/api/v4/pdp/get_pc",
            {"shop_id": shop_id, "item_id": item_id},
        )
        data = payload.get("data") or {}
        item = data.get("item") or data
        category_metadata = category_metadata or {}
        product = ProductRecord(
            product_id=str(item.get("itemid") or item_id),
            name=str(item.get("title") or item.get("name") or f"Shopee item {item_id}"),
            category_id=category_metadata.get("category_id", ""),
            category=category_metadata.get("category_name", self.config.category),
            category_url=category_metadata.get("category_url", ""),
            crawl_batch_id=category_metadata.get("crawl_batch_id", self.config.batch_id),
            description=str(item.get("description") or ""),
            price=_normalize_price(item.get("price")),
            avg_rating=_safe_float((item.get("item_rating") or {}).get("rating_star")),
            review_count=_safe_int(item.get("cmt_count") or item.get("historical_sold")),
            image_url=_first_image(item),
            product_url=product_url,
            source="shopee",
        )
        product_dir = ensure_dir(Path(raw_dir) if raw_dir is not None else self.output_dir / "products")
        raw_file = product_dir / f"{product.product_id}.json"
        product.raw_file = str(raw_file)
        write_json(raw_file, {"record": product.model_dump(), "raw_payload": payload})
        return product

    def crawl_reviews(self, products: list[ProductRecord] | None = None) -> int:
        if products is None:
            products = [
                ProductRecord(product_id=parse_shopee_ids(url)[1], name=f"Shopee item {index}", product_url=url)
                for index, url in enumerate(self.config.product_urls, start=1)
            ]

        total = 0
        review_dir = ensure_dir(self.output_dir / "reviews")
        for product in products[: self.config.max_products]:
            shop_id, item_id = parse_shopee_ids(product.product_url)
            output_file = review_dir / f"{product.product_id}.jsonl"
            offset = 0
            while offset < self.config.max_reviews_per_product:
                limit = min(50, self.config.max_reviews_per_product - offset)
                payload = self.get_json(
                    "https://shopee.vn/api/v2/item/get_ratings",
                    {
                        "filter": 0,
                        "flag": 1,
                        "itemid": item_id,
                        "limit": limit,
                        "offset": offset,
                        "shopid": shop_id,
                        "type": 0,
                    },
                )
                ratings = payload.get("data", {}).get("ratings") or []
                if not ratings:
                    break
                records = [_rating_to_record(product, rating, output_file) for rating in ratings]
                total += append_jsonl(output_file, [record.model_dump() for record in records])
                offset += len(ratings)
                sleep_random(self.config.delay_seconds.min, self.config.delay_seconds.max)
        return total


def _rating_to_record(product: ProductRecord, rating: dict[str, Any], raw_file: Path) -> ReviewRecord:
    created_at = rating.get("ctime")
    created_at_text = (
        datetime.fromtimestamp(created_at, tz=timezone.utc).isoformat()
        if isinstance(created_at, int)
        else None
    )
    review = ReviewRecord(
        review_id=str(rating.get("cmtid") or rating.get("rating_id") or ""),
        product_id=product.product_id,
        text=str(rating.get("comment") or ""),
        rating=int(rating.get("rating_star") or 0),
        created_at=created_at_text,
        source="shopee",
        variant=str(rating.get("product_items") or rating.get("variation") or ""),
        helpful_count=_safe_int(rating.get("like_count")) or 0,
        media_count=len(rating.get("images") or []) + len(rating.get("videos") or []),
        review_url=product.product_url,
        category_id=product.category_id,
        category=product.category,
        category_url=product.category_url,
        crawl_batch_id=product.crawl_batch_id,
        raw_file=str(raw_file),
        raw_payload=rating,
    )
    return review.with_dedupe_key()


def _normalize_price(value: Any) -> float | None:
    number = _safe_float(value)
    if number is None:
        return None
    return number / 100000 if number > 1000000 else number


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_image(item: dict[str, Any]) -> str | None:
    images = item.get("images") or []
    if not images:
        return None
    image = images[0]
    return f"https://down-vn.img.susercontent.com/file/{image}" if isinstance(image, str) else None
