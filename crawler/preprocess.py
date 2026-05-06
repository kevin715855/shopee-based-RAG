from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import clean_text, ensure_dir, sentiment_from_rating


PRODUCT_COLUMNS = [
    "product_id",
    "name",
    "category_id",
    "category",
    "category_url",
    "crawl_batch_id",
    "description",
    "price",
    "avg_rating",
    "review_count",
    "image_url",
    "product_url",
    "source",
    "crawl_time",
    "raw_file",
]

REVIEW_COLUMNS = [
    "review_id",
    "product_id",
    "text",
    "text_normalized",
    "rating",
    "sentiment",
    "created_at",
    "source",
    "category_id",
    "category",
    "category_url",
    "crawl_batch_id",
    "variant",
    "helpful_count",
    "media_count",
    "review_url",
    "crawl_time",
    "raw_file",
    "clean_status",
    "dedupe_key",
    "document_text",
]


def preprocess(raw_dir: str | Path = "data/raw/shopee", processed_dir: str | Path = "data/processed") -> dict[str, int]:
    raw_dir = Path(raw_dir)
    processed_dir = ensure_dir(processed_dir)

    categories = _load_category_records(raw_dir / "categories")
    products = _load_product_records(raw_dir / "products")
    reviews = _load_review_records(raw_dir / "reviews")

    categories_df = pd.DataFrame(categories)
    products_df = pd.DataFrame(products)
    reviews_df = pd.DataFrame(reviews)

    if categories_df.empty:
        categories_df = pd.DataFrame(columns=[
            "category_id",
            "category_name",
            "category_url",
            "batch_id",
            "product_count",
            "raw_file",
        ])
    else:
        categories_df = categories_df.reindex(columns=[
            "category_id",
            "category_name",
            "category_url",
            "batch_id",
            "product_count",
            "raw_file",
        ])
        categories_df = categories_df.drop_duplicates(subset=["category_id", "category_url"], keep="last")

    if products_df.empty:
        products_df = pd.DataFrame(columns=PRODUCT_COLUMNS)
    else:
        products_df = products_df.reindex(columns=PRODUCT_COLUMNS)
        products_df = products_df.drop_duplicates(subset=["source", "product_id", "category_id"], keep="last")

    if reviews_df.empty:
        reviews_df = pd.DataFrame(columns=REVIEW_COLUMNS)
        raw_count = 0
    else:
        raw_count = len(reviews_df)
        reviews_df["text"] = reviews_df["text"].fillna("").map(clean_text)
        reviews_df["rating"] = pd.to_numeric(reviews_df["rating"], errors="coerce").fillna(0).astype(int)
        reviews_df["dedupe_key"] = reviews_df.apply(
            lambda row: f"{row.get('source', 'shopee')}:{row.get('product_id')}:{row.get('review_id')}",
            axis=1,
        )
        reviews_df["clean_status"] = "clean"
        reviews_df["text_normalized"] = reviews_df["text"].str.lower()
        reviews_df["sentiment"] = reviews_df["rating"].map(sentiment_from_rating)
        product_names = products_df.set_index("product_id")["name"].to_dict() if not products_df.empty else {}
        reviews_df["document_text"] = reviews_df.apply(
            lambda row: _document_text(row, product_names.get(row["product_id"], "")),
            axis=1,
        )
        reviews_df = reviews_df[reviews_df["text"].str.len() >= 10]
        reviews_df = reviews_df[reviews_df["rating"].between(1, 5)]
        reviews_df = reviews_df.drop_duplicates(subset=["dedupe_key"], keep="last")
        reviews_df = reviews_df.reindex(columns=REVIEW_COLUMNS)

    categories_df.to_csv(processed_dir / "categories.csv", index=False, encoding="utf-8-sig")
    products_df.to_csv(processed_dir / "products.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(reviews).to_csv(processed_dir / "reviews.csv", index=False, encoding="utf-8-sig")
    reviews_df.to_csv(processed_dir / "reviews_clean.csv", index=False, encoding="utf-8-sig")

    return {
        "categories": len(categories_df),
        "products": len(products_df),
        "reviews_raw": raw_count,
        "reviews_clean": len(reviews_df),
        "reviews_removed": raw_count - len(reviews_df),
    }


def _load_product_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file in sorted(path.glob("*.json")):
        with file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        record = payload.get("record", payload)
        record["raw_file"] = record.get("raw_file") or str(file)
        rows.append(record)
    return rows


def _load_category_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file in sorted(path.glob("*.json")):
        with file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        record = payload.get("record", payload)
        record["raw_file"] = record.get("raw_file") or str(file)
        rows.append(record)
    return rows


def _load_review_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file in sorted(path.glob("*.jsonl")):
        with file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row.pop("raw_payload", None)
                row["raw_file"] = row.get("raw_file") or str(file)
                rows.append(row)
    return rows


def _document_text(row: pd.Series, product_name: str) -> str:
    return f"Sản phẩm: {product_name}. Rating: {row['rating']} sao. Review: {row['text']}"
