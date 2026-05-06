from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .preprocess import PRODUCT_COLUMNS, REVIEW_COLUMNS


def validate_dataset(processed_dir: str | Path = "data/processed") -> dict[str, Any]:
    processed_dir = Path(processed_dir)
    products_path = processed_dir / "products.csv"
    reviews_path = processed_dir / "reviews_clean.csv"
    categories_path = processed_dir / "categories.csv"

    errors: list[str] = []
    if not categories_path.exists():
        errors.append(f"Missing {categories_path}")
        categories = pd.DataFrame()
    else:
        categories = pd.read_csv(categories_path)

    if not products_path.exists():
        errors.append(f"Missing {products_path}")
        products = pd.DataFrame(columns=PRODUCT_COLUMNS)
    else:
        products = pd.read_csv(products_path)

    if not reviews_path.exists():
        errors.append(f"Missing {reviews_path}")
        reviews = pd.DataFrame(columns=REVIEW_COLUMNS)
    else:
        reviews = pd.read_csv(reviews_path)

    missing_product_columns = sorted(set(PRODUCT_COLUMNS) - set(products.columns))
    missing_review_columns = sorted(set(REVIEW_COLUMNS) - set(reviews.columns))
    if missing_product_columns:
        errors.append(f"Missing product columns: {', '.join(missing_product_columns)}")
    if missing_review_columns:
        errors.append(f"Missing review columns: {', '.join(missing_review_columns)}")

    duplicate_products = 0
    if {"source", "product_id"}.issubset(products.columns):
        duplicate_products = int(products.duplicated(subset=["source", "product_id"]).sum())
        if duplicate_products:
            errors.append(f"Duplicate products: {duplicate_products}")

    missing_category_metadata = 0
    enforce_category_metadata = False
    if "category_id" in products.columns:
        enforce_category_metadata = bool((products["category_id"].fillna("").astype(str).str.len() > 0).any())
    if "category_url" in products.columns:
        enforce_category_metadata = enforce_category_metadata or bool((products["category_url"].fillna("").astype(str).str.len() > 0).any())
    if enforce_category_metadata and {"category_id", "category"}.issubset(products.columns):
        missing_category_metadata = int(
            (products["category_id"].fillna("").astype(str).str.len() == 0).sum()
            + (products["category"].fillna("").astype(str).str.len() == 0).sum()
        )
        if missing_category_metadata:
            errors.append(f"Missing category metadata in products: {missing_category_metadata}")

    duplicate_reviews = 0
    if {"source", "product_id", "review_id"}.issubset(reviews.columns):
        duplicate_reviews = int(reviews.duplicated(subset=["source", "product_id", "review_id"]).sum())
        if duplicate_reviews:
            errors.append(f"Duplicate reviews: {duplicate_reviews}")

    short_reviews = 0
    if "text" in reviews.columns:
        short_reviews = int((reviews["text"].fillna("").str.len() < 10).sum())
        if short_reviews:
            errors.append(f"Reviews shorter than 10 chars: {short_reviews}")

    invalid_ratings = 0
    if "rating" in reviews.columns:
        rating = pd.to_numeric(reviews["rating"], errors="coerce")
        invalid_ratings = int((~rating.between(1, 5)).sum())
        if invalid_ratings:
            errors.append(f"Invalid ratings: {invalid_ratings}")

    return {
        "ok": not errors,
        "errors": errors,
        "product_count": len(products),
        "review_count": len(reviews),
        "category_count": len(categories),
        "duplicate_products": duplicate_products,
        "duplicate_reviews": duplicate_reviews,
        "missing_category_metadata": missing_category_metadata,
        "short_reviews": short_reviews,
        "invalid_ratings": invalid_ratings,
        "rating_distribution": _value_counts(reviews, "rating"),
        "sentiment_distribution": _value_counts(reviews, "sentiment"),
    }


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    return {str(key): int(value) for key, value in df[column].value_counts(dropna=False).sort_index().items()}
