from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import ensure_dir, write_json
from .validate import validate_dataset


def write_phase1_report(processed_dir: str | Path = "data/processed", report_dir: str | Path = "data/reports") -> Path:
    processed_dir = Path(processed_dir)
    report_dir = ensure_dir(report_dir)
    validation = validate_dataset(processed_dir)
    categories_path = processed_dir / "categories.csv"
    categories = pd.read_csv(categories_path) if categories_path.exists() else pd.DataFrame()
    reviews_path = processed_dir / "reviews_clean.csv"
    examples: list[dict[str, object]] = []
    if reviews_path.exists():
        examples = pd.read_csv(reviews_path).head(5).to_dict(orient="records")

    payload = {
        "phase": "Phase 1 - Crawl Data",
        "validation": validation,
        "category_count": len(categories),
        "sample_clean_reviews": examples,
    }
    output = report_dir / "phase1_report.json"
    write_json(output, payload)
    return output
