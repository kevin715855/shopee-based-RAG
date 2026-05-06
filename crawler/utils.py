from __future__ import annotations

import json
import random
import re
import time
import hashlib
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_dir(Path(path).parent)
    count = 0
    with Path(path).open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def sleep_random(min_seconds: float, max_seconds: float) -> None:
    time.sleep(random.uniform(min_seconds, max_seconds))


def clean_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    text = re.sub(r"[\u0000-\u001f\u007f-\u009f]", " ", text)
    text = re.sub(r"[\U00010000-\U0010ffff]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sentiment_from_rating(rating: int) -> str:
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def stable_hash(value: str, length: int = 12) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return digest[:length]
