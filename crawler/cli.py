from __future__ import annotations

import argparse
import json
import sys

from .browser import discover_category_products, open_shopee_session
from .config import load_config
from .preprocess import preprocess
from .shopee import ShopeeCrawler
from .utils import ensure_dir, stable_hash, write_json
from .validate import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="ShopeeFeed crawler pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("crawl-products", "crawl-reviews", "crawl-seed-categories", "discover-categories", "crawl-category-discovery"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", default="configs/crawl.example.yaml")
        command.add_argument("--raw-dir", default="data/raw/shopee")

    session_parser = subparsers.add_parser("open-shopee-session")
    session_parser.add_argument("--config", default="configs/crawl.example.yaml")

    preprocess_parser = subparsers.add_parser("preprocess-reviews")
    preprocess_parser.add_argument("--raw-dir", default="data/raw/shopee")
    preprocess_parser.add_argument("--processed-dir", default="data/processed")

    validate_parser = subparsers.add_parser("validate-dataset")
    validate_parser.add_argument("--processed-dir", default="data/processed")

    args = parser.parse_args()

    if args.command == "crawl-products":
        config = load_config(args.config)
        crawler = ShopeeCrawler(config, args.raw_dir)
        result = crawler.crawl_products()
        _print({"products": len(result)})
    elif args.command == "crawl-reviews":
        config = load_config(args.config)
        crawler = ShopeeCrawler(config, args.raw_dir)
        products = crawler.crawl_products()
        count = crawler.crawl_reviews(products)
        _print({"reviews": count})
    elif args.command == "crawl-seed-categories":
        config = load_config(args.config)
        crawler = ShopeeCrawler(config, args.raw_dir)
        products = _crawl_seed_categories(crawler, config.categories)
        reviews = crawler.crawl_reviews(products)
        _print({"categories": len(config.categories), "products": len(products), "reviews": reviews})
    elif args.command == "discover-categories":
        config = load_config(args.config)
        categories = _discover_categories(config, args.raw_dir)
        _print(
            {
                "categories": len(categories),
                "products": sum(len(category["product_urls"]) for category in categories),
            },
        )
    elif args.command == "crawl-category-discovery":
        config = load_config(args.config)
        crawler = ShopeeCrawler(config, args.raw_dir)
        categories = _discover_categories(config, args.raw_dir)
        products = _crawl_discovered_categories(crawler, categories)
        reviews = crawler.crawl_reviews(products)
        _print({"categories": len(categories), "products": len(products), "reviews": reviews})
    elif args.command == "open-shopee-session":
        config = load_config(args.config)
        open_shopee_session(user_data_dir=config.user_data_dir, timeout_ms=config.timeout_seconds * 1000)
        _print({"user_data_dir": config.user_data_dir})
    elif args.command == "preprocess-reviews":
        _print(preprocess(args.raw_dir, args.processed_dir))
    elif args.command == "validate-dataset":
        result = validate_dataset(args.processed_dir)
        _print(result)
        if not result["ok"]:
            raise SystemExit(1)


def _print(payload: object) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8") + b"\n")


def _crawl_seed_categories(crawler: ShopeeCrawler, categories: list[object]) -> list[object]:
    products: list[object] = []
    seen: set[str] = set()
    for category in categories[: crawler.config.max_categories]:
        category_url = getattr(category, "category_url", "")
        category_name = getattr(category, "category_name", "")
        category_id = getattr(category, "category_id", "") or stable_hash(category_url or category_name)
        category_data = {
            "category_id": category_id,
            "category_name": category_name,
            "category_url": category_url,
            "crawl_batch_id": crawler.config.batch_id,
        }
        for url in getattr(category, "product_urls", [])[: crawler.config.max_products_per_category]:
            if url in seen and crawler.config.dedupe_across_categories:
                continue
            seen.add(url)
            products.append(crawler.crawl_product(url, category_metadata=category_data))
            if len(products) >= crawler.config.max_products:
                return products
    return products


def _discover_categories(config: object, raw_dir: str) -> list[dict[str, object]]:
    category_dir = ensure_dir(f"{raw_dir}/categories")
    categories: list[dict[str, object]] = []
    for index, category in enumerate(getattr(config, "categories", [])[: getattr(config, "max_categories")], start=1):
        category_url = getattr(category, "category_url", "")
        if not category_url:
            continue
        category_name = getattr(category, "category_name", "")
        category_id = getattr(category, "category_id", "") or stable_hash(category_url)
        print(f"[discover {index}] {category_name}: {category_url}", file=sys.stderr, flush=True)
        discovery = discover_category_products(
            category_url,
            timeout_ms=getattr(config, "timeout_seconds") * 1000,
            max_scrolls=getattr(config, "max_pages_per_category"),
            max_products=getattr(config, "max_products_per_category"),
            user_data_dir=getattr(config, "user_data_dir"),
        )
        product_urls = discovery.product_urls[: getattr(config, "max_products_per_category")]
        record = {
            "category_id": category_id,
            "category_name": category_name,
            "category_url": category_url,
            "batch_id": getattr(config, "batch_id"),
            "product_count": len(product_urls),
            "product_urls": product_urls,
        }
        write_json(
            category_dir / f"{index:03d}_{category_id}.json",
            {
                "record": record,
                "raw_payload": {
                    "network_product_urls": discovery.network_product_urls,
                    "html_product_urls": discovery.html_product_urls,
                    "api_statuses": discovery.api_statuses,
                    "blocked_responses": discovery.blocked_responses,
                    "html_length": len(discovery.html),
                },
            },
        )
        print(f"[discover {index}] found {len(product_urls)} product URLs", file=sys.stderr, flush=True)
        categories.append(record)
    return categories


def _crawl_discovered_categories(crawler: ShopeeCrawler, categories: list[dict[str, object]]) -> list[object]:
    products: list[object] = []
    seen: set[str] = set()
    for category in categories:
        category_data = {
            "category_id": str(category.get("category_id", "")),
            "category_name": str(category.get("category_name", "")),
            "category_url": str(category.get("category_url", "")),
            "crawl_batch_id": str(category.get("batch_id", crawler.config.batch_id)),
        }
        for url in list(category.get("product_urls", [])):
            if not isinstance(url, str) or not url.strip():
                continue
            if url in seen and crawler.config.dedupe_across_categories:
                continue
            seen.add(url)
            print(f"[product {len(products) + 1}] {category_data['category_name']}: {url}", file=sys.stderr, flush=True)
            products.append(crawler.crawl_product(url, category_metadata=category_data))
            if len(products) >= crawler.config.max_products:
                return products
    return products


if __name__ == "__main__":
    main()
