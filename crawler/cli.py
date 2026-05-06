from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .category import ShopeeCategoryCrawler
from .browser import discover_category_products, save_shopee_session
from .preprocess import preprocess
from .report import write_phase1_report
from .sample_data import create_sample_data
from .shopee import ShopeeCrawler
from .utils import ensure_dir, write_json
from .validate import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="ShopeeFeed Phase 1 crawl data pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("crawl-products", "crawl-reviews"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", default="configs/crawl.example.yaml")
        command.add_argument("--raw-dir", default="data/raw/shopee")

    for name in ("crawl-categories", "crawl-category-products", "crawl-category-reviews", "crawl-phase2", "debug-category"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", default="configs/crawl.example.yaml")
        command.add_argument("--raw-dir", default="data/raw/shopee")

    session_parser = subparsers.add_parser("save-shopee-session")
    session_parser.add_argument("--config", default="configs/crawl.example.yaml")

    preprocess_parser = subparsers.add_parser("preprocess-reviews")
    preprocess_parser.add_argument("--raw-dir", default="data/raw/shopee")
    preprocess_parser.add_argument("--processed-dir", default="data/processed")

    validate_parser = subparsers.add_parser("validate-dataset")
    validate_parser.add_argument("--processed-dir", default="data/processed")

    sample_parser = subparsers.add_parser("sample-data")
    sample_parser.add_argument("--raw-dir", default="data/raw/shopee")

    report_parser = subparsers.add_parser("phase1-report")
    report_parser.add_argument("--processed-dir", default="data/processed")
    report_parser.add_argument("--report-dir", default="data/reports")

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
    elif args.command == "preprocess-reviews":
        _print(preprocess(args.raw_dir, args.processed_dir))
    elif args.command == "validate-dataset":
        result = validate_dataset(args.processed_dir)
        _print(result)
        if not result["ok"]:
            raise SystemExit(1)
    elif args.command == "sample-data":
        _print(create_sample_data(args.raw_dir))
    elif args.command == "phase1-report":
        path = write_phase1_report(args.processed_dir, args.report_dir)
        _print({"report": str(Path(path))})
    elif args.command == "save-shopee-session":
        config = load_config(args.config)
        save_shopee_session(
            login_url=config.session_login_url,
            auth_state_path=config.auth_state_path,
            user_data_dir=config.user_data_dir,
            timeout_ms=config.timeout_seconds * 1000,
        )
        _print({"auth_state_path": config.auth_state_path, "user_data_dir": config.user_data_dir})
    elif args.command == "crawl-categories":
        config = load_config(args.config)
        categories = ShopeeCategoryCrawler(config, args.raw_dir).crawl_categories()
        _write_phase2_checkpoint(config.batch_id, {"categories": len(categories)})
        _print({"categories": len(categories)})
    elif args.command == "crawl-category-products":
        config = load_config(args.config)
        category_crawler = ShopeeCategoryCrawler(config, args.raw_dir)
        crawler = ShopeeCrawler(config, args.raw_dir)
        categories = category_crawler.crawl_categories()
        products = _crawl_category_products(crawler, categories)
        _write_phase2_checkpoint(config.batch_id, {"categories": len(categories), "products": len(products)})
        _print({"categories": len(categories), "products": len(products)})
    elif args.command == "crawl-category-reviews":
        config = load_config(args.config)
        category_crawler = ShopeeCategoryCrawler(config, args.raw_dir)
        crawler = ShopeeCrawler(config, args.raw_dir)
        categories = category_crawler.crawl_categories()
        products = _crawl_category_products(crawler, categories)
        count = crawler.crawl_reviews(products)
        _write_phase2_checkpoint(
            config.batch_id,
            {"categories": len(categories), "products": len(products), "reviews": count},
        )
        _print({"categories": len(categories), "products": len(products), "reviews": count})
    elif args.command == "crawl-phase2":
        config = load_config(args.config)
        category_crawler = ShopeeCategoryCrawler(config, args.raw_dir)
        crawler = ShopeeCrawler(config, args.raw_dir)
        categories = category_crawler.crawl_categories()
        products = _crawl_category_products(crawler, categories)
        reviews = crawler.crawl_reviews(products)
        _write_phase2_checkpoint(
            config.batch_id,
            {"categories": len(categories), "products": len(products), "reviews": reviews},
        )
        _print({"categories": len(categories), "products": len(products), "reviews": reviews})
    elif args.command == "debug-category":
        config = load_config(args.config)
        results = []
        for url in config.category_urls[: config.max_categories]:
            discovery = discover_category_products(
                url,
                timeout_ms=config.timeout_seconds * 1000,
                max_scrolls=config.max_pages_per_category,
                max_products=config.max_products_per_category,
                auth_state_path=config.auth_state_path,
                user_data_dir=config.user_data_dir,
            )
            product_urls = discovery.product_urls
            results.append(
                {
                    "url": url,
                    "title": discovery.title,
                    "html_length": len(discovery.html),
                    "product_urls": len(product_urls),
                    "network_product_urls": len(discovery.network_product_urls),
                    "html_product_urls": len(discovery.html_product_urls),
                    "api_statuses": _compact_statuses(discovery.api_statuses),
                    "blocked_responses": discovery.blocked_responses[:5],
                    "sample_product_urls": product_urls[:5],
                    "diagnostic": _category_diagnostic(discovery),
                },
            )
        _print({"categories": results})


def _print(payload: object) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8") + b"\n")


def _write_phase2_checkpoint(batch_id: str, payload: dict[str, int]) -> None:
    state_dir = ensure_dir("data/state")
    write_json(state_dir / f"{batch_id}.checkpoint.json", payload)


def _compact_statuses(statuses: dict[str, int]) -> dict[str, int]:
    compact: dict[str, int] = {}
    for url, status in statuses.items():
        key = url.split("?")[0].replace("https://shopee.vn", "")
        compact[key] = status
    return compact


def _category_diagnostic(discovery: object) -> str:
    product_urls = getattr(discovery, "product_urls", [])
    if product_urls:
        return "ok"
    blocked = getattr(discovery, "blocked_responses", [])
    if blocked:
        return "Shopee blocked one or more listing/category API responses."
    statuses = getattr(discovery, "api_statuses", {})
    if statuses:
        return "Shopee returned API responses, but no shopid/itemid objects were found."
    return "No relevant Shopee listing API responses were observed."


def _crawl_category_products(crawler: ShopeeCrawler, categories: list[object]) -> list[object]:
    products: list[object] = []
    seen: set[str] = set()
    for category in categories:
        category_data = {
            "category_id": getattr(category, "category_id", ""),
            "category_name": getattr(category, "category_name", ""),
            "category_url": getattr(category, "category_url", ""),
            "crawl_batch_id": getattr(category, "batch_id", ""),
        }
        for url in getattr(category, "product_urls", []):
            if url in seen and crawler.config.dedupe_across_categories:
                continue
            seen.add(url)
            products.append(crawler.crawl_product(url, category_metadata=category_data))
    return products


if __name__ == "__main__":
    main()
