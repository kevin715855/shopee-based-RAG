# ShopeeFeed

ShopeeFeed is a review-based RAG demo for crawling e-commerce product reviews, cleaning the dataset, and preparing product/review evidence for later embedding, retrieval, reranking, and answer generation.

The repository currently contains:

- React/Vite front end demo for product review QA.
- Python crawler pipeline for Shopee product metadata and reviews.
- Phase 2 category crawler with Playwright network discovery and Shopee session support.

## Quick Start

Install front-end dependencies and run the demo:

```bash
npm install
npm run dev
```

Build the front end:

```bash
npm run build
```

Install crawler dependencies:

```bash
pip install -r requirements-crawler.txt
python -m playwright install chromium
```

Run the offline sample pipeline:

```bash
python -m crawler.cli sample-data
python -m crawler.cli preprocess-reviews
python -m crawler.cli validate-dataset
python -m crawler.cli phase1-report
```

## Shopee Product Crawl

Add product URLs to `configs/crawl.example.yaml`, then run:

```bash
python -m crawler.cli crawl-products --config configs/crawl.example.yaml
python -m crawler.cli crawl-reviews --config configs/crawl.example.yaml
python -m crawler.cli preprocess-reviews
python -m crawler.cli validate-dataset
python -m crawler.cli phase1-report
```

If Shopee blocks direct API calls, the product crawler falls back to rendering the product page with Playwright and capturing the browser's `api/v4/pdp/get_pc` response.

## Shopee Category Crawl

Add category URLs to `configs/crawl.example.yaml`.

Create a Shopee browser session first:

```bash
python -m crawler.cli save-shopee-session --config configs/crawl.example.yaml
```

This opens Chromium. Log in to Shopee in the opened browser, then return to the terminal and press `Enter`. The session is stored under `data/session/` and `data/browser/`, both ignored by git.

Check whether category discovery can see product URLs:

```bash
python -m crawler.cli debug-category --config configs/crawl.example.yaml
```

Run category crawl:

```bash
python -m crawler.cli crawl-phase2 --config configs/crawl.example.yaml
python -m crawler.cli preprocess-reviews
python -m crawler.cli validate-dataset
python -m crawler.cli phase1-report
```

Shopee may still return `403` for listing APIs. In that case, use `product_urls` seeds in the config.

## Outputs

Generated crawl outputs are ignored by git:

- `data/raw/shopee/products/*.json`
- `data/raw/shopee/reviews/*.jsonl`
- `data/processed/products.csv`
- `data/processed/reviews.csv`
- `data/processed/reviews_clean.csv`
- `data/reports/phase1_report.json`

Tracked placeholders keep the folder structure available without committing raw data, browser profiles, cookies, or reports.

## Dataset Contract

Product fields:

`product_id`, `name`, `category_id`, `category`, `category_url`, `crawl_batch_id`, `description`, `price`, `avg_rating`, `review_count`, `image_url`, `product_url`, `source`, `crawl_time`, `raw_file`

Clean review fields:

`review_id`, `product_id`, `text`, `text_normalized`, `rating`, `sentiment`, `created_at`, `source`, `category_id`, `category`, `category_url`, `crawl_batch_id`, `variant`, `helpful_count`, `media_count`, `review_url`, `crawl_time`, `raw_file`, `clean_status`, `dedupe_key`, `document_text`

