# ShopeeFeed

ShopeeFeed is a review-based RAG demo for crawling Shopee product reviews, cleaning the dataset, and preparing product/review evidence for later embedding, retrieval, reranking, and answer generation.

The repository contains:

- React/Vite front end demo for product review QA.
- Python crawler pipeline for Shopee product metadata and reviews.
- Seed-category crawler for Shopee product/review collection from real product URLs.
- Category-page discovery crawler that requires a logged-in Shopee browser session.

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

## Recommended Shopee Workflow

Paste real Shopee product URLs under each category in `configs/crawl.example.yaml`, then crawl those products and reviews.

```yaml
categories:
  - category_name: "Điện Thoại & Phụ Kiện"
    category_url: ""
    product_urls:
      - "https://shopee.vn/product/313873802/19259739210"
```

Run:

```bash
python -m crawler.cli crawl-seed-categories --config configs/crawl.example.yaml
python -m crawler.cli preprocess-reviews
python -m crawler.cli validate-dataset
```

This keeps category metadata on every product and review without depending on Shopee category listing APIs.

## Category Page Discovery

Category discovery opens real Shopee category pages with Playwright and reuses a logged-in browser profile. Log in before running discovery; without a fresh session, Shopee commonly returns `403` or renders the page without product URLs.

Create or refresh the Shopee session:

```bash
python -m crawler.cli open-shopee-session --config configs/crawl.example.yaml
```

Log in inside the opened browser, return to the terminal, then press `Enter`. The session profile is stored in `data/browser/shopee-profile`, which is ignored by git.

Add category URLs to `configs/crawl.example.yaml`:

```yaml
categories:
  - category_name: "Balo & Túi Ví Nam"
    category_url: "https://shopee.vn/Balo-T%C3%BAi-V%C3%AD-Nam-cat.11035741"
    product_urls: []
```

Discover product URLs only:

```bash
python -m crawler.cli discover-categories --config configs/crawl.example.yaml
```

Discover category product URLs, then crawl product metadata and reviews:

```bash
python -m crawler.cli crawl-category-discovery --config configs/crawl.example.yaml
python -m crawler.cli preprocess-reviews
python -m crawler.cli validate-dataset
```

Discovery is controlled by `max_categories`, `max_products_per_category`, and `max_pages_per_category`.

## Direct Product Crawl

Add product URLs to `configs/crawl.example.yaml`, then run:

```bash
python -m crawler.cli crawl-products --config configs/crawl.example.yaml
python -m crawler.cli crawl-reviews --config configs/crawl.example.yaml
python -m crawler.cli preprocess-reviews
python -m crawler.cli validate-dataset
```

If Shopee blocks direct API calls, the product crawler falls back to rendering the product page with Playwright and using the browser context for the product/review API calls.

## Outputs

Generated crawl outputs are ignored by git:

- `data/raw/shopee/products/*.json`
- `data/raw/shopee/reviews/*.jsonl`
- `data/processed/categories.csv`
- `data/processed/products.csv`
- `data/processed/reviews.csv`
- `data/processed/reviews_clean.csv`

Tracked placeholders keep the folder structure available without committing raw data, browser profiles, or cookies.

## Dataset Contract

Product fields:

`product_id`, `name`, `category_id`, `category`, `category_url`, `crawl_batch_id`, `description`, `price`, `avg_rating`, `review_count`, `image_url`, `product_url`, `source`, `crawl_time`, `raw_file`

Clean review fields:

`review_id`, `product_id`, `text`, `text_normalized`, `rating`, `sentiment`, `created_at`, `source`, `category_id`, `category`, `category_url`, `crawl_batch_id`, `variant`, `helpful_count`, `media_count`, `review_url`, `crawl_time`, `raw_file`, `clean_status`, `dedupe_key`, `document_text`
