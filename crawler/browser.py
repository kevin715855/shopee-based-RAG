from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re
from urllib.parse import quote, urlencode, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup


@dataclass(slots=True)
class CategoryDiscoveryResult:
    html: str = ""
    title: str = ""
    network_product_urls: list[str] = field(default_factory=list)
    html_product_urls: list[str] = field(default_factory=list)
    api_statuses: dict[str, int] = field(default_factory=dict)
    blocked_responses: list[str] = field(default_factory=list)

    @property
    def product_urls(self) -> list[str]:
        return sorted(set(self.network_product_urls + self.html_product_urls))


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def render_html(
    url: str,
    timeout_ms: int = 20000,
    wait_until: str = "domcontentloaded",
    auth_state_path: str | None = None,
    user_data_dir: str | None = None,
) -> str:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed. Run `pip install -r requirements-crawler.txt` "
            "and `python -m playwright install chromium`.",
        ) from exc

    normalized_url = normalize_url_for_browser(url)
    with sync_playwright() as playwright:
        context, browser = _open_context(
            playwright,
            headless=True,
            auth_state_path=auth_state_path,
            user_data_dir=user_data_dir,
        )
        page = context.new_page()
        page.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in {"image", "media", "font"}
            else route.continue_(),
        )
        try:
            page.goto(normalized_url, wait_until=wait_until, timeout=timeout_ms)
            page.wait_for_timeout(2500)
        except PlaywrightTimeoutError:
            # Shopee often keeps network connections open. The partial DOM is still useful.
            pass
        html = page.content()
        context.close()
        if browser is not None:
            browser.close()
        return html


def discover_category_products(
    url: str,
    timeout_ms: int = 20000,
    max_scrolls: int = 2,
    max_products: int = 20,
    auth_state_path: str | None = None,
    user_data_dir: str | None = None,
) -> CategoryDiscoveryResult:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed. Run `pip install -r requirements-crawler.txt` "
            "and `python -m playwright install chromium`.",
        ) from exc

    normalized_url = normalize_url_for_browser(url)
    result = CategoryDiscoveryResult()
    network_urls: set[str] = set()

    with sync_playwright() as playwright:
        context, browser = _open_context(
            playwright,
            headless=True,
            auth_state_path=auth_state_path,
            user_data_dir=user_data_dir,
        )
        page = context.new_page()
        page.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in {"image", "media", "font"}
            else route.continue_(),
        )

        def on_response(response: Any) -> None:
            response_url = response.url
            if not _is_relevant_shopee_response(response_url):
                return
            result.api_statuses[response_url] = response.status
            if response.status in {401, 403, 429}:
                result.blocked_responses.append(response_url)
            try:
                payload = response.json()
            except Exception:  # noqa: BLE001 - non-JSON responses are expected on Shopee pages
                return
            for product_url in extract_product_urls_from_payload(payload):
                network_urls.add(product_url)

        page.on("response", on_response)
        try:
            page.goto(normalized_url, wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_timeout(2500)
        except PlaywrightTimeoutError:
            pass

        for _ in range(max(0, max_scrolls)):
            if len(network_urls) >= max_products:
                break
            page.mouse.wheel(0, 1400)
            page.wait_for_timeout(1500)

        result.html = page.content()
        context.close()
        if browser is not None:
            browser.close()

    result.title = parse_title_from_html(result.html)
    result.html_product_urls = extract_product_urls_from_html(result.html)[:max_products]
    result.network_product_urls = sorted(network_urls)[:max_products]
    return result


def save_shopee_session(
    login_url: str,
    auth_state_path: str,
    user_data_dir: str,
    timeout_ms: int = 120000,
) -> None:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed. Run `pip install -r requirements-crawler.txt` "
            "and `python -m playwright install chromium`.",
        ) from exc

    Path(auth_state_path).parent.mkdir(parents=True, exist_ok=True)
    Path(user_data_dir).mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
            locale="vi-VN",
            user_agent=DEFAULT_USER_AGENT,
            viewport={"width": 1366, "height": 900},
        )
        page = context.pages[0] if context.pages else context.new_page()
        try:
            page.goto(normalize_url_for_browser(login_url), wait_until="domcontentloaded", timeout=timeout_ms)
        except PlaywrightTimeoutError:
            pass
        print("Login Shopee in the opened browser window, then return here and press Enter.")
        input()
        context.storage_state(path=auth_state_path)
        context.close()


def fetch_json_with_browser_context(
    url: str,
    params: dict[str, Any],
    timeout_ms: int = 20000,
    auth_state_path: str | None = None,
    user_data_dir: str | None = None,
) -> Any:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed. Run `pip install -r requirements-crawler.txt` "
            "and `python -m playwright install chromium`.",
        ) from exc

    query = urlencode(params)
    target_url = f"{url}?{query}" if query else url
    with sync_playwright() as playwright:
        context, browser = _open_context(
            playwright,
            headless=True,
            auth_state_path=auth_state_path,
            user_data_dir=user_data_dir,
        )
        response = context.request.get(
            target_url,
            headers={
                "accept": "application/json",
                "accept-language": "vi-VN,vi;q=0.9,en;q=0.8",
                "referer": "https://shopee.vn/",
            },
            timeout=timeout_ms,
        )
        if not response.ok:
            context.close()
            if browser is not None:
                browser.close()
            raise RuntimeError(f"Browser context request failed: HTTP {response.status} {target_url}")
        payload = response.json()
        context.close()
        if browser is not None:
            browser.close()
        return payload


def fetch_product_api_by_rendering_page(
    product_url: str,
    api_path: str,
    timeout_ms: int = 30000,
    auth_state_path: str | None = None,
    user_data_dir: str | None = None,
) -> Any:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed. Run `pip install -r requirements-crawler.txt` "
            "and `python -m playwright install chromium`.",
        ) from exc

    payloads: list[Any] = []
    with sync_playwright() as playwright:
        context, browser = _open_context(
            playwright,
            headless=True,
            auth_state_path=auth_state_path,
            user_data_dir=user_data_dir,
        )
        page = context.new_page()

        def on_response(response: Any) -> None:
            if api_path not in response.url or response.status != 200:
                return
            try:
                payloads.append(response.json())
            except Exception:
                return

        page.on("response", on_response)
        try:
            page.goto(normalize_url_for_browser(product_url), wait_until="domcontentloaded", timeout=timeout_ms)
        except PlaywrightTimeoutError:
            pass
        page.wait_for_timeout(7000)
        context.close()
        if browser is not None:
            browser.close()

    if not payloads:
        raise RuntimeError(f"No {api_path} response observed while rendering {product_url}")
    return payloads[0]


def normalize_url_for_browser(url: str) -> str:
    parts = urlsplit(url.strip())
    path = quote(parts.path, safe="/%.-_~")
    query = quote(parts.query, safe="=&?/%.-_~:+")
    return urlunsplit((parts.scheme, parts.netloc, path, query, parts.fragment))


def _open_context(
    playwright: Any,
    headless: bool,
    auth_state_path: str | None,
    user_data_dir: str | None,
) -> tuple[Any, Any | None]:
    if user_data_dir:
        Path(user_data_dir).mkdir(parents=True, exist_ok=True)
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=headless,
            args=["--disable-blink-features=AutomationControlled"],
            locale="vi-VN",
            user_agent=DEFAULT_USER_AGENT,
            viewport={"width": 1366, "height": 900},
        )
        return context, None

    browser = playwright.chromium.launch(
        headless=headless,
        args=["--disable-blink-features=AutomationControlled"],
    )
    state = auth_state_path if auth_state_path and Path(auth_state_path).exists() else None
    context = browser.new_context(
        locale="vi-VN",
        user_agent=DEFAULT_USER_AGENT,
        viewport={"width": 1366, "height": 900},
        storage_state=state,
    )
    return context, browser


def parse_title_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("title")
    return title.get_text(strip=True) if title else ""


def extract_product_urls_from_html(html: str, base_url: str = "https://shopee.vn") -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls: set[str] = set()
    for tag in soup.find_all("a", href=True):
        href = str(tag["href"])
        if "shopee.vn" in href and "-i." in href:
            urls.add(_normalize_shopee_url(href, base_url))
        elif href.startswith("/"):
            joined = urljoin(base_url, href)
            if "-i." in joined:
                urls.add(_normalize_shopee_url(joined, base_url))

    for match in re.findall(r"https?://(?:[a-z0-9-]+\.)?shopee\.vn/[^\"'\\s<>]+?-i\.\d+\.\d+(?:\?[^\"'\\s<>]+)?", html, flags=re.I):
        urls.add(_normalize_shopee_url(match, base_url))
    return sorted(urls)


def extract_product_urls_from_payload(payload: Any) -> list[str]:
    urls: set[str] = set()
    for item in _walk_dicts(payload):
        candidate = item.get("item_basic") if isinstance(item.get("item_basic"), dict) else item
        shop_id = candidate.get("shopid") or candidate.get("shop_id")
        item_id = candidate.get("itemid") or candidate.get("item_id")
        if shop_id and item_id:
            urls.add(f"https://shopee.vn/product/{shop_id}/{item_id}")
    return sorted(urls)


def _walk_dicts(value: Any) -> list[dict[str, Any]]:
    dicts: list[dict[str, Any]] = []
    if isinstance(value, dict):
        dicts.append(value)
        for child in value.values():
            dicts.extend(_walk_dicts(child))
    elif isinstance(value, list):
        for child in value:
            dicts.extend(_walk_dicts(child))
    return dicts


def _is_relevant_shopee_response(url: str) -> bool:
    if "shopee.vn/api/" not in url:
        return False
    tokens = (
        "search",
        "category",
        "recommend",
        "item",
        "collection",
        "homepage",
    )
    return any(token in url for token in tokens)


def _normalize_shopee_url(url: str, base_url: str) -> str:
    cleaned = url.split("#", 1)[0]
    return cleaned if cleaned.startswith("http") else urljoin(base_url, cleaned)
