"""Autonomous web crawler for Lattice-Grow self-directed learning.

Crawls seed URLs in a background thread, extracts text via trafilatura,
and feeds it back to the training pipeline through a callback. Designed
for old laptops: single-threaded, sequential requests, configurable delay,
same-domain-only by default, hard cap on pages fetched.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Callable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
import trafilatura

logger = logging.getLogger(__name__)

DEFAULT_MAX_PAGES = 50
DEFAULT_MAX_DEPTH = 2
DEFAULT_DELAY = 2.0
DEFAULT_FETCH_TIMEOUT = 15
MAX_TEXT_PER_PAGE = 500_000  # chars, matches ingestion.py safety limit
USER_AGENT = "LatticeGrow/0.1 (research prototype; autonomous learning mode)"


@dataclass
class CrawlerConfig:
    """Configuration for AutonomousCrawler."""

    max_pages: int = DEFAULT_MAX_PAGES
    max_depth: int = DEFAULT_MAX_DEPTH
    delay_between_requests: float = DEFAULT_DELAY
    same_domain_only: bool = True
    fetch_timeout: int = DEFAULT_FETCH_TIMEOUT


class _LinkExtractor(HTMLParser):
    """Minimal HTML parser that collects href values from <a> tags."""

    def __init__(self) -> None:
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.links.append(value)


def _extract_links(html: str) -> List[str]:
    """Parse HTML and return all href values from <a> tags."""
    parser = _LinkExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    return parser.links


def _is_valid_url(url: str) -> bool:
    """Check if URL has http(s) scheme."""
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


class AutonomousCrawler:
    """Background web crawler that feeds extracted text to a callback.

    Usage:
        crawler = AutonomousCrawler(config)
        crawler.start(
            seed_urls=["https://example.com/topic"],
            text_callback=lambda text: pending_texts.append(text),
            status_callback=lambda msg: print(msg),
        )
        # ... later ...
        crawler.stop()
    """

    def __init__(self, config: Optional[CrawlerConfig] = None) -> None:
        self.config = config or CrawlerConfig()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pages_fetched = 0
        self._lock = threading.Lock()
        self._current_status: str = "Idle"

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def pages_fetched(self) -> int:
        with self._lock:
            return self._pages_fetched

    @property
    def status(self) -> str:
        with self._lock:
            return self._current_status

    def start(
        self,
        seed_urls: List[str],
        text_callback: Callable[[str], None],
        status_callback: Callable[[str], None],
    ) -> None:
        """Start crawling in a background daemon thread.

        Args:
            seed_urls: Initial URLs to crawl.
            text_callback: Called with extracted text for each page.
            status_callback: Called with human-readable status updates.
        """
        if self.is_running:
            logger.warning("Crawler already running, ignoring start()")
            return

        self._stop_event.clear()
        with self._lock:
            self._pages_fetched = 0
            self._current_status = "Starting..."

        self._thread = threading.Thread(
            target=self._crawl_loop,
            args=(seed_urls, text_callback, status_callback),
            daemon=True,
            name="lattice-crawler",
        )
        self._thread.start()
        logger.info("Crawler started with %d seed URLs", len(seed_urls))

    def stop(self) -> None:
        """Signal the crawler to stop after the current page."""
        self._stop_event.set()
        logger.info("Crawler stop requested")

    def _set_status(self, msg: str, status_callback: Callable[[str], None]) -> None:
        with self._lock:
            self._current_status = msg
        try:
            status_callback(msg)
        except Exception:
            pass

    def _fetch_page(self, url: str) -> Tuple[str, str]:
        """Fetch a URL and extract text + raw HTML.

        Returns:
            (extracted_text, raw_html) — both empty strings on failure.
        """
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded)
                if text:
                    return text[:MAX_TEXT_PER_PAGE], downloaded

            # Fallback: raw requests
            resp = requests.get(
                url,
                timeout=self.config.fetch_timeout,
                headers={"User-Agent": USER_AGENT},
            )
            resp.raise_for_status()
            raw = resp.text
            # Try trafilatura extraction on raw content
            text = trafilatura.extract(raw)
            if text:
                return text[:MAX_TEXT_PER_PAGE], raw
            # Last resort: use raw text directly
            return raw[:MAX_TEXT_PER_PAGE], raw

        except Exception as e:
            logger.warning("Failed to fetch %s: %s", url, e)
            return "", ""

    def _crawl_loop(
        self,
        seed_urls: List[str],
        text_callback: Callable[[str], None],
        status_callback: Callable[[str], None],
    ) -> None:
        """Main crawl loop running in background thread."""
        cfg = self.config
        visited: Set[str] = set()
        # Queue entries: (url, depth)
        queue: List[Tuple[str, int]] = []

        # Seed the queue
        allowed_domain: Optional[str] = None
        for url in seed_urls:
            url = url.strip()
            if not url or not _is_valid_url(url):
                continue
            queue.append((url, 0))
            if cfg.same_domain_only and allowed_domain is None:
                allowed_domain = urlparse(url).netloc

        self._set_status(
            f"Crawling: {len(queue)} seeds, max {cfg.max_pages} pages",
            status_callback,
        )

        while queue and not self._stop_event.is_set():
            with self._lock:
                if self._pages_fetched >= cfg.max_pages:
                    break

            url, depth = queue.pop(0)

            if url in visited:
                continue
            visited.add(url)

            # Same-domain filter
            if cfg.same_domain_only and allowed_domain:
                if urlparse(url).netloc != allowed_domain:
                    continue

            self._set_status(
                f"Fetching ({self._pages_fetched + 1}/{cfg.max_pages}): {url[:80]}",
                status_callback,
            )

            text, raw_html = self._fetch_page(url)

            if text and text.strip():
                try:
                    text_callback(text)
                except Exception as e:
                    logger.error("text_callback failed for %s: %s", url, e)

                with self._lock:
                    self._pages_fetched += 1

            # Discover links if under max depth
            if depth < cfg.max_depth and raw_html and not self._stop_event.is_set():
                links = _extract_links(raw_html)
                new_count = 0
                for link in links:
                    abs_url = urljoin(url, link)
                    if (
                        _is_valid_url(abs_url)
                        and abs_url not in visited
                        and (not cfg.same_domain_only or urlparse(abs_url).netloc == allowed_domain)
                    ):
                        queue.append((abs_url, depth + 1))
                        new_count += 1
                if new_count:
                    logger.debug("Discovered %d new links from %s", new_count, url)

            # Rate limiting
            if queue and not self._stop_event.is_set():
                time.sleep(cfg.delay_between_requests)

        final_msg = f"Crawl complete: {self._pages_fetched} pages fetched"
        if self._stop_event.is_set():
            final_msg += " (stopped early)"
        self._set_status(final_msg, status_callback)
        logger.info(final_msg)
