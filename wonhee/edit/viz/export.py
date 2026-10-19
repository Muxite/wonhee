"""Headless-screenshot a self-contained viz HTML page to a PNG.

Playwright is an optional extra (`uv sync --extra viz`, then `uv run playwright install chromium`), imported
lazily so the rest of wonhee never needs a browser. The force-directed graph sets window.__vizkitSettled once
its simulation has ended and the final fit has run; waiting for that flag instead of a fixed delay is what
stopped small graphs from being captured mid-drift (fixed in vizkit 2026-10-19, kept here).
"""
from __future__ import annotations

import os
from pathlib import Path


class ExportUnavailable(RuntimeError):
    """Playwright or its Chromium is missing."""


def export_png(html_path: Path, png_path: Path, width: int = 1600, height: int = 1000,
               settle_ms: int = 3500, max_settle_ms: int = 30000) -> None:
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise ExportUnavailable("PNG export needs Playwright: uv sync --extra viz && uv run playwright install "
                                "chromium") from e
    # WONHEE_CHROMIUM_PATH points at an already-installed Chromium when the bundled one does not match
    exe = os.environ.get("WONHEE_CHROMIUM_PATH") or None
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(executable_path=exe)
        except PlaywrightError as e:
            raise ExportUnavailable(f"Chromium did not start ({str(e).splitlines()[0]}); run `uv run playwright "
                                    "install chromium` or set WONHEE_CHROMIUM_PATH") from e
        try:
            page = browser.new_page(viewport={"width": width, "height": height})
            page.goto(html_path.resolve().as_uri())
            page.wait_for_timeout(settle_ms)
            if page.evaluate("'__vizkitSettled' in window"):
                page.wait_for_function("window.__vizkitSettled === true", timeout=max_settle_ms)
            page.screenshot(path=str(png_path))
        finally:
            browser.close()
