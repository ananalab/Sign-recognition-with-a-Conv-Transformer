"""Screenshot of the browser demo replaying an LSFB-ISOL example clip.

Serves app/web locally, opens it in headless Chromium (Playwright), replays example clips until
the requested gloss comes up and saves the page to paper/figures/demo.png and
reports/figures/demo.png.

    uv run python scripts/screenshot_demo.py [--gloss EQUIPE]
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
from pathlib import Path

from playwright.async_api import async_playwright

