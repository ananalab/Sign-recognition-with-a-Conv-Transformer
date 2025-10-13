"""End-to-end check that the browser demo predicts exactly like PyTorch.

Serves app/web locally, opens it in headless Chromium (Playwright), classifies every example clip
through the demo's own code path (JavaScript preprocessing + ONNX Runtime Web) and compares the
probabilities with the PyTorch checkpoint. Writes reports/browser_parity.json.

    uv run python scripts/check_browser_parity.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

