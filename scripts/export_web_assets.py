"""Generate the JavaScript constants and the parity fixtures of the browser demo.

    uv run python scripts/export_web_assets.py [--model asl=models/<run>/best.pt ...]

- ``app/web/js/landmarks.js``: unified subset, mirror permutation and skeleton edges
- ``tests/fixtures/js_parity.json``: raw sequences + expected ``preprocess()`` outputs
- with ``--model``: ``app/web/models/<id>/`` (INT8 ONNX, labels, preprocessing config) and
  ``app/web/models/index.json``
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

