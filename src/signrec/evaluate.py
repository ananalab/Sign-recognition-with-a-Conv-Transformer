"""Evaluate a checkpoint on a split and write a Markdown + JSON report.

    uv run python -m signrec.evaluate --ckpt models/<run>/best.pt --data data/processed/<ds> \
        --split splits/<ds>_signer.json --subset val [--gifs 20]

Outputs ``reports/eval/<run>_<subset>.{md,json,npz}`` (the npz keeps probabilities for figures).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

