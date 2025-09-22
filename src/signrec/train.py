"""Training entry point.

    uv run python -m signrec.train --config configs/train.yaml [key=value ...]

Writes ``models/<run>/best.pt`` (weights + everything needed for inference) and
``reports/runs/<run>/{config.yaml,history.csv,metrics.json}``; also logs to MLflow (``mlruns/``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

