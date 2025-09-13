"""Classical baselines on temporal statistics of the preprocessed landmarks.

    uv run python scripts/run_baseline.py --data data/processed/kaggle_asl \
        --split splits/kaggle_asl_signer.json [--models logreg lgbm] [--eval-test]

Writes ``reports/baselines/<dataset>_<split>.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

