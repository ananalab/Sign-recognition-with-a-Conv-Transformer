"""Write experiments/lsfb_{scratch,transfer}.yaml: LSFB learning curves (scratch vs. ASL).

The number of epochs grows as the training set shrinks so that every run performs a
comparable number of updates; small runs load data in the main process.

    uv run python scripts/gen_lsfb_experiments.py
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SIZES = {5: 300, 10: 200, 25: 120, 50: 80, 100: 60, "all": 60}  # examples per sign: epochs
SEEDS = (0, 1, 2)
PRETRAINED = "models/asl-transformer-s0/best.pt"
DATA = ["data.root=data/processed/lsfb_isol", "data.split=splits/lsfb_isol_signer.json",
        "train.cache_val=true"]  # fmt: skip


