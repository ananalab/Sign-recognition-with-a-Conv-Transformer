"""Synthetic fixtures (no real data in the repository)."""

from __future__ import annotations

# onnxruntime must be loaded before torch: in the reverse order, the process intermittently
# aborts at interpreter exit on macOS ("recursive_mutex lock failed") once both have run.
import onnxruntime  # noqa: F401  # isort: skip

import numpy as np
import pandas as pd
import pytest

from signrec.io.kaggle import KAGGLE_OFFSETS

SIZES = {"face": 468, "left_hand": 21, "pose": 33, "right_hand": 21}


def make_kaggle_long(n_frames: int = 5, missing_left: bool = True, seed: int = 0) -> pd.DataFrame:
    """Kaggle-like long table whose x encodes the 543-point index (for order checks)."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_frames):
        for t in ("face", "left_hand", "pose", "right_hand"):
            for i in range(SIZES[t]):
                g = KAGGLE_OFFSETS[t] + i
                x = np.nan if (missing_left and t == "left_hand") else float(g)
                rows.append((100 + f, f"{100 + f}-{t}-{i}", t, i, x, rng.random(), rng.random()))
    return pd.DataFrame(rows, columns=["frame", "row_id", "type", "landmark_index", "x", "y", "z"])


