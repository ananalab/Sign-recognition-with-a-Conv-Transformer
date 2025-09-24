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

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from signrec.baseline import build_stat_matrix  # torch-free: avoids an OpenMP clash with LightGBM
from signrec.metrics import summary
from signrec.preprocess import PreprocessConfig
from signrec.splits import load_split, split_indices
from signrec.store import PackedDataset

ROOT = Path(__file__).resolve().parents[1]


def fit_logreg(X: np.ndarray, y: np.ndarray, seed: int) -> LogisticRegression:
    clf = LogisticRegression(C=0.5, max_iter=500, random_state=seed)
    return clf.fit(X, y)


