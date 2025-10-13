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

def missing_rate(seq: np.ndarray) -> float:
    """Share of frames where neither hand is detected."""
    return float((~hand_presence(seq).any(axis=1)).mean())


def cpu_latency(model: torch.nn.Module, x: np.ndarray, m: np.ndarray, n: int = 200) -> dict:
    model = model.cpu().eval()
    torch.set_num_threads(1)
    xt, mt = torch.from_numpy(x[None]), torch.from_numpy(m[None])
    times = []
    with torch.no_grad():
        for _ in range(10):
            model(xt, mt)
        for _ in range(n):
            t0 = time.perf_counter()
            model(xt, mt)
            times.append((time.perf_counter() - t0) * 1e3)
    return {
        "model_p50_ms": float(np.median(times)),
        "model_p95_ms": float(np.percentile(times, 95)),
    }


