"""Paired significance tests between models evaluated on the same clips -> reports/stats.json.

Every comparison uses the per-clip predictions saved by signrec.evaluate (reports/eval/*.npz)
for the three seeds of each model:

- per-clip correctness averaged over seeds, compared with a paired bootstrap, either
  hierarchical over signers (``signer``) or over clips within signers (``clip``);
- an exact McNemar test on the predictions of the seed ensembles;
- the mean and standard deviation over seeds of the accuracy difference.

Comparisons: Conv-Transformer vs. BiGRU on the ASL test signers, and the ASL-initialised LSFB
models vs. the model trained from scratch on the LSFB test signers.

    uv run python scripts/stats_tests.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from signrec.metrics import mcnemar, paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "reports" / "eval"
SEEDS = (0, 1, 2)


def main() -> None:
    res: dict = {}
    tr, gru = load("asl-transformer", "test"), load("asl-gru", "test")
    if tr and gru:
        res["asl_transformer_vs_gru_test"] = compare(tr, gru, ("signer", "clip"))
    scratch = load("lsfb-scratch-nall", "test")
    for other in ("ft", "ftfreeze"):
        b = load(f"lsfb-{other}-nall", "test")
        if scratch and b:
            res[f"lsfb_{other}_vs_scratch_test"] = compare(b, scratch, ("signer", "clip"))
    out = ROOT / "reports" / "stats.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out.relative_to(ROOT)} ({', '.join(res)})")


if __name__ == "__main__":
    main()
