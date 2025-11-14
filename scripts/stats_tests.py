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

