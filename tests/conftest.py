"""Synthetic fixtures (no real data in the repository)."""

from __future__ import annotations

# onnxruntime must be loaded before torch: in the reverse order, the process intermittently
# aborts at interpreter exit on macOS ("recursive_mutex lock failed") once both have run.
import onnxruntime  # noqa: F401  # isort: skip

import numpy as np
import pandas as pd
import pytest

from signrec.io.kaggle import KAGGLE_OFFSETS

