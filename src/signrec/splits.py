"""Signer-independent train/val/test splits, frozen as JSON in ``splits/``.

A split file stores the *signer* lists (and, for the leakage study, explicit sample ids), so it is
small, human-readable and versioned.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


