"""Classical baselines: per-coordinate temporal statistics + linear model / gradient boosting."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

