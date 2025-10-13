"""PyTorch dataset over a packed dataset: augmentation (training only), then preprocess()."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

