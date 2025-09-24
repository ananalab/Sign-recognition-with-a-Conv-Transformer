"""Training entry point.

    uv run python -m signrec.train --config configs/train.yaml [key=value ...]

Writes ``models/<run>/best.pt`` (weights + everything needed for inference) and
``reports/runs/<run>/{config.yaml,history.csv,metrics.json}``; also logs to MLflow (``mlruns/``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from signrec.augment import AugmentConfig
from signrec.dataset import SignDataset
from signrec.metrics import summary
from signrec.models import build_model
from signrec.models.common import count_parameters
from signrec.preprocess import PreprocessConfig
from signrec.splits import load_split, split_indices
from signrec.store import PackedDataset

ROOT = Path(__file__).resolve().parents[2]


