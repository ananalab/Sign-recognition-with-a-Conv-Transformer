"""ONNX export, dynamic INT8 quantisation and CPU benchmark of a trained checkpoint.

    uv run python -m signrec.export --ckpt models/<run>/best.pt [--out app/web/models]

The exported graph takes ``x [B, T, F] float32`` and ``mask [B, T] float32`` (1 = real frame)
and returns ``probs [B, C]``; a float mask keeps the model easy to feed from JavaScript.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn

from signrec.checkpoint import load_checkpoint
from signrec.preprocess import PreprocessConfig, preprocess
from signrec.splits import load_split, split_indices
from signrec.store import PackedDataset


