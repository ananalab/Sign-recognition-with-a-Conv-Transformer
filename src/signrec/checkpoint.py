"""Loading of training checkpoints (torch only, no ONNX dependency)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

