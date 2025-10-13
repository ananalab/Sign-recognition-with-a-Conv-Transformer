"""The script run on Kaggle must behave exactly like the package code."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

from signrec.io.kaggle import load_kaggle_sequence

ROOT = Path(__file__).resolve().parents[1]
ENTRY = "# " + "-" * 64 + " kernel entry point"


def _load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


