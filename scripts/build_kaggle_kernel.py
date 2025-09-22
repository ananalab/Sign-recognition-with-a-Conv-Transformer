"""Build (and optionally push) the Kaggle script that converts asl-signs to the unified format.

The kernel inlines the exact source of ``landmarks.py``, ``io/kaggle.py`` and ``store.py``,
so the conversion run on Kaggle is the code of this repository (checked by tests).

    uv run python scripts/build_kaggle_kernel.py [--push]
    kaggle kernels status <user>/asl-signs-unified-subset
    kaggle kernels output <user>/asl-signs-unified-subset -p data/processed/
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

