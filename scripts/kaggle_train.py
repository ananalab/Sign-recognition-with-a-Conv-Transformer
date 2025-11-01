"""Run a list of training experiments on a Kaggle GPU and fetch the results.

The current source tree (src/, configs/, splits/) is embedded in the kernel script as a tarball,
so what runs on Kaggle is exactly the local code. Data come from Kaggle sources:
the conversion kernel output (asl-signs) and/or a private dataset with the processed LSFB data.

    uv run python scripts/kaggle_train.py --experiments experiments/asl_main.yaml --push
    uv run python scripts/kaggle_train.py --experiments experiments/asl_main.yaml --fetch

Experiment file (YAML)::

    name: asl-main               # kernel slug suffix
    sources: [kaggle_asl]        # kaggle_asl and/or lsfb_isol
    pretrained_from: []          # optional kernels whose models/*/best.pt are made available
    runs:
      - run_name: asl-transformer-s0
        overrides: [seed=0]
        evaluate: [val, test]    # optional
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

RUNNER = r'''
import base64, glob, io, json, os, shutil, subprocess, sys, tarfile

CODE = "__CODE__"
SPEC = json.loads(r"""__SPEC__""")
WORK = "/kaggle/working/repo"
OUT = "/kaggle/working/out"

tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE)), mode="r:gz").extractall(WORK)
os.makedirs(f"{WORK}/data/processed", exist_ok=True)
for name in SPEC["sources"]:
    slug = name.replace("_", "-")
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}/index.parquet", recursive=True)
                  + glob.glob(f"/kaggle/input/**/{slug}-unified/index.parquet", recursive=True))
    if not hits:
        sys.exit(f"source {name} not mounted")
    os.symlink(os.path.dirname(hits[0]), f"{WORK}/data/processed/{name}")
for kernel in SPEC.get("pretrained_from", []):
    pattern = f"/kaggle/input/**/{kernel.split('/')[-1]}/**/models/*/best.pt"
    for ck in glob.glob(pattern, recursive=True):
        dst = f"{WORK}/models/{os.path.basename(os.path.dirname(ck))}"
        os.makedirs(dst, exist_ok=True)
        shutil.copy(ck, f"{dst}/best.pt")
for mod in ("omegaconf", "tabulate"):
    try:
        __import__(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", mod], check=True)

def pack_code() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for item in PACKAGED:
            tar.add(ROOT / item, arcname=item, filter=_skip_cache)
    return base64.b64encode(buf.getvalue()).decode()


def _skip_cache(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    return None if "__pycache__" in info.name else info


if __name__ == "__main__":
    main()
