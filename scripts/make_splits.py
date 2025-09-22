"""Freeze the evaluation splits in splits/ (versioned).

uv run python scripts/make_splits.py --dataset kaggle_asl
uv run python scripts/make_splits.py --dataset lsfb_isol
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from signrec.splits import random_split, save_split, signer_split, split_indices

def lsfb_selection_report(index: pd.DataFrame, split: dict) -> None:
    """Vocabulary-selection statistics quoted in the paper."""
    inst = pd.read_csv(ROOT / "data/raw/lsfb_isol/instances.csv")
    selected = inst[inst["sign"].isin(set(index["label"]))]
    tr = split_indices(index, split)["train"]
    per_class = index.iloc[tr]["label"].value_counts()
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "lsfb_selection.json").write_text(
        json.dumps(
            {
                "all_instances": int(len(inst)),
                "all_glosses": int(inst["sign"].nunique()),
                "all_signers": int(inst["signer"].nunique()),
                "selected_instances": int(len(selected)),
                "kept_instances": int(len(index)),
                "dropped_share": float(1 - len(index) / len(selected)),
            },
            indent=1,
        )
    )
    (reports / "lsfb_train_per_class.json").write_text(
        json.dumps(
            {
                "min": int(per_class.min()),
                "median": float(per_class.median()),
                "max": int(per_class.max()),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
