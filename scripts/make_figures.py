"""Generate every figure and LaTeX table of the paper from data/ and reports/.

    uv run python scripts/make_figures.py [--only data|results]

Figures go to paper/figures/ (PDF + PNG), tables to paper/tables/. Nothing is typed by hand:
each number in the paper comes from a file written by the training/evaluation scripts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

def sign_gifs(dss: dict[str, PackedDataset]) -> None:
    """Animated skeleton of a typical clip for a few signs, in reports/figures/."""
    for name, ds in dss.items():
        for sign in GIF_SIGNS.get(name, []):
            out = ROOT / "reports" / "figures" / f"{name}_{sign}.gif"
            viz.save_gif(ds.sequence(typical_clip(ds, sign)), out, title=sign)


def data_analyses(dss: dict[str, PackedDataset]) -> dict:
    """Numbers quoted in the data section, written to reports/dataset_analyses.json."""
    out: dict = {}
    if "kaggle_asl" in dss:
        ds = dss["kaggle_asl"]
        out["nose_ratio_asl"] = nose_height_ratio(ds, 1.0)
        # frames without any hand: before/after the sign (removed by trimming) or within it
        rng = np.random.default_rng(0)
        outside = total = 0
        for r in rng.permutation(len(ds))[:5000]:
            s = ds.sequence(int(r))
            none = np.isnan(s[:, :42, 0]).all(1)
            if none.all():
                outside += int(none.sum())
            elif none.any():
                idx = np.flatnonzero(~none)
                outside += int(none[: idx[0]].sum() + none[idx[-1] + 1 :].sum())
            total += int(none.sum())
        out["asl_handless_outside_sign_share"] = outside / max(total, 1)
        T = PreprocessConfig().T
        lengths = [
            trim_handless(ds.sequence(int(r))).shape[0] for r in rng.permutation(len(ds))[:5000]
        ]
        out["asl_longer_than_T_share"] = float(np.mean(np.array(lengths) > T))
        out["asl_handless_sample_clips"] = 5000
    if "lsfb_isol" in dss:
        ds = dss["lsfb_isol"]
        out["nose_ratio_lsfb_raw"] = nose_height_ratio(ds, 1.0)
        out["nose_ratio_lsfb_corrected"] = nose_height_ratio(ds, ds.y_scale)
        out["nose_ratio_lsfb_four_three"] = nose_height_ratio(ds, 0.75)
        sp = _load(ROOT / "splits" / "lsfb_isol_signer.json")
        if sp:
            test = ds.index[ds.index.signer_id.astype(str).isin(sp["signers"]["test"])]
            counts = test.label.value_counts()
            out["lsfb_test_top10_share"] = float(counts.iloc[:10].sum() / counts.sum())
    (ROOT / "reports" / "dataset_analyses.json").write_text(json.dumps(out, indent=1))
    return out


def fig_training_curves() -> None:
    fig, ax = plt.subplots(figsize=(viz.COL_W, 1.9))
    drawn = False
    for k, (prefix, name) in enumerate(
        (("asl-transformer", "Conv-Transformer"), ("asl-gru", "BiGRU"))
    ):
        hist = [pd.read_csv(ROOT / "reports" / "runs" / r / "history.csv") for r in seeds(prefix)]
        if not hist:
            continue
        n = min(len(h) for h in hist)
        acc = np.stack([h["val_top1"].to_numpy()[:n] for h in hist]) * 100
        ep = np.arange(1, n + 1)
        ax.plot(ep, acc.mean(0), color=viz.SERIES[k], label=name)
        if len(hist) > 1:
            ax.fill_between(ep, acc.min(0), acc.max(0), color=viz.SERIES[k], alpha=0.15, lw=0)
        drawn = True
    if not drawn:
        plt.close(fig)
        return
    ax.set(xlabel="epoch", ylabel="validation top-1 (%)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    viz.save(fig, FIG / "training_curves")


