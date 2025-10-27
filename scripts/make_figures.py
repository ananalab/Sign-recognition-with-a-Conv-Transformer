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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from signrec import viz
from signrec.landmarks import POSE_POS, SLICES
from signrec.metrics import summary
from signrec.preprocess import PreprocessConfig, canonicalize_hand, normalize, trim_handless
from signrec.store import PackedDataset

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
DATASETS = {"kaggle_asl": "ASL (Google ISLR)", "lsfb_isol": "LSFB-ISOL"}
GIF_SIGNS = {"kaggle_asl": ["hello", "thankyou"], "lsfb_isol": ["AUSSI"]}  # shown in the README


def available() -> dict[str, PackedDataset]:
    out = {}
    for name in DATASETS:
        p = ROOT / "data" / "processed" / name
        if (p / "index.parquet").exists():
            out[name] = PackedDataset(p)
    return out


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


def numbers_from_data(nums: Numbers) -> None:
    stats = _load(ROOT / "reports" / "dataset_stats.json") or {}
    if "kaggle_asl" in stats:
        a = stats["kaggle_asl"]
        nums["numAslSamples"] = f"{a['n_samples']:,}"
        nums["numAslClasses"] = a["n_classes"]
        nums["numAslSigners"] = a["n_signers"]
    if "lsfb_isol" in stats:
        b = stats["lsfb_isol"]
        nums["numLsfbClasses"] = b["n_classes"]
        nums["numLsfbFps"] = int(b["fps"])
    for key, name in (("Asl", "kaggle_asl"), ("Lsfb", "lsfb_isol")):
        if name in stats:
            st = stats[name]
            nums[f"num{key}DurMedian"] = f"{st['duration_median_s']:.2f}"
            nums[f"num{key}NoHandPct"] = f"{100 * st['missing']['any_hand']:.1f}"
            nums[f"num{key}FramesMedian"] = f"{st['frames_median']:.0f}"
    an = _load(ROOT / "reports" / "dataset_analyses.json") or {}
    for key, k in (("NoseAsl", "nose_ratio_asl"), ("NoseLsfbRaw", "nose_ratio_lsfb_raw"),
                   ("NoseLsfbCorr", "nose_ratio_lsfb_corrected"),
                   ("NoseLsfbFourThree", "nose_ratio_lsfb_four_three")):  # fmt: skip
        if k in an:
            nums[f"num{key}"] = f"{an[k]:.2f}"
    if "asl_handless_outside_sign_share" in an:
        nums["numHandlessOutside"] = pct(an["asl_handless_outside_sign_share"])
        nums["numHandlessSample"] = f"{an['asl_handless_sample_clips']:,}"
    if "lsfb_test_top10_share" in an:
        nums["numLsfbTopTenShare"] = pct(an["lsfb_test_top10_share"], 0)
    per_class = _load(ROOT / "reports" / "lsfb_train_per_class.json")
    if per_class:
        nums["numLsfbMedianPerClass"] = f"{per_class['median']:.0f}"
    lsfb_meta = _load(ROOT / "reports" / "lsfb_selection.json")
    if lsfb_meta:
        nums["numLsfbAllInstances"] = f"{lsfb_meta['all_instances']:,}"
        nums["numLsfbAllGlosses"] = f"{lsfb_meta['all_glosses']:,}"
        nums["numLsfbAllSigners"] = lsfb_meta["all_signers"]
        nums["numLsfbDroppedPct"] = f"{100 * lsfb_meta['dropped_share']:.1f}"
    bp = _load(ROOT / "reports" / "browser_parity.json")
    if bp:
        nums["numBrowserSame"] = f"{bp['same_top1']}/{bp['n_clips']}"
        nums["numBrowserDiff"] = (
            f"{bp['max_abs_prob_diff']:.0e}".replace("e-0", "\\times10^{-").replace(
                "e-", "\\times10^{-"
            )
            + "}"
        )
        nums["numExampleAcc"] = pct(bp["top1_on_examples"])
    js_parity_report()
    par = _load(ROOT / "reports" / "js_parity.json")
    if par:
        e = max(r["maxDiff"] for r in par)
        mant, exp = f"{e:.0e}".split("e")
        nums["numJsParity"] = f"${mant}\\times10^{{{int(exp)}}}$"


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


def fig_per_signer_and_missing() -> None:
    reps = {name: [eval_report(r, "test") for r in seeds(prefix)]
            for prefix, name in (("asl-transformer", "Conv-Transformer"), ("asl-gru", "BiGRU"))}  # fmt: skip
    reps = {k: [r for r in v if r] for k, v in reps.items()}
    reps = {k: v for k, v in reps.items() if v}
    if not reps:
        return
    fig, axes = plt.subplots(
        1, 2, figsize=(viz.FULL_W, 1.95), gridspec_kw={"width_ratios": [1, 1.25]}
    )
    ax = axes[0]
    signers = sorted({d["g"] for r in next(iter(reps.values())) for d in r["by_signer"]})
    for k, (name, rs) in enumerate(reps.items()):
        acc = (
            np.array([[{d["g"]: d["acc"] for d in r["by_signer"]}[s] for s in signers] for r in rs])
            * 100
        )
        x = np.arange(len(signers)) + (k - 0.5) * 0.18
        ax.errorbar(x, acc.mean(0), yerr=acc.std(0) if len(rs) > 1 else None, fmt="o", ms=4,
                    color=viz.SERIES[k], label=name, capsize=2, lw=1)  # fmt: skip
    ax.set_xticks(np.arange(len(signers)), [f"signer {s}" for s in signers])
    ax.set_ylabel("test top-1 (%)")
    ax.set_title("(a) accuracy per unseen signer")
    ax.set_xlim(-0.5, len(signers) - 0.5)
    ax = axes[1]
    first = next(iter(reps.values()))
    bins = [d["bin"] for d in first[0]["by_missing"]]
    w = 0.38
    for k, (name, rs) in enumerate(reps.items()):
        acc = np.array([[d["acc"] for d in r["by_missing"]] for r in rs]) * 100
        n = [d["n"] for d in rs[0]["by_missing"]]
        x = np.arange(len(bins)) + (k - 0.5) * w
        ax.bar(x, np.nan_to_num(acc.mean(0)), w * 0.92, color=viz.SERIES[k], label=name)
    ax.set_xticks(np.arange(len(bins)), [f"{b}\n(n={c:,})" for b, c in zip(bins, n, strict=True)])
    ax.set_xlabel("share of frames with no hand detected")
    ax.set_ylabel("test top-1 (%)")
    handles, labels = ax.get_legend_handles_labels()
    ax.set_title("(b) accuracy vs. missing hands")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.9))
    fig.legend(handles, labels, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.0))
    viz.save(fig, FIG / "signers_missing")


