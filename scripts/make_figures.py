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


def part_missing_rates(ds: PackedDataset, max_n: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Share of frames where each body part is entirely missing (random subsample)."""
    rng = np.random.default_rng(seed)
    rows = rng.permutation(len(ds))[:max_n]
    acc = {k: [] for k in ("left_hand", "right_hand", "any_hand", "pose", "lips")}
    for r in rows:
        s = ds.sequence(int(r))[..., 0]
        miss = {
            p: np.isnan(s[:, SLICES[p]]).all(1) for p in ("left_hand", "right_hand", "pose", "lips")
        }
        miss["any_hand"] = miss["left_hand"] & miss["right_hand"]
        for k, v in miss.items():
            acc[k].append(v.mean())
    return pd.DataFrame({k: [np.mean(v)] for k, v in acc.items()})


def fig_dataset_overview(dss: dict[str, PackedDataset]) -> dict:
    """(a) examples per class, (b) clip duration, (c) examples per signer."""
    fig, axes = plt.subplots(1, 3, figsize=(viz.FULL_W, 1.9))
    stats = {}
    for k, (name, ds) in enumerate(dss.items()):
        idx = ds.index
        color = viz.SERIES[k]
        per_class = idx.label.value_counts().to_numpy()
        axes[0].plot(np.arange(1, len(per_class) + 1), per_class, color=color, label=DATASETS[name])
        dur = idx.n_frames / idx.fps
        bins = np.linspace(0, 6, 61)
        axes[1].hist(
            dur.clip(upper=6), bins=bins, histtype="step", density=True, color=color, lw=1.2
        )
        per_signer = np.sort(idx.signer_id.value_counts().to_numpy())[::-1]
        axes[2].plot(np.arange(1, len(per_signer) + 1), per_signer, color=color)
        stats[name] = {
            "n_samples": int(len(idx)),
            "n_classes": int(idx.label.nunique()),
            "n_signers": int(idx.signer_id.nunique()),
            "per_class_min": int(per_class.min()),
            "per_class_median": float(np.median(per_class)),
            "per_class_max": int(per_class.max()),
            "duration_median_s": float(dur.median()),
            "frames_median": float(idx.n_frames.median()),
            "frames_p95": float(idx.n_frames.quantile(0.95)),
            "fps": float(idx.fps.iloc[0]),
        }
    axes[0].set(
        xlabel="class rank", ylabel="examples", title="(a) examples per class", yscale="log"
    )
    axes[1].set(
        xlabel="clip duration (s; last bin ≥ 6 s)", ylabel="density", title="(b) clip duration"
    )
    axes[2].set(
        xlabel="signer rank", ylabel="examples", title="(c) examples per signer", yscale="log"
    )
    axes[0].legend(loc="upper right")
    fig.tight_layout(w_pad=1.2)
    viz.save(fig, FIG / "dataset_overview")
    return stats


def fig_missing(dss: dict[str, PackedDataset]) -> pd.DataFrame:
    rates = pd.concat({DATASETS[n]: part_missing_rates(ds) for n, ds in dss.items()}).droplevel(1)
    parts = ["left_hand", "right_hand", "any_hand", "pose", "lips"]
    nice = ["left\nhand", "right\nhand", "both\nhands", "upper\nbody", "lips"]
    fig, ax = plt.subplots(figsize=(viz.COL_W, 1.9))
    w = 0.38
    x = np.arange(len(parts))
    for k, (name, row) in enumerate(rates.iterrows()):
        vals = row[parts].to_numpy() * 100
        bars = ax.bar(x + (k - 0.5) * w, vals, w * 0.92, color=viz.SERIES[k], label=name)
        ax.bar_label(bars, fmt="%.0f", fontsize=6, padding=1.5, color=viz.INK_2)
    ax.set_xticks(x, nice)
    ax.set_ylabel("frames with part missing (%)")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    viz.save(fig, FIG / "missing_parts")
    return rates


def fig_normalisation(dss: dict[str, PackedDataset], n: int = 400, seed: int = 0) -> None:
    """Per-clip mean positions of the nose, shoulders and dominant wrist in three coordinate frames.

    (a) image coordinates; (b) body-centred, without aspect correction; (c) body-centred with the
    per-dataset aspect correction. Each point is one clip.
    """
    rng = np.random.default_rng(seed)
    cfg = PreprocessConfig()
    joints = [
        POSE_POS["nose"],
        POSE_POS["l_shoulder"],
        POSE_POS["r_shoulder"],
        SLICES["right_hand"].start,
    ]
    fig, axes = plt.subplots(1, 3, figsize=(viz.FULL_W, 2.05))
    for k, (name, ds) in enumerate(dss.items()):
        pts = {0: [], 1: [], 2: []}
        for r in rng.permutation(len(ds))[:n]:
            seq = canonicalize_hand(trim_handless(ds.sequence(int(r))))
            iso = seq.copy()
            iso[..., 1] *= ds.y_scale
            frames = (
                seq,
                normalize(seq, cfg.min_shoulder_width),
                normalize(iso, cfg.min_shoulder_width),
            )
            for j, arr in enumerate(frames):
                pts[j] += [np.nanmean(arr[:, idx, :2], axis=0) for idx in joints]
        for j, ax in enumerate(axes):
            p = np.array(pts[j])
            ax.scatter(
                p[:, 0], p[:, 1], s=1.2, color=viz.SERIES[k], alpha=0.35, lw=0, label=DATASETS[name]
            )
    axes[0].set(
        title="(a) image coordinates",
        xlabel="x / width",
        ylabel="y / height",
        xlim=(-0.1, 1.1),
        ylim=(1.6, -0.1),
    )
    axes[1].set(
        title="(b) body frame, no aspect correction",
        xlabel="shoulder widths",
        xlim=(-2.2, 2.2),
        ylim=(3.0, -1.6),
    )
    axes[2].set(
        title="(c) body frame, aspect-corrected",
        xlabel="shoulder widths",
        xlim=(-2.2, 2.2),
        ylim=(3.0, -1.6),
    )
    for ax in axes:
        ax.set_aspect("equal")
    for ax in axes[1:]:
        ax.annotate("nose", xy=(0, -0.5), xytext=(1.0, -1.25), fontsize=6.5, color=viz.INK_2,
                    arrowprops={"arrowstyle": "-", "color": viz.INK_3, "lw": 0.6})  # fmt: skip
    leg = axes[0].legend(loc="lower left", markerscale=5)
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.tight_layout(w_pad=0.8)
    viz.save(fig, FIG / "normalisation")


def typical_clip(ds: PackedDataset, label: str | None) -> int:
    """Row of a clip with a visible hand in most frames and a typical length (~30 frames)."""
    idx = ds.index
    cand = idx.index[idx.label == label] if label in set(idx.label) else idx.index
    best, best_score = int(cand[0]), -1.0
    for r in cand[:300]:
        s = ds.sequence(int(r))
        score = (~np.isnan(s[:, 21:42, 0]).all(1) | ~np.isnan(s[:, :21, 0]).all(1)).mean()
        score -= abs(len(s) - 30) / 300
        if score > best_score:
            best, best_score = int(r), score
    return best


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


def eval_report(run: str, subset: str) -> dict | None:
    return _load(ROOT / "reports" / "eval" / f"{run}_{subset}.json")


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


def table_datasets() -> None:
    stats = _load(ROOT / "reports" / "dataset_stats.json") or {}
    splits = {"kaggle_asl": "kaggle_asl_signer.json", "lsfb_isol": "lsfb_isol_signer.json"}
    rows = []
    for name, st in stats.items():
        sp = _load(ROOT / "splits" / splits[name])
        s = sp["signers"] if sp else {"train": [], "val": [], "test": []}
        rows.append(
            f"{DATASETS[name]} & {st['n_classes']} & {st['n_samples']:,} & {st['n_signers']} & "
            f"{len(s['train'])}/{len(s['val'])}/{len(s['test'])} & {st['per_class_min']}--"
            f"{st['per_class_max']} & {st['duration_median_s']:.2f}\\,s & {int(st['fps'])} & "
            f"{100 * st['missing']['any_hand']:.0f}\\,\\% \\\\"
        )
    body = "\n".join(rows)
    (TAB / "datasets.tex").write_text(
        "\\begin{tabular}{lrrrcrrrr}\n\\toprule\n"
        " & & & & Signers & Clips & & & No \\\\\n"
        "Dataset & Classes & Clips & Signers & tr/val/test & per class & Duration & fps & hand \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n"
    )


def significance(nums: Numbers) -> None:
    """Macros from reports/stats.json (paired tests between models)."""
    st = stats()
    c = st.get("asl_transformer_vs_gru_test")
    if c:
        b, bc = c["bootstrap_signer"], c["bootstrap_clip"]
        nums["numTrGruDiff"] = f"{100 * b['diff']:.1f}"
        nums["numTrGruCiLo"] = f"{100 * b['ci_low']:.1f}"
        nums["numTrGruCiHi"] = f"{100 * b['ci_high']:.1f}"
        nums["numTrGruClipCiLo"] = f"{100 * bc['ci_low']:.1f}"
        nums["numTrGruClipCiHi"] = f"{100 * bc['ci_high']:.1f}"
        nums["numTrGruMcnemar"] = fmt_p(c["mcnemar_ensemble"]["p"])
    for key, name in (("ft", "Ft"), ("ftfreeze", "Freeze")):
        c = st.get(f"lsfb_{key}_vs_scratch_test")
        if c:
            b = c["bootstrap_signer"]
            nums[f"numLsfb{name}Diff"] = f"{100 * b['diff']:+.1f}"
            nums[f"numLsfb{name}CiLo"] = f"{100 * b['ci_low']:+.1f}"
            nums[f"numLsfb{name}CiHi"] = f"{100 * b['ci_high']:+.1f}"
            nums[f"numLsfb{name}P"] = fmt_p(b["p"])


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


def transfer_checks(nums: Numbers) -> None:
    """Linear probe and low-learning-rate fine-tuning (validation top-1, seed 0)."""
    for key, run in (("ProbeAll", "lsfb-probe-nall-s0"), ("ProbeFive", "lsfb-probe-n5-s0"),
                     ("LowlrTwentyFive", "lsfb-ftlowlr-n25-s0"),
                     ("ScratchTwentyFive", "lsfb-scratch-n25-s0"), ("FtTwentyFive", "lsfb-ft-n25-s0"),
                     ("ScratchFive", "lsfb-scratch-n5-s0"), ("FtFive", "lsfb-ft-n5-s0")):  # fmt: skip
        m = run_metrics(run)
        if m:
            nums[f"num{key}"] = pct(m["best_val"]["top1"])
    base = _load(ROOT / "reports" / "baselines" / "lsfb_isol_lsfb_isol_signer.json")
    if base:
        nums["numLsfbLogregVal"] = pct(base["logreg"]["val"]["top1"])
        nums["numLsfbLogregTest"] = pct(base["logreg"]["test"]["top1"])
        nums["numLsfbLogregTestFOne"] = pct(base["logreg"]["test"]["macro_f1"])


def export_tables(nums: Numbers) -> None:
    """Size and latency (native and browser) of FP32 / INT8 models; INT8 accuracy if exported."""
    lat = _load(ROOT / "reports" / "latency.json")
    if not lat:
        return
    acc = {}
    for mid in ("asl", "lsfb"):
        rep = _load(ROOT / "app" / "web" / "models" / mid / "export_report.json")
        if rep:
            acc[rep["run"].split("-s")[0]] = rep
    rows = []
    for arch, name in (("asl-transformer", "Conv-Transformer"), ("asl-gru", "BiGRU")):
        for tag, prec in (("fp32", "FP32"), ("int8", "INT8")):
            e = lat.get(f"{arch}_{tag}")
            if not e:
                continue
            web = e.get("browser", {})
            top1 = acc.get(arch, {}).get(tag, {}).get("val_top1")
            rows.append(
                f"{name} & {prec} & {e['size_mb']:.1f}\\,MB & {e['native']['p50_ms']:.1f} & "
                f"{web.get('p50_ms', float('nan')):.1f} & {web.get('p95_ms', float('nan')):.1f} & "
                f"{pct(top1) if top1 is not None else '--'} \\\\"
            )
    (TAB / "export.tex").write_text(
        "\\begin{tabular}{llrrrrr}\n\\toprule\n"
        " & & & Native & \\multicolumn{2}{c}{Browser (WebAssembly)} & \\\\\n\\cmidrule(lr){5-6}\n"
        "Model & Precision & Size & p50 (ms) & p50 (ms) & p95 (ms) & Val.\\ top-1 \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    t8 = lat.get("asl-transformer_int8")
    if t8:
        nums["numIntSizeMB"] = f"{t8['size_mb']:.1f}"
        nums["numIntLatency"] = f"{t8.get('browser', t8['native'])['p50_ms']:.0f}"
        nums["numFpSizeMB"] = f"{lat['asl-transformer_fp32']['size_mb']:.1f}"
        nums["numIntNative"] = f"{t8['native']['p50_ms']:.1f}"
        nums["numFpNative"] = f"{lat['asl-transformer_fp32']['native']['p50_ms']:.1f}"
        fp = lat["asl-transformer_fp32"]
        nums["numFpBrowser"] = f"{fp.get('browser', fp['native'])['p50_ms']:.0f}"
        nums["numIntSlowdown"] = f"{t8['native']['p50_ms'] / fp['native']['p50_ms']:.1f}"
        nums["numIntShrink"] = f"{fp['size_mb'] / t8['size_mb']:.1f}"
        g = lat.get("asl-gru_fp32")
        if g:
            nums["numGruBrowser"] = f"{g.get('browser', g['native'])['p50_ms']:.0f}"


