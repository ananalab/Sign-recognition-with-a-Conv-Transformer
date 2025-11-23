"""Figure style for the paper and skeleton visualisation (static strips and animated GIFs)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from signrec.landmarks import SLICES, skeleton_edges  # noqa: E402

# Categorical palette, readable with the common colour-vision deficiencies.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, INK_3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"

PART_COLORS = {"left_hand": SERIES[1], "right_hand": SERIES[0], "pose": INK_3, "lips": SERIES[4]}

# Half and full text width of the single-column A4 article (inches).
COL_W, FULL_W = 3.05, 6.3


def draw_frame(ax: plt.Axes, frame: np.ndarray, lw: float = 1.0, point_size: float = 2.0) -> list:
    """Draw one ``[L, 3]`` frame (image coordinates, y down) on ``ax``; returns the artists.

    Hands and lips are drawn as lines only; body joints are drawn as points.
    """
    artists = []
    for (a, b), c in zip(_EDGES, _EDGE_COLORS, strict=True):
        pa, pb = frame[a, :2], frame[b, :2]
        if np.isnan(pa).any() or np.isnan(pb).any():
            continue
        width = lw * (0.6 if c == PART_COLORS["lips"] else 1.0)
        (ln,) = ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=c, lw=width, solid_capstyle="round")
        artists.append(ln)
    body = frame[SLICES["pose"]]
    ok = ~np.isnan(body[:, 0])
    artists.append(ax.scatter(body[ok, 0], body[ok, 1], s=point_size, c=INK_2, lw=0, zorder=3))
    return artists


def _bounds(seq: np.ndarray, pad: float = 0.05) -> tuple[float, float, float, float]:
    xy = seq[..., :2].reshape(-1, 2)
    xy = xy[~np.isnan(xy).any(1)]
    if len(xy) == 0:
        return 0, 1, 0, 1
    lo, hi = np.percentile(xy, 1, axis=0), np.percentile(xy, 99, axis=0)
    span = (hi - lo).max() / 2 + pad
    c = (lo + hi) / 2
    return c[0] - span, c[0] + span, c[1] - span, c[1] + span


def _setup(ax: plt.Axes, bounds: tuple[float, float, float, float]) -> None:
    x0, x1, y0, y1 = bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)  # image y axis points down
    ax.set_aspect("equal")
    ax.axis("off")


def save_gif(seq: np.ndarray, path: str | Path, title: str = "", fps: int = 15) -> None:
    """Animated skeleton GIF (README / error analysis)."""
    fig, ax = plt.subplots(figsize=(3, 3))
    b = _bounds(seq)

    def update(t: int) -> list:
        ax.clear()
        _setup(ax, b)
        ax.set_title(f"{title}  [{t + 1}/{len(seq)}]", fontsize=8)
        return draw_frame(ax, seq[t], lw=1.6, point_size=4)

    anim = FuncAnimation(fig, update, frames=len(seq), blit=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(fig)
