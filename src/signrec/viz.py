"""Figure style for the paper and skeleton visualisation (static strips and animated GIFs)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

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


