"""Unified landmark subset shared by every dataset, the trainer and the demo.

Order is frozen (tested): left hand (21) | right hand (21) | upper body (7) | lips (40).
Indices refer to the MediaPipe Holistic numbering of each part
(hands: 21-point hand model, pose: 33-point BlazePose, face: 468/478-point face mesh).
"""

from __future__ import annotations

from typing import Final

# MediaPipe hand model: wrist, then 4 joints per finger (thumb to pinky).
HAND_IDX: Final[tuple[int, ...]] = tuple(range(21))

# BlazePose upper body: nose, shoulders, elbows, wrists. Hips are excluded: they are
# usually out of frame in webcam and LSFB recordings (y > 1).
POSE_IDX: Final[tuple[int, ...]] = (0, 11, 12, 13, 14, 15, 16)
POSE_NAMES: Final[tuple[str, ...]] = (
    "nose",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
)

# Face-mesh lip contour (outer then inner ring), 40 points.
LIPS_IDX: Final[tuple[int, ...]] = (
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
)  # fmt: skip

# (part name, source part in MediaPipe output, indices within that part)
GROUPS: Final[tuple[tuple[str, str, tuple[int, ...]], ...]] = (
    ("left_hand", "left_hand", HAND_IDX),
    ("right_hand", "right_hand", HAND_IDX),
    ("pose", "pose", POSE_IDX),
    ("lips", "face", LIPS_IDX),
)

N_LANDMARKS: Final[int] = sum(len(idx) for _, _, idx in GROUPS)  # 89


def part_slices() -> dict[str, slice]:
    """Return the slice of each part in the unified landmark axis."""
    out: dict[str, slice] = {}
    start = 0
    for name, _, idx in GROUPS:
        out[name] = slice(start, start + len(idx))
        start += len(idx)
    return out


