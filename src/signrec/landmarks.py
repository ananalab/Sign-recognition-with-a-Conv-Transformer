"""Unified landmark subset shared by every dataset, the trainer and the demo.

Order is frozen (tested): left hand (21) | right hand (21) | upper body (7) | lips (40).
Indices refer to the MediaPipe Holistic numbering of each part
(hands: 21-point hand model, pose: 33-point BlazePose, face: 468/478-point face mesh).
"""

from __future__ import annotations

from typing import Final

# MediaPipe hand model: wrist, then 4 joints per finger (thumb to pinky).
HAND_IDX: Final[tuple[int, ...]] = tuple(range(21))

