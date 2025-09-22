"""Preprocessing: raw unified sequence ``[T, L, 3]`` to model features ``[T_out, F]`` + mask.

:func:`preprocess` is used for training, evaluation and, through a line-by-line JavaScript port
checked by fixtures, the browser demo. All steps are pure numpy.

Order of operations: trim, canonical hand (mirror), spatial normalisation, fixed length,
features, NaN to 0.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from signrec.landmarks import N_LANDMARKS, POSE_POS, SLICES, mirror_permutation

_MIRROR = np.asarray(mirror_permutation())
_LH, _RH = SLICES["left_hand"], SLICES["right_hand"]
_EPS = 1e-6


@dataclass(frozen=True)
class PreprocessConfig:
    T: int = 64
    length_mode: str = "resample"  # resample | pad
    use_z: bool = False
    parts: tuple[str, ...] = field(default=("left_hand", "right_hand", "pose", "lips"))
    trim: bool = True
    canonical_hand: bool = True
    hand_local: bool = True
    velocity: bool = True
    min_shoulder_width: float = 0.02

    @classmethod
    def from_dict(cls, d: dict) -> PreprocessConfig:
        return cls(**{**d, "parts": tuple(d.get("parts", cls.parts))})

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: object) -> PreprocessConfig:
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) | overrides)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["parts"] = list(self.parts)
        return d

    @property
    def n_dims(self) -> int:
        return 3 if self.use_z else 2

    @property
    def landmark_ids(self) -> np.ndarray:
        return np.concatenate([np.arange(N_LANDMARKS)[SLICES[p]] for p in self.parts])

    @property
    def n_features(self) -> int:
        n = len(self.landmark_ids) * self.n_dims
        if self.hand_local:
            n += 21 * self.n_dims
        return n * (2 if self.velocity else 1)


