import numpy as np
import pytest

from conftest import random_sequence
from signrec.augment import AugmentConfig, augment_raw, make_augmenter, spatial, time_warp
from signrec.landmarks import N_LANDMARKS, SLICES
from signrec.preprocess import (
    PreprocessConfig,
    canonicalize_hand,
    fix_length,
    mirror,
    preprocess,
    resample,
    trim_handless,
)

CFG = PreprocessConfig()


def test_config_from_yaml_matches_defaults() -> None:
    assert PreprocessConfig.from_yaml("configs/preprocess.yaml") == CFG


@pytest.mark.parametrize("shift,scale", [((0.1, -0.05), 1.0), ((0.0, 0.0), 1.7), ((0.2, 0.1), 0.6)])
def test_invariant_to_translation_and_scale(shift, scale) -> None:
    seq = random_sequence()
    moved = seq.copy()
    moved[..., :2] = (seq[..., :2] - 0.5) * scale + 0.5 + np.array(shift)
    a, _ = preprocess(seq, CFG)
    b, _ = preprocess(moved, CFG)
    np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-4)


def test_mirror_twice_is_identity() -> None:
    seq = random_sequence(nan_left=True)
    np.testing.assert_allclose(mirror(mirror(seq)), seq, equal_nan=True, atol=1e-6)


def test_canonical_hand_moves_dominant_to_right_slot() -> None:
    seq = random_sequence()
    seq[:, SLICES["right_hand"]] = np.nan  # only the left hand is signing
    out = canonicalize_hand(seq)
    assert not np.isnan(out[:, SLICES["right_hand"]]).any()
    assert np.isnan(out[:, SLICES["left_hand"]]).all()


