import numpy as np

from conftest import make_kaggle_long
from signrec.io.kaggle import load_kaggle_sequence, long_to_frames, unified_from_kaggle_index
from signrec.io.lsfb import parts_to_unified
from signrec.landmarks import LIPS_IDX, N_LANDMARKS, POSE_IDX, SLICES
from signrec.store import PackedDataset, PackedWriter


def test_kaggle_dense_shape_and_nan() -> None:
    dense = long_to_frames(make_kaggle_long(n_frames=4))
    assert dense.shape == (4, 543, 3)
    assert np.isnan(dense[:, 468:489, 0]).all()  # missing left hand
    assert not np.isnan(dense[:, 522:, 0]).any()


def test_kaggle_unified_order(kaggle_parquet) -> None:
    seq = load_kaggle_sequence(kaggle_parquet)
    assert seq.shape == (5, N_LANDMARKS, 3) and seq.dtype == np.float32
    x = seq[0, :, 0]
    assert np.isnan(x[SLICES["left_hand"]]).all()
    np.testing.assert_array_equal(x[SLICES["right_hand"]], 522 + np.arange(21))
    np.testing.assert_array_equal(x[SLICES["pose"]], 489 + np.array(POSE_IDX))
    np.testing.assert_array_equal(x[SLICES["lips"]], np.array(LIPS_IDX))
    assert len(unified_from_kaggle_index()) == N_LANDMARKS


