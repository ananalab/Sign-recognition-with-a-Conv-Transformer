import numpy as np

from signrec import landmarks as lm


def test_subset_size_and_order() -> None:
    assert lm.N_LANDMARKS == 21 + 21 + 7 + 40
    s = lm.SLICES
    assert [k for k in s] == ["left_hand", "right_hand", "pose", "lips"]
    assert s["left_hand"] == slice(0, 21)
    assert s["right_hand"] == slice(21, 42)
    assert s["pose"] == slice(42, 49)
    assert s["lips"] == slice(49, 89)
    assert len(set(lm.LIPS_IDX)) == 40


