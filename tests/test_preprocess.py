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

