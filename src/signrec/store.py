"""Packed on-disk storage of the unified format.

A processed dataset lives in ``data/processed/<dataset>/``:

- ``frames.npy``  float16 ``[sum(T_i), L, 3]``, all sequences concatenated (memory-mapped on read);
- ``index.parquet`` one row per sample:
  ``sample_id, dataset, label, signer_id, n_frames, fps, offset``;
- ``labels.json``  label to class index;
- ``meta.json`` (optional) dataset-level constants, e.g. ``y_scale`` (see ``preprocess``).

``PackedDataset.sequence(i)`` returns a ``[T, L, 3]`` float32 array (NaN = missing landmark).
One packed file instead of ~10^5 small ``.npy`` files keeps loading fast on a laptop.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

INDEX_COLUMNS = ["sample_id", "dataset", "label", "signer_id", "n_frames", "fps", "offset"]


