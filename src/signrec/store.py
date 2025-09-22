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

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

INDEX_COLUMNS = ["sample_id", "dataset", "label", "signer_id", "n_frames", "fps", "offset"]


@dataclass
class PackedWriter:
    """Accumulates sequences in memory and writes them as one packed dataset."""

    dataset: str
    chunks: list[np.ndarray] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    _offset: int = 0

    def add(self, seq: np.ndarray, sample_id: str, label: str, signer_id: str, fps: float) -> None:
        if seq.ndim != 3 or seq.shape[-1] != 3 or seq.shape[0] == 0:
            raise ValueError(f"expected [T>0, L, 3], got {seq.shape}")
        self.chunks.append(seq.astype(np.float16))
        self.rows.append(
            {
                "sample_id": sample_id,
                "dataset": self.dataset,
                "label": label,
                "signer_id": str(signer_id),
                "n_frames": int(seq.shape[0]),
                "fps": float(fps),
                "offset": self._offset,
            }
        )
        self._offset += seq.shape[0]

    def write(
        self, out_dir: str | Path, labels: Iterable[str] | None = None, meta: dict | None = None
    ) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        frames = np.concatenate(self.chunks, axis=0) if self.chunks else np.zeros((0, 0, 3))
        np.save(out / "frames.npy", frames.astype(np.float16))
        index = pd.DataFrame(self.rows, columns=INDEX_COLUMNS)
        index.to_parquet(out / "index.parquet", index=False)
        names = sorted(set(index["label"])) if labels is None else list(labels)
        (out / "labels.json").write_text(json.dumps({n: i for i, n in enumerate(names)}, indent=1))
        if meta:
            (out / "meta.json").write_text(json.dumps(meta, indent=1))
        return out


