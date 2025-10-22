"""ONNX export, dynamic INT8 quantisation and CPU benchmark of a trained checkpoint.

    uv run python -m signrec.export --ckpt models/<run>/best.pt [--out app/web/models]

The exported graph takes ``x [B, T, F] float32`` and ``mask [B, T] float32`` (1 = real frame)
and returns ``probs [B, C]``; a float mask keeps the model easy to feed from JavaScript.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn

from signrec.checkpoint import load_checkpoint
from signrec.preprocess import PreprocessConfig, preprocess
from signrec.splits import load_split, split_indices
from signrec.store import PackedDataset


class _ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model(x, mask > 0.5).softmax(-1)


def export_onnx(model: nn.Module, T: int, n_features: int, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(1, T, n_features)
    m = torch.ones(1, T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            _ExportWrapper(model).eval(),  # wrapper must be in eval mode: export restores it
            (x, m),
            str(out),
            input_names=["x", "mask"],
            output_names=["probs"],
            dynamic_axes={"x": {0: "batch"}, "mask": {0: "batch"}, "probs": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
    model.eval()
    return out


def quantize(fp32: str | Path, int8: str | Path) -> Path:
    """Dynamic INT8 quantisation of the weight matrices, keeping the input projection in FP32.

    The input features have rare but large outliers (|x| up to ~200 while 99.9 % are below ~2), so
    a per-tensor dynamic activation scale on the first layer destroys their resolution (validation
    top-1 drops from 68 % to 41 % on ASL). Only matmuls with constant weights are quantised.
    """
    graph = onnx.load(str(fp32)).graph
    first = [n.name for n in graph.node if n.op_type in ("MatMul", "Gemm") and "/proj/" in n.name]
    quantize_dynamic(
        str(fp32),
        str(int8),
        weight_type=QuantType.QInt8,
        nodes_to_exclude=first[:1],
        extra_options={"MatMulConstBOnly": True},
    )
    return Path(int8)


def session(path: str | Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def onnx_predict(sess: ort.InferenceSession, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return sess.run(["probs"], {"x": x.astype(np.float32), "mask": mask.astype(np.float32)})[0]


