import json
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

import signrec.train as tr
from conftest import random_sequence
from signrec.models import build_model
from signrec.preprocess import PreprocessConfig
from signrec.splits import save_split, signer_split
from signrec.store import PackedWriter
from signrec.train import load_config, load_encoder, train

PCFG = PreprocessConfig(T=32)
SMALL = {
    "gru": {"dim": 64, "hidden": 64, "layers": 1, "dropout": 0.0},
    "transformer": {"max_len": 32, "dim": 64, "stages": 1, "convs_per_stage": 1, "heads": 4,
                    "dropout": 0.0},
}  # fmt: skip


@pytest.mark.parametrize("name", ["gru", "transformer"])
def test_model_overfits_minibatch(name) -> None:
    torch.manual_seed(0)
    model = build_model(name, PCFG.n_features, 8, **SMALL[name])
    x = torch.randn(16, PCFG.T, PCFG.n_features)
    m = torch.ones(16, PCFG.T, dtype=torch.bool)
    m[:4, 20:] = False
    y = torch.arange(16) % 8
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(200):
        loss = torch.nn.functional.cross_entropy(model(x, m), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if loss.item() < 0.05:
            break
    assert loss.item() < 0.05


def test_padding_does_not_change_output() -> None:
    torch.manual_seed(0)
    model = build_model("transformer", PCFG.n_features, 5, **SMALL["transformer"]).eval()
    x = torch.randn(1, PCFG.T, PCFG.n_features)
    m = torch.ones(1, PCFG.T, dtype=torch.bool)
    m[:, 20:] = False
    x2 = x.clone()
    x2[:, 20:] = 7.0  # garbage in padded frames
    torch.testing.assert_close(model(x * m[..., None], m), model(x2, m), atol=1e-4, rtol=1e-4)


@pytest.fixture
def toy_dataset(tmp_path):
    w = PackedWriter("toy")
    rng = np.random.default_rng(0)
    for k in range(120):
        c = k % 3
        seq = random_sequence(T=int(rng.integers(10, 30)), seed=k)
        seq[:, 21:42, 1] += 0.2 * c  # class-dependent hand height
        w.add(seq, f"id{k}", f"sign{c}", f"P{k % 6}", 30.0)
    root = w.write(tmp_path / "toy")
    index = pd.read_parquet(root / "index.parquet")
    save_split(signer_split(index, n_val=1, n_test=1), tmp_path / "split.json")
    return tmp_path


