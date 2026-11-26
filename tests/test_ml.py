import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from cardiomag.data.dataset import MCGDataset
from cardiomag.ml import metrics as M
from cardiomag.ml.registry import available, build_model
from cardiomag.ml.train import load_config, train


class ExternalModel(nn.Module):
    """Plug-in example: any nn.Module with this constructor signature works via `target:`."""
    task = "classify"

    def __init__(self, n_sensors, n_times, n_outputs, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(n_sensors * n_times, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_outputs))

    def forward(self, x):
        return self.net(x)


def _fake_dataset(root, S=9, T=40, n=60, seed=0):
    """Synthetic shards in the generator's format: anomalies have a shifted / reshaped bump."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    root.mkdir(parents=True, exist_ok=True)
    for split, nn_ in (("train", n), ("val", n // 2), ("test", n // 2)):
        y = rng.integers(0, 4, nn_)
        y[: nn_ // 2] = 0
        X = np.zeros((nn_, S, T), np.float32)
        for i in range(nn_):
            c = 15 + (8 if y[i] else 0)
            w = rng.normal(size=S)
            X[i] = w[:, None] * np.exp(-((t - c) / 4.0) ** 2)
        (root / split).mkdir(exist_ok=True)
        meta = [json.dumps(dict(heart=f"h{split}", label=["healthy", "a", "b", "c"][k])) for k in y]
        np.savez(root / split / "shard_000.npz", X=X, y=y, anomaly=(y > 0).astype(int),
                 heart=np.array([f"h{split}"] * nn_), meta=np.array(meta))
    json.dump(dict(classes=["healthy", "a", "b", "c"], t_ms=list(range(0, 2 * T, 2)), n_sensors=S, n_times=T),
              open(root / "dataset.json", "w"))
    return root


def test_auroc():
    assert M.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert M.auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert abs(M.auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) - 0.5) < 1e-12


@pytest.mark.parametrize("name", ["cnn1d", "resnet1d", "conv_ae"])
def test_builtin_models_forward(name):
    m = build_model({"name": name}, 12, 45, 4)
    out = m(torch.randn(3, 12, 45))
    assert out.shape == ((3, 4) if m.task == "classify" else (3, 12, 45))


def test_registry_lists_builtins():
    assert {"cnn1d", "resnet1d", "conv_ae"} <= set(available())


def test_external_model_and_training(tmp_path):
    root = _fake_dataset(tmp_path / "ds")
    ds = MCGDataset(root, "train", noise_fT_rtHz=5, augment={"time_shift": 2})
    x, y = ds[0]
    assert x.shape == (9, 40) and y.dtype == torch.long
    cfg = dict(run_name="t", out_dir=str(tmp_path / "runs"), seed=0, device="cpu",
               data=dict(root=str(root), target="anomaly"),
               model={"target": "tests.test_ml:ExternalModel", "kwargs": {"hidden": 8}},
               optim=dict(lr=3e-3, epochs=12, batch_size=16, patience=20))
    res = train(cfg)
    assert res["test"]["auroc_anomaly"] > 0.9


def test_autoencoder_training(tmp_path):
    root = _fake_dataset(tmp_path / "ds2", seed=1)
    cfg = dict(run_name="ae", out_dir=str(tmp_path / "runs"), seed=0, device="cpu",
               data=dict(root=str(root), target="anomaly", normalize="sample_rms"),
               model={"name": "conv_ae", "kwargs": {"width": 16, "latent": 4, "code": 8}},
               optim=dict(lr=3e-3, epochs=15, batch_size=16, patience=30))
    res = train(cfg)
    assert res["test"]["auroc_anomaly"] > 0.7


def test_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\nb: {c: 2}\n")
    cfg = load_config(p, ["b.c=5", "d.e=hello"])
    assert cfg["b"]["c"] == 5 and cfg["d"]["e"] == "hello"
