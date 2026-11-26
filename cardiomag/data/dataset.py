"""PyTorch datasets over generated shards (see cardiomag.data.generate).

    ds = MCGDataset("data/atrial_ptb49", "train", target="anomaly", noise_fT_rtHz=20)
    x, y = ds[0]          # x: (S, T) float32, y: int64

Transforms (applied on the fly, in this order): optional acquisition model (cardiomag.data.acquisition:
SNR-matched noise for simulations, low-pass, RMS-onset alignment + crop, identical for real data) or
plain sensor noise (white + 1/f, given fT/sqrt(Hz)),
amplitude jitter, time shift, channel dropout, per-sample normalisation. Real recordings
(`split="real"`) are healthy measured data on the same layout and get the same normalisation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

TARGETS = ("anomaly", "label")


def load_split(root, split):
    root = Path(root)
    if split == "real":
        f = root / "real" / "real.npz"
        if not f.exists():
            return None
        z = np.load(f)
        return dict(X=z["X"], y=z["y"], anomaly=z["anomaly"], heart=z["name"],
                    meta=[json.dumps(dict(heart=str(n), label="healthy", real=True)) for n in z["name"]])
    shards = sorted((root / split).glob("shard_*.npz"))
    if not shards:
        return None
    parts = [np.load(s, allow_pickle=False) for s in shards]
    return {k: np.concatenate([p[k] for p in parts]) for k in ("X", "y", "anomaly", "heart", "meta")}


class MCGDataset(Dataset):
    def __init__(self, root, split, target="anomaly", normalize="sample_rms", noise_fT_rtHz=0.0,
                 pink_fraction=0.3, augment=None, healthy_only=False, seed=0, hearts=None, acquisition=None):
        """hearts: optional list of heart ids; if given, the split is rebuilt from ALL generated shards
        (train+val+test) keeping only these hearts (e.g. leave-one-heart-out cross-validation)."""
        assert target in TARGETS
        self.root, self.split, self.target = Path(root), split, target
        self.info = json.load(open(self.root / "dataset.json"))
        if hearts is not None and split != "real":
            parts = [p for p in (load_split(root, s) for s in ("train", "val", "test")) if p is not None]
            d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            sel = np.isin(d["heart"], list(hearts))
            d = {k: v[sel] for k, v in d.items()}
        else:
            d = load_split(root, split)
        if d is None or len(d["y"]) == 0:
            raise FileNotFoundError(f"no data for split {split!r} in {root}")
        keep = np.ones(len(d["y"]), bool)
        if healthy_only:
            keep = d["anomaly"] == 0
        self.X = d["X"][keep].astype(np.float32)
        self.y = d["y"][keep].astype(np.int64)
        self.anomaly = d["anomaly"][keep].astype(np.int64)
        self.heart = d["heart"][keep]
        self.meta = [json.loads(m) for m in np.asarray(d["meta"])[keep]]
        self.normalize = normalize
        self.dt_ms = float(np.diff(self.info["t_ms"][:2])[0])
        self.noise_sd_pT = noise_fT_rtHz * 1e-3 * np.sqrt((1000.0 / self.dt_ms) / 2.0)
        self.pink = pink_fraction
        self.augment = augment or {}
        self.rng = np.random.default_rng(seed)
        self.train_mode = split == "train"
        from .acquisition import Acquisition
        self.acq = Acquisition(self.info["t_ms"], acquisition) if acquisition else None

    @property
    def n_classes(self):
        return 2 if self.target == "anomaly" else len(self.info["classes"])

    @property
    def shape(self):
        return (self.X.shape[1], self.acq.n_out) if self.acq else self.X.shape[1:]

    def targets(self):
        return self.anomaly if self.target == "anomaly" else self.y

    def __len__(self):
        return self.X.shape[0]

    def _noise(self, x):
        if self.noise_sd_pT <= 0:
            return x
        n = self.rng.normal(0, self.noise_sd_pT, x.shape)
        if self.pink > 0:
            rw = np.cumsum(self.rng.normal(0, 1, x.shape), axis=-1)
            rw -= rw.mean(axis=-1, keepdims=True)
            rw /= rw.std(axis=-1, keepdims=True) + 1e-12
            n = np.sqrt(1 - self.pink) * n + np.sqrt(self.pink) * self.noise_sd_pT * rw
        return x + n.astype(np.float32)

    def __getitem__(self, i):
        x = self.X[i].copy()
        if self.acq is not None:
            jit = self.augment.get("onset_jitter_ms", 0.0) if self.train_mode else 0.0
            x = self.acq(x, self.rng, simulated=self.split != "real", jitter_ms=jit)
        elif self.split != "real":
            x = self._noise(x)
        if self.train_mode and self.augment:
            a = self.augment
            if a.get("amp_jitter"):
                x *= np.float32(np.exp(self.rng.normal(0, a["amp_jitter"])))
            if a.get("time_shift"):
                s = int(self.rng.integers(-a["time_shift"], a["time_shift"] + 1))
                x = np.roll(x, s, axis=-1)
                if s > 0:
                    x[:, :s] = x[:, s:s + 1]
                elif s < 0:
                    x[:, s:] = x[:, s - 1:s]
            if a.get("channel_dropout"):
                m = self.rng.random(x.shape[0]) < a["channel_dropout"]
                x[m] = 0.0
        if self.normalize == "sample_rms":
            x = x / (np.sqrt((x ** 2).mean()) + 1e-6)
        elif self.normalize == "per_channel":
            x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)
        elif self.normalize == "pT_scale":
            x = x / 10.0
        t = self.anomaly[i] if self.target == "anomaly" else self.y[i]
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), torch.tensor(t, dtype=torch.long)
