"""Sim-to-real gap diagnosis for a generated dataset: simulated healthy vs measured healthy recordings.

    python scripts/diagnose_gap.py --data data/whole_kiel36 --config configs/train/whole_cnn.yaml

Applies the training config's acquisition model to both domains, then compares:
  - RMS-over-sensors time course (normalised)       -> timing / duration differences
  - spectrum                                          -> noise / bandwidth differences
  - |corr| of RMS-peak maps (sim-sim vs real-sim)     -> spatial / geometry differences
Writes runs/gap/<dataset>.png and prints summary numbers.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardiomag.data.dataset import MCGDataset  # noqa: E402
from cardiomag.ml.train import load_config  # noqa: E402


def stack(ds, n=None):
    idx = range(len(ds)) if n is None else np.random.default_rng(0).choice(len(ds), min(n, len(ds)), replace=False)
    return np.stack([ds[i][0].numpy() for i in idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    dc = cfg["data"]
    kw = dict(normalize="sample_rms", acquisition=dc.get("acquisition"))
    sim = MCGDataset(a.data, "train", healthy_only=True, **kw)
    real = MCGDataset(a.data, "real", **kw)
    hs, rs = stack(sim, 400), stack(real)
    dt = sim.dt_ms
    t = np.arange(hs.shape[2]) * dt - (dc.get("acquisition") or {}).get("pre_ms", 0)
    rms_h, rms_r = np.sqrt((hs ** 2).mean(1)), np.sqrt((rs ** 2).mean(1))
    f = np.fft.rfftfreq(hs.shape[2], d=dt * 1e-3)
    ph = (np.abs(np.fft.rfft(hs, axis=-1)) ** 2).mean((0, 1))
    pr = (np.abs(np.fft.rfft(rs, axis=-1)) ** 2).mean((0, 1))

    def peakmap(X):
        i = np.argmax(np.sqrt((X ** 2).mean(1)), axis=1)
        return np.stack([X[k, :, i[k]] for k in range(len(X))])
    mh, mr = peakmap(hs), peakmap(rs)
    cc = lambda u, v: abs(np.corrcoef(u, v)[0, 1])
    within = [cc(mh[i], mh[j]) for i in range(0, len(mh), 17) for j in range(3, len(mh), 19) if i != j]
    cross = [cc(m, s) for m in mr for s in mh[::7]]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].fill_between(t, np.percentile(rms_h, 10, 0), np.percentile(rms_h, 90, 0), alpha=.3, label="sim healthy 10-90%")
    for x in rms_r:
        ax[0].plot(t, x, "k", lw=1)
    ax[0].set_xlabel("ms from detected onset"); ax[0].set_title("RMS over sensors (normalised)"); ax[0].legend()
    ax[1].semilogy(f, ph / ph.sum(), label="sim"); ax[1].semilogy(f, pr / pr.sum(), "k", label="real")
    ax[1].set_xlabel("Hz"); ax[1].set_title("spectrum"); ax[1].legend()
    ax[2].hist(within, 20, alpha=.5, density=True, label="sim-sim"); ax[2].hist(cross, 20, alpha=.5, density=True, label="real-sim")
    ax[2].set_title("|corr| of RMS-peak maps"); ax[2].legend()
    out = Path("runs/gap"); out.mkdir(parents=True, exist_ok=True)
    name = Path(a.data).name
    fig.suptitle(f"{name}: simulated healthy vs measured healthy (after the acquisition model)")
    fig.tight_layout(); fig.savefig(out / f"{name}.png", dpi=110)
    summary = dict(dataset=name, real=len(rs), sim=len(hs),
                   rms_peak_ms_real=[float(t[np.argmax(x)]) for x in rms_r], rms_peak_ms_sim_median=float(np.median(t[np.argmax(rms_h, 1)])),
                   hf_power_above_40Hz=dict(sim=float(ph[f > 40].sum() / ph.sum()), real=float(pr[f > 40].sum() / pr.sum())),
                   map_corr_median=dict(sim_sim=float(np.median(within)), real_sim=float(np.median(cross))))
    json.dump(summary, open(out / f"{name}.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
