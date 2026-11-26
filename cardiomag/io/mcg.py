"""Measured MCG datasets as R-peak-aligned median beats on known sensor layouts.

PTB (Koch et al. 2011): 49 lowest-layer Z SQUIDs, fT, 1 kHz, 100 s. Stored values are B . anterior
  (inferred from QRS axes; the paper does not state the polarity - see Analysis/ptb_compare.py).
Kiel Cardio (PhysioNet): 4 QuSpin OPMs x 25 trials, 200 Hz preprocessed WFDB (format 212, pT).
  Channels 0/2/4/6 measure -y (posterior) -> B . anterior = -value. Kiel x is mapped to the patient's
  left with KIEL_X_TO_LEFT = -1 (array over the left chest); this sign is not documented.

Each record yields a median beat B(t) (pT) per sensor in the same channel order as the simulated
presets (cardiomag.physics.sensors: ptb49 / kiel36), plus P-wave and QRS onsets.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np

from .. import paths
from ..physics.sensors import kiel36_layout, ptb49_layout

PRE, POST = 450, 550
KIEL_X_TO_LEFT = -1


# ------------------------------------------------------------------------------------- beat processing
def _movmean(x, w):
    c = np.cumsum(np.pad(x, ((0, 0), (w // 2 + 1, w // 2)), mode="edge"), axis=1)
    return (c[:, w:] - c[:, :-w]) / w


def detect_r_peaks(X, fs, refractory_s=0.3):
    """Greedy R detection on multichannel data (C, T): highest QRS-energy peaks with exclusion."""
    e = (np.diff(X, axis=1) ** 2).sum(axis=0)
    e = np.convolve(e, np.ones(5) / 5, mode="same")
    excl = int(refractory_s * fs)
    taken = np.zeros(e.size, bool)
    peaks = []
    for i in np.argsort(e)[::-1]:
        if len(peaks) >= int(e.size / fs / 0.33):
            break
        if not taken[i]:
            peaks.append(i)
            taken[max(0, i - excl):i + excl] = True
    peaks = np.array(peaks)
    ref = np.median(np.sort(e[peaks])[::-1][:max(5, int(e.size / fs * 0.8))])
    return np.sort(peaks[(e[peaks] > 0.35 * ref) & (e[peaks] < 8 * ref)])


def median_beat(X, fs, R):
    """(C, T) at 1 kHz on t = -PRE..POST-1 ms; baseline from the TP window."""
    pre_s, post_s = int(PRE * fs / 1000), int(POST * fs / 1000)
    R = R[(R > pre_s) & (R < X.shape[1] - post_s)]
    seg = np.stack([X[:, r - pre_s:r + post_s] for r in R])
    seg -= seg[:, :, : max(1, int(70 * fs / 1000))].mean(axis=2, keepdims=True)
    med = np.median(seg, axis=0)
    ts = np.arange(-pre_s, post_s) * 1000.0 / fs
    t1 = np.arange(-PRE, POST)
    return np.stack([np.interp(t1, ts, m) for m in med]), R.size


def p_onset(avg, search_end=-70):
    """P-wave onset (ms rel. R) with re-baselining on the 40 ms before onset."""
    t = np.arange(-PRE, POST)
    for _ in range(3):
        rms = np.sqrt((avg ** 2).mean(axis=0))
        rs = np.convolve(rms, np.ones(9) / 9, mode="same")
        s = (t >= -330) & (t <= search_end)
        base = rs[s].min()
        ip = np.flatnonzero(s)[np.argmax(rs[s])]
        lvl = base + 0.10 * (rs[ip] - base)
        a = ip
        while rs[a] > lvl and t[a] > -330:
            a -= 1
        avg = avg - avg[:, max(a - 40, 0):a].mean(axis=1, keepdims=True)
    return int(t[a]), avg


def qrs_onset(avg):
    t = np.arange(-PRE, POST)
    rms = np.sqrt((avg ** 2).mean(axis=0))
    i = PRE + int(np.argmax(rms[PRE - 60:PRE + 60])) - 60
    lvl = 0.08 * rms[i] + 0.9 * np.median(rms[(t > -150) & (t < -60)])
    while rms[i] > lvl and t[i] > -120:
        i -= 1
    return int(t[i])


def window(avg, onset_ms, t0_ms, t1_ms, dt_ms):
    """Resample (C, 1 kHz beat) to the model grid relative to onset."""
    t = np.arange(-PRE, POST)
    tq = onset_ms + np.arange(t0_ms, t1_ms, dt_ms)
    return np.stack([np.interp(tq, t, c) for c in avg])


# ------------------------------------------------------------------------------------------ PTB
def load_ptb():
    """Median beat (49, 1000) pT on the ptb49 layout order, P and QRS onsets."""
    _, names = ptb49_layout()
    X = []
    for nm in names:
        f = glob.glob(os.path.join(paths.PTB, "MCG_part*", "*", "MCG_data_CH_*", f"MCG_CH_{nm[:2]}",
                                   f"MCG_CH_{nm}.txt"))[0]
        X.append(np.loadtxt(f) * 1e-3)
    X = np.array(X)
    X = X - _movmean(X, 1501)
    R = detect_r_peaks(X, 1000.0)
    avg, nb = median_beat(X, 1000.0, R)
    p0, avg_p = p_onset(avg, search_end=-40)
    return dict(name="ptb_01", beat=avg, beat_p=avg_p, p_onset=p0, qrs_onset=qrs_onset(avg), beats=nb,
                preset="ptb49")


# ------------------------------------------------------------------------------------------ Kiel
def _read212(stem):
    h = open(stem + ".hea").read().splitlines()
    n_ch, fs, n = map(int, h[0].split()[1:4])
    gains, base = [], []
    for line in h[1:1 + n_ch]:
        m = re.match(r"([\d.eE+-]+)\((-?\d+)\)", line.split()[2])
        gains.append(float(m.group(1)))
        base.append(int(m.group(2)))
    pos = {int(m.group(1)): np.array(m.group(2).split(), float)
           for m in re.finditer(r"position sensor (\d) \[cm\]>: \[([^\]]+)\]", "\n".join(h))}
    b = np.fromfile(stem + ".dat", dtype=np.uint8)
    b = b[: len(b) // 3 * 3].reshape(-1, 3).astype(np.int32)
    s = np.empty(2 * len(b), np.int32)
    s[0::2] = b[:, 0] | ((b[:, 1] & 0x0F) << 8)
    s[1::2] = b[:, 2] | ((b[:, 1] & 0xF0) << 4)
    s[s > 2047] -= 4096
    return ((s[: n * n_ch].reshape(n, n_ch) - np.array(base)) / np.array(gains)).T, fs, pos


def load_kiel(subject: int):
    """Median beat (36, 1000) pT on the kiel36 layout order (repeated positions averaged)."""
    layout, _ = kiel36_layout()
    beats = {}
    for stem in sorted(glob.glob(os.path.join(paths.KIEL, f"subject{subject}_preprocessed_trial*.hea"))):
        X, fs, pos = _read212(stem[:-4])
        R = detect_r_peaks(X, fs)
        avg, _ = median_beat(X[[0, 2, 4, 6]], fs, R)
        for c, sensor in enumerate((0, 1, 2, 3)):
            x_cm, _, z_cm = pos[sensor]
            key = (KIEL_X_TO_LEFT * x_cm * 10.0, z_cm * 10.0)
            beats.setdefault(key, []).append(-avg[c])
    keys = np.array(list(beats.keys()))
    keys_c = keys - keys.mean(axis=0)
    order = [int(np.argmin(np.linalg.norm(keys_c - p, axis=1))) for p in layout]
    avg = np.array([np.mean(beats[tuple(keys[i])], axis=0) for i in order])
    p0, avg_p = p_onset(avg, search_end=-70)
    return dict(name=f"kiel_{subject:02d}", beat=avg, beat_p=avg_p, p_onset=p0, qrs_onset=qrs_onset(avg),
                preset="kiel36", positions_mm=keys[order])


def real_windows(preset: str, kind: str, t0_ms: float, t1_ms: float, dt_ms: float):
    """Real healthy recordings on a preset as model inputs: list of (name, X (S,T) pT)."""
    recs = []
    if preset == "ptb49" and paths.PTB.exists():
        recs.append(load_ptb())
    if preset == "kiel36" and paths.KIEL.exists():
        recs += [load_kiel(s) for s in range(1, 8)]
    out = []
    for r in recs:
        if kind == "atria":
            w = window(r["beat_p"], r["p_onset"], t0_ms, t1_ms, dt_ms)
            # the simulation has no ventricles: fade the signal out before the QRS starts
            tq = r["p_onset"] + np.arange(t0_ms, t1_ms, dt_ms)
            fade = np.clip((r["qrs_onset"] - 10 - tq) / 20.0, 0, 1)
            out.append((r["name"], w * fade[None, :]))
        else:
            out.append((r["name"], window(r["beat"], r["qrs_onset"], t0_ms, t1_ms, dt_ms)))
    return out
