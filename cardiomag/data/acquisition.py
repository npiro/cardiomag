"""Acquisition-chain model applied IDENTICALLY to simulated and measured windows.

Why: a detector trained on clean simulations flags every real healthy recording (see the domain-gap
diagnosis in the README). The main causes were (1) onset definition - simulations started at the first
activated cell, real windows at an RMS threshold - (2) noise / bandwidth, and (3) windows of real atrial
data running into the QRS. This module removes (1)-(3) by construction:

  stored window (long, e.g. -40..240 ms)
      -> [sim only]  sensor noise at a signal-to-noise ratio sampled from the measured range
      -> low-pass (zero-phase Butterworth)
      -> onset = first time the smoothed RMS over sensors exceeds `onset_frac` of its rise
         (same detector as for the real median beats), searched backwards from the RMS peak
      -> crop [onset - pre_ms, onset - pre_ms + len_ms)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def lowpass(x, fc_hz, fs_hz, order=4):
    if not fc_hz or fc_hz >= fs_hz / 2:
        return x
    sos = butter(order, fc_hz, fs=fs_hz, output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def detect_onset(x, dt_ms, frac=0.10, smooth_ms=8, search_from_ms=None, t0_ms=0.0):
    """Index of the activation onset in x (S, T): RMS over sensors, smoothed, walk back from the peak
    until it drops below base + frac * (peak - base)."""
    rms = np.sqrt((x ** 2).mean(axis=0))
    k = max(1, int(round(smooth_ms / dt_ms)))
    rs = np.convolve(rms, np.ones(k) / k, mode="same")
    i0 = 0 if search_from_ms is None else max(0, int(round((search_from_ms - t0_ms) / dt_ms)))
    ip = i0 + int(np.argmax(rs[i0:]))
    base = rs[: max(ip, 1)].min()
    lvl = base + frac * (rs[ip] - base)
    a = ip
    while a > 0 and rs[a] > lvl:
        a -= 1
    return a


def snr_noise(x, rng, snr_range, pink_fraction=0.3):
    """Additive white + 1/f noise with RMS = signal RMS / SNR, SNR ~ log-uniform in snr_range."""
    lo, hi = snr_range
    snr = np.exp(rng.uniform(np.log(lo), np.log(hi)))
    sd = np.sqrt((x ** 2).mean()) / snr
    n = rng.normal(0, sd, x.shape)
    if pink_fraction > 0:
        rw = np.cumsum(rng.normal(0, 1, x.shape), axis=-1)
        rw -= rw.mean(axis=-1, keepdims=True)
        rw /= rw.std(axis=-1, keepdims=True) + 1e-12
        n = np.sqrt(1 - pink_fraction) * n + np.sqrt(pink_fraction) * sd * rw
    return (x + n).astype(np.float32)


class Acquisition:
    """Config (dataset `acquisition:` block):
        lowpass_hz: 80          zero-phase low-pass for both domains
        onset_frac: 0.10        RMS-onset threshold
        pre_ms: 10              ms kept before the detected onset
        len_ms: 120             model window length
        snr: [8, 40]            simulated noise, signal RMS / noise RMS (measured data: none added)
    """

    def __init__(self, t_ms, cfg):
        self.t = np.asarray(t_ms, float)
        self.dt = float(self.t[1] - self.t[0])
        self.fs = 1000.0 / self.dt
        self.lp = cfg.get("lowpass_hz", 80)
        self.frac = cfg.get("onset_frac", 0.10)
        self.pre = cfg.get("pre_ms", 10)
        self.n_out = int(round(cfg.get("len_ms", 120) / self.dt))
        self.snr = cfg.get("snr")
        self.search_from = cfg.get("search_from_ms")

    def __call__(self, x, rng=None, simulated=True, jitter_ms=0.0):
        if simulated and self.snr and rng is not None:
            x = snr_noise(x, rng, self.snr)
        x = lowpass(x, self.lp, self.fs)
        a = detect_onset(x, self.dt, self.frac, search_from_ms=self.search_from, t0_ms=self.t[0])
        s = a - int(round(self.pre / self.dt))
        if jitter_ms and rng is not None:
            s += int(round(rng.normal(0, jitter_ms) / self.dt))
        s = int(np.clip(s, 0, x.shape[1] - self.n_out))
        return x[:, s:s + self.n_out]
