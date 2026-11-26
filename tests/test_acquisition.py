import numpy as np

from cardiomag.data.acquisition import Acquisition, detect_onset, lowpass


def _bump(T=140, at=40, S=6, dt=2.0):
    t = np.arange(T) * dt
    w = np.random.default_rng(0).normal(size=S)
    return (w[:, None] * np.exp(-((t - at * dt - 30) / 12.0) ** 2)).astype(np.float32)


def test_onset_alignment_is_shift_invariant():
    acq = Acquisition(np.arange(-40, 240, 2.0), dict(lowpass_hz=80, pre_ms=10, len_ms=100))
    a = acq(_bump(at=30), simulated=False)
    b = acq(_bump(at=45), simulated=False)
    assert a.shape == (6, 50) and np.allclose(a, b, atol=1e-3)


def test_snr_noise_level():
    rng = np.random.default_rng(1)
    acq = Acquisition(np.arange(-40, 240, 2.0), dict(lowpass_hz=None, snr=[10, 10], len_ms=270))
    x = _bump()
    y = acq(x, rng=rng, simulated=True)
    assert y.shape[1] == 135


def test_lowpass_removes_high_frequency():
    t = np.arange(500) * 2e-3
    x = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 200 * t)
    y = lowpass(x[None], 40, 500)[0]
    assert np.abs(y - np.sin(2 * np.pi * 5 * t))[50:-50].max() < 0.05
