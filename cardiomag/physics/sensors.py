"""Sensor arrays. Positions are defined in an array frame (u = patient's left, v = superior, mm) and
placed on the chest: array centre at (cx, cy) in the body frame, plane at skin + standoff, measuring
along `axis` (default the chest normal = anterior). A flat array sits at the highest skin point under
its footprint plus the standoff (like a rigid OPM/SQUID array).

Presets
  grid      : N x M grid, spacing mm
  ptb49     : PTB 304-channel SQUID system, the 49 lowest-layer Z sensors (Koch et al. 2011)
  kiel36    : Kiel Cardio OPM grid, 6 x 6 positions at 3 cm (QuSpin QZFM, 25 trials x 4 sensors)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

# PTB module centres (x left, y towards feet, mm) and Z1..Z3 offsets; positions present in the public data
_PTB_MODULES = {"A": (-86.6, -66), "B": (-86.6, -116), "C": (-86.6, -166), "D": (-43.3, -41), "E": (-43.3, -91),
                "F": (-43.3, -141), "G": (-43.3, -191), "H": (0, -16), "I": (0, -66), "K": (0, -116),
                "L": (0, -166), "M": (0, -216), "N": (43.3, -41), "O": (43.3, -91), "P": (43.3, -141),
                "R": (43.3, -191), "S": (86.6, -66), "T": (86.6, -116), "U": (86.6, -166)}
_PTB_Z = {"Z1": (16.7, 0), "Z2": (-8.3, -14.4), "Z3": (-8.3, 14.4)}
_PTB_PRESENT = {"Z1": "ABCDEFGHIKLNPRSTU", "Z2": "ABCDEFHIKLNPRSTU", "Z3": "ABCDFGHIKLNPRSTU"}


def ptb49_layout():
    """(49, 2) mm in (left, superior) relative to the array centre, plus channel names."""
    xy, names = [], []
    for sq, mods in _PTB_PRESENT.items():
        for m in mods:
            mx, my = _PTB_MODULES[m]
            ox, oy = _PTB_Z[sq]
            xy.append((mx + ox, -(my + oy)))          # PTB y points to the feet -> superior = -y
            names.append(f"{sq}{m}")
    xy = np.array(xy)
    return xy - np.array([0.0, 116.0]), names           # centre of the module grid (y_PTB = -116)


def kiel36_layout():
    xs = np.arange(6) * 30.0
    g = np.array([(x, y) for y in xs for x in xs])
    return g - g.mean(axis=0), [f"K{i:02d}" for i in range(36)]


@dataclass
class SensorArray:
    preset: str = "grid"
    nx: int = 8
    ny: int = 8
    spacing_mm: float = 25.0
    centre_mm: tuple = (0.0, 0.0)      # (left, superior) in the body frame, relative to heart centroid
    standoff_mm: float = 10.0          # sensor (cell centre) above the skin
    rotation_deg: float = 0.0          # in-plane rotation
    axis: str = "normal"               # "normal" | "left" | "superior"
    names: list = field(default_factory=list)

    def layout_mm(self):
        if self.preset == "grid":
            gx = (np.arange(self.nx) - (self.nx - 1) / 2) * self.spacing_mm
            gy = (np.arange(self.ny) - (self.ny - 1) / 2) * self.spacing_mm
            X, Y = np.meshgrid(gx, gy)
            xy = np.column_stack([X.ravel(), Y.ravel()])
            names = [f"G{i:03d}" for i in range(xy.shape[0])]
        elif self.preset == "ptb49":
            xy, names = ptb49_layout()
        elif self.preset == "kiel36":
            xy, names = kiel36_layout()
        else:
            raise ValueError(f"unknown preset {self.preset}")
        a = np.radians(self.rotation_deg)
        Rz = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        self.names = names
        return xy @ Rz.T

    def place(self, skin_fn):
        """Sensor positions (S,3) and measurement directions (S,3) in the body frame (m)."""
        xy = (self.layout_mm() + np.array(self.centre_mm)) * 1e-3
        z_skin = np.max(skin_fn(xy[:, 0], xy[:, 1]))
        pos = np.column_stack([xy, np.full(xy.shape[0], z_skin + self.standoff_mm * 1e-3)])
        n = {"normal": (0, 0, 1), "left": (1, 0, 0), "superior": (0, 1, 0)}[self.axis]
        return pos, np.tile(np.array(n, float), (xy.shape[0], 1))

    @property
    def n_sensors(self):
        return self.layout_mm().shape[0]

    def to_dict(self):
        d = asdict(self)
        d.pop("names", None)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        for k in ("centre_mm",):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


def default_centre_mm(kind: str, preset: str):
    """Nominal array centre relative to the heart (tissue) centroid, (left, superior) in mm.
    ptb49 : centred on the sternum at the V1/V2 (4th intercostal) level, as in the PTB recordings
    kiel36: over the left upper chest, 0-15 cm below the shoulder line (Kiel README geometry)
    grid  : over the heart"""
    table = {("atria", "ptb49"): (5.0, -76.0), ("atria", "kiel36"): (65.0, 90.0), ("atria", "grid"): (0.0, -20.0),
             ("ventricles", "ptb49"): (-30.0, 0.0), ("ventricles", "kiel36"): (35.0, 130.0),
             ("ventricles", "grid"): (20.0, -10.0)}
    return table[(kind, preset)]


# ------------------------------------------------------------------------------------------- noise
def white_noise_std(density_fT_rtHz: float, fs_hz: float) -> float:
    """Std (T) of white noise with the given amplitude spectral density, sampled at fs (bandwidth fs/2)."""
    return density_fT_rtHz * 1e-15 * np.sqrt(fs_hz / 2.0)


def add_noise(signal_T: np.ndarray, rng: np.random.Generator, density_fT_rtHz: float, fs_hz: float,
              pink_fraction: float = 0.3) -> np.ndarray:
    """White + 1/f-ish noise (cumulative-sum random walk, rescaled) on (..., T) signals."""
    sd = white_noise_std(density_fT_rtHz, fs_hz)
    n = rng.normal(0, sd, signal_T.shape)
    if pink_fraction > 0:
        rw = np.cumsum(rng.normal(0, 1, signal_T.shape), axis=-1)
        rw -= rw.mean(axis=-1, keepdims=True)
        rw /= rw.std(axis=-1, keepdims=True) + 1e-12
        n = np.sqrt(1 - pink_fraction) * n + np.sqrt(pink_fraction) * sd * rw
    return signal_T + n
