"""Low-level mesh readers: openCARP (.pts/.elem/.lon/.vtx/.igb), legacy ASCII VTK, EnSight Gold binary.

All readers return plain numpy arrays. Coordinates are returned in the file's units; callers convert.
"""
from __future__ import annotations

import os
import re

import numpy as np


# ---------------------------------------------------------------- openCARP
def read_pts(path: str) -> np.ndarray:
    return np.loadtxt(path, skiprows=1, dtype=np.float64)


def read_elem(path: str) -> tuple[np.ndarray, np.ndarray]:
    a = np.loadtxt(path, skiprows=1, usecols=(1, 2, 3, 4, 5), dtype=np.int64)
    return a[:, :4], a[:, 4]


def read_lon(path: str) -> np.ndarray:
    f = np.loadtxt(path, skiprows=1, usecols=(0, 1, 2), dtype=np.float64)
    n = np.linalg.norm(f, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return f / n


def read_vtx(path: str) -> np.ndarray:
    lines = open(path).read().split()
    n = int(lines[0])
    return np.array(lines[2:2 + n], dtype=np.int64)


def read_igb(path: str, n_nodes: int) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f4", offset=1024)
    n_t = raw.size // n_nodes
    return raw[: n_t * n_nodes].reshape(n_t, n_nodes)


def read_par_regions(path: str) -> dict[int, dict]:
    """gregion blocks of an openCARP .par file: name, tag IDs, eikonal velocities (mm/s)."""
    txt = open(path).read()
    out = {}
    for idx in sorted({int(i) for i in re.findall(r"gregion\[(\d+)\]", txt)}):
        def g(key):
            m = re.search(rf"gregion\[{idx}\]\.{re.escape(key)}\s*=\s*(\S+)", txt)
            return m.group(1) if m else None
        ids = [int(x) for x in re.findall(rf"gregion\[{idx}\]\.ID\[\d+\]\s*=\s*(\d+)", txt)]
        out[idx] = dict(name=g("name"), ids=ids, vl=float(g("dream.vel_l")), vt=float(g("dream.vel_t")))
    return out


# ---------------------------------------------------------------- legacy VTK (ASCII)
def _block(lines, i, n_lines):
    """Parse lines[i:i+n_lines] (one record per line) into a flat float array, in chunks."""
    parts = [np.fromstring(" ".join(lines[j:min(j + 200000, i + n_lines)]), sep=" ")
             for j in range(i, i + n_lines, 200000)]
    return np.concatenate(parts)


def read_vtk_ascii(path: str, want_cells: bool = True) -> dict:
    """Legacy ASCII VTK UNSTRUCTURED_GRID reader (points, tetra cells, POINT/CELL scalars + vectors).
    Assumes one record per line (as written by VTK); fast enough for ~500k points / 2.5M cells."""
    with open(path, "r") as fh:
        lines = fh.read().splitlines()
    out = {"point_data": {}, "cell_data": {}}
    i, section, n_pts, n_cells = 0, None, 0, 0
    while i < len(lines):
        tok = lines[i].split()
        i += 1
        if not tok:
            continue
        key = tok[0].upper()
        if key == "POINTS":
            n_pts = int(tok[1])
            arr = _block(lines, i, n_pts)
            out["points"] = arr.reshape(n_pts, 3)
            i += n_pts
        elif key == "CELLS":
            n_cells = int(tok[1])
            if want_cells:
                arr = _block(lines, i, n_cells)
                out["cells"] = arr.reshape(n_cells, 5)[:, 1:5].astype(np.int64)
            i += n_cells
        elif key == "CELL_TYPES":
            i += int(tok[1])
        elif key == "POINT_DATA":
            section, n_cur = "point_data", int(tok[1])
        elif key == "CELL_DATA":
            section, n_cur = "cell_data", int(tok[1])
        elif key == "SCALARS":
            ncomp = int(tok[3]) if len(tok) > 3 else 1
            if lines[i].upper().startswith("LOOKUP_TABLE"):
                i += 1
            arr = _block(lines, i, n_cur)
            out[section][tok[1]] = arr.reshape(n_cur, ncomp) if ncomp > 1 else arr
            i += n_cur
        elif key == "VECTORS":
            arr = _block(lines, i, n_cur)
            out[section][tok[1]] = arr.reshape(n_cur, 3)
            i += n_cur
    return out


# ---------------------------------------------------------------- EnSight Gold binary (meshtool output)
class _Bin:
    def __init__(self, path):
        self.b = open(path, "rb").read()
        self.p = 0

    def s(self):
        v = self.b[self.p:self.p + 80].split(b"\x00")[0].decode("ascii", "ignore").strip()
        self.p += 80
        return v

    def i(self, n=1):
        v = np.frombuffer(self.b, dtype="<i4", count=n, offset=self.p)
        self.p += 4 * n
        return v if n > 1 else int(v[0])

    def f(self, n):
        v = np.frombuffer(self.b, dtype="<f4", count=n, offset=self.p)
        self.p += 4 * n
        return v


_NPE = {"tetra4": 4, "tria3": 3, "hexa8": 8, "penta6": 6, "pyramid5": 5, "quad4": 4, "bar2": 2, "point": 1}


def read_ensight_geo(path: str) -> dict:
    r = _Bin(path)
    assert r.s().lower().startswith("c binary"), "not a C Binary EnSight file"
    r.s(); r.s()
    node_id = r.s().lower()
    elem_id = r.s().lower()
    pts, cells = [], {}
    while r.p < len(r.b):
        t = r.s().lower()
        if t.startswith("extents"):
            r.f(6)
            continue
        if t != "part":
            raise ValueError(f"unexpected token {t!r}")
        r.i(); r.s()
        assert r.s().lower() == "coordinates"
        nn = r.i()
        if "given" in node_id or "ignore" in node_id:
            r.i(nn)
        xyz = np.stack([r.f(nn), r.f(nn), r.f(nn)], axis=1).astype(np.float64)
        pts.append(xyz)
        while r.p < len(r.b):
            save = r.p
            et = r.s().lower()
            if et == "part":
                r.p = save
                break
            ne = r.i()
            if "given" in elem_id or "ignore" in elem_id:
                r.i(ne)
            conn = r.i(ne * _NPE[et]).reshape(ne, _NPE[et]).astype(np.int64) - 1
            cells.setdefault(et, []).append(conn)
    return {"points": np.concatenate(pts), "cells": {k: np.concatenate(v) for k, v in cells.items()}}


def read_ensight_var(path: str, n: int, ncomp: int = 1) -> np.ndarray:
    """Per-node or per-element variable of a single-part file (meshtool layout: components blocked)."""
    r = _Bin(path)
    r.s()
    assert r.s().lower() == "part"
    r.i()
    r.s()  # 'coordinates' or element type
    v = r.f(n * ncomp).astype(np.float64)
    return v.reshape(ncomp, n).T if ncomp > 1 else v


def exists_all(*paths: str) -> bool:
    return all(os.path.exists(p) for p in paths)
