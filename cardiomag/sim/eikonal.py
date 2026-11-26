"""Anisotropic eikonal activation on tetrahedral meshes via graph shortest paths.

Edge travel time for an edge d (mm) in an element with fibre f and velocities (v_l, v_t) (mm/ms):
    t = sqrt( (d.f)^2 / v_l^2 + (|d|^2 - (d.f)^2) / v_t^2 )
Shortest paths on the edge graph overestimate the continuum travel time (zig-zag paths); a global
correction factor `GRAPH_FACTOR` is calibrated against the reference DREAM activation of the
biatrial models (see tests / scripts/validate_eikonal.py).

Stimuli are (node indices, onset ms) pairs, implemented with a virtual source node.
Blocked tissue: nodes with `blocked=True` are removed (never activate, AT = inf).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

GRAPH_FACTOR = 1.08  # calibrated: MAE 0.9-1.4 ms vs reference DREAM activation, biatrial S1-S4


class EdgeGraph:
    """Unique mesh edges with the fibre / element of one adjacent tetrahedron (reused across scenarios)."""

    def __init__(self, points_m, tets, fibres):
        pairs = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        e = tets[:, pairs].reshape(-1, 2)
        e.sort(axis=1)
        elem = np.repeat(np.arange(tets.shape[0]), 6)
        key = e[:, 0] * (tets.max() + 1) + e[:, 1]
        _, first = np.unique(key, return_index=True)
        self.i, self.j = e[first, 0], e[first, 1]
        self.elem = elem[first]
        d = (points_m[self.j] - points_m[self.i]) * 1e3      # mm
        f = fibres[self.elem]
        self.dl2 = (d * f).sum(1) ** 2
        self.dt2 = np.maximum((d * d).sum(1) - self.dl2, 0)
        self.n = points_m.shape[0]

    def solve(self, vl_elem, vt_elem, stimuli, blocked_nodes=None, fast_nodes=None, v_fast=None,
              factor=None):
        """Activation times (ms) per node. vl/vt per element (mm/ms). fast_nodes: edges with both ends in
        the fast layer conduct isotropically at v_fast (Purkinje-like endocardial layer)."""
        factor = GRAPH_FACTOR if factor is None else factor
        vl = vl_elem[self.elem] * factor
        vt = vt_elem[self.elem] * factor
        w = np.sqrt(self.dl2 / vl ** 2 + self.dt2 / vt ** 2)
        if fast_nodes is not None and v_fast:
            ff = fast_nodes[self.i] & fast_nodes[self.j]
            w[ff] = np.sqrt(self.dl2[ff] + self.dt2[ff]) / (v_fast * factor)
        ok = np.isfinite(w) & (w > 0)
        if blocked_nodes is not None:
            ok &= ~blocked_nodes[self.i] & ~blocked_nodes[self.j]
        src = self.n
        si, so = [], []
        for nodes, onset in stimuli:
            nodes = np.asarray(nodes)
            if blocked_nodes is not None:
                nodes = nodes[~blocked_nodes[nodes]]
            si.append(nodes)
            so.append(np.full(nodes.size, max(float(onset), 1e-6)))
        si, so = np.concatenate(si), np.concatenate(so)
        rows = np.concatenate([self.i[ok], np.full(si.size, src)])
        cols = np.concatenate([self.j[ok], si])
        vals = np.concatenate([w[ok], so])
        G = sp.csr_matrix((vals, (rows, cols)), shape=(self.n + 1, self.n + 1))
        at = dijkstra(G, directed=False, indices=src)
        return at[: self.n]


def vm_from_at(at_ms, t_ms, v_rest=-85.0, v_peak=20.0, tau_up=1.0, plateau_decay_ms=300.0):
    """Template action potential (mV) for nodes with activation times at_ms at times t_ms (T,) -> (T,N).
    Depolarisation only (sigmoid upstroke + slow plateau decay); repolarisation is outside the window."""
    dt = (t_ms[:, None] - at_ms[None, :]).astype(np.float32)
    up = 0.5 * (1 + np.tanh(dt / tau_up))
    decay = np.exp(-np.clip(dt, 0, None) / plateau_decay_ms)
    v = v_rest + (v_peak - v_rest) * up * (0.85 + 0.15 * decay)
    v[:, ~np.isfinite(at_ms)] = v_rest
    return v
