"""Healthy and abnormal activation scenarios on the heart meshes.

ATRIA (biatrial S1-S4). In these models the left and right atrium are separate mesh components joined
through interatrial connections, represented by left-atrial sites (BB1-3 Bachmann's bundle, CS
coronary sinus, PC posterior) that are activated with a delay after the nearest right-atrial tissue.
We model each connection as (LA nodes, RA entry nodes, delay) with delays taken from the reference
(DREAM) activation, and couple both atria iteratively, so activation can cross in either direction.
  healthy     : sinus node (SAN1/SAN2) activation, CV and timing jitter
  slow_patch  : fibrotic patch (r 10-20 mm) with CV x 0.2-0.4
  ia_block    : interatrial block - Bachmann's bundle connections removed (LA via CS/PC only)
  ectopic     : ectopic focus in the LA or RA (>= 25 mm from the SAN) instead of the sinus node

VENTRICLES (Strocchi healthy / heart failure). Myocardium v_l 0.5, v_t 0.2 mm/ms; a thin fast endocardial
layer (UVC transmural <= 0.05) conducting at 1.0-1.5 mm/ms stands in for the Purkinje system, triggered
at the LV and RV mid-septum (normal QRS ~80-100 ms).
  normal      : LV + RV septal roots, both fast layers
  lbbb        : left bundle branch block - LV root and LV fast layer removed
  rbbb        : right bundle branch block - RV root and RV fast layer removed
  scar        : non-conducting, non-current-generating LV region (r 15-25 mm)
  (heart-failure anatomies are simulated with normal or LBBB conduction; anatomy group is a label)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..io.hearts import HeartMesh
from .eikonal import EdgeGraph

ATRIAL_CLASSES = ["healthy", "slow_patch", "ia_block", "ectopic"]
VENTRICULAR_CLASSES = ["normal", "lbbb", "rbbb", "scar"]


@dataclass
class Activation:
    at_ms: np.ndarray           # (N,) activation time, inf = never
    sigma_scale: np.ndarray     # (E,) conductivity multiplier (0 in scar)
    label: str
    params: dict


def _elem_cv(h: HeartMesh, scale=1.0):
    vl = np.array([h.cv[h.region_names[r]][0] for r in h.region]) * scale
    vt = np.array([h.cv[h.region_names[r]][1] for r in h.region]) * scale
    return vl, vt


# =========================================================================================== atria
class AtrialSimulator:
    def __init__(self, h: HeartMesh):
        assert h.kind == "atria"
        self.h = h
        self.g = EdgeGraph(h.points, h.tets, h.fibres)
        self.name_to_region = {v: k for k, v in h.region_names.items()}
        la_r, ra_r = self.name_to_region["LA"], self.name_to_region["RA"]
        self.la_nodes = np.unique(h.tets[h.region == la_r])
        ra_like = [self.name_to_region[n] for n in ("RA", "CT", "PM") if n in self.name_to_region]
        self.ra_nodes = np.unique(h.tets[np.isin(h.region, ra_like)])
        self.blocked = np.zeros(h.n_nodes, bool)
        if "BO" in h.sites:
            self.blocked[h.sites["BO"]] = True
        ref = h.reference_at
        t0 = np.nanmin(ref)
        self.san = [k for k in h.sites if k.startswith("SAN")]
        self.san_onset = {k: float(np.nanmin(ref[h.sites[k]]) - t0) for k in self.san}
        # connection entry on the RA side: the nearest RA tissue that is activated >= 3 ms before the
        # LA site in the reference (the connection runs from there); delay = reference time difference
        ra_at = ref[self.ra_nodes]
        self.connections = []
        for k, nodes in h.sites.items():
            if k == "BO" or k in self.san:
                continue
            t_site = np.nanmin(ref[nodes])
            c = h.points[nodes].mean(axis=0)
            d = np.linalg.norm(h.points[self.ra_nodes] - c, axis=1)
            cand = np.flatnonzero(np.isfinite(ra_at) & (ra_at <= t_site - 3.0))
            if cand.size == 0:
                cand = np.arange(self.ra_nodes.size)
            near = cand[np.argsort(d[cand])[:20]]
            ra_entry = self.ra_nodes[near]
            delay = float(t_site - np.nanmin(ref[ra_entry]))
            self.connections.append(dict(name=k, la=nodes, ra=ra_entry, delay=max(delay, 2.0),
                                         dist_mm=float(d[near].mean() * 1e3)))
        self.vl0, self.vt0 = _elem_cv(h)

    def _solve(self, vl, vt, primary, connections, blocked):
        stim = list(primary)
        at = self.g.solve(vl, vt, stim, blocked_nodes=blocked)
        for _ in range(3):  # couple both atria through the connections
            extra = []
            for c in connections:
                t_ra, t_la = np.min(at[c["ra"]]), np.min(at[c["la"]])
                if np.isfinite(t_ra):
                    extra.append((c["la"], t_ra + c["delay"]))
                if np.isfinite(t_la):
                    extra.append((c["ra"], t_la + c["delay"]))
            new = self.g.solve(vl, vt, stim + extra, blocked_nodes=blocked)
            if np.allclose(np.where(np.isfinite(new), new, -1), np.where(np.isfinite(at), at, -1), atol=0.05):
                at = new
                break
            at = new
        return at

    def sample(self, label: str, rng: np.random.Generator) -> Activation:
        h = self.h
        cv_scale = rng.uniform(0.9, 1.1)
        vl, vt = self.vl0 * cv_scale, self.vt0 * cv_scale
        sigma_scale = np.ones(h.tets.shape[0])
        conns = [dict(c, delay=c["delay"] * rng.uniform(0.85, 1.15)) for c in self.connections]
        sinus = [(h.sites[k], self.san_onset[k] + rng.normal(0, 2.0)) for k in self.san]
        params = dict(cv_scale=cv_scale)
        primary = sinus
        if label == "healthy":
            pass
        elif label == "slow_patch":
            pool = self.la_nodes if rng.random() < 0.5 else self.ra_nodes
            c = h.points[rng.choice(pool)]
            r = rng.uniform(0.010, 0.020)
            f = rng.uniform(0.2, 0.4)
            cen = h.points[h.tets].mean(axis=1)
            inside = np.linalg.norm(cen - c, axis=1) < r
            vl, vt = vl.copy(), vt.copy()
            vl[inside] *= f
            vt[inside] *= f
            params.update(centre_m=c.tolist(), radius_m=r, cv_factor=f, n_elem=int(inside.sum()))
        elif label == "ia_block":
            conns = [c for c in conns if not c["name"].startswith("BB")]
            params.update(removed=[c["name"] for c in self.connections if c["name"].startswith("BB")])
        elif label == "ectopic":
            san_c = h.points[np.concatenate([h.sites[k] for k in self.san])].mean(axis=0)
            if rng.random() < 0.6:
                pool = self.la_nodes
            else:
                pool = self.ra_nodes[np.linalg.norm(h.points[self.ra_nodes] - san_c, axis=1) > 0.025]
            focus = rng.choice(pool)
            ball = np.flatnonzero(np.linalg.norm(h.points - h.points[focus], axis=1) < 0.003)
            primary = [(ball, 0.0)]
            params.update(focus_m=h.points[focus].tolist(), side="LA" if np.isin(focus, self.la_nodes) else "RA")
        else:
            raise ValueError(label)
        at = self._solve(vl, vt, primary, conns, self.blocked)
        at = at - np.nanmin(at[np.isfinite(at)])
        return Activation(at_ms=at, sigma_scale=sigma_scale, label=label, params=params)

    def reference(self) -> Activation:
        h = self.h
        at = h.reference_at - np.nanmin(h.reference_at)
        at = np.where(np.isnan(at), np.inf, at)
        return Activation(at_ms=at, sigma_scale=np.ones(h.tets.shape[0]), label="healthy", params={"reference": True})


# =========================================================================================== ventricles
class VentricularSimulator:
    def __init__(self, h: HeartMesh):
        assert h.kind == "ventricles"
        self.h = h
        self.g = EdgeGraph(h.points, h.tets, h.fibres)
        nf = h.node_fields
        lv_nodes = np.unique(h.tets[h.region == 0])
        rv_nodes = np.unique(h.tets[h.region == 1])
        self.lv = np.zeros(h.n_nodes, bool)
        self.lv[lv_nodes] = True
        self.rv = np.zeros(h.n_nodes, bool)
        self.rv[rv_nodes] = True
        self.rv &= ~self.lv  # septal interface nodes belong to the LV
        rho, z = nf["rho"], nf["z"]
        valid = (rho >= 0) & (rho <= 1)
        self.endo = valid & (rho <= 0.05)
        self.z = np.where((z >= 0) & (z <= 1), z, np.nan)
        lv_endo = np.flatnonzero(self.endo & self.lv)
        rv_endo = np.flatnonzero(self.endo & self.rv)
        d_lv, _ = cKDTree(h.points[rv_endo]).query(h.points[lv_endo])
        d_rv, _ = cKDTree(h.points[lv_endo]).query(h.points[rv_endo])
        mid_lv = (self.z[lv_endo] > 0.35) & (self.z[lv_endo] < 0.65)
        mid_rv = (self.z[rv_endo] > 0.35) & (self.z[rv_endo] < 0.65)
        self.lv_root = lv_endo[mid_lv & (d_lv < np.percentile(d_lv[mid_lv], 15))]
        self.rv_root = rv_endo[mid_rv & (d_rv < np.percentile(d_rv[mid_rv], 15))]
        self.lv_nodes = lv_nodes
        self.vl0, self.vt0 = _elem_cv(h)

    def sample(self, label: str, rng: np.random.Generator) -> Activation:
        h = self.h
        cv_scale = rng.uniform(0.9, 1.1)
        vl, vt = self.vl0 * cv_scale, self.vt0 * cv_scale
        v_fast = rng.uniform(1.0, 1.5)
        fast = self.endo.copy()
        stim = [(self.lv_root, rng.uniform(0, 3)), (self.rv_root, rng.uniform(3, 8))]
        blocked = np.zeros(h.n_nodes, bool)
        sigma_scale = np.ones(h.tets.shape[0])
        params = dict(cv_scale=cv_scale, v_fast=v_fast)
        if label == "normal":
            pass
        elif label == "lbbb":
            fast &= ~self.lv
            stim = [(self.rv_root, 0.0)]
        elif label == "rbbb":
            fast &= ~self.rv
            stim = [(self.lv_root, 0.0)]
        elif label == "scar":
            ok = np.isfinite(self.z[self.lv_nodes]) & (self.z[self.lv_nodes] > 0.2) & (self.z[self.lv_nodes] < 0.85)
            c = h.points[rng.choice(self.lv_nodes[ok])]
            r = rng.uniform(0.015, 0.025)
            blocked = np.linalg.norm(h.points - c, axis=1) < r
            blocked &= self.lv
            sigma_scale[blocked[h.tets].any(axis=1)] = 0.0
            params.update(centre_m=c.tolist(), radius_m=r, n_nodes=int(blocked.sum()))
        else:
            raise ValueError(label)
        at = self.g.solve(vl, vt, stim, blocked_nodes=blocked, fast_nodes=fast, v_fast=v_fast)
        at = at - np.nanmin(at[np.isfinite(at)])
        return Activation(at_ms=at, sigma_scale=sigma_scale, label=label, params=params)


def make_simulator(h: HeartMesh):
    return AtrialSimulator(h) if h.kind == "atria" else VentricularSimulator(h)
