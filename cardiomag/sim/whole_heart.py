"""Whole-beat simulation (P wave, PR interval, QRS, ST-T) on the Strocchi four-chamber meshes.

Anatomy (Strocchi labels): 1 LV, 2 RV, 3 LA, 4 RA; landmarks from the inlet/valve tags:
16/23 superior vena cava (SVC), 17/24 inferior vena cava (IVC), 7 mitral, 8 tricuspid valve plane.
Electrical insulation: the mesh shares nodes between atria and ventricles and between LA and RA; nodes
are duplicated per compartment (ventricles, LA, RA), so the only couplings are the ones modelled:
  - sinus node (SAN): RA tissue at the lateral SVC junction, t = 0
  - interatrial: Bachmann's bundle (superior septal interface), fossa ovalis (mid), coronary sinus
    (inferior) - each an (RA nodes, LA nodes, delay) connection; activation can cross both ways
  - AV node: RA tissue at the septal tricuspid ring; His/ventricular septal roots fire AV-delay later
  - ventricles: myocardium + thin fast endocardial layer (as cardiomag.sim.scenarios)
Repolarisation: RT = AT + APD with ventricular APD longer at the apex (-40 ms apex->base) and
slightly longer at the endocardium (-20 ms endo->epi), shortened for late activation - this gives the
concordant T wave seen in measured MCG (R- and T-peak maps correlate +0.65..0.70 here, PTB +0.88).
Atrial APD ~ 200 ms (atrial Ta wave during the QRS). Phase-3 width tau_rep = 30 ms.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .. import paths
from ..io import hearts as H
from ..io import meshio
from .eikonal import EdgeGraph
from .scenarios import Activation

WHOLE_CLASSES = ["normal", "av_block", "lbbb", "rbbb", "scar", "ia_block", "ectopic_atrial", "long_qt"]
LV, RV, LA, RA = 0, 1, 2, 3


# ------------------------------------------------------------------------------------------- mesh
def _raw_whole(cohort, cid):
    cache = paths.cache_dir("strocchi") / f"whole_{cohort}_{cid}.npz"
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    if cohort == "healthy":
        m = meshio.read_vtk_ascii(str(paths.STROCCHI_HEALTHY / f"{cid}.vtk"))
        pts, tets, tags, fib = m["points"], m["cells"], m["cell_data"]["ID"].astype(np.int64), m["cell_data"]["fibres"]
        pd = m["point_data"]
        rho, z = pd["RHO.dat"], pd["Z.dat"]
    else:
        d = paths.STROCCHI_HF / cid
        g = meshio.read_ensight_geo(str(d / f"{cid}.geo"))
        pts, tets = g["points"], g["cells"]["tetra4"]
        ne, nn = tets.shape[0], pts.shape[0]
        tags = meshio.read_ensight_var(str(d / f"{cid}.tags.ens"), ne).astype(np.int64)
        fib = meshio.read_ensight_var(str(d / f"{cid}.fiber.ens"), ne, 3)
        rho = meshio.read_ensight_var(str(d / f"{cid}_uvc_transmural.ens"), nn)
        z = meshio.read_ensight_var(str(d / f"{cid}_uvc_longitudinal.ens"), nn)
    land = {name: np.unique(tets[np.isin(tags, t)]) for name, t in
            (("svc", [16, 23]), ("ivc", [17, 24]), ("tv", [8]), ("mv", [7]))}
    # duplicate nodes per compartment: ventricles (1,2), LA (3), RA (4)
    groups = [(np.isin(tags, [1, 2]), "V"), (tags == 3, "LA"), (tags == 4, "RA")]
    P, T, TAG, FIB, RHO, Z, ORIG, GRP = [], [], [], [], [], [], [], []
    off = 0
    for gi, (mask, _) in enumerate(groups):
        tt = tets[mask]
        used = np.unique(tt)
        remap = np.full(pts.shape[0], -1, np.int64)
        remap[used] = np.arange(used.size) + off
        P.append(pts[used]); T.append(remap[tt]); TAG.append(tags[mask]); FIB.append(fib[mask])
        RHO.append(rho[used] if gi == 0 else np.full(used.size, -10.0))
        Z.append(z[used] if gi == 0 else np.full(used.size, -10.0))
        ORIG.append(used); GRP.append(np.full(used.size, gi))
        off += used.size
    out = dict(pts=np.concatenate(P) * 1e-3, tets=np.concatenate(T), tags=np.concatenate(TAG),
               fib=np.concatenate(FIB), rho=np.concatenate(RHO), z=np.concatenate(Z),
               orig=np.concatenate(ORIG), grp=np.concatenate(GRP))
    for k, v in land.items():
        out["land_" + k] = v
    np.savez(cache, **out)
    return out


def load_whole_heart(cohort="healthy", cid="01", chest_wall_m=0.02) -> H.HeartMesh:
    d = _raw_whole(cohort, cid)
    pts, tets, tags, fib, grp, orig = d["pts"], d["tets"], d["tags"], d["fib"], d["grp"], d["orig"]
    # canonical body orientation from the ventricles (same as load_strocchi)
    lvn = np.unique(tets[tags == 1])
    z = d["z"]
    apex = pts[lvn[z[lvn] < np.nanpercentile(z[lvn], 3)]].mean(0)
    base = pts[lvn[z[lvn] > np.nanpercentile(z[lvn], 97)]].mean(0)
    rv_c, lv_c = pts[np.unique(tets[tags == 2])].mean(0), pts[lvn].mean(0)
    R = H.rotation_from_pairs(apex - base, rv_c - lv_c, H.APEX_DIR, H.RV_DIR)
    vent_c = pts[np.unique(tets[np.isin(tags, [1, 2])])].mean(0)
    body = (pts - vent_c) @ R.T                      # origin at the ventricular centroid
    region = np.select([tags == 1, tags == 2, tags == 3], [LV, RV, LA], RA)
    h = H.HeartMesh(name=f"whole_{cohort}_{cid}", kind="whole", group="healthy" if cohort == "healthy" else "heart_failure",
                    points=body, tets=tets, fibres=fib @ R.T, region=region,
                    region_names={LV: "LV", RV: "RV", LA: "LA", RA: "RA"},
                    cv={"LV": (0.5, 0.2), "RV": (0.5, 0.2), "LA": (1.1, 0.6), "RA": (1.1, 0.6)},  # atrial bulk CV incl. fast tracts
                    node_fields=dict(rho=d["rho"], z=d["z"], grp=grp),
                    skin_fn=H.HalfSpaceSkin(body[:, 2].max() + chest_wall_m),
                    sigma_region_scale={"LV": H._kvent(), "RV": H._kvent()},
                    frame=dict(R=R, centre=vent_c, cohort=cohort, cid=cid))
    # landmarks: map original node ids -> duplicated ids per group
    def nodes_of(orig_ids, g):
        sel = np.flatnonzero((grp == g) & np.isin(orig, orig_ids))
        return sel
    ra_nodes = np.flatnonzero(grp == 2)
    la_nodes = np.flatnonzero(grp == 1)
    svc_ra = nodes_of(d["land_svc"], 2)
    tv_ra = nodes_of(d["land_tv"], 2)
    # SAN: lateral (patient-right) part of the RA-SVC junction
    if svc_ra.size:
        lat = body[svc_ra, 0]
        san_c = body[svc_ra[lat <= np.percentile(lat, 25)]].mean(0)
    else:
        san_c = body[ra_nodes[np.argmax(body[ra_nodes, 1])]]
    san = ra_nodes[np.linalg.norm(body[ra_nodes] - san_c, axis=1) < 0.004]
    # AV node: RA tissue on the tricuspid ring closest to the ventricular septum
    vent = np.flatnonzero(grp == 0)
    sept_c = body[vent][np.argsort(np.abs(body[vent, 0] - np.median(body[vent, 0])))[:2000]].mean(0)
    ring = tv_ra if tv_ra.size else ra_nodes
    avc = body[ring[np.argmin(np.linalg.norm(body[ring] - sept_c, axis=1))]]
    avn = ra_nodes[np.linalg.norm(body[ra_nodes] - avc, axis=1) < 0.004]
    # interatrial interface: LA/RA node pairs that were the same original node
    shared = np.intersect1d(orig[la_nodes], orig[ra_nodes])
    conns = []
    if shared.size:
        la_s = la_nodes[np.isin(orig[la_nodes], shared)]
        ra_s = ra_nodes[np.isin(orig[ra_nodes], shared)]
        y = body[la_s, 1]
        for name, lo, hi, delay in (("BB", 75, 100, 4.0), ("FO", 40, 60, 12.0), ("CS", 0, 20, 10.0)):
            sel = (y >= np.percentile(y, lo)) & (y <= np.percentile(y, hi))
            c = body[la_s[sel]].mean(0)
            la_c = la_s[np.linalg.norm(body[la_s] - c, axis=1) < 0.005]
            ra_c = ra_s[np.linalg.norm(body[ra_s] - c, axis=1) < 0.005]
            if la_c.size and ra_c.size:
                conns.append(dict(name=name, la=la_c, ra=ra_c, delay=delay))
    h.sites = dict(SAN=san, AVN=avn)
    h.connections = conns
    return h


# ------------------------------------------------------------------------------------------- simulator
class WholeHeartSimulator:
    def __init__(self, h: H.HeartMesh):
        assert h.kind == "whole"
        self.h = h
        self.g = EdgeGraph(h.points, h.tets, h.fibres)
        grp = h.node_fields["grp"]
        self.vent = grp == 0
        self.la_nodes, self.ra_nodes = np.flatnonzero(grp == 1), np.flatnonzero(grp == 2)
        rho, z = h.node_fields["rho"], h.node_fields["z"]
        valid = self.vent & (rho >= 0) & (rho <= 1)
        self.rho = np.where(valid, rho, np.nan)
        self.z = np.where(valid & (z >= 0) & (z <= 1), z, np.nan)
        lv_n = np.unique(h.tets[h.region == LV])
        rv_n = np.unique(h.tets[h.region == RV])
        self.lv = np.zeros(h.n_nodes, bool); self.lv[lv_n] = True
        self.rv = np.zeros(h.n_nodes, bool); self.rv[rv_n] = True
        self.rv &= ~self.lv
        self.endo = valid & (rho <= 0.05)
        lv_endo, rv_endo = np.flatnonzero(self.endo & self.lv), np.flatnonzero(self.endo & self.rv)
        d_lv, _ = cKDTree(h.points[rv_endo]).query(h.points[lv_endo])
        d_rv, _ = cKDTree(h.points[lv_endo]).query(h.points[rv_endo])
        mid_lv = (self.z[lv_endo] > 0.35) & (self.z[lv_endo] < 0.65)
        mid_rv = (self.z[rv_endo] > 0.35) & (self.z[rv_endo] < 0.65)
        self.lv_root = lv_endo[mid_lv & (d_lv < np.percentile(d_lv[mid_lv], 15))]
        self.rv_root = rv_endo[mid_rv & (d_rv < np.percentile(d_rv[mid_rv], 15))]
        self.lv_nodes = lv_n
        names = h.region_names
        self.vl0 = np.array([h.cv[names[r]][0] for r in h.region])
        self.vt0 = np.array([h.cv[names[r]][1] for r in h.region])

    def _atria(self, vl, vt, primary, conns):
        at = self.g.solve(vl, vt, primary)
        for _ in range(3):
            extra = []
            for c in conns:
                tr, tl = np.min(at[c["ra"]]), np.min(at[c["la"]])
                if np.isfinite(tr):
                    extra.append((c["la"], tr + c["delay"]))
                if np.isfinite(tl):
                    extra.append((c["ra"], tl + c["delay"]))
            new = self.g.solve(vl, vt, primary + extra)
            done = np.allclose(np.nan_to_num(new, posinf=-1), np.nan_to_num(at, posinf=-1), atol=0.05)
            at = new
            if done:
                break
        return at

    def sample(self, label: str, rng: np.random.Generator) -> Activation:
        h = self.h
        cv_scale = rng.uniform(0.9, 1.1)
        vl, vt = self.vl0 * cv_scale, self.vt0 * cv_scale
        conns = [dict(c, delay=c["delay"] * rng.uniform(0.8, 1.2)) for c in h.connections]
        av_delay = rng.uniform(50, 90)
        v_fast = rng.uniform(1.0, 1.5)
        fast = self.endo.copy()
        blocked = np.zeros(h.n_nodes, bool)
        sigma_scale = np.ones(h.tets.shape[0])
        apd_v0, apd_a0 = rng.uniform(260, 300), rng.uniform(180, 220)
        primary = [(h.sites["SAN"], 0.0)]
        v_roots = "both"
        params = dict(cv_scale=cv_scale, av_delay=av_delay, apd_v=apd_v0)
        if label == "av_block":
            av_delay = rng.uniform(160, 240)
            params["av_delay"] = av_delay
        elif label == "ia_block":
            conns = [c for c in conns if c["name"] != "BB"]
        elif label == "ectopic_atrial":
            pool = self.la_nodes if rng.random() < 0.6 else self.ra_nodes
            f = rng.choice(pool)
            primary = [(np.flatnonzero(np.linalg.norm(h.points - h.points[f], axis=1) < 0.003), 0.0)]
            params["focus_side"] = "LA" if f in set(self.la_nodes[:1]) or np.isin(f, self.la_nodes) else "RA"
        elif label == "lbbb":
            fast &= ~self.lv
            v_roots = "rv"
        elif label == "rbbb":
            fast &= ~self.rv
            v_roots = "lv"
        elif label == "scar":
            ok = np.isfinite(self.z[self.lv_nodes]) & (self.z[self.lv_nodes] > 0.2) & (self.z[self.lv_nodes] < 0.85)
            c = h.points[rng.choice(self.lv_nodes[ok])]
            blocked = (np.linalg.norm(h.points - c, axis=1) < rng.uniform(0.015, 0.025)) & self.lv
            sigma_scale[blocked[h.tets].any(axis=1)] = 0.0
        elif label == "long_qt":
            apd_v0 += rng.uniform(70, 120)
            params["apd_v"] = apd_v0
        elif label != "normal":
            raise ValueError(label)
        at_a = self._atria(vl, vt, primary, conns)
        t_his = float(np.min(at_a[h.sites["AVN"]])) + av_delay
        roots = {"both": [(self.lv_root, t_his + rng.uniform(0, 3)), (self.rv_root, t_his + rng.uniform(3, 8))],
                 "rv": [(self.rv_root, t_his)], "lv": [(self.lv_root, t_his)]}[v_roots]
        at_v = self.g.solve(vl, vt, roots, blocked_nodes=blocked, fast_nodes=fast, v_fast=v_fast)
        at = np.where(self.vent, at_v, at_a)
        # repolarisation: APD longer at the apex (z -> 0) and endocardium (rho -> 0)
        rho = np.nan_to_num(self.rho, nan=0.5)
        zz = np.nan_to_num(self.z, nan=0.5)
        apd = np.where(self.vent, apd_v0 - 20.0 * rho - 40.0 * (zz - 0.5) - 0.3 * (at - np.nanmin(at[self.vent])),
                       apd_a0)
        params.update(pr_ms=t_his, qrs_start=t_his)
        act = Activation(at_ms=at, sigma_scale=sigma_scale, label=label, params=params)
        act.rt_ms = at + apd
        return act


def vm_full(at_ms, rt_ms, t_ms, v_rest=-85.0, v_peak=20.0, tau_up=1.0, tau_rep=30.0):
    """Template AP with depolarisation at AT and repolarisation at RT: (T, N) mV."""
    t = t_ms[:, None].astype(np.float32)
    up = 0.5 * (1 + np.tanh((t - at_ms[None, :]) / tau_up))
    down = 0.5 * (1 + np.tanh((t - rt_ms[None, :]) / tau_rep))
    frac = np.clip((t - at_ms[None, :]) / np.maximum(rt_ms - at_ms, 1)[None, :], 0, 1)
    plateau = 1.0 - 0.15 * frac                       # slow phase-2 decline
    v = v_rest + (v_peak - v_rest) * up * (1 - down) * plateau
    v[:, ~np.isfinite(at_ms)] = v_rest
    return v.astype(np.float32)


def field_series(W, at_ms, rt_ms, t_ms, chunk=60):
    """B (S, T) [same units as W * V] computed in time chunks (full-beat Vm does not fit in memory)."""
    out = np.zeros((W.shape[0], t_ms.size), np.float64)
    for i in range(0, t_ms.size, chunk):
        vm = vm_full(at_ms, rt_ms, t_ms[i:i + chunk]) * 1e-3
        out[:, i:i + chunk] = W @ vm.T
    return out
