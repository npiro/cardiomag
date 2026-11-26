"""Heart models as a common `HeartMesh`: tetrahedral mesh in metres, fibres, region labels and an
anatomical body frame (x = patient's left, y = superior, z = anterior; origin at the tissue centroid).

Sources
  biatrial  : Zenodo 17143584 (S1..S4), atria inside a torso, ECG-calibrated eikonal parameters
  strocchi  : Zenodo 4590294 (healthy 01..20) and 3890034 (heart failure 01..24), four-chamber CT
              meshes; only the ventricles (tags 1 = LV, 2 = RV) are kept.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .. import paths
from . import meshio


@dataclass
class HeartMesh:
    name: str
    kind: str                     # "atria" | "ventricles"
    group: str                    # "healthy" | "heart_failure" | "patient"
    points: np.ndarray            # (N,3) m, body frame
    tets: np.ndarray              # (E,4)
    fibres: np.ndarray            # (E,3) unit, body frame
    region: np.ndarray            # (E,) int label (see region_names)
    region_names: dict            # label -> name
    cv: dict = field(default_factory=dict)       # region name -> (v_l, v_t) mm/ms
    node_fields: dict = field(default_factory=dict)  # e.g. UVC coordinates, site masks
    sites: dict = field(default_factory=dict)        # named node-index arrays (e.g. SAN)
    skin_fn: object = None        # callable (x, y) -> anterior skin coordinate z (m)
    reference_at: np.ndarray | None = None          # provided activation times (ms), if any
    reference_vm: str | None = None                 # path to provided Vm (.igb), if any
    sigma_region_scale: dict | None = None          # per-region conductivity factor (e.g. K_VENT)
    connections: list = field(default_factory=list)  # interatrial connections (whole heart)
    frame: dict | None = None                        # body transform: body = (orig_m - centre) @ R.T

    @property
    def n_nodes(self):
        return self.points.shape[0]

    def summary(self):
        return dict(name=self.name, kind=self.kind, group=self.group, nodes=int(self.n_nodes),
                    tets=int(self.tets.shape[0]), regions={int(k): v for k, v in self.region_names.items()})


# ------------------------------------------------------------------------------------------- frames
def _orthonormal(a, b):
    a = a / np.linalg.norm(a)
    b = b - b.dot(a) * a
    b = b / np.linalg.norm(b)
    return a, b, np.cross(a, b)


def rotation_from_pairs(src_a, src_b, dst_a, dst_b):
    """Rotation R with R @ src_a ~ dst_a and R @ src_b ~ dst_b (both pairs orthonormalised)."""
    S = np.stack(_orthonormal(src_a, src_b), axis=1)
    D = np.stack(_orthonormal(dst_a, dst_b), axis=1)
    return D @ S.T


def rot_about(axis, deg):
    axis = axis / np.linalg.norm(axis)
    a = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


# ------------------------------------------------------------------------------------------- torso
def _torso_points(path):
    cache = paths.cache_dir("torso") / (path.stem + "_points.npy")
    if cache.exists():
        return np.load(cache)
    with open(path) as fh:
        lines = []
        for line in fh:
            if line.startswith("POINTS"):
                n = int(line.split()[1])
                break
        txt = " ".join(next(fh) for _ in range(n))
    pts = np.fromstring(txt, sep=" ").reshape(n, 3) * 1e-6
    np.save(cache, pts)
    return pts


class SkinMap:
    """Anterior skin height z(x, y) on a 5 mm grid (body frame, m), from torso points."""

    def __init__(self, torso_body, step=0.005):
        x, y, z = torso_body.T
        self.x0, self.y0, self.step = x.min(), y.min(), step
        nx = int((x.max() - x.min()) / step) + 1
        ny = int((y.max() - y.min()) / step) + 1
        ix = ((x - self.x0) / step).astype(int)
        iy = ((y - self.y0) / step).astype(int)
        grid = np.full((ny, nx), -np.inf)
        np.maximum.at(grid, (iy, ix), z)
        # fill holes with nearest valid neighbour (simple dilation passes)
        for _ in range(6):
            bad = ~np.isfinite(grid)
            if not bad.any():
                break
            pad = np.pad(grid, 1, constant_values=-np.inf)
            neigh = np.max(np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]]), axis=0)
            grid[bad] = neigh[bad]
        self.grid = grid

    def __call__(self, x, y):
        ix = np.clip(((np.asarray(x) - self.x0) / self.step).astype(int), 0, self.grid.shape[1] - 1)
        iy = np.clip(((np.asarray(y) - self.y0) / self.step).astype(int), 0, self.grid.shape[0] - 1)
        return self.grid[iy, ix]

    def to_dict(self):
        g = np.where(np.isfinite(self.grid), self.grid, np.nan)
        return dict(x0=float(self.x0), y0=float(self.y0), step=float(self.step), grid=g.tolist())


class HalfSpaceSkin:
    """Flat chest (half-space boundary) at constant anterior coordinate z_skin."""

    def __init__(self, z_skin):
        self.z = float(z_skin)

    def __call__(self, x, y):
        return np.full(np.shape(x), self.z)


# ------------------------------------------------------------------------------------------- biatrial
BIATRIAL_PATIENTS = ["S1", "S2", "S3", "S4"]


def load_biatrial(patient: str = "S1", with_torso: bool = True) -> HeartMesh:
    base = paths.BIATRIAL / patient
    cache = paths.cache_dir("biatrial") / f"{patient}.npz"
    if cache.exists():
        d = np.load(cache)
        pts, tets, tags, fib = d["pts"], d["tets"], d["tags"], d["fib"]
    else:
        pts = meshio.read_pts(str(base / "mshdata" / f"{patient}_atria.pts")) * 1e-6
        tets, tags = meshio.read_elem(str(base / "mshdata" / f"{patient}_atria.elem"))
        fib = meshio.read_lon(str(base / "mshdata" / f"{patient}_atria.lon"))
        np.savez(cache, pts=pts, tets=tets, tags=tags, fib=fib)
    regions = meshio.read_par_regions(str(base / "simdata" / f"{patient}_parameters.par"))
    region = np.full(tags.shape, -1, dtype=np.int64)
    names, cv = {}, {}
    for idx, r in regions.items():
        region[np.isin(tags, r["ids"])] = idx
        names[idx] = r["name"]
        cv[r["name"]] = (r["vl"] / 1000.0, r["vt"] / 1000.0)  # mm/ms
    if (region < 0).any():  # unmapped tags -> bulk atrial tissue of the nearest kind
        la = [k for k, v in names.items() if v == "LA"][0]
        region[region < 0] = la

    # body frame from torso PCA + ECG electrodes (see Analysis/cardiac_bfield.py for the derivation)
    torso = _torso_points(base / "mshdata" / f"{patient}_atria_torso.vtk") if with_torso else None
    el = json.load(open(paths.BIATRIAL / "electrode_nodes.json"))
    if torso is not None:
        E = {k: torso[v] for k, v in el.items()}
        tc = torso.mean(axis=0)
        w, V = np.linalg.eigh(np.cov((torso[::20] - tc).T))
        V = V[:, ::-1]
        s_ax, l_ax, a_ax = V[:, 0], V[:, 1], V[:, 2]
        if (0.5 * (E["R"] + E["L"]) - E["F"]).dot(s_ax) < 0:
            s_ax = -s_ax
        if (E["L"] - E["R"]).dot(l_ax) < 0:
            l_ax = -l_ax
        if (0.5 * (E["V1"] + E["V2"]) - tc).dot(a_ax) < 0:
            a_ax = -a_ax
        Rm = np.vstack([l_ax, s_ax, a_ax])
    else:
        Rm = np.eye(3)
    p0 = pts[tets[:, 0]]
    vol = np.abs(np.linalg.det(np.stack([pts[tets[:, i]] - p0 for i in (1, 2, 3)], axis=2))) / 6
    centre = (pts[tets].mean(axis=1) * vol[:, None]).sum(0) / vol.sum()
    to_body = lambda X: (X - centre) @ Rm.T
    body = to_body(pts)
    skin = SkinMap(to_body(torso)) if torso is not None else HalfSpaceSkin(body[:, 2].max() + 0.03)

    sites = {}
    for f in (base / "activation_sites").glob("*_i.vtx"):
        sites[f.stem.replace("_i", "")] = meshio.read_vtx(str(f))
    lat = np.loadtxt(base / "simdata" / "vm_activation_seq.dat")
    lat[lat > 1e6] = np.nan
    return HeartMesh(name=f"biatrial_{patient}", kind="atria", group="patient", points=body, tets=tets,
                     fibres=fib @ Rm.T, region=region, region_names=names, cv=cv, sites=sites,
                     skin_fn=skin, reference_at=lat, reference_vm=str(base / "simdata" / "vm.igb"))


# ------------------------------------------------------------------------------------------- strocchi
STROCCHI_HEALTHY_IDS = [f"{i:02d}" for i in range(1, 21)]
STROCCHI_HF_IDS = [f"{i:02d}" for i in range(1, 25)]

# canonical heart orientation in the body frame (left, superior, anterior): long axis (base->apex)
# points left, inferior and anterior (~45 deg from the sagittal plane, ~30 deg below horizontal);
# the RV lies anterior-right of the LV.
APEX_DIR = np.array([0.61, -0.50, 0.61])
RV_DIR = np.array([-0.45, 0.05, 0.89])


def _strocchi_raw(cohort: str, cid: str):
    cache = paths.cache_dir("strocchi") / f"{cohort}_{cid}.npz"
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    if cohort == "healthy":
        m = meshio.read_vtk_ascii(str(paths.STROCCHI_HEALTHY / f"{cid}.vtk"))
        pts, tets = m["points"], m["cells"]
        tags = m["cell_data"]["ID"].astype(np.int64)
        fib = m["cell_data"]["fibres"]
        pd = m["point_data"]
        rho, z, phi, v = pd["RHO.dat"], pd["Z.dat"], pd["PHI.dat"], pd["V.dat"]
    else:
        d = paths.STROCCHI_HF / cid
        g = meshio.read_ensight_geo(str(d / f"{cid}.geo"))
        pts, tets = g["points"], g["cells"]["tetra4"]
        ne, nn = tets.shape[0], pts.shape[0]
        tags = meshio.read_ensight_var(str(d / f"{cid}.tags.ens"), ne).astype(np.int64)
        fib = meshio.read_ensight_var(str(d / f"{cid}.fiber.ens"), ne, 3)
        rho = meshio.read_ensight_var(str(d / f"{cid}_uvc_transmural.ens"), nn)
        z = meshio.read_ensight_var(str(d / f"{cid}_uvc_longitudinal.ens"), nn)
        phi = meshio.read_ensight_var(str(d / f"{cid}_uvc_rotational.ens"), nn)
        v = meshio.read_ensight_var(str(d / f"{cid}_uvc_intraventricular.ens"), nn)
    keep = np.isin(tags, [1, 2])
    tets, tags, fib = tets[keep], tags[keep], fib[keep]
    used = np.unique(tets)
    remap = np.full(pts.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    out = dict(pts=pts[used] * 1e-3, tets=remap[tets], tags=tags, fib=fib,
               rho=rho[used], z=z[used], phi=phi[used], v=v[used])
    np.savez(cache, **out)
    return out


def load_strocchi(cohort: str = "healthy", cid: str = "01", rot_jitter_deg: float = 0.0,
                  rng: np.random.Generator | None = None, chest_wall_m: float = 0.02) -> HeartMesh:
    """Ventricles of a Strocchi mesh in a canonical body orientation; the chest is a flat half-space
    `chest_wall_m` in front of the most anterior ventricular point (no torso in this dataset)."""
    d = _strocchi_raw(cohort, cid)
    pts, tets, tags, fib = d["pts"], d["tets"], d["tags"], d["fib"]
    lv = tags == 1
    lv_nodes = np.unique(tets[lv])
    zc = d["z"]
    apex = pts[lv_nodes[zc[lv_nodes] < np.nanpercentile(zc[lv_nodes], 3)]].mean(axis=0)
    base = pts[lv_nodes[zc[lv_nodes] > np.nanpercentile(zc[lv_nodes], 97)]].mean(axis=0)
    rv_c = pts[np.unique(tets[tags == 2])].mean(axis=0)
    lv_c = pts[lv_nodes].mean(axis=0)
    R = rotation_from_pairs(apex - base, rv_c - lv_c, APEX_DIR, RV_DIR)
    if rot_jitter_deg and rng is not None:
        for ax in np.eye(3):
            R = rot_about(ax, rng.normal(0, rot_jitter_deg)) @ R
    centre = pts.mean(axis=0)
    body = (pts - centre) @ R.T
    region = np.where(tags == 1, 0, 1)
    frame = dict(R=R, centre=centre, cohort=cohort, cid=cid)
    nf = dict(rho=d["rho"], z=d["z"], phi=d["phi"], v=d["v"])
    group = "healthy" if cohort == "healthy" else "heart_failure"
    return HeartMesh(name=f"strocchi_{cohort}_{cid}", kind="ventricles", group=group, points=body,
                     tets=tets, fibres=fib @ R.T, region=region, region_names={0: "LV", 1: "RV"},
                     cv={"LV": (0.5, 0.2), "RV": (0.5, 0.2)}, node_fields=nf,
                     sigma_region_scale={"LV": _kvent(), "RV": _kvent()}, frame=frame,
                     skin_fn=HalfSpaceSkin(body[:, 2].max() + chest_wall_m))


def _kvent():
    from ..physics.leadfield import K_VENT
    return K_VENT


def available_hearts() -> dict:
    out = {"biatrial": [p for p in BIATRIAL_PATIENTS if (paths.BIATRIAL / p).exists()],
           "strocchi_healthy": [c for c in STROCCHI_HEALTHY_IDS
                                if (paths.STROCCHI_HEALTHY / f"{c}.vtk").exists()],
           "strocchi_hf": [c for c in STROCCHI_HF_IDS if (paths.STROCCHI_HF / c / f"{c}.geo").exists()]}
    return out
