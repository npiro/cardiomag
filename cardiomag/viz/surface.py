"""Heart surfaces for display: boundary triangles of a tetrahedral mesh, simplified by vertex clustering.

    verts (V,3) m, tris (F,3), tri_label (F,)   -- label = region/tag of the tetrahedron the face came from
"""
from __future__ import annotations

import numpy as np

from .. import paths
from ..io import meshio

STROCCHI_LABELS = {1: "LV", 2: "RV", 3: "LA", 4: "RA", 5: "aorta", 6: "pulmonary artery"}


def boundary_faces(tets, labels, points=None):
    """Faces used by exactly one tetrahedron, oriented outwards when points are given."""
    F = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    opp = np.concatenate([tets[:, 3], tets[:, 2], tets[:, 1], tets[:, 0]])
    lab = np.tile(labels, 4)
    key = np.sort(F, axis=1)
    _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    keep = idx[cnt == 1]
    F, opp, lab = F[keep], opp[keep], lab[keep]
    if points is not None:
        n = np.cross(points[F[:, 1]] - points[F[:, 0]], points[F[:, 2]] - points[F[:, 0]])
        inward = ((points[opp] - points[F[:, 0]]) * n).sum(1) > 0
        F[inward] = F[inward][:, [0, 2, 1]]
    return F, lab


def simplify(points, faces, labels, cell=0.004):
    """Vertex clustering on a `cell` grid (m): merges vertices, drops degenerate / duplicate faces."""
    key = np.floor(points / cell).astype(np.int64)
    used = np.unique(faces)
    _, cid = np.unique(key[used], axis=0, return_inverse=True)
    cmap = np.full(points.shape[0], -1, np.int64)
    cmap[used] = cid.ravel()
    nc = cid.max() + 1
    verts = np.stack([np.bincount(cid.ravel(), weights=points[used][:, k], minlength=nc) for k in range(3)], 1)
    verts /= np.bincount(cid.ravel(), minlength=nc)[:, None]
    f = cmap[faces]
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f, labels = f[ok], labels[ok]
    _, u = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    return verts, f[u], labels[u]


def strocchi_full_surface(frame, cell=0.004):
    """Four-chamber anatomy (tags 1-6: ventricles, atria, aorta, pulmonary artery) in a HeartMesh body frame."""
    cohort, cid = frame["cohort"], frame["cid"]
    cache = paths.cache_dir("surfaces") / f"strocchi_{cohort}_{cid}_{int(cell * 1e4)}.npz"
    if cache.exists():
        d = np.load(cache)
        verts, tris, lab = d["verts"], d["tris"], d["lab"]
    else:
        if cohort == "healthy":
            m = meshio.read_vtk_ascii(str(paths.STROCCHI_HEALTHY / f"{cid}.vtk"))
            pts, tets, tags = m["points"], m["cells"], m["cell_data"]["ID"].astype(np.int64)
        else:
            g = meshio.read_ensight_geo(str(paths.STROCCHI_HF / cid / f"{cid}.geo"))
            pts, tets = g["points"], g["cells"]["tetra4"]
            tags = meshio.read_ensight_var(str(paths.STROCCHI_HF / cid / f"{cid}.tags.ens"), tets.shape[0]).astype(np.int64)
        keep = np.isin(tags, list(STROCCHI_LABELS))
        tets, tags = tets[keep], tags[keep]
        pts = pts * 1e-3
        F, lab = boundary_faces(tets, tags, pts)
        verts, tris, lab = simplify(pts, F, lab, cell)
        np.savez(cache, verts=verts, tris=tris, lab=lab)
    body = (verts - frame["centre"]) @ frame["R"].T
    return body, tris, lab, {k: v for k, v in STROCCHI_LABELS.items()}


def mesh_surface(h, cell=0.004):
    """Surface of the simulated mesh itself (e.g. biatrial atria), labelled by region."""
    F, lab = boundary_faces(h.tets, h.region, h.points)
    verts, tris, lab = simplify(h.points, F, lab, cell)
    return verts, tris, lab, dict(h.region_names)
