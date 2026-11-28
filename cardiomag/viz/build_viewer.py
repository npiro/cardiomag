"""Build the standalone interactive viewer (single HTML file, no external dependencies).

    cardiomag-viewer --out viewer/cardiomag_viewer.html [--quick]

Embedded per simulated source (gzip + base64, decoded in the browser):
  - display nodes (subsample) with Vm per frame (uint8),
  - current dipole moments lumped into voxels (int16 + scale) -> the browser computes B for ANY
    sensor array the user places (Biot-Savart from the voxel dipoles, normal or tangential component),
  - chest surface (anterior skin height map) for placing the array.
Measured sources (PTB 49-ch SQUID, Kiel 36-pos OPM) carry their median beats on their layouts.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path

import numpy as np

from ..io import hearts as H
from ..io import meshio
from ..io.mcg import load_kiel, load_ptb
from ..physics import leadfield as lf
from ..physics.sensors import default_centre_mm, kiel36_layout, ptb49_layout
from ..sim.eikonal import vm_from_at
from ..sim.scenarios import AtrialSimulator, VentricularSimulator
from ..sim.whole_heart import WholeHeartSimulator, load_whole_heart, vm_full
from .surface import mesh_surface, strocchi_full_surface

TEMPLATE = Path(__file__).with_name("viewer_template.html")


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(gzip.compress(np.ascontiguousarray(arr).tobytes(), 6)).decode("ascii")


def _surface_nodes(points, n, rng):
    """Random subsample of mesh nodes (whole volume). Only used for sources without a surface mesh."""
    idx = rng.choice(points.shape[0], size=min(n, points.shape[0]), replace=False)
    return idx


def _vertex_nodes(h, surf, max_mm=6.0):
    """For each surface vertex, the nearest mesh node of the same chamber (-1: none, e.g. vessels, or a
    chamber this model does not simulate). Matching by chamber keeps e.g. an atrial vertex at the AV
    junction from picking up the coincident ventricular node copy."""
    from scipy.spatial import cKDTree
    verts, tris, lab, names = surf
    nv = verts.shape[0]
    # vertex label = most frequent label of its faces
    cnt = np.zeros((nv, int(lab.max()) + 1), np.int32)
    np.add.at(cnt, (tris.ravel(), np.repeat(lab, 3)), 1)
    vlab = cnt.argmax(1)
    node_reg = np.full(h.points.shape[0], -1)
    node_reg[h.tets.ravel()] = np.repeat(h.region, 4)
    rid = {v: k for k, v in h.region_names.items()}
    out = np.full(nv, -1, np.int64)
    for L, name in names.items():
        sel = np.flatnonzero(vlab == L)
        if not sel.size or name not in rid:
            continue
        cand = np.flatnonzero(node_reg == rid[name])
        if not cand.size:
            continue
        d, j = cKDTree(h.points[cand]).query(verts[sel])
        ok = d * 1e3 <= max_mm
        out[sel[ok]] = cand[j[ok]]
    return out


def _surface_payload(verts, tris, lab, names):
    return dict(verts=_b64((verts * 1e3).astype(np.float32)), n_verts=int(verts.shape[0]),
                tris=_b64(tris.astype(np.uint32)), n_tris=int(tris.shape[0]), lab=_b64(lab.astype(np.uint8)),
                names={int(k): v for k, v in names.items()})


def _source(h, at_ms, sigma_scale, label, title, t_ms, rng, n_disp=4500, voxel=0.008, disp_every=2, rt_ms=None,
            surf=None):
    if rt_ms is not None:
        vm = np.concatenate([vm_full(at_ms, rt_ms, t_ms[i:i + 50]) for i in range(0, t_ms.size, 50)])
    else:
        vm = vm_from_at(at_ms, t_ms).astype(np.float32)        # (T, N) mV
    sig = lf.sigma_for(h, scale=sigma_scale)
    centres, m = lf.voxel_moments(h.points, h.tets, sig, vm * 1e-3, voxel=voxel)
    extra = {}
    if surf is not None:      # Vm on the anatomy surface vertices (255 = no data); no volume dots
        vn = _vertex_nodes(h, surf)
        sv = np.clip((vm[::disp_every][:, np.maximum(vn, 0)] + 90.0) / 120.0 * 254, 0, 254).astype(np.uint8)
        sv[:, vn < 0] = 255
        extra = dict(sv=_b64(sv), n_sv=int(vn.size))
        n_disp = 0
    disp = _surface_nodes(h.points, n_disp, rng)
    vq = np.clip((vm[::disp_every, disp] + 90.0) / 120.0 * 255, 0, 255).astype(np.uint8)
    ms = float(np.abs(m).max()) or 1.0
    mq = np.round(m / ms * 32767).astype(np.int16)
    P = m.sum(axis=1)
    return dict(id=f"{h.name}:{label}", title=title, kind=h.kind, group=h.group, label=label,
                t_ms=t_ms.tolist(), disp_every=disp_every,
                pts=_b64((h.points[disp] * 1e3).astype(np.float32)), n_pts=int(disp.size),
                vm=_b64(vq), vox=_b64((centres * 1e3).astype(np.float32)), n_vox=int(centres.shape[0]),
                m=_b64(mq), m_scale=ms, dipole_uAm=(np.linalg.norm(P, axis=1) * 1e6).round(4).tolist(), **extra)


def _skin(h):
    if isinstance(h.skin_fn, H.SkinMap):
        g = h.skin_fn
        grid = np.where(np.isfinite(g.grid), g.grid, np.nan) * 1e3
        return dict(type="map", x0=g.x0 * 1e3, y0=g.y0 * 1e3, step=g.step * 1e3, ny=grid.shape[0], nx=grid.shape[1],
                    grid=_b64(np.nan_to_num(grid, nan=-9999).astype(np.float32)))
    return dict(type="flat", z=h.skin_fn.z * 1e3)


def artifact_body(html: str) -> str:
    """Strip the document skeleton (claude.ai adds its own <!doctype>/<html>/<head>/<body>)."""
    import re
    html = re.sub(r"(?is)<!doctype html>\s*", "", html)
    html = re.sub(r"(?is)</?html[^>]*>|</?head>|</?body>", "", html)
    html = re.sub(r'(?i)<meta charset="utf-8">\s*|<meta name="viewport"[^>]*>\s*', "", html)
    # artifact viewers cannot download files: keep only the clipboard export
    html = re.sub(r"(?s)/\*DOWNLOAD\*/.*?/\*/DOWNLOAD\*/", "", html)
    html = html.replace('<button class="primary" id="export">Download CSV</button>', "")
    html = html.replace('<button id="copycsv">Copy CSV</button>', '<button class="primary" id="copycsv">Copy CSV</button>')
    return html.strip() + "\n"


def build(out: Path, quick=False, seed=0, artifact=False):
    rng = np.random.default_rng(seed)
    sources, skins, surfaces = [], {}, {}
    t_at = np.arange(-10, 170, 2.0)
    t_v = np.arange(-10, 200, 2.0)
    pts = ["S1"] if quick else ["S1", "S2", "S3", "S4"]
    for p in pts:
        h = H.load_biatrial(p)
        sim = AtrialSimulator(h)
        skins[h.name] = _skin(h)
        surf_raw = mesh_surface(h)
        surfaces[h.name] = _surface_payload(*surf_raw)
        ref = sim.reference()
        sources.append(dict(_source(h, ref.at_ms, ref.sigma_scale, "healthy", f"Atria {p} - healthy (reference)",
                                    t_at, rng, surf=surf_raw), skin=h.name, surface=h.name))
        if p == "S1":
            for lab, title in (("slow_patch", "fibrotic slow-conduction patch"), ("ia_block", "interatrial block"),
                               ("ectopic", "ectopic focus")):
                a = sim.sample(lab, rng)
                sources.append(dict(_source(h, a.at_ms, a.sigma_scale, lab, f"Atria {p} - {title}", t_at, rng, surf=surf_raw),
                                    skin=h.name, surface=h.name))
        print(f"  atria {p} done", flush=True)
    vent = [("healthy", "01", ["normal", "lbbb", "rbbb", "scar"])]
    if not quick:
        vent.append(("hf", "01", ["normal"]))
    for coh, cid, labels in vent:
        if not (H.paths.STROCCHI_HEALTHY.exists() if coh == "healthy" else (H.paths.STROCCHI_HF / cid).exists()):
            continue
        h = H.load_strocchi(coh, cid)
        sim = VentricularSimulator(h)
        skins[h.name] = _skin(h)
        skey = f"anatomy_{coh}_{cid}"
        surf_raw = strocchi_full_surface(h.frame)
        if skey not in surfaces:
            surfaces[skey] = _surface_payload(*surf_raw)
        nice = {"normal": "normal conduction", "lbbb": "left bundle branch block", "rbbb": "right bundle branch block",
                "scar": "LV scar"}
        for lab in labels:
            a = sim.sample(lab, rng)
            anat = "healthy anatomy" if coh == "healthy" else "heart-failure anatomy"
            sources.append(dict(_source(h, a.at_ms, a.sigma_scale, lab, f"Ventricles {cid} ({anat}) - {nice[lab]}",
                                        t_v, rng, voxel=0.009, surf=surf_raw), skin=h.name, surface=skey))
        print(f"  ventricles {coh} {cid} done", flush=True)

    # whole beats (P - QRS - T) on four-chamber anatomies
    t_w = np.arange(-50, 650, 4.0)
    whole = [("healthy", "01", ["normal", "lbbb", "av_block", "long_qt"])]
    if not quick:
        whole.append(("hf", "01", ["normal"]))
    nice_w = {"normal": "normal sinus beat", "lbbb": "left bundle branch block", "av_block": "1st-degree AV block",
              "long_qt": "long QT"}
    for coh, cid, labels in whole:
        try:
            h = load_whole_heart(coh, cid)
        except FileNotFoundError:
            continue
        sim = WholeHeartSimulator(h)
        skins[h.name] = _skin(h)
        skey = f"anatomy_{coh}_{cid}"
        surf_raw = strocchi_full_surface(h.frame)
        if skey not in surfaces:   # same mesh, same canonical frame as the ventricle-only loader
            surfaces[skey] = _surface_payload(*surf_raw)
        anat = "healthy anatomy" if coh == "healthy" else "heart-failure anatomy"
        for lab in labels:
            a = sim.sample(lab, rng)
            sources.append(dict(_source(h, a.at_ms, a.sigma_scale, lab, f"Whole heart {cid} ({anat}) - {nice_w[lab]}",
                                        t_w, rng, voxel=0.009, disp_every=1, rt_ms=a.rt_ms, surf=surf_raw),
                                skin=h.name, surface=skey))
        print(f"  whole heart {coh} {cid} done", flush=True)

    measured = []
    try:
        r = load_ptb()
        xy, names = ptb49_layout()
        measured.append(dict(id="ptb", title="Measured - PTB 49-ch SQUID (healthy volunteer)", preset="ptb49",
                             layout=xy.round(2).tolist(), names=names, beat=_b64(r["beat"][:, ::2].astype(np.float32)),
                             n_ch=len(names), t_ms=list(range(-450, 550, 2)), p_onset=r["p_onset"],
                             qrs_onset=r["qrs_onset"], standoff=45, centre=default_centre_mm("atria", "ptb49")))
    except Exception as e:  # noqa: BLE001
        print("  PTB skipped:", e)
    for s in range(1, 8):
        try:
            r = load_kiel(s)
            xy, names = kiel36_layout()
            measured.append(dict(id=f"kiel{s}", title=f"Measured - Kiel OPM subject {s} (healthy)", preset="kiel36",
                                 layout=xy.round(2).tolist(), names=names,
                                 beat=_b64(r["beat"][:, ::2].astype(np.float32)), n_ch=len(names),
                                 t_ms=list(range(-450, 550, 2)), p_onset=r["p_onset"], qrs_onset=r["qrs_onset"],
                                 standoff=20, centre=default_centre_mm("atria", "kiel36")))
        except Exception as e:  # noqa: BLE001
            print(f"  Kiel {s} skipped:", e)
    presets = dict(ptb49=ptb49_layout()[0].round(2).tolist(), kiel36=kiel36_layout()[0].round(2).tolist())
    centres = {f"{k}|{p}": default_centre_mm(k, p) for k in ("atria", "ventricles", "whole")
               for p in ("grid", "ptb49", "kiel36")}
    payload = dict(sources=sources, measured=measured, skins=skins, surfaces=surfaces, presets=presets, centres=centres,
                   k_cal=lf.K_CAL, version="0.1")
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(payload))
    if artifact:
        html = artifact_body(html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"[viewer] {out} ({out.stat().st_size / 1e6:.1f} MB, {len(sources)} simulated + {len(measured)} measured)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="viewer/cardiomag_viewer.html")
    ap.add_argument("--quick", action="store_true", help="S1 + one ventricle only (fast build)")
    ap.add_argument("--artifact", action="store_true", help="omit the document skeleton (for claude.ai Artifacts)")
    ap.add_argument("--from-html", default=None, help="re-template an existing built viewer (reuse its data)")
    a = ap.parse_args(argv)
    if a.from_html:
        import re
        old = Path(a.from_html).read_text()
        data = re.search(r"const DATA = (\{.*?\});\nconst \$ =", old, re.S).group(1)
        html = TEMPLATE.read_text().replace("/*__DATA__*/null", data)
        Path(a.out).write_text(artifact_body(html) if a.artifact else html)
        print(f"[viewer] {a.out} re-templated from {a.from_html}")
        return
    build(Path(a.out), quick=a.quick, artifact=a.artifact)


if __name__ == "__main__":
    main()
