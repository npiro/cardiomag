"""Promotional mock-up video: heart current flow, magnetic field on a sensor array, and the current
reconstructed from the array (minimum-norm inverse), for a healthy beat and a left bundle branch block.

    python scripts/promo_video.py compute      # simulate + forward + inverse -> .cache/promo.npz
    python scripts/promo_video.py render       # frames + ffmpeg -> ../promo/cardiomag_promo.mp4

All signals are simulated (whole-heart Strocchi mesh 01). The detector badge is illustrative.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / ".cache" / "promo.npz"
OUT = ROOT.parent / "promo" / "cardiomag_promo.mp4"
MU0_4PI = 1e-7
CASES = [("normal", "Healthy heart"), ("lbbb", "Left bundle branch block")]


def compute():
    from cardiomag.sim import whole_heart as wh
    from cardiomag.physics import leadfield as lf
    from cardiomag.physics.sensors import SensorArray
    from cardiomag.viz.surface import strocchi_full_surface

    h = wh.load_whole_heart("healthy", "01")
    sim = wh.WholeHeartSimulator(h)
    sig = lf.sigma_for(h)
    arr = SensorArray(preset="grid", nx=8, ny=8, spacing_mm=25.0, centre_mm=(0.0, 0.0), standoff_mm=10.0)
    pos, nrm = arr.place(h.skin_fn)
    verts, tris, lab, _ = strocchi_full_surface(h.frame, cell=0.006)

    # dense display plane at the sensor height
    g = np.linspace(-0.11, 0.11, 56)
    GX, GY = np.meshgrid(g, g)
    plane = np.column_stack([GX.ravel(), GY.ravel(), np.full(GX.size, pos[0, 2])])

    def bz(points, m):           # points (P,3), sources c (V,3), m (T,V,3) -> (T,P) normal field
        R = points[:, None, :] - c[None]
        L = MU0_4PI * np.cross(R, np.array([0, 0, 1.0])) / (np.linalg.norm(R, axis=2) ** 3)[..., None]
        return np.einsum("pvi,tvi->tp", L, m)

    out = dict(verts=verts, tris=tris, lab=lab, sensors=pos, grid=g)
    for key, _ in CASES:
        act = sim.sample(key, np.random.default_rng(11))
        t0 = float(np.nanmin(act.at_ms))
        t = np.arange(t0 - 20, t0 + 560, 2.0)
        ms = []
        for i in range(0, t.size, 20):
            vm = wh.vm_full(act.at_ms, act.rt_ms, t[i:i + 20]) * 1e-3
            c, m = lf.voxel_moments(h.points, h.tets, sig, vm, voxel=0.008)
            ms.append(m.astype(np.float32))
            print(key, i, "/", t.size, flush=True)
        m = np.concatenate(ms)
        out[f"{key}_t"] = t - t0
        out[f"{key}_m"] = m
        out[f"{key}_B"] = bz(pos, m)
        out[f"{key}_Bmap"] = bz(plane, m).reshape(len(t), g.size, g.size)
    out["c"] = c

    # inverse: depth-weighted minimum-norm estimate on a coarse source grid from noisy sensor data
    key3 = np.floor(c / 0.014).astype(int)
    _, sid = np.unique(key3, axis=0, return_inverse=True)
    sid = sid.ravel()
    ns = sid.max() + 1
    src = np.stack([np.bincount(sid, weights=c[:, k], minlength=ns) for k in range(3)], 1) / np.bincount(sid)[:, None]
    R = pos[:, None, :] - src[None]
    L = MU0_4PI * np.cross(R, nrm[:, None, :]) / (np.linalg.norm(R, axis=2) ** 3)[..., None]   # (S,K,3)
    S = L.shape[0]
    A = L.reshape(S, -1)
    w = np.repeat(np.linalg.norm(L, axis=(0, 2)) ** -0.8, 3)
    Aw = A * w
    G = Aw @ Aw.T
    lam = 0.02 * np.trace(G) / S
    K = (Aw.T @ np.linalg.inv(G + lam * np.eye(S))) * w[:, None]
    rng = np.random.default_rng(0)
    for key, _ in CASES:
        B = out[f"{key}_B"]
        Bn = B + rng.normal(0, 0.01 * np.abs(B).max(), B.shape)
        out[f"{key}_Bn"] = Bn
        out[f"{key}_rec"] = (Bn @ K.T).reshape(len(B), ns, 3).astype(np.float32)
    out["src"] = src
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, **out)
    print("saved", CACHE)


# ----------------------------------------------------------------------------------------- render
BG, FG, DIM = "#0b0f17", "#e8edf5", "#7d8798"
ACC = {"normal": "#38d39f", "lbbb": "#ff8a3d"}
CH = {1: "#c95b6a", 2: "#5b7fc9", 3: "#c9846a", 4: "#7a6ac9", 5: "#b8b8b8", 6: "#8fb3c9"}


def render(fps=30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import FancyBboxPatch

    d = dict(np.load(CACHE))
    verts, tris, lab, c, src, g = d["verts"], d["tris"], d["lab"], d["c"], d["src"], d["grid"]
    sens = d["sensors"]
    T = min(d[f"{k}_t"].size for k, _ in CASES)
    t = d["normal_t"][:T]
    mmax = max(np.linalg.norm(d[f"{k}_m"], axis=2).max() for k, _ in CASES)
    bmax = max(np.percentile(np.abs(d[f"{k}_Bmap"]), 99.5) for k, _ in CASES)
    cm_b = matplotlib.colors.LinearSegmentedColormap.from_list(
        "bfield", ["#7fd6ff", "#1f6fd1", "#0f1626", "#d1314f", "#ffb36b"])
    gam = lambda x: np.sign(x) * np.abs(x) ** 0.5            # dynamic-range compression (P and T visible)
    rmax = max(np.linalg.norm(d[f"{k}_rec"], axis=2).max() for k, _ in CASES)
    rms = {k: np.sqrt((d[f"{k}_Bn"][:T] ** 2).mean(1)) * 1e12 for k, _ in CASES}
    qrs_end = {}
    for k, _ in CASES:
        r = rms[k]; iq = np.argmax(np.where(t < 300, r, 0))
        qrs_end[k] = t[iq + np.argmax(r[iq:] < 0.25 * r[iq])]
    fcol = np.array([matplotlib.colors.to_rgba(CH.get(int(l), "#999999"), 0.07) for l in lab])
    fcol_dim = fcol.copy(); fcol_dim[:, 3] = 0.025
    tmp = Path(OUT.parent / "_frames"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.edgecolor": DIM})

    def project(p, az):
        ca, sa = np.cos(az), np.sin(az)
        x = p[:, 0] * ca - p[:, 2] * sa
        z = p[:, 0] * sa + p[:, 2] * ca
        return np.column_stack([x, p[:, 1]]), z

    def heart(ax, az, dim=False):
        P, z = project(verts, az)
        tz = z[tris].mean(1); o = np.argsort(tz)
        ax.add_collection(PolyCollection(P[tris[o]], facecolors=(fcol_dim if dim else fcol)[o], edgecolors="none"))
        return P

    def axstyle(ax, lim=0.095):
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.axis("off")

    frame_id = [0]

    def save(fig):
        fig.savefig(tmp / f"f{frame_id[0]:05d}.png", dpi=100, facecolor=BG)
        plt.close(fig); frame_id[0] += 1

    def title_card(alpha, lines, n):
        for i in range(n):
            a = min(1, (i + 1) / (0.4 * n), (n - i) / (0.3 * n)) * alpha
            fig = plt.figure(figsize=(19.2, 10.8), facecolor=BG)
            for y, txt, sz, col, wt in lines:
                fig.text(0.5, y, txt, ha="center", va="center", fontsize=sz, color=col, alpha=a, weight=wt)
            save(fig)

    only = [int(x) for x in os.environ.get("PROMO_FRAMES", "").split(",") if x]
    if not only:
      title_card(1, [(0.56, "CardioMag", 76, FG, "bold"),
                     (0.45, "Seeing the heart's electrical currents through their magnetic field", 26, DIM, "normal")],
               int(2.5 * fps))

    az0 = np.radians(-20)
    for i in (only or range(T)):
        az = az0 + np.radians(35) * np.sin(2 * np.pi * i / T)
        fig = plt.figure(figsize=(19.2, 10.8), facecolor=BG)
        fig.text(0.035, 0.945, "CardioMag", fontsize=30, weight="bold", color=FG)
        fig.text(0.175, 0.948, "simulated whole-heart beat  ·  64-sensor magnetometer array", fontsize=16, color=DIM)
        fig.text(0.965, 0.948, f"t = {t[i]:5.0f} ms", fontsize=22, color=FG, ha="right", family="DejaVu Sans Mono")
        heads = ["Cardiac current flow", "Magnetic field at the sensor array", "Current reconstructed from the array"]
        for j, hd in enumerate(heads):
            fig.text(0.20 + j * 0.27, 0.885, hd, fontsize=17, color=FG, ha="center", weight="bold")
        for r, (k, name) in enumerate(CASES):
            y0 = 0.475 - r * 0.385
            fig.text(0.012, y0 + 0.19, name, fontsize=17, color=ACC[k], rotation=90, va="center", weight="bold")
            # 1 - true current
            ax = fig.add_axes([0.065, y0, 0.27, 0.37]); axstyle(ax)
            heart(ax, az)
            m = d[f"{k}_m"][i]; mag = np.linalg.norm(m, axis=1)
            sel = np.argsort(mag)[-260:]; sel = sel[mag[sel] > max(0.25 * mag.max(), 0.003 * mmax)]
            if sel.size:
                P, _ = project(c[sel], az); V, _ = project(m[sel], az)
                a = np.clip(mag[sel] / mmax, 0, 1) ** 0.4
                a = a / a.max() * max(a.max(), 0.7)                 # keep P / T currents visible
                U = V / np.maximum(mag[sel], 1e-30)[:, None] * a[:, None]
                ax.scatter(P[:, 0], P[:, 1], s=220 * a + 4, c=a, cmap="inferno", vmin=0, vmax=1.0, alpha=0.15, lw=0)
                ax.quiver(P[:, 0], P[:, 1], U[:, 0], U[:, 1], a, cmap="inferno", clim=(0.0, 1.0),
                          scale=26, width=0.0045, headwidth=3.2, alpha=0.95)
            # 2 - field map
            ax = fig.add_axes([0.335, y0 + 0.07, 0.27, 0.30]); axstyle(ax, 0.11)
            Bm = d[f"{k}_Bmap"][i]
            Bg = np.sign(Bm) * np.clip(np.abs(Bm) / bmax, 0, 1) ** 0.6
            ax.imshow(Bg, extent=(g[0], g[-1], g[0], g[-1]), origin="lower", cmap=cm_b, vmin=-1, vmax=1,
                      interpolation="bicubic")
            lv = [v for v in np.linspace(-1, 1, 11) if abs(v) > 0.05]
            ax.contour(g, g, Bg, levels=lv, colors="w", linewidths=0.5, alpha=0.25)
            heart(ax, 0.0, dim=True)
            ax.scatter(sens[:, 0], sens[:, 1], s=26, facecolor=BG, edgecolor=FG, lw=0.8, alpha=0.9)
            # trace inset
            ins = fig.add_axes([0.37, y0 + 0.005, 0.2, 0.055]); ins.axis("off")
            ins.plot(t, rms[k], color=ACC[k], lw=1.2, alpha=0.35)
            ins.plot(t[:i + 1], rms[k][:i + 1], color=ACC[k], lw=1.8)
            ins.axvline(t[i], color=FG, lw=0.8, alpha=0.6); ins.set_xlim(t[0], t[-1])
            ins.set_ylim(0, max(rms[kk].max() for kk, _ in CASES) * 1.1)
            # 3 - reconstruction
            ax = fig.add_axes([0.605, y0, 0.27, 0.37]); axstyle(ax)
            Pv = heart(ax, 0.0)
            rec = d[f"{k}_rec"][i]; rm = np.linalg.norm(rec, axis=1)
            a = np.clip(rm / rmax, 0, 1) ** 0.5
            sel = (rm > 0.45 * rm.max()) & (a > 0.08)
            ax.scatter(src[sel, 0], src[sel, 1], s=700 * a[sel] ** 2, c=a[sel], cmap="viridis", vmin=0, vmax=1,
                       alpha=0.22, lw=0)
            if sel.any():
                U = rec[sel, :2] / np.maximum(rm[sel], 1e-30)[:, None] * a[sel, None]
                ax.quiver(src[sel, 0], src[sel, 1], U[:, 0], U[:, 1], a[sel], cmap="viridis", clim=(0, 1),
                          scale=22, width=0.006, headwidth=3.5)
            # detector badge
            bx = fig.add_axes([0.885, y0 + 0.12, 0.1, 0.13]); bx.axis("off")
            decided = t[i] > qrs_end["normal"] + 30
            if not decided:
                txt, col = "analysing…", DIM
            elif k == "normal":
                txt, col = "normal", ACC["normal"]
            else:
                txt, col = "ANOMALY\ndetected", ACC["lbbb"]
            bx.add_patch(FancyBboxPatch((0.05, 0.1), 0.9, 0.8, boxstyle="round,pad=0.02", fc="none", ec=col, lw=2,
                                        transform=bx.transAxes))
            bx.text(0.5, 0.72, "AI detector", ha="center", fontsize=12, color=DIM, transform=bx.transAxes)
            bx.text(0.5, 0.38, txt, ha="center", va="center", fontsize=16, color=col, weight="bold",
                    transform=bx.transAxes)
        fig.text(0.5, 0.012, "Simulated data (Strocchi et al. heart mesh, CC BY 4.0) · illustrative mock-up",
                 fontsize=11, color=DIM, ha="center")
        save(fig)
        if i % 25 == 0:
            print("frame", i, "/", T, flush=True)

    if only:
        print("preview frames in", tmp); return
    title_card(1, [(0.58, "Contactless · radiation-free · millisecond resolution", 30, FG, "bold"),
                   (0.47, "CardioMag", 60, ACC["normal"], "bold")], int(3 * fps))
    OUT.parent.mkdir(exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(OUT)],
                   check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    {"compute": compute, "render": render}[sys.argv[1]]()
