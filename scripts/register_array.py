"""Register a sensor layout (ptb49 | kiel36) on a whole-heart model: scan the array centre and axis
signs, and compare simulated R-, T- and P-peak maps with the measured healthy median beats (map
correlations, averaged over subjects).

    python scripts/register_array.py --preset kiel36 --hearts 01 02
    python scripts/register_array.py --preset kiel36 --cx -40 0 40 80 --cy 0 30 60   # refine

Writes runs/gap/<preset>_registration.json (best first). Put the winner in
cardiomag.physics.sensors.default_centre_mm[("whole", preset)] (and ("ventricles", preset)).
"""
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cardiomag.sim.whole_heart as wh  # noqa: E402
from cardiomag.physics import leadfield as lf  # noqa: E402
from cardiomag.physics.sensors import ptb49_layout, kiel36_layout  # noqa: E402
from cardiomag.io.mcg import load_ptb, load_kiel  # noqa: E402

GRIDS = {"ptb49": ([-120, -60, 0, 60, 120], [-80, -40, 0]),
         "kiel36": ([-40, 0, 40, 80, 120], [-20, 10, 40, 70, 100])}


def peak_maps(r):
    """Measured R-, T- and P-peak maps (S,) of one median beat on t = -450..549 ms rel. R."""
    tt = np.arange(-450, 550); B = r["beat"]; rms = np.sqrt((B ** 2).mean(0))
    iq = 450 + np.argmax(rms[390:510]) - 60
    ts = (tt > 150) & (tt < 450); it = np.flatnonzero(ts)[np.argmax(rms[ts])]
    ps = (tt > -300) & (tt < -60); ip = np.flatnonzero(ps)[np.argmax(rms[ps])]
    return B[:, iq], B[:, it], r["beat_p"][:, ip]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="ptb49", choices=list(GRIDS))
    ap.add_argument("--hearts", nargs="*", default=["01"])
    ap.add_argument("--cx", nargs="*", type=float); ap.add_argument("--cy", nargs="*", type=float)
    ap.add_argument("--no-flips", action="store_true")
    a = ap.parse_args()
    recs = [load_ptb()] if a.preset == "ptb49" else [load_kiel(s) for s in range(1, 8)]
    refs = [peak_maps(r) for r in recs]
    lay, _ = ptb49_layout() if a.preset == "ptb49" else kiel36_layout()
    cxs, cys = a.cx or GRIDS[a.preset][0], a.cy or GRIDS[a.preset][1]
    flips = [(1, 1)] if a.no_flips else list(itertools.product((1, -1), (1, -1)))
    out = []
    for cid in a.hearts:
        h = wh.load_whole_heart("healthy", cid); sim = wh.WholeHeartSimulator(h)
        act = sim.sample("normal", np.random.default_rng(5)); tm = np.arange(-150, 800, 2.0); pr = act.params["pr_ms"]
        av = act.at_ms[sim.vent]; qd = np.percentile(av, 98) - av.min()
        sig = lf.conductivity_tensors(h.fibres, h.region, h.region_names, h.cv)
        for (sx, sy), cx, cy in itertools.product(flips, cxs, cys):
            xy = (lay * np.array([sx, sy]) + np.array([cx, cy])) * 1e-3
            z = np.max(h.skin_fn(xy[:, 0], xy[:, 1])) + 0.045
            pos = np.column_stack([xy, np.full(len(xy), z)]); nrm = np.tile([0, 0, 1.0], (len(xy), 1))
            W = lf.node_leadfield(h.points, h.tets, sig, pos, nrm); Bs = wh.field_series(W, act.at_ms, act.rt_ms, tm)
            rr = np.sqrt((Bs ** 2).mean(0)); q = (tm >= pr) & (tm <= pr + qd); T = tm > pr + qd + 60; P = tm < pr
            sm = [Bs[:, np.flatnonzero(m)[np.argmax(rr[m])]] for m in (q, T, P)]
            c = np.array([[np.corrcoef(ref[k], sm[k])[0, 1] for k in range(3)] for ref in refs])  # (subjects, 3)
            R, Tc, Pc = c.mean(0)
            out.append(dict(heart=cid, sx=sx, sy=sy, cx=cx, cy=cy, R=R, T=Tc, P=Pc, score=R + Tc + Pc,
                            per_subject=c.round(3).tolist()))
            print(cid, sx, sy, cx, cy, "R %.2f T %.2f P %.2f" % (R, Tc, Pc), flush=True)
    # aggregate over hearts
    agg = {}
    for d in out:
        agg.setdefault((d["sx"], d["sy"], d["cx"], d["cy"]), []).append(d)
    summ = sorted([dict(sx=k[0], sy=k[1], cx=k[2], cy=k[3], **{m: float(np.mean([d[m] for d in v])) for m in ("R", "T", "P", "score")})
                   for k, v in agg.items()], key=lambda d: -d["score"])
    Path("runs/gap").mkdir(parents=True, exist_ok=True)
    json.dump(dict(preset=a.preset, hearts=a.hearts, ranking=summ, all=out),
              open(f"runs/gap/{a.preset}_registration.json", "w"), indent=1)
    print("BEST:"); [print(" ", d) for d in summ[:6]]


if __name__ == "__main__":
    main()
