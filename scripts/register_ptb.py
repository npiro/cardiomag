"""Register the PTB 49-sensor layout on a whole-heart model: scan array centre and axis signs and
compare simulated R-, T- and P-peak maps with the measured PTB median beat (map correlations)."""
import itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cardiomag.sim.whole_heart as wh
from cardiomag.physics import leadfield as lf
from cardiomag.physics.sensors import ptb49_layout
from cardiomag.io.mcg import load_ptb

tt = np.arange(-450, 550); r = load_ptb(); B = r["beat"]; rms = np.sqrt((B ** 2).mean(0))
iq = 450 + np.argmax(rms[390:510]) - 60
ts = (tt > 150) & (tt < 450); it = np.flatnonzero(ts)[np.argmax(rms[ts])]
ps = (tt > -300) & (tt < -60); ip = np.flatnonzero(ps)[np.argmax(rms[ps])]
rq, rt, rp = B[:, iq], B[:, it], r["beat_p"][:, ip]
lay, _ = ptb49_layout()
out = []
for cid in sys.argv[1:] or ["01"]:
    h = wh.load_whole_heart("healthy", cid); sim = wh.WholeHeartSimulator(h)
    a = sim.sample("normal", np.random.default_rng(5)); tm = np.arange(-150, 800, 2.0); pr = a.params["pr_ms"]
    av = a.at_ms[sim.vent]; qd = np.percentile(av, 98) - av.min()
    sig = lf.conductivity_tensors(h.fibres, h.region, h.region_names, h.cv)
    for sx, sy in itertools.product((1, -1), (1, -1)):
        for cx, cy in itertools.product([-120, -60, 0, 60, 120], [-80, -40, 0]):
            xy = (lay * np.array([sx, sy]) + np.array([cx, cy])) * 1e-3
            z = np.max(h.skin_fn(xy[:, 0], xy[:, 1])) + 0.045
            pos = np.column_stack([xy, np.full(len(xy), z)]); nrm = np.tile([0, 0, 1.0], (len(xy), 1))
            W = lf.node_leadfield(h.points, h.tets, sig, pos, nrm); Bs = wh.field_series(W, a.at_ms, a.rt_ms, tm)
            rr = np.sqrt((Bs ** 2).mean(0)); q = (tm >= pr) & (tm <= pr + qd); T = tm > pr + qd + 60; P = tm < pr
            c = [np.corrcoef(ref, Bs[:, np.flatnonzero(m)[np.argmax(rr[m])]])[0, 1] for ref, m in ((rq, q), (rt, T), (rp, P))]
            out.append(dict(heart=cid, sx=sx, sy=sy, cx=cx, cy=cy, R=c[0], T=c[1], P=c[2], score=c[0] + c[1] + c[2]))
            print(cid, sx, sy, cx, cy, "R %.2f T %.2f P %.2f" % tuple(c), flush=True)
out.sort(key=lambda d: -d["score"])
Path("runs/gap").mkdir(parents=True, exist_ok=True)
json.dump(out, open("runs/gap/ptb_registration.json", "w"), indent=1)
print("BEST:", out[:5])
