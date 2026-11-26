"""Generate labelled MCG datasets from simulated healthy / abnormal hearts.

    cardiomag-generate --config configs/datasets/atrial_ptb49.yaml [--out data/atrial_ptb49] [--workers 4]

Output directory layout
    dataset.json                 config, classes, sensor layout, time axis, per-split counts
    <split>/shard_000.npz        X (n,S,T) float32 pT, y (n,) class, anomaly (n,), heart (n,), meta json
    real/real.npz                measured healthy recordings on the same layout (if available)

Per heart: a few "placements" (heart rotation jitter + array centre / standoff / rotation jitter). The
sensor lead field W (S x nodes) is computed once per placement; each sample is an activation map ->
template Vm -> B = W @ Vm. Scars subtract the scar elements' contribution from W.
Noise is NOT baked in: it is added on the fly by the PyTorch dataset (so the noise level is a training
choice).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from ..io import hearts as H
from ..io.mcg import real_windows
from ..physics import leadfield as lf
from ..physics.sensors import SensorArray, default_centre_mm
from ..sim.eikonal import vm_from_at
from ..sim.scenarios import ATRIAL_CLASSES, VENTRICULAR_CLASSES, make_simulator
from ..sim.whole_heart import WHOLE_CLASSES, WholeHeartSimulator, field_series, load_whole_heart

CLASSES = {"atria": ATRIAL_CLASSES, "ventricles": VENTRICULAR_CLASSES, "whole": WHOLE_CLASSES}


def _load_heart(spec: str, rng=None, rot_deg=0.0):
    src, cid = spec.split(":")
    if src == "biatrial":
        return H.load_biatrial(cid)
    if src == "strocchi_healthy":
        return H.load_strocchi("healthy", cid, rot_jitter_deg=rot_deg, rng=rng)
    if src == "strocchi_hf":
        return H.load_strocchi("hf", cid, rot_jitter_deg=rot_deg, rng=rng)
    if src == "whole_healthy":
        return load_whole_heart("healthy", cid)
    if src == "whole_hf":
        return load_whole_heart("hf", cid)
    raise ValueError(spec)


def _rotate_heart(h, rng, deg):
    if not deg:
        return h
    R = np.eye(3)
    for ax in np.eye(3):
        R = H.rot_about(ax, rng.normal(0, deg)) @ R
    h.points = h.points @ R.T
    h.fibres = h.fibres @ R.T
    return h


def generate_heart(job):
    """Worker: the samples of one (heart, placement) job. Returns dict of arrays.
    Jobs are split by placement so the pool stays balanced across hearts of uneven size."""
    cfg, spec, split, seed = job["cfg"], job["spec"], job["split"], job["seed"]
    rng = np.random.default_rng(seed)
    kind = cfg["kind"]
    classes = CLASSES[kind]
    tcfg = cfg["time"]
    t_ms = np.arange(tcfg["t0_ms"], tcfg["t1_ms"], tcfg["dt_ms"], dtype=float)
    jit = cfg.get("placement_jitter", {})
    n_place = int(cfg.get("placements_per_heart", 3))
    per_class = cfg["samples_per_heart"]
    group_hf = spec.startswith(("strocchi_hf", "whole_hf"))
    hf_classes = cfg.get("hf_conduction", ["normal", "lbbb"])

    base = _load_heart(spec)
    sim = WholeHeartSimulator(base) if kind == "whole" else make_simulator(base)
    X, y, anomaly, meta = [], [], [], []
    t_start = time.time()
    for p in job.get("placements", range(n_place)):
        h = _load_heart(spec)
        h = _rotate_heart(h, rng, jit.get("heart_rot_deg", 0.0))
        a = dict(cfg["array"])
        c0 = a.pop("centre_mm", None) or default_centre_mm(kind, a.get("preset", "grid"))
        a["centre_mm"] = (c0[0] + rng.normal(0, jit.get("centre_mm", 0.0)),
                          c0[1] + rng.normal(0, jit.get("centre_mm", 0.0)))
        a["standoff_mm"] = a.get("standoff_mm", 10.0) + abs(rng.normal(0, jit.get("standoff_mm", 0.0)))
        a["rotation_deg"] = a.get("rotation_deg", 0.0) + rng.normal(0, jit.get("rotation_deg", 0.0))
        arr = SensorArray.from_dict(a)
        pos, nrm = arr.place(h.skin_fn)
        sig = lf.sigma_for(h)
        W = lf.node_leadfield(h.points, h.tets, sig, pos, nrm)
        labels = classes if not group_hf else hf_classes
        for lab in labels:
            n = int(per_class.get(lab, 0)) if not group_hf else int(cfg.get("hf_samples_per_class", 10))
            n = int(np.ceil(n / n_place))
            for _ in range(n):
                act = sim.sample(lab, rng)
                Wd = W
                if (act.sigma_scale < 1).any():
                    el = np.flatnonzero(act.sigma_scale < 1)
                    Wd = W - lf.node_leadfield(h.points, h.tets, sig, pos, nrm, elements=el) * \
                        (1 - act.sigma_scale[el]).mean()
                if kind == "whole":                                   # full beat incl. repolarisation
                    B = field_series(Wd, act.at_ms, act.rt_ms, t_ms) * 1e12
                else:
                    vm = vm_from_at(act.at_ms, t_ms) * 1e-3          # (T, N) volts
                    B = (Wd @ vm.T) * 1e12                            # (S, T) pT
                X.append(B.astype(np.float32))
                yi = classes.index(lab)
                y.append(yi)
                anomaly.append(int(lab not in ("healthy", "normal") or group_hf))
                meta.append(dict(heart=spec, group=h.group, label=lab, placement=p, array=arr.to_dict(),
                                 params={k: v for k, v in act.params.items() if not isinstance(v, list)}))
    return dict(spec=spec, split=split, X=np.stack(X), y=np.array(y), anomaly=np.array(anomaly), meta=meta,
                seconds=time.time() - t_start)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--scale", type=float, default=1.0, help="multiply samples per class (quick tests: 0.1)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    if args.scale != 1.0:
        cfg["samples_per_heart"] = {k: max(1, int(round(v * args.scale))) for k, v in cfg["samples_per_heart"].items()}
        cfg["hf_samples_per_class"] = max(1, int(round(cfg.get("hf_samples_per_class", 10) * args.scale)))
    out = Path(args.out or Path("data") / cfg["name"])
    out.mkdir(parents=True, exist_ok=True)
    kind = cfg["kind"]
    classes = CLASSES[kind]
    jobs = []
    seed = int(cfg.get("seed", 0))
    for split, specs in cfg["splits"].items():
        for spec in specs:
            h_idx = len({j["spec"] for j in jobs})
            n_place = int(cfg.get("placements_per_heart", 3))
            for p in range(n_place):  # one job per (heart, placement), deterministic seed
                jobs.append(dict(cfg=cfg, spec=spec, split=split, placements=[p],
                                 seed=seed + h_idx * 7919 + p * 104729))
    print(f"[generate] {cfg['name']}: {len({j['spec'] for j in jobs})} hearts, {len(jobs)} jobs "
          f"(heart x placement), {args.workers} workers -> {out}", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(generate_heart, jobs):
            print(f"  {r['spec']:24s} {r['split']:5s} placement {r['meta'][0]['placement']}  "
                  f"{r['X'].shape[0]:4d} samples  {r['seconds']:.0f}s", flush=True)
            results.append(r)
    counts = {}
    for split in cfg["splits"]:
        rs = [r for r in results if r["split"] == split]
        if not rs:
            continue
        sd = out / split
        sd.mkdir(exist_ok=True)
        X = np.concatenate([r["X"] for r in rs])
        meta = sum([r["meta"] for r in rs], [])
        np.savez_compressed(sd / "shard_000.npz", X=X, y=np.concatenate([r["y"] for r in rs]),
                            anomaly=np.concatenate([r["anomaly"] for r in rs]),
                            heart=np.array([m["heart"] for m in meta]),
                            meta=np.array([json.dumps(m) for m in meta]))
        counts[split] = {c: int(sum(m["label"] == c for m in meta)) for c in classes}
        counts[split]["anomaly"] = int(np.concatenate([r["anomaly"] for r in rs]).sum())
        counts[split]["total"] = int(X.shape[0])
    arr = SensorArray.from_dict({k: v for k, v in cfg["array"].items() if k != "centre_mm"})
    tc = cfg["time"]
    real = real_windows(arr.preset, kind, tc["t0_ms"], tc["t1_ms"], tc["dt_ms"])
    if real:
        (out / "real").mkdir(exist_ok=True)
        np.savez_compressed(out / "real" / "real.npz", X=np.stack([x for _, x in real]).astype(np.float32),
                            name=np.array([n for n, _ in real]), y=np.zeros(len(real), int),
                            anomaly=np.zeros(len(real), int))
    info = dict(name=cfg["name"], kind=kind, classes=classes, config=cfg, counts=counts,
                n_sensors=int(results[0]["X"].shape[1]), n_times=int(results[0]["X"].shape[2]),
                layout_mm=arr.layout_mm().tolist(), sensor_names=arr.names,
                t_ms=np.arange(tc["t0_ms"], tc["t1_ms"], tc["dt_ms"]).tolist(),
                real=[n for n, _ in real], units="pT", created=time.strftime("%Y-%m-%d %H:%M"))
    json.dump(info, open(out / "dataset.json", "w"), indent=2)
    print(f"[generate] done: {counts}; real recordings: {[n for n, _ in real]}", flush=True)


if __name__ == "__main__":
    main()
