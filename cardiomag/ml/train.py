"""Train / evaluate models on generated MCG datasets.

    cardiomag-train --config configs/train/atrial_cnn.yaml [--set optim.epochs=5 data.root=...]
    cardiomag-eval  --run runs/atrial_cnn

Config (YAML):
    run_name: atrial_cnn
    out_dir: runs
    seed: 0
    device: auto                    # auto | cuda | mps | cpu
    data:
      root: data/atrial_ptb49
      target: anomaly               # anomaly (binary) | label (scenario class)
      normalize: sample_rms
      noise_fT_rtHz: 20             # sensor noise added on the fly
      augment: {amp_jitter: 0.1, time_shift: 3, channel_dropout: 0.05}
    model: {name: cnn1d, kwargs: {width: 64}}      # or {target: "my_pkg.models:MyNet", kwargs: {...}}
    optim: {lr: 1e-3, weight_decay: 1e-4, epochs: 40, batch_size: 64, patience: 8}
    workers: 0

Classifier models are trained with cross-entropy (class-balanced weights). Reconstruction models
(task = "reconstruct") are trained on healthy samples only; the anomaly score is the reconstruction
error and the detection threshold is the 95th percentile of healthy validation scores.
Outputs: <out_dir>/<run_name>/{config.yaml, metrics.csv, best.pt, results.json, scores_<split>.csv}
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from ..data.dataset import MCGDataset
from . import metrics as M
from .registry import build_model


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set(cfg, dotted, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(value)


def load_config(path, overrides=()):
    cfg = yaml.safe_load(open(path))
    for o in overrides:
        k, _, v = o.partition("=")
        _set(cfg, k, v)
    return cfg


def _datasets(cfg, task):
    dc = cfg["data"]
    common = dict(target=dc.get("target", "anomaly"), normalize=dc.get("normalize", "sample_rms"),
                  noise_fT_rtHz=dc.get("noise_fT_rtHz", 0.0), seed=cfg.get("seed", 0))
    healthy_only = task == "reconstruct"
    ds = {"train": MCGDataset(dc["root"], "train", augment=dc.get("augment"), healthy_only=healthy_only, **common)}
    for s in ("val", "test", "real"):
        try:
            ds[s] = MCGDataset(dc["root"], s, **common)
        except FileNotFoundError:
            pass
    return ds


@torch.no_grad()
def predict(model, loader, device, task):
    model.eval()
    scores, probs, ys = [], [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        if task == "reconstruct":
            s = ((x - out) ** 2).mean(dim=(1, 2))
            scores.append(s.float().cpu().numpy())
        else:
            p = torch.softmax(out.float(), dim=1).cpu().numpy()
            probs.append(p)
            scores.append(1.0 - p[:, 0])  # anomaly score = 1 - P(class 0 = healthy/normal)
        ys.append(y.numpy())
    return (np.concatenate(scores), np.concatenate(probs) if probs else None, np.concatenate(ys))


def evaluate_split(model, ds, device, task, threshold, bs=256):
    if ds is None:
        return None
    dl = DataLoader(ds, batch_size=bs, shuffle=False)
    scores, probs, y = predict(model, dl, device, task)
    res = dict(n=int(len(ds)))
    if len(np.unique(ds.anomaly)) > 1:
        res["auroc_anomaly"] = M.auroc(scores, ds.anomaly)
    res.update(M.detection_report(scores, ds.anomaly, ds.y, ds.info["classes"], threshold))
    if probs is not None:
        pred = probs.argmax(1)
        res["balanced_accuracy"] = M.balanced_accuracy(pred, y, probs.shape[1])
        res["confusion"] = M.confusion(pred, y, probs.shape[1]).tolist()
    return res, scores


def train(cfg):
    torch.manual_seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))
    out = Path(cfg.get("out_dir", "runs")) / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out / "config.yaml", "w"), sort_keys=False)
    device = pick_device(cfg.get("device", "auto"))

    from .registry import get_model_class
    task = getattr(get_model_class(cfg["model"]), "task", "classify")
    ds = _datasets(cfg, task)
    tr = ds["train"]
    S, T = tr.shape
    model = build_model(cfg["model"], S, T, tr.n_classes).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    oc = cfg.get("optim", {})
    opt = torch.optim.AdamW(model.parameters(), lr=float(oc.get("lr", 1e-3)),
                            weight_decay=float(oc.get("weight_decay", 1e-4)))
    epochs = int(oc.get("epochs", 30))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(oc.get("lr", 1e-3)), total_steps=epochs *
                                                max(1, int(np.ceil(len(tr) / int(oc.get("batch_size", 64))))))
    if task == "classify":
        counts = np.bincount(tr.targets(), minlength=tr.n_classes).astype(float)
        w = torch.tensor(counts.sum() / np.maximum(counts, 1) / tr.n_classes, dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=w)
    else:
        loss_fn = nn.MSELoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    dl = DataLoader(tr, batch_size=int(oc.get("batch_size", 64)), shuffle=True, drop_last=len(tr) > 64,
                    num_workers=int(cfg.get("workers", 0)))
    print(f"[train] {cfg['run_name']}: {type(model).__name__} ({n_par / 1e3:.0f}k params, task={task}) on {device}; "
          f"train {len(tr)} x {S} sensors x {T} samples; val {len(ds.get('val', []))}", flush=True)

    best, best_ep, patience = -np.inf, -1, int(oc.get("patience", 10))
    log = open(out / "metrics.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["epoch", "train_loss", "val_metric", "seconds"])
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                out_ = model(x)
                loss = loss_fn(out_, y) if task == "classify" else loss_fn(out_, x)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach()) * x.shape[0]
            n += x.shape[0]
        val = ds.get("val")
        if val is not None:
            scores, probs, y = predict(model, DataLoader(val, batch_size=256), device, task)
            if cfg["data"].get("target", "anomaly") == "label" and probs is not None:
                metric = M.balanced_accuracy(probs.argmax(1), y, probs.shape[1])
            else:
                metric = M.auroc(scores, val.anomaly)
        else:
            metric = -tot / n
        wr.writerow([ep, tot / n, metric, time.time() - t0])
        log.flush()
        print(f"  epoch {ep:3d}  loss {tot / n:.4f}  val {metric:.4f}  ({time.time() - t0:.1f}s)", flush=True)
        if metric > best:
            best, best_ep = metric, ep
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": ep, "shape": (S, T),
                        "n_outputs": tr.n_classes}, out / "best.pt")
        elif ep - best_ep >= patience:
            print(f"  early stop (best epoch {best_ep})", flush=True)
            break
    log.close()
    return evaluate(out)


def evaluate(run_dir):
    run_dir = Path(run_dir)
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    device = pick_device(cfg.get("device", "auto"))
    S, T = ck["shape"]
    model = build_model(cfg["model"], S, T, ck["n_outputs"]).to(device)
    model.load_state_dict(ck["model"])
    from .registry import get_model_class
    task = getattr(get_model_class(cfg["model"]), "task", "classify")
    ds = _datasets(cfg, task)
    # threshold: 95th percentile of healthy validation scores (fixed false-positive budget)
    ref = ds.get("val") or ds["train"]
    sc, _, _ = predict(model, DataLoader(ref, batch_size=256), device, task)
    healthy = sc[ref.anomaly == 0]
    thr = float(np.percentile(healthy, 95)) if healthy.size else 0.5
    results = dict(run=cfg["run_name"], best_epoch=int(ck["epoch"]), threshold_from="val healthy p95",
                   threshold=thr, classes=ref.info["classes"], target=cfg["data"].get("target", "anomaly"))
    for split in ("val", "test", "real"):
        if split in ds:
            r, s = evaluate_split(model, ds[split], device, task, thr)
            results[split] = r
            with open(run_dir / f"scores_{split}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["index", "heart", "label", "anomaly", "score"])
                for i, (h, m, a, v) in enumerate(zip(ds[split].heart, ds[split].meta, ds[split].anomaly, s)):
                    w.writerow([i, h, m.get("label"), int(a), float(v)])
    json.dump(results, open(run_dir / "results.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in results.items() if k in ("test", "real")}, indent=1)[:3000], flush=True)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. optim.epochs=5 device=cpu")
    a = ap.parse_args(argv)
    train(load_config(a.config, a.set))


def eval_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    evaluate(ap.parse_args(argv).run)


if __name__ == "__main__":
    main()
