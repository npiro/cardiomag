"""Leave-one-heart-out cross-validation: each heart is the test set once, the next one validation.

    python scripts/cross_validate.py --config configs/train/atrial_cnn.yaml [--set optim.epochs=30]

Writes runs/<run_name>_cv/summary.json with per-fold and pooled test AUROC / detection rates.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # works even if the editable install is skipped
from cardiomag.ml.train import load_config, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    base = load_config(a.config, a.set)
    info = json.load(open(Path(base["data"]["root"]) / "dataset.json"))
    hearts = [h for s in ("train", "val", "test") for h in info["config"]["splits"].get(s, [])]
    out = Path(base.get("out_dir", "runs")) / f"{base['run_name']}_cv"
    folds = []
    for i, test_h in enumerate(hearts):
        val_h = hearts[(i + 1) % len(hearts)]
        train_h = [h for h in hearts if h not in (test_h, val_h)]
        cfg = json.loads(json.dumps(base))
        cfg["run_name"] = f"fold{i}_{test_h.replace(':', '_')}"
        cfg["out_dir"] = str(out)
        cfg["data"]["split_hearts"] = {"train": train_h, "val": [val_h], "test": [test_h]}
        print(f"=== fold {i}: test {test_h}, val {val_h}, train {train_h}", flush=True)
        r = train(cfg)
        folds.append(dict(test=test_h, val=val_h, **{k: v for k, v in r["test"].items() if k != "confusion"},
                          real=r.get("real")))
    aur = [f["auroc_anomaly"] for f in folds if "auroc_anomaly" in f]
    summary = dict(config=a.config, folds=folds, auroc_mean=float(np.mean(aur)), auroc_std=float(np.std(aur)))
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "folds"}, indent=1))
    for f in folds:
        print(f"  test {f['test']:22s} AUROC {f.get('auroc_anomaly', float('nan')):.3f}  FPR {f['false_positive_rate']:.2f}  "
              + "  ".join(f"{k[8:]} {v:.2f}" for k, v in f.items() if k.startswith("flagged_") and k != "flagged_healthy"))


if __name__ == "__main__":
    main()
