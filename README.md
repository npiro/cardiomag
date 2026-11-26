# cardiomag

Cardiac magnetic-field modelling and anomaly detection:

- **Physics**: primary cardiac current `J = −σᵢ∇Vₘ` on tetrahedral heart meshes and the magnetic field at
  any sensor array (exact element-wise Biot–Savart, assembled as a linear map `B = W·Vₘ`).
- **Simulated hearts**: an anisotropic eikonal solver (validated to ~1 ms against reference simulations)
  generating healthy and abnormal activation on real anatomies: four atrial patient models and the
  Strocchi healthy / heart-failure ventricle cohorts.
- **Measured data**: PTB 49-channel SQUID and Kiel 36-position OPM recordings (healthy volunteers).
- **Viewer**: a single self-contained HTML file showing currents, simulated and measured fields, with a
  movable virtual sensor array whose signals you can export.
- **ML**: dataset generation, PyTorch datasets, pluggable models, a trainer that runs on CUDA, MPS or CPU,
  and recipes for remote GPUs.

```
raw datasets ──► io/ (meshes, MCG recordings) ──► sim/ (eikonal + scenarios) ──► physics/ (B = W·Vm)
                                                                                   │
           viewer/cardiomag_viewer.html ◄── viz/ ◄────────────────────────────────┤
                                                                                   ▼
     runs/<name>/ ◄── ml/train.py ◄── data/dataset.py (noise, augment) ◄── data/generate.py (shards)
```

## Quick start

```bash
environment/setup_env.sh                 # .venv with PyTorch (MPS on Apple silicon, CPU otherwise)
source .venv/bin/activate
pytest -q                                # physics, simulator, ML smoke tests
cardiomag-generate --config configs/datasets/atrial_ptb49.yaml       # -> data/atrial_ptb49
cardiomag-train --config configs/train/atrial_cnn.yaml               # -> runs/atrial_cnn
cardiomag-viewer --out viewer/cardiomag_viewer.html                  # standalone HTML
```

Raw data are looked up in `$CARDIOMAG_DATA` (default: `../cardiac-mcg-data`, the folder created by the
download step). Parsed meshes are cached in `.cache/`.

| Dataset | Used for | Folder under `$CARDIOMAG_DATA` |
|---|---|---|
| Biatrial ECG-calibrated models, Zenodo 17143584 (S1–S4) | atrial simulations, torso/chest | `zenodo_17143584_biatrial/` |
| Strocchi healthy four-chamber meshes, Zenodo 4590294 (01–20) | healthy ventricles | `zenodo_4590294_strocchi_healthy/` |
| Strocchi heart-failure meshes, Zenodo 3890034 (01–24; 01–02 downloaded) | heart-failure ventricles | `zenodo_3890034_strocchi_hf/` |
| PTB reference dataset (Koch et al. 2011) | measured healthy MCG, calibration | `ptb_koch2011/` |
| Kiel Cardio OPM database (PhysioNet) | measured healthy MCG | `kiel_cardio/` |

## Simulated conditions

| Kind | Class | What is simulated |
|---|---|---|
| atria | `healthy` | sinus-node activation; CV, timing and placement jitter |
| atria | `slow_patch` | fibrotic patch (r 10–20 mm) with conduction velocity × 0.2–0.4 |
| atria | `ia_block` | interatrial block: Bachmann's bundle connections removed |
| atria | `ectopic` | ectopic focus in the LA or RA instead of the sinus node |
| ventricles | `normal` | LV + RV septal activation with a fast endocardial (Purkinje-like) layer |
| ventricles | `lbbb` / `rbbb` | left / right bundle branch block |
| ventricles | `scar` | non-conducting, non-current-generating LV region (r 15–25 mm) |
| ventricles | heart-failure anatomy | Strocchi HF meshes with normal or LBBB conduction (anomalous by anatomy) |

The binary target `anomaly` is 1 for every non-healthy class and for heart-failure anatomies. Splits are by
heart, so no anatomy appears in two splits. Signals are stored noise-free; sensor noise (white + 1/f,
fT/√Hz) is added on the fly by the dataset.

Physics and calibration notes:
- **Conductivity.** σᵢ is uniform with fibre anisotropy from the conduction velocities. The effective
  scale K_cal = 0.33 is calibrated against the measured PTB P-wave over four patient models (see
  `cardiomag/physics/leadfield.py`).
- **Field component.** The normal component is exact for a flat (half-space) chest. Tangential
  components omit volume currents.
- **Ventricles.** The Strocchi meshes have no torso, so the chest is a flat half-space 2 cm in front of
  the heart, in a canonical anatomical orientation with ±6° jitter. Complete RBBB produces long
  activation times (≈160–210 ms) because no right-sided conduction remains.
- **Repolarisation.** It is not modelled. Windows are 170 ms (P-wave) and 210 ms (QRS) from onset.

## Training on your own model

Any `torch.nn.Module` with the constructor `Model(n_sensors, n_times, n_outputs, **kwargs)` plugs in:

```yaml
model:
  target: "my_package.models:MyTransformer"   # importable path
  kwargs: {depth: 4}
```

Set the class attribute `task = "reconstruct"` for unsupervised models: they train on healthy samples
only, and the anomaly score is the reconstruction error. Classifiers use `task = "classify"` (the
default). Override any config value from the command line, e.g.
`cardiomag-train --config ... --set optim.epochs=10 data.noise_fT_rtHz=50 device=cpu`.

Each run writes `runs/<name>/` containing `config.yaml`, `metrics.csv`, `best.pt`, `results.json` and
per-sample `scores_*.csv`. `results.json` reports on the test split (held-out hearts) and the **real**
split, which holds measured healthy recordings on the same layout and checks for false positives:
- AUROC;
- the detection rate per class at a threshold of the 95th percentile of healthy validation scores;
- for classifiers, the confusion matrix.

## Remote GPUs

- **SkyPilot** (RunPod, Lambda, AWS, GCP, Azure…): `sky launch -c cardiomag remote/skypilot.yaml --env CONFIG=configs/train/ventricular_cnn.yaml`.
- **Any SSH box**: `remote/run_ssh.sh user@host configs/train/ventricular_cnn.yaml`, then `remote/run_ssh.sh user@host --fetch`.
- **Docker**: `docker build -f environment/Dockerfile -t cardiomag .` and `docker run --gpus all -v $PWD/data:/work/data -v $PWD/runs:/work/runs cardiomag cardiomag-train --config ...`.

Generated datasets (`data/`, tens of MB) are uploaded. The raw downloads are not; generate locally first.

## Layout

```
cardiomag/io        meshes (openCARP, VTK, EnSight), heart loaders + body frames, PTB/Kiel MCG
cardiomag/physics   conductivity, lead fields (B = W·Vm), sensor arrays, noise
cardiomag/sim       eikonal solver, action-potential template, healthy/abnormal scenarios
cardiomag/data      dataset generation (shards), PyTorch dataset
cardiomag/ml        model registry + built-ins (cnn1d, resnet1d, conv_ae), trainer, metrics
cardiomag/viz       standalone HTML viewer builder + template
configs/            datasets/*.yaml, train/*.yaml
environment/        setup_env.sh, Dockerfile
remote/             skypilot.yaml, run_ssh.sh
tests/              pytest + headless viewer check (tests/js)
```
