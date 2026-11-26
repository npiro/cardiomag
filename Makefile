PY := .venv/bin/python
.PHONY: env test data train viewer

env:            ## create .venv (CUDA: make env CUDA=12.4)
	bash environment/setup_env.sh $(if $(CUDA),--cuda $(CUDA),)

test:
	$(PY) -m pytest -q

data:           ## generate all datasets
	for c in atrial_ptb49 atrial_kiel36 ventricular_ptb49 ventricular_kiel36; do \
	  $(PY) -m cardiomag.data.generate --config configs/datasets/$$c.yaml; done

train:          ## train every config in configs/train
	for c in configs/train/*.yaml; do $(PY) -m cardiomag.ml.train --config $$c; done

viewer:
	$(PY) -m cardiomag.viz.build_viewer --out viewer/cardiomag_viewer.html
