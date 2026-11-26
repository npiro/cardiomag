# GPU training image. Build from the repo root:
#   docker build -f environment/Dockerfile -t cardiomag:latest .
# Run (datasets mounted read-only, runs written back):
#   docker run --gpus all -v $PWD/data:/work/data:ro -v $PWD/runs:/work/runs cardiomag:latest \
#       cardiomag-train --config configs/train/ventricular_cnn.yaml
# Pick a base image whose CUDA version your provider's driver supports.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /work
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml README.md ./
COPY cardiomag ./cardiomag
RUN pip install -e ".[dev]"
COPY configs ./configs
COPY tests ./tests
CMD ["cardiomag-train", "--help"]
