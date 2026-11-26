"""Where the raw data and caches live. Override with environment variables.

CARDIOMAG_DATA   root holding the downloaded datasets (default: ../cardiac-mcg-data next to the repo)
CARDIOMAG_CACHE  parsed-mesh / lead-field cache (default: <repo>/.cache)
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("CARDIOMAG_DATA", REPO.parent / "cardiac-mcg-data")).expanduser()
CACHE = Path(os.environ.get("CARDIOMAG_CACHE", REPO / ".cache")).expanduser()

BIATRIAL = DATA / "zenodo_17143584_biatrial" / "atria_models_zenodo"
STROCCHI_HEALTHY = DATA / "zenodo_4590294_strocchi_healthy"
STROCCHI_HF = DATA / "zenodo_3890034_strocchi_hf"
PTB = DATA / "ptb_koch2011"
KIEL = DATA / "kiel_cardio" / "data"


def cache_dir(*parts: str) -> Path:
    p = CACHE.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
