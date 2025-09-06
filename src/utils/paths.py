import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

def ensure_dirs():
    for d in [DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, EXPERIMENTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
