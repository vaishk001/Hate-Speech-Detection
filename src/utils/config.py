# src/utils/config.py
from pathlib import Path

ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

CLEAN_CSV = DATA_DIR / "clean_data.csv"
DB_PATH = DATA_DIR / "app.db"

TRANSFORMER_LOCAL = MODELS_DIR / "transformer" / "distilbert_local"
DISTILBERT_LOCAL = MODELS_DIR / "transformer" / "distilbert_local"
INDICBERT_LOCAL = MODELS_DIR / "transformer" / "indicbert_local"
BASELINE_DIR = MODELS_DIR / "baseline"
EMBED_CACHE = MODELS_DIR / "transformer" / "embeddings.joblib"
DISTILBERT_SKLEARN = BASELINE_DIR / "logreg_distilbert.joblib"
