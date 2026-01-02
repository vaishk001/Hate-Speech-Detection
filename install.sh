#!/usr/bin/env bash
set -euo pipefail

echo "========================================="
echo "  KRIXION Hate Speech Detection - Setup"
echo "========================================="

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
VENDOR_DIR="$ROOT_DIR/vendor"
MODELS_TRANSFORMER_DIR="$ROOT_DIR/models/transformer/distilbert_local"
SCRIPTS_DIR="$ROOT_DIR/scripts"

# 1) ensure python3 available
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found on PATH. Install Python 3.10+ and re-run."
  exit 2
fi

# 2) create virtual env if missing
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment at $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
else
  echo "Virtual environment already exists at $VENV_DIR"
fi

# 3) activate venv
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# 4) install dependencies (offline-first)
if [ -d "$VENDOR_DIR" ] && compgen -G "$VENDOR_DIR/*.whl" > /dev/null; then
  echo "Installing dependencies from vendor/ (offline mode)..."
  python -m pip install --no-index --find-links "$VENDOR_DIR" -r requirements.txt
else
  echo "vendor/ not found or empty — installing from PyPI (requires internet)..."
  python -m pip install -r requirements.txt
fi

# 5) check & download transformer model (only if download script exists and model missing)
if [ ! -d "$MODELS_TRANSFORMER_DIR" ]; then
  if [ -f "$SCRIPTS_DIR/download_transformer.py" ]; then
    echo "Transformer model not found locally — attempting to download (this may require internet)..."
    python "$SCRIPTS_DIR/download_transformer.py"
  else
    echo "Transformer model missing and download script not found."
    echo "Place a transformer model under models/transformer/distilbert_local or add scripts/download_transformer.py"
  fi
else
  echo "Transformer model present at $MODELS_TRANSFORMER_DIR"
fi

# 6) initialize local SQLite DB
echo "Initializing local SQLite DB (data/app.db)..."
python - <<PY
from src.utils.db import init_db
init_db()
print("DB initialized -> data/app.db")
PY

# 7) run preflight check if present
if [ -f "$SCRIPTS_DIR/preflight_check.py" ]; then
  echo "Running preflight check..."
  python "$SCRIPTS_DIR/preflight_check.py"
else
  echo "Preflight check script not found at scripts/preflight_check.py — skipping."
fi

echo
echo "========================================="
echo "  INSTALLATION COMPLETE ✔"
echo
echo "To run the app now (Linux/macOS):"
echo "  source .venv/bin/activate"
echo "  python app.py"
echo
echo "If you want to run in background (nohup):"
echo "  nohup python app.py &"
echo
echo "Notes:"
echo " - For fully offline operation ensure vendor/ contains wheel files and"
echo "   models/transformer/distilbert_local exists (or the download script can fetch it)."
echo " - If transformer download fails due to Hugging Face gating, obtain the model manually"
echo "   and place it at models/transformer/distilbert_local/"
echo "========================================="

exit 0
