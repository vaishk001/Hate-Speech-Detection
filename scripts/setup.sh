#!/bin/bash
# scripts/setup.sh - Setup project environment

set -e

echo "Setting up KRIXION Hate Speech Detection..."

# Create directories
mkdir -p data/raw models/baseline models/transformer reports logs

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Initialize database
echo "Initializing database..."
python - <<PYTHON
from src.utils.db import init_db
init_db()
print("✓ Database initialized")
PYTHON

# Run preflight checks
echo ""
echo "Running preflight checks..."
python scripts/preflight_check.py

echo ""
echo "✓ Setup complete!"
echo "To activate the environment, run: source .venv/bin/activate"
echo "To start the app, run: python app.py"
