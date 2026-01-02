#!/bin/bash
# scripts/predict.sh - Run inference on text

set -e

MODEL=${1:-baseline}
TEXT=${2:-}

if [ -z "$TEXT" ]; then
    echo "Usage: $0 {baseline|bilstm|textcnn|hecan} \"text to classify\""
    exit 1
fi

python - <<PYTHON
from src.api.predict import predict_text
result = predict_text("$TEXT", model_type="$MODEL")
print(f"Model: $MODEL")
print(f"Text: $TEXT")
print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.3f}")
PYTHON
