#!/bin/bash
# scripts/train.sh - Train hate speech detection models

set -e

MODEL=${1:-baseline}
SAMPLE=${2:-}
EPOCHS=${3:-10}

echo "Training $MODEL model..."

if [ "$MODEL" = "baseline" ]; then
    python -m src.training.train_baseline ${SAMPLE:+--sample $SAMPLE}
elif [ "$MODEL" = "bilstm" ]; then
    python -m src.training.train_deep --model bilstm ${SAMPLE:+--sample $SAMPLE} --epochs $EPOCHS
elif [ "$MODEL" = "textcnn" ]; then
    python -m src.training.train_deep --model textcnn ${SAMPLE:+--sample $SAMPLE} --epochs $EPOCHS
elif [ "$MODEL" = "hecan" ]; then
    python -m src.training.train_deep --model hecan ${SAMPLE:+--sample $SAMPLE} --epochs $EPOCHS
elif [ "$MODEL" = "all" ]; then
    echo "Training all models..."
    python -m src.training.train_baseline ${SAMPLE:+--sample $SAMPLE}
    python -m src.training.train_deep --model bilstm ${SAMPLE:+--sample $SAMPLE} --epochs $EPOCHS
    python -m src.training.train_deep --model textcnn ${SAMPLE:+--sample $SAMPLE} --epochs $EPOCHS
    python -m src.training.train_deep --model hecan ${SAMPLE:+--sample $SAMPLE} --epochs $EPOCHS
else
    echo "Usage: $0 {baseline|bilstm|textcnn|hecan|all} [sample_size] [epochs]"
    exit 1
fi

echo "✓ Training complete"
