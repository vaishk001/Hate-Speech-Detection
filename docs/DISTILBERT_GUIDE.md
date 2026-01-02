# DistilBERT Model Guide

## Overview
DistilBERT provides the **highest accuracy** for multilingual hate speech detection (English, Hindi, Hinglish).

## Setup

### 1. Download DistilBERT Model
Run the download script while online to fetch the model files:

```bash
python scripts/download_transformer.py
```

This will download `distilbert-base-multilingual-cased` to `models/transformer/distilbert_local/`

### 2. Train DistilBERT Classifier
After downloading the model, train the classifier:

```bash
python src/training/train_distilbert.py
```

Optional arguments:
- `--sample N`: Use only N samples for quick testing
- `--batch_size N`: Batch size for embedding generation (default: 16)
- `--max_length N`: Maximum sequence length (default: 128)

Example:
```bash
python src/training/train_distilbert.py --batch_size 32 --max_length 128
```

## Usage

### In the UI
1. Navigate to the **Classify** page
2. Select **distilbert** from the model dropdown
3. Enter your text and click **CLASSIFY**

### Programmatically
```python
from src.api.predict import predict_text_with_model

result = predict_text_with_model("Your text here", model_name="distilbert")
print(result)
# Output: {'label': 0, 'score': 0.95, 'model_name': 'distilbert', 'latency_ms': 120, 'lang': 'en'}
```

## Model Files
- **Transformer Model**: `models/transformer/distilbert_local/`
- **Trained Classifier**: `models/baseline/logreg_distilbert.joblib`
- **Classification Report**: `reports/classification_report_distilbert.json`
- **Confusion Matrix**: `reports/confusion_matrix_distilbert.png`

## Performance
DistilBERT typically provides:
- Better accuracy on complex/ambiguous texts
- Improved multilingual understanding
- Slightly higher latency (~100-200ms) compared to baseline models
- Better handling of code-mixed (Hinglish) text

## Troubleshooting

### Model Not Found Error
If you see "DistilBERT model not found", ensure:
1. You've run `python scripts/download_transformer.py`
2. The directory `models/transformer/distilbert_local/` exists and contains model files
3. You've trained the classifier with `python src/training/train_distilbert.py`

### Out of Memory
If training fails with OOM:
- Reduce batch size: `--batch_size 8`
- Use sampling: `--sample 5000`
- Reduce max length: `--max_length 64`

### Slow Inference
For faster inference:
- Ensure you're using CPU mode (default)
- Consider using ensemble or baseline models for real-time applications
- DistilBERT is best for batch processing or when accuracy is critical

## When to Use DistilBERT

✅ **Use DistilBERT when:**
- Text is multilingual (English/Hindi/Hinglish)
- Sentences are complex or ambiguous
- Highest accuracy is needed
- Can afford ~100-200ms latency

❌ **Don't use DistilBERT when:**
- Need real-time predictions (<10ms)
- Text is simple English only
- Running on resource-constrained devices
