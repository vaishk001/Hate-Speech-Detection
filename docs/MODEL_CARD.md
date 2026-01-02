# Model Card - KRIXION Hate Speech Detection

## Model Overview

This project uses **9 classification models** for hate speech detection:

### 1. Logistic Regression (Baseline)
**File:** `models/baseline/logreg.joblib` | **Speed:** ⚡⚡⚡ Fast (50-100ms)
- TF-IDF + LogisticRegression
- Best for: Real-time, simple texts

### 2. Naive Bayes
**File:** `models/baseline/nb.joblib` | **Speed:** ⚡⚡⚡ Very Fast (5-10ms)
- TF-IDF + MultinomialNB
- Best for: Fastest predictions, probabilistic outputs

### 3. SVC (Support Vector Classifier)
**File:** `models/baseline/svc.joblib` | **Speed:** ⚡⚡⚡ Fast (5-10ms)
- TF-IDF + LinearSVC
- Best for: Balanced speed/accuracy

### 4. Random Forest
**File:** `models/baseline/rf.joblib` | **Speed:** ⚡⚡ Medium (100-150ms)
- TF-IDF + RandomForestClassifier
- Best for: Robust predictions, feature importance

### 5. CNN (Convolutional Neural Network)
**File:** `models/baseline/cnn.pth` | **Speed:** ⚡ Slow (3000-4000ms)
- Deep learning model for pattern recognition
- Best for: Local feature patterns

### 6. BiLSTM (Bidirectional LSTM)
**File:** `models/baseline/bilstm.pth` | **Speed:** ⚡⚡ Medium (30-50ms)
- Sequential model with bidirectional context
- Best for: Long-range dependencies

### 7. HECAN (Hierarchical Attention)
**File:** `models/baseline/hecan.pth` | **Speed:** ⚡⚡ Medium (30-50ms)
- Hierarchical attention network
- Best for: Word and sentence level analysis

### 8. DistilBERT (Transformer)
**File:** `models/baseline/logreg_distilbert.joblib` | **Speed:** ⚡ Slow (100-200ms)
- DistilBERT embeddings + LogisticRegression
- Best for: **Multilingual (English/Hindi/Hinglish), highest accuracy**
- Model: `distilbert-base-multilingual-cased`
- Size: ~500 MB (transformer) + ~5 MB (classifier)

### 9. Ensemble
**Virtual Model** | **Speed:** ⚡⚡ Medium (50-100ms)
- Combines predictions from multiple models
- Best for: Production use, balanced accuracy

**Training Data:**

- Dataset: Combined hate speech datasets
- Total samples: 68 (after preprocessing)
- Languages: English (en), Hindi (hi), Hinglish (hi-en)
- Split: 85% train, 15% test

**Performance (on test set):**

- Accuracy: 0.636
- Macro F1: 0.389
- Latency: 50-100ms (CPU)

**Classes:**

- 0: Normal (non-offensive)
- 1: Offensive (mild hate speech)
- 2: Hate (severe hate speech)

## Quick Model Selection Guide

| Use Case | Recommended Model |
|----------|------------------|
| Real-time classification | Logistic Regression, Naive Bayes, SVC |
| Multilingual text | **DistilBERT** |
| Complex/ambiguous text | DistilBERT, Ensemble |
| Production deployment | Ensemble |
| Fastest predictions | Naive Bayes |
| Highest accuracy | **DistilBERT** |

## Check Model Status

```bash
# Check all 9 models
python scripts/check_all_models.py

# Check DistilBERT only
python scripts/check_distilbert.py
```

## Training Procedure

### Baseline Model

```bash
python -m src.training.train_baseline
```

**Steps:**

1. Load preprocessed data from `data/clean_data.csv`
2. Split into train/test (85/15)
3. Train TF-IDF vectorizer on training text
4. Train LogisticRegression, MultinomialNB, LinearSVC
5. Evaluate on test set
6. Save best model (LogisticRegression) to `models/baseline/logreg.joblib`
7. Generate reports: classification_report.json, confusion_matrix.png

### Deep Learning Models (CNN, BiLSTM, HECAN)

```bash
python -m src.training.train_deep_models
```

### DistilBERT Model

```bash
# Step 1: Download model (requires internet)
python scripts/download_transformer.py

# Step 2: Train classifier
python src/training/train_distilbert.py --batch_size 16 --max_length 128
```

**Steps:**
1. Download `distilbert-base-multilingual-cased` from Hugging Face
2. Generate mean-pooled embeddings (768 dimensions)
3. Train LogisticRegression on embeddings
4. Save to `models/baseline/logreg_distilbert.joblib`
5. Generate reports: classification_report_distilbert.json, confusion_matrix_distilbert.png

## Model Selection in UI

In the **Classify** page, select from dropdown:
- `logistic_regression` (default)
- `naive_bayes`
- `svc`
- `random_forest`
- `cnn`
- `bilstm`
- `hecan`
- `distilbert` ⭐ NEW
- `ensemble`

## Programmatic Usage

```python
from src.api.predict import predict_text_with_model

# Use specific model
result = predict_text_with_model("Your text", model_name="distilbert")
print(result)  # {'label': 0, 'score': 0.95, 'model_name': 'distilbert', ...}
```

## Admin Panel

### Access Routes

- **Public Admin:** `http://localhost:8080/admin`
- **Secure Admin (Hidden):** `http://localhost:8080/krixion-admin-secure`

### Features

**Dataset Upload**
- Upload CSV files (format: text,label)
- Label values: 0 (Normal), 1 (Offensive), 2 (Hate)
- Saved to: `data/batch_upload.csv`

**Model Training**
- Trigger training from UI
- Runs: `python -m src.training.train_transformer_embeddings`
- Real-time status updates

**Log Viewer**
- View last 100 lines of `logs/app.log`
- Real-time updates
- Refresh functionality

### Usage

1. Start app: `python app.py`
2. Navigate to `/admin` or `/krixion-admin-secure`
3. Upload dataset or trigger training
4. Monitor progress in logs

## Limitations

1. **Small Training Set:** Only 68 samples (insufficient for production)
2. **Class Imbalance:** 25 Normal, 43 Offensive/Hate
3. **Language Coverage:** Primarily English, limited Hindi/Hinglish
4. **CPU-Only:** No GPU optimization
5. **Context:** Single-sentence classification (no conversation context)

## Ethical Considerations

- **Bias:** Models trained on limited data may exhibit bias
- **False Positives:** May flag legitimate content as hate speech
- **False Negatives:** May miss subtle or coded hate speech
- **Use Case:** Designed for content moderation assistance, not automated decisions
- **Human Review:** Always recommended for borderline cases

## License

- **Baseline Model:** Custom (project-specific)
- **DistilBERT:** Apache 2.0
- **Training Data:** See DATA_SOURCES.md

## Version History

- **v1.1** (Current): DistilBERT Integration
  - Added DistilBERT model support
  - Total models: 9 (8 trained + 1 ensemble)
  - Model status checker scripts
  - Updated UI with model selection
  
- **v1.0** (2025-11-28): Initial release
  - Baseline models: LogisticRegression, NaiveBayes, SVC, RandomForest
  - Deep learning: CNN, BiLSTM, HECAN
  - Ensemble model
  - 3-class classification (Normal/Offensive/Hate)
  - Admin panel with dataset upload and training control

## Contact

For questions or issues, contact: [Your Name/Organization]
