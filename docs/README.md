# KRIXION Hate Speech Detection - Documentation

## Installation

### Prerequisites

- Python 3.13 or higher
- pip package manager
- Internet connection (for initial setup only)

### Step 1: Clone Repository

```bash
cd D:\Projects
git clone <repository-url> KRIXION_HateSpeechDetection
cd KRIXION_HateSpeechDetection
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:

- nicegui (Web UI framework)
- pandas (Data processing)
- numpy (Numerical operations)
- scikit-learn (Machine learning)
- joblib (Model serialization)
- matplotlib (Visualization)
- seaborn (Statistical plots)
- transformers (Hugging Face transformers)
- torch (PyTorch)

### Step 4: Download Transformer Model (Online)

```bash
python -m scripts.download_transformer
```

This downloads DistilBERT to `models/transformer/distilbert_local` for offline use.

### Step 5: Prepare Data

```bash
python -m src.data.load_data
```

Loads and preprocesses data from `data/raw/*.csv` into `data/clean_data.csv`.

### Step 6: Train Models

**Baseline Model (TF-IDF + Logistic Regression):**

```bash
python -m src.training.train_baseline
```

**DistilBERT Model (DistilBERT embeddings + Logistic Regression):**

```bash
python src/training/train_distilbert.py --batch_size 16 --max_length 128
```

**Alternative Transformer Model:**

```bash
python -m src.training.train_transformer_embeddings --sample 2000 --batch_size 8
```

### Step 7: Initialize Database

```bash
python -c "from src.utils.db import init_db; init_db()"
```

## Running the Application

### Start Web Server

```bash
python app.py
```

Access the application at: `http://localhost:8080`

### Available Routes

- `/` - Home page (text classification)
- `/history` - Prediction history
- `/analytics` - Analytics dashboard
- `/admin` - Admin panel

## Offline Testing

### Preflight Check

Verify all components are ready:

```bash
python scripts/preflight_check.py
```

Checks:

- Baseline model exists
- Transformer model exists
- Database initialized
- Tokenizer loads successfully

### Test Prediction (CLI)

```bash
python -c "from src.api.predict import predict_text; print(predict_text('You are an idiot'))"
```

Expected output:

```python
{'label': 1, 'score': 0.78, 'model_name': 'baseline-logreg', 'latency_ms': 150, 'lang': 'en'}
```

### Test Data Loading

```bash
python -m src.data.load_data
```

### Test Training (Small Sample)

```bash
python -m src.training.train_baseline
python -m src.training.train_transformer_embeddings --sample 100 --batch_size 4
```

## Project Structure

```
KRIXION_HateSpeechDetection/
├── app.py                 # Main application entry
├── data/
│   ├── raw/              # Raw CSV datasets
│   ├── clean_data.csv    # Preprocessed data
│   └── app.db           # SQLite database
├── models/
│   ├── baseline/        # Baseline models (.joblib)
│   └── transformer/     # Transformer models
├── reports/             # Training reports & metrics
├── src/
│   ├── api/            # Prediction API
│   ├── data/           # Data loading & preprocessing
│   ├── models/         # Model definitions
│   ├── training/       # Training scripts
│   ├── ui/             # NiceGUI pages
│   └── utils/          # Database & utilities
├── scripts/            # Helper scripts
└── docs/              # Documentation
```

## Troubleshooting

### Import Errors

Ensure virtual environment is activated:

```bash
.venv\Scripts\activate
```

### Model Not Found

Run training scripts:

```bash
python -m src.training.train_baseline
```

### Database Errors

Reinitialize database:

```bash
python -c "from src.utils.db import init_db; init_db()"
```

### Port Already in Use

Change port in `app.py`:

```python
ui.run(host='0.0.0.0', port=8081)
```

## Available Models (9 Total)

| Model | Speed | Best For |
|-------|-------|----------|
| Logistic Regression | ⚡⚡⚡ Fast | Real-time, simple texts |
| Naive Bayes | ⚡⚡⚡ Very Fast | Fastest predictions |
| SVC | ⚡⚡⚡ Fast | Balanced performance |
| Random Forest | ⚡⚡ Medium | Robust predictions |
| CNN | ⚡ Slow | Pattern recognition |
| BiLSTM | ⚡⚡ Medium | Sequential context |
| HECAN | ⚡⚡ Medium | Hierarchical attention |
| **DistilBERT** | ⚡ Slow | **Multilingual, highest accuracy** |
| Ensemble | ⚡⚡ Medium | Production use |

## Check Model Status

```bash
# Check all models
python scripts/check_all_models.py

# Check DistilBERT
python scripts/check_distilbert.py
```

## Performance Notes

- Baseline models: ~5-100ms latency
- Deep learning: ~30-4000ms latency
- DistilBERT: ~100-200ms latency (CPU)
- Database: SQLite (suitable for <10K predictions/day)
- Web server: Supports concurrent users via async handlers
