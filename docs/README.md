# Hate Speech Detection - Technical Documentation

KRIXION is a complete Trust & Safety AI solution that uses 10 different machine learning models to detect Hate Speech and Offensive Language in Text and Images.

## Core Capabilities Added in V2.0

1. **Multimodal OCR (Meme Detection):** Upload an image, and the system uses EasyOCR to extract text and analyze it for hate speech.
2. **Auto-Moderation Engine:** Outputs actionable business logic (e.g., "Flag for Review", "Auto-Delete").
3. **Toxicity Redaction:** Generates a censored, "Safe Filter" version of the text.
4. **Advanced Hinglish Sarcasm:** Rule-based context engine detecting Hindi/Hinglish sarcasm and group targeting.

## Installation

### Prerequisites
- Python 3.13 or higher
- pip package manager
- Internet connection (for initial setup only)

### Step 1: Clone Repository
```bash
cd D:\Projects
git clone <repository-url>HateSpeechDetection
cd HateSpeechDetection
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
- `nicegui` (Web UI framework)
- `pandas`, `numpy` (Data processing)
- `scikit-learn` (Machine learning)
- `transformers`, `torch` (Deep Learning & NLP)
- `easyocr`, `Pillow` (OCR & Image Processing)

### Step 4: Prepare Data & Train Models

**Load Data:**
```bash
python -m src.data.load_data
```

**Train Baseline Models (Logistic Regression, RF, SVC):**
```bash
python -m src.training.train_baseline
```

**Train HinglishBERT (Transformer):**
```bash
python -m src.training.train_hinglish_bert --epochs 2
```

### Step 5: Initialize Database & Run

```bash
python -c "from src.utils.db import init_db; init_db()"
python app.py
```
Access the application at: `http://localhost:10000`

---

## Available Models (10 Total)

| Model | Speed | Best For |
|-------|-------|----------|
| Logistic Regression | ⚡⚡⚡ Fast | Real-time, simple texts |
| Naive Bayes | ⚡⚡⚡ Very Fast | Fastest predictions |
| SVC | ⚡⚡⚡ Fast | Balanced performance |
| Random Forest | ⚡⚡ Medium | Robust predictions |
| CNN | ⚡ Slow | Pattern recognition |
| BiLSTM | ⚡⚡ Medium | Sequential context |
| HECAN | ⚡⚡ Medium | Hierarchical attention |
| DistilBERT | ⚡ Slow | Multilingual text |
| **HinglishBERT** | ⚡ Slow | **Hindi/Hinglish contextual accuracy** |
| Ensemble | ⚡⚡ Medium | Production use |

## Project Structure

```
HateSpeechDetection/
├── app.py                 # Main application entry
├── data/                  # Datasets & app.db SQLite database
├── models/                # Baseline & Transformer models
├── src/
│   ├── api/predict.py     # Core Prediction Router
│   ├── ui/                # NiceGUI interface pages
│   ├── utils/ocr.py       # EasyOCR Engine
│   └── utils/sarcasm_context.py # Sarcasm & Context Rules
└── docs/                  # Documentation
```

## Troubleshooting

- **EasyOCR Not Working:** Ensure `easyocr` and `Pillow` are installed (`pip install easyocr Pillow`).
- **Model Not Found:** Rerun `python -m src.training.train_baseline` or `train_hinglish_bert`.
- **Database Errors:** Delete `data/app.db` and reinitialize.
- **Port in Use:** Change port in `app.py` `ui.run(port=8081)`.
