# 🛡️ KRIXION — Advanced Multimodal Hate Speech Detection & Moderation System

> A professional Trust & Safety platform built with NiceGUI, Scikit-learn, PyTorch, DistilBERT, and EasyOCR.

---

## 📌 Project Overview

**KRIXION** is a full-stack, multimodal hate speech detection and moderation system designed for real-world platforms. It classifies text and images into three categories:

| Label | Class | Description | Moderation Action |
|-------|-------|-------------|-------------------|
| `0` | ✅ Normal | Non-offensive, neutral, or positive content | Allow Content |
| `1` | ⚠️ Offensive | Personal attacks, insults, mild aggression, sarcasm | Flag for Human Review |
| `2` | 🚫 Hate | Violent threats, discrimination, dehumanization | Auto-Delete & Warn |

### 🌟 Key Standout Features

1. **Multilingual & Hinglish Support:** Native support for English, Hindi, and Hinglish (code-mixed) using a fine-tuned HinglishBERT model and custom-curated datasets.
2. **Auto-Moderation & Toxicity Redaction Engine:** Simulates a real-world Trust & Safety pipeline. It automatically recommends a moderation action (Allow, Flag, Delete) and generates a "Safe Filter" redacted version of the text by censoring offensive words.
3. **Multimodal OCR (Meme Analysis):** Hate speech often hides in memes and screenshots. KRIXION features an EasyOCR-powered image upload pipeline that extracts text from images and runs it through the hate speech classifiers.
4. **Sarcasm & Context Analysis Engine:** A sophisticated rule-based engine that detects sarcasm (e.g., "waah bhai kya baat hai"), group targeting ("in logo ko"), and self-deprecation to dynamically adjust the severity score, reducing false positives.

---

## 🤖 Models — 10 Classification Architecture

KRIXION utilizes a 10-model architecture, allowing the system to balance inference speed with accuracy depending on the use case.

| # | Model | Type | Speed | Best For |
|---|-------|------|-------|----------|
| 1 | **Logistic Regression** | TF-IDF + LogReg | ⚡⚡⚡ Fast (1ms) | Real-time, simple texts |
| 2 | **Naive Bayes** | TF-IDF + MultinomialNB | ⚡⚡⚡ Very Fast (5ms) | Fastest baseline predictions |
| 3 | **SVC** | TF-IDF + LinearSVC | ⚡⚡⚡ Fast (5ms) | Balanced speed/accuracy |
| 4 | **Random Forest** | TF-IDF + RF | ⚡⚡ Medium (100ms) | Robust predictions |
| 5 | **CNN (TextCNN)** | Deep Learning | ⚡ Slow (3000ms) | Local pattern recognition |
| 6 | **BiLSTM** | Deep Learning | ⚡⚡ Medium (30ms) | Sequential/long-range context |
| 7 | **HECAN** | Hierarchical Attention | ⚡⚡ Medium (30ms) | Word & sentence level analysis |
| 8 | **DistilBERT** | Transformer | ⚡ Slow (150ms) | Multilingual text |
| 9 | **HinglishBERT** ⭐ | Transformer | ⚡ Slow (150ms) | **Hindi/Hinglish accuracy** |
| 10 | **Ensemble** | Virtual (combined) | ⚡⚡ Medium (50ms) | Production use |

---

## 🔬 Deep Dive: Trust & Safety Engines

### 1. Auto-Moderation & Redaction
Instead of just returning a probability score, KRIXION outputs business logic:
- **Redaction:** Identifies words like "stupid", "idiot", or severe slurs and censors them (e.g., `s*****`).
- **Action Recommendation:** If hate density is high, recommends "Auto-Delete". If context risk is medium, recommends "Flag for Review".

### 2. Sarcasm & Context Analysis
A rule-based layer running **after** every model prediction to refine labels based on linguistic signals:
- **Hinglish Sarcasm:** Catches phrases like *"waah bhai"*, *"bahut badhiya"* used maliciously.
- **Group Targeting Context:** Phrases like *"in logo ko"* or *"their kind"* automatically elevate an Offensive rating to a Hate rating.
- **Humour / Self-Deprecation:** Downgrades the risk if the user is joking or insulting themselves.

### 3. Image-Based Meme Analysis (OCR)
Powered by EasyOCR, the Classify Dashboard allows drag-and-drop of images. The system extracts Hindi and English text from the pixels and feeds it into the NLP pipeline, solving the critical issue of text-filter evasion via screenshots.

---

## 📊 Dataset & Training

Our dataset has been heavily customized to handle edge cases that typical models fail at:
- **Total Samples:** ~684 (Cleaned, balanced)
- **Data Sources:** 
  - `dataset_english_large.csv`: Core English hate/offensive text.
  - `dataset_hindi_hinglish.csv`: Curated Hinglish phrases and code-mixed insults.
  - `dataset_sarcasm_nuanced.csv`: Edge cases involving irony and contextual abuse.

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Web UI** | NiceGUI (Reactive Python UI) |
| **ML Baseline** | Scikit-learn |
| **Deep Learning** | PyTorch |
| **Transformers** | HuggingFace (DistilBERT, HingBERT) |
| **OCR / Vision**| EasyOCR, Pillow |
| **Database** | SQLite (`data/app.db`) |

---

## 🚀 Installation & Quick Start

### 1. Clone & Setup Environment
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train Models & Prepare Data
```bash
python -m src.data.load_data
python -m src.training.train_hinglish_bert --epochs 2
python -m src.training.train_baseline
```

### 3. Initialize Database & Run
```bash
python -c "from src.utils.db import init_db; init_db()"
python app.py
```
Access the application at `http://localhost:10000` (or the port specified in your console).

---

## 🔒 Ethical Considerations
- **Bias:** Models trained on limited data may exhibit systematic bias.
- **Context:** While the Sarcasm engine handles many edge cases, highly nuanced human context can still be misinterpreted.
- **Intended Use:** This tool is designed to *assist* human moderators via the Auto-Moderation Engine, not replace them entirely.

---
*Last Updated: May 2026 | Version: 2.0 (Multimodal OCR & Auto-Moderation)*
