"""
Baseline model training script for KRIXION Hate Speech Detection.

Trains multiple baseline classifiers:
- Logistic Regression (TF-IDF features)
- Multinomial Naive Bayes (TF-IDF features)
- Support Vector Classifier (TF-IDF features)

Label mapping (consistent with KRIXION brief):
- 0: Normal (non-offensive)
- 1: Offensive
- 2: Hate speech

Usage:
    python -m src.training.train_baseline [--sample N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Wrap scikit-learn imports so we can provide a clear error if the package is missing
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
except Exception as e:
    raise RuntimeError(
        "scikit-learn is required to run this script. Install it with 'pip install scikit-learn' "
        f"or ensure your PYTHONPATH is configured correctly. Original error: {e}"
    ) from e


DATA_PATH = Path('data/clean_data.csv')
MODELS_DIR = Path('models/baseline')
REPORTS_DIR = Path('reports')

# Label mapping per KRIXION brief
LABEL_MAP = {
    0: 'Normal',
    1: 'Offensive',
    2: 'Hate',
}


def load_data(path: Path, sample: int | None = None) -> tuple[list[str], list[int]]:
    """Load cleaned dataset from CSV.
    
    Args:
        path: Path to clean_data.csv.
        sample: Optional number of rows to sample for quick dev.
    
    Returns:
        (texts, labels) where labels are mapped to 0/1/2.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    df = pd.read_csv(path)
    
    if sample is not None and sample > 0:
        df = df.sample(n=min(sample, len(df)), random_state=42)
    
    # Map label column to 0/1/2
    # Assuming current labels are 0 (non-hate) and 1 (hate)
    # For KRIXION brief we need 0=Normal, 1=Offensive, 2=Hate
    # If data already has proper labels, pass through; otherwise map binary to 0 and 2
    def map_label(label: int) -> int:
        # Pass through if already in expected range
        if label in {0, 1, 2}:
            return int(label)
        # If numeric but outside expected range, clamp into [0,2]
        try:
            iv = int(label)
            if iv <= 0:
                return 0
            if iv == 1:
                return 1
            return 2
        except Exception:
            # Non-numeric fallback: treat common tokens
            s = str(label).strip().lower()
            if s in {'hate', 'hateful', 'hostile'}:
                return 2
            if s in {'offensive', 'abusive', 'toxic', 'insult'}:
                return 1
            return 0
    
    df['label'] = df['label'].apply(map_label)
    
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    return texts, labels


def build_pipeline(classifier_name: str) -> Pipeline:
    """Build a TF-IDF + classifier pipeline.
    
    Args:
        classifier_name: One of 'logreg', 'nb', 'svc'.
    
    Returns:
        sklearn Pipeline.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        strip_accents='unicode',
        lowercase=True,
        sublinear_tf=True,
    )
    
    if classifier_name == 'logreg':
        clf = LogisticRegression(max_iter=500, random_state=42, solver='saga', multi_class='multinomial')
    elif classifier_name == 'nb':
        clf = MultinomialNB(alpha=0.1)
    elif classifier_name == 'svc':
        clf = SVC(kernel='linear', probability=True, random_state=42)
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    
    return Pipeline([
        ('tfidf', vectorizer),
        ('clf', clf),
    ])


def train_and_evaluate(
    X_train: list[str],
    y_train: list[int],
    X_val: list[str],
    y_val: list[int],
    X_test: list[str],
    y_test: list[int],
) -> None:
    """Train multiple baseline models and generate reports.
    
    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data (currently unused, available for tuning).
        X_test, y_test: Test data for final evaluation.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    classifiers = {
        'logreg': 'Logistic Regression',
        'nb': 'Multinomial Naive Bayes',
        'svc': 'Support Vector Classifier',
    }
    
    results: Dict[str, Any] = {}
    
    for clf_key, clf_name in classifiers.items():
        print(f"\nTraining {clf_name}...")
        
        pipeline = build_pipeline(clf_key)
        pipeline.fit(X_train, y_train)
        
        # Save model
        model_path = MODELS_DIR / f'{clf_key}.joblib'
        joblib.dump(pipeline, model_path)
        print(f"  Saved to {model_path}")
        
        # Predict on test set
        y_pred = pipeline.predict(X_test)
        
        # Classification report
        report = classification_report(
            y_test,
            y_pred,
            target_names=[LABEL_MAP.get(i, f'Class_{i}') for i in sorted(set(y_test))],
            output_dict=True,
            zero_division=0,
        )
        results[clf_key] = {
            'name': clf_name,
            'report': report,
        }
        
        # Print summary
        print(f"  Accuracy: {report['accuracy']:.3f}")
        print(f"  Macro F1: {report['macro avg']['f1-score']:.3f}")
    
    # Save consolidated classification report
    report_path = REPORTS_DIR / 'classification_report_baseline.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved classification reports to {report_path}")
    
    # Generate confusion matrix for best model (logreg by default)
    best_model_key = 'logreg'
    best_pipeline = joblib.load(MODELS_DIR / f'{best_model_key}.joblib')
    y_pred_best = best_pipeline.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred_best)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    labels_present = sorted(set(y_test))
    label_names = [LABEL_MAP.get(i, f'Class_{i}') for i in labels_present]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.title(f'Confusion Matrix - {classifiers[best_model_key]}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = REPORTS_DIR / 'confusion_matrix_baseline.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Train baseline classifiers for hate speech detection.')
    parser.add_argument('--sample', type=int, default=None, help='Use only N rows for quick dev testing')
    args = parser.parse_args()
    
    print("Loading data...")
    texts, labels = load_data(DATA_PATH, sample=args.sample)
    print(f"Loaded {len(texts)} samples")
    
    # Check label distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("Label distribution:")
    for lbl, cnt in zip(unique, counts):
        print(f"  {LABEL_MAP.get(lbl, f'Class_{lbl}')}: {cnt}")
    
    # If dataset is too small, skip stratification
    min_samples_needed = 10  # Minimum for meaningful split
    if len(texts) < min_samples_needed:
        print(f"\nWarning: Only {len(texts)} samples available. Need at least {min_samples_needed} for training.")
        print("Please load more data into data/raw/ and run: python -m src.data.load_data")
        return
    
    # Stratified split: 70% train, 15% val, 15% test
    # If too small for stratification, use simple split
    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=0.15, random_state=42, stratify=labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=(0.15 / 0.85), random_state=42, stratify=y_temp
        )
    except ValueError:
        # Fallback to non-stratified split for small datasets
        print("\nWarning: Dataset too small for stratified split. Using simple random split.")
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=0.15, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=(0.15 / 0.85), random_state=42
        )
    
    print(f"\nSplit sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    # Train and evaluate
    train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test)
    
    print("\n✓ Baseline training complete!")


if __name__ == '__main__':
    main()
