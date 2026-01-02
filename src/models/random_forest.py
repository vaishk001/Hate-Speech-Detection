# src/models/random_forest.py
from pathlib import Path
from typing import List, Tuple
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def build_rf_pipeline(n_estimators: int = 100, max_features: int = 10000) -> Pipeline:
    """Build TF-IDF + Random Forest pipeline."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            strip_accents='unicode',
            lowercase=True,
            sublinear_tf=True,
        )),
        ('clf', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
        )),
    ])


def train_and_save(X: List[str], y: List[int], out_path: str) -> None:
    """Train Random Forest and save to disk."""
    pipeline = build_rf_pipeline()
    pipeline.fit(X, y)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)


def load_model(path: str) -> Pipeline:
    """Load trained Random Forest pipeline."""
    return joblib.load(path)


def predict_text(model: Pipeline, texts: List[str]) -> List[Tuple[int, float]]:
    """Predict labels and confidence scores."""
    if not texts:
        return []
    predictions = model.predict(texts)
    probas = model.predict_proba(texts)
    results = []
    for pred, proba in zip(predictions, probas):
        confidence = proba[int(pred)]
        results.append((int(pred), float(confidence)))
    return results
