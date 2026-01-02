"""
Baseline model for KRIXION Hate Speech Detection.

Provides a simple TF-IDF + Logistic Regression pipeline for text classification.

Usage:
    from src.models.baseline import build_tfidf_logreg_pipeline, train_and_save, load_model, predict_text
    
    # Train
    pipeline = build_tfidf_logreg_pipeline(max_features=10000)
    train_and_save(X_train, y_train, 'models/baseline/logreg.pkl')
    
    # Predict
    model = load_model('models/baseline/logreg.pkl')
    predictions = predict_text(model, ['sample text', 'another text'])
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING, Any

try:
    import joblib  # type: ignore
except Exception:
    # Fallback to scikit-learn's vendored joblib for older sklearn versions,
    # otherwise raise a clear error that joblib must be installed.
    try:
        from sklearn.externals import joblib  # type: ignore
    except Exception as e:
        raise ImportError("joblib is required; install it with 'pip install joblib'") from e

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
else:
    Pipeline = Any


def build_tfidf_logreg_pipeline(max_features: int = 10000) -> Pipeline:
    """Build a TF-IDF + Logistic Regression pipeline.
    
    Args:
        max_features: Maximum number of TF-IDF features to extract.
    
    Returns:
        sklearn Pipeline ready for fitting.
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            strip_accents='unicode',
            lowercase=True,
            sublinear_tf=True,
        )),
        ('clf', LogisticRegression(
            max_iter=200,
            solver='lbfgs',
            random_state=42,
            class_weight='balanced',
        )),
    ])


def train_and_save(X: List[str], y: List[int], out_path: str) -> None:
    """Train a baseline pipeline and save to disk.
    
    Args:
        X: List of text samples.
        y: List of integer labels (0 or 1).
        out_path: Path to save the trained model (joblib format).
    """
    if not X or not y:
        raise ValueError("X and y must be non-empty lists")
    if len(X) != len(y):
        raise ValueError(f"X and y must have same length, got {len(X)} vs {len(y)}")
    
    pipeline = build_tfidf_logreg_pipeline()
    pipeline.fit(X, y)
    
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_p)


def load_model(path: str) -> Pipeline:
    """Load a trained baseline pipeline from disk.
    
    Args:
        path: Path to the saved model file.
    
    Returns:
        Loaded sklearn Pipeline.
    """
    return joblib.load(path)


def predict_text(model: Pipeline, texts: List[str]) -> List[Tuple[int, float]]:
    """Predict labels and confidence scores for a list of texts.
    
    Args:
        model: Trained sklearn Pipeline.
        texts: List of text strings to classify.
    
    Returns:
        List of (label, confidence) tuples. Label is 0 or 1,
        confidence is the probability of the predicted class.
    """
    if not texts:
        return []
    
    predictions = model.predict(texts)
    probas = model.predict_proba(texts)
    
    results = []
    for pred, proba in zip(predictions, probas):
        # confidence = probability of the predicted class
        confidence = proba[pred]
        results.append((int(pred), float(confidence)))
    
    return results


if __name__ == '__main__':
    # Smoke test
    X_sample = [
        'this is a normal message',
        'you are terrible and hateful',
        'have a nice day',
        'i hate you so much',
    ]
    y_sample = [0, 1, 0, 1]
    
    print("Building and training baseline pipeline...")
    pipeline = build_tfidf_logreg_pipeline(max_features=100)
    pipeline.fit(X_sample, y_sample)
    
    print("Testing predictions...")
    test_texts = ['nice message', 'hate speech example']
    preds = predict_text(pipeline, test_texts)
    
    for text, (label, conf) in zip(test_texts, preds):
        print(f"  '{text}' -> label={label}, confidence={conf:.3f}")
    
    print("Baseline model smoke test complete.")
