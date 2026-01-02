# src/training/evaluate.py
import json
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.utils.config import CLEAN_CSV, REPORTS_DIR, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {0: 'Normal', 1: 'Offensive', 2: 'Hate'}
LATENCY_REPORT_PATH = REPORTS_DIR / 'latency_report.json'


def evaluate_model(model_path: str, X_test: List[str], y_test: List[int], model_name: str = 'baseline') -> dict:
    """Evaluate a trained model and generate reports.
    
    Args:
        model_path: Path to saved model (.joblib).
        X_test: Test texts.
        y_test: Test labels.
        model_name: Name for report files.
    
    Returns:
        Dictionary with report and confusion matrix.
    """
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # Save JSON report
    report_path = REPORTS_DIR / f'classification_report_{model_name}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved classification report to {report_path}")
    
    # Save confusion matrix PNG
    plt.figure(figsize=(8, 6))
    labels_present = sorted(set(y_test))
    label_names = [LABEL_MAP.get(i, f'Class_{i}') for i in labels_present]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = REPORTS_DIR / f'confusion_matrix_{model_name}.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {cm_path}")
    
    return {'report': report, 'confusion_matrix': cm.tolist()}


def measure_latency(model, X_test: List[str], samples: int = 10) -> Dict[str, float]:
    """Measure inference latency for a model."""
    test_samples = X_test[:samples]
    times = []
    
    for text in test_samples:
        start = time.time()
        model.predict([text])
        times.append((time.time() - start) * 1000)  # Convert to ms
    
    return {
        'min': min(times),
        'max': max(times),
        'mean': np.mean(times),
        'p95': np.percentile(times, 95)
    }


def generate_latency_report(X_test: List[str]) -> None:
    """Generate latency report for all available models."""
    report = {
        'metadata': {
            'generated_at': pd.Timestamp.now().isoformat(),
            'test_samples': len(X_test),
            'hardware': 'CPU'
        },
        'models': {}
    }
    
    # Test baseline models
    for model_name in ['logreg', 'nb', 'svc']:
        model_path = MODELS_DIR / 'baseline' / f'{model_name}.joblib'
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                latency = measure_latency(model, X_test)
                report['models'][f'baseline_{model_name}'] = {
                    'type': 'baseline',
                    'inference_ms': latency
                }
                logger.info(f"Measured latency for {model_name}: {latency['mean']:.2f}ms")
            except Exception as e:
                logger.warning(f"Could not measure latency for {model_name}: {e}")
    
    # Save report
    with open(LATENCY_REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Latency report saved to {LATENCY_REPORT_PATH}")


def evaluate_baseline(sample: int = None) -> None:
    """Evaluate baseline model on clean data."""
    if not CLEAN_CSV.exists():
        logger.error(f"No clean_data.csv found at {CLEAN_CSV}")
        return
    
    df = pd.read_csv(CLEAN_CSV)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)
    
    X_test = df['text'].astype(str).tolist()
    y_test = df['label'].astype(int).tolist()
    
    model_path = MODELS_DIR / 'baseline' / 'logreg.joblib'
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Train first with: python -m src.training.train_baseline")
        return
    
    evaluate_model(str(model_path), X_test, y_test, 'baseline')
    generate_latency_report(X_test)
    logger.info("Baseline evaluation complete")


if __name__ == "__main__":
    evaluate_baseline()
