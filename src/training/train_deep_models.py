#!/usr/bin/env python3
"""
Train Random Forest, CNN, and BiLSTM models for hate speech detection.

Usage:
    python -m src.training.train_deep_models --model rf
    python -m src.training.train_deep_models --model cnn --epochs 15
    python -m src.training.train_deep_models --model bilstm --epochs 15
    python -m src.training.train_deep_models --model all
"""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json


def load_data(sample: int = None):
    """Load and prepare data."""
    data_path = Path("data/clean_data.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found. Run 'python -m src.data.load_data' first.")
    
    df = pd.read_csv(data_path)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)
    
    X = df['text'].fillna('').tolist()
    y = df['label'].tolist()
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest model."""
    print("\n=== Training Random Forest ===")
    from src.models.random_forest import train_and_save, load_model, predict_text
    
    model_path = "models/baseline/rf.joblib"
    train_and_save(X_train, y_train, model_path)
    print(f"[OK] Model saved to {model_path}")
    
    model = load_model(model_path)
    predictions = [pred for pred, _ in predict_text(model, X_test)]
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    return predictions


def train_cnn(X_train, y_train, X_test, y_test, epochs=10):
    """Train CNN model."""
    print(f"\n=== Training CNN ({epochs} epochs) ===")
    from src.models.textcnn import train_and_save, load_model, predict_text
    
    model_path = "models/baseline/cnn.pth"
    train_and_save(X_train, y_train, model_path, epochs=epochs)
    print(f"[OK] Model saved to {model_path}")
    
    model = load_model(model_path)
    predictions = [pred for pred, _ in predict_text(model, X_test)]
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    return predictions


def train_bilstm(X_train, y_train, X_test, y_test, epochs=10):
    """Train BiLSTM model."""
    print(f"\n=== Training BiLSTM ({epochs} epochs) ===")
    from src.models.bilstm import train_and_save, load_model, predict_text
    
    model_path = "models/baseline/bilstm.pth"
    train_and_save(X_train, y_train, model_path, epochs=epochs)
    print(f"[OK] Model saved to {model_path}")
    
    model = load_model(model_path)
    predictions = [pred for pred, _ in predict_text(model, X_test)]
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    return predictions


def train_hecan(X_train, y_train, X_test, y_test, epochs=10):
    """Train HECAN model."""
    print(f"\n=== Training HECAN ({epochs} epochs) ===")
    from src.models.hecan import train_and_save, load_model, predict_text
    
    model_path = "models/baseline/hecan.pth"
    train_and_save(X_train, y_train, model_path, epochs=epochs)
    print(f"[OK] Model saved to {model_path}")
    
    model = load_model(model_path)
    predictions = [pred for pred, _ in predict_text(model, X_test)]
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    return predictions


def save_confusion_matrix(y_test, predictions, model_name):
    """Save confusion matrix plot."""
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    labels = ['Normal', 'Offensive', 'Hate'] if cm.shape[0] == 3 else ['Normal', 'Hate/Offensive']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    Path("reports").mkdir(exist_ok=True)
    plt.savefig(f"reports/confusion_matrix_{model_name.lower()}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Confusion matrix saved to reports/confusion_matrix_{model_name.lower()}.png")


def main():
    parser = argparse.ArgumentParser(description="Train deep learning models")
    parser.add_argument("--model", choices=["rf", "cnn", "bilstm", "hecan", "all"], default="all", help="Model to train")
    parser.add_argument("--sample", type=int, help="Sample N rows for faster training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for deep models")
    args = parser.parse_args()
    
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data(args.sample)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    if args.model in ["rf", "all"]:
        predictions = train_random_forest(X_train, y_train, X_test, y_test)
        save_confusion_matrix(y_test, predictions, "RandomForest")
    
    if args.model in ["cnn", "all"]:
        predictions = train_cnn(X_train, y_train, X_test, y_test, args.epochs)
        save_confusion_matrix(y_test, predictions, "CNN")
    
    if args.model in ["bilstm", "all"]:
        predictions = train_bilstm(X_train, y_train, X_test, y_test, args.epochs)
        save_confusion_matrix(y_test, predictions, "BiLSTM")
    
    if args.model in ["hecan", "all"]:
        predictions = train_hecan(X_train, y_train, X_test, y_test, args.epochs)
        save_confusion_matrix(y_test, predictions, "HECAN")
    
    print("\n[OK] Training complete!")


if __name__ == "__main__":
    main()
