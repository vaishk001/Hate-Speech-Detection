import argparse
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModel
import torch


MODEL_DIR = Path("models/transformer/distilbert_local")
OUT_PATH = Path("models/baseline/logreg_transformer.joblib")
CLEAN_CSV = Path("data/clean_data.csv")
REPORT_JSON = Path("reports/classification_report_transformer.json")
CONFUSION_PNG = Path("reports/confusion_matrix_transformer.png")


def load_data(path: Path, sample: int = 0):
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='latin-1')
    df = df.dropna(subset=['text']).reset_index(drop=True)
    if sample and sample < len(df):
        df = df.sample(sample, random_state=42)
    X = df['text'].astype(str).tolist()
    y = df['label'].astype(int).tolist()
    return X, y

def embed_texts(texts, tokenizer, model, device='cpu', batch_size=16, max_length=128):
    model.to(device)
    model.eval()
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        with torch.no_grad():
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            # mean-pool last_hidden_state (batch, seq, dim) -> (batch, dim)
            last = outputs.last_hidden_state
            mask = inputs.get('attention_mask', None)
            if mask is not None:
                mask = mask.unsqueeze(-1).expand(last.size()).float()
                summed = (last * mask).sum(1)
                denom = mask.sum(1).clamp(min=1e-9)
                pooled = summed / denom
            else:
                pooled = last.mean(1)
            embeddings.append(pooled.cpu().numpy())
    return np.vstack(embeddings)

def main(args):
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Transformer dir not found: {MODEL_DIR}. Run scripts/download_transformer.py while online.")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
        model = AutoModel.from_pretrained(str(MODEL_DIR), local_files_only=True)

    X_texts, y = load_data(CLEAN_CSV, sample=args.sample)
    print(f"Loaded {len(X_texts)} examples")

    emb = embed_texts(X_texts, tokenizer, model, device='cpu', batch_size=args.batch_size, max_length=args.max_length)
    print(f"Embeddings shape: {emb.shape}")

    X_train, X_test, y_train, y_test = train_test_split(emb, y, test_size=0.15, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, OUT_PATH)
    print(f"Saved transformer-logreg to: {OUT_PATH}")

    y_pred = clf.predict(X_test)
    report_dict = classification_report(y_test, y_pred, output_dict=True, digits=4)
    print(classification_report(y_test, y_pred, digits=4))
    
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2)
    print(f"Saved classification report to: {REPORT_JSON}")
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xlabel='Predicted',
           ylabel='True',
           title='Confusion Matrix (Transformer)')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2. else "black")
    plt.tight_layout()
    fig.savefig(CONFUSION_PNG, dpi=100)
    plt.close(fig)
    print(f"Saved confusion matrix to: {CONFUSION_PNG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="use a sample size for dev")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()
    main(args)
