#!/usr/bin/env python3
"""
Train a Hinglish-specific transformer for hate speech detection.

Uses L3Cube-Pune's HingBERT (l3cube-pune/hing-bert) — a BERT model pretrained
on 52M+ Hindi-English code-mixed sentences — and fine-tunes it with a 3-class
classification head (Normal=0, Offensive=1, Hate=2).

Training strategy:
  1. Freeze encoder layers initially, train only the classifier head (warm-up)
  2. Unfreeze and fine-tune the full model with a lower learning rate
  3. Use class-weighted cross-entropy to handle label imbalance
  4. Apply learning rate scheduling + early stopping

Usage:
    python -m src.training.train_hinglish_bert
    python -m src.training.train_hinglish_bert --epochs 15 --lr 2e-5
    python -m src.training.train_hinglish_bert --sample 200  # quick test run
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ── Constants ────────────────────────────────────────────────────────────────

# HuggingFace model ID — L3Cube-Pune HingBERT (pretrained on Hinglish corpus)
HF_MODEL_ID = "l3cube-pune/hing-bert"

# Local save directory for the fine-tuned model
SAVE_DIR = Path("models/transformer/hinglish_bert")
CLEAN_CSV = Path("data/clean_data.csv")
REPORT_JSON = Path("reports/classification_report_hinglish.json")
CONFUSION_PNG = Path("reports/confusion_matrix_hinglish.png")

NUM_LABELS = 3
LABEL_NAMES = {0: "Normal", 1: "Offensive", 2: "Hate"}


# ── Dataset ──────────────────────────────────────────────────────────────────

class HateSpeechDataset(Dataset):
    """PyTorch dataset that tokenizes text on-the-fly."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_data(path: Path, sample: int = 0):
    """Load clean CSV. Returns texts list and labels list."""
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    df = df.dropna(subset=["text"]).reset_index(drop=True)
    if sample and sample < len(df):
        df = df.sample(sample, random_state=42)

    return df["text"].astype(str).tolist(), df["label"].astype(int).tolist()


def compute_class_weights(labels, num_classes=3, device="cpu"):
    """Compute inverse-frequency class weights for balanced training."""
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 1)
        weights.append(total / (num_classes * cnt))
    return torch.tensor(weights, dtype=torch.float32).to(device)


def save_confusion_matrix(y_true, y_pred, save_path: Path):
    """Plot and save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    labels = [LABEL_NAMES.get(i, str(i)) for i in range(cm.shape[0])]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix - HinglishBERT",
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2.0 else "black",
            )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  [OK] Confusion matrix -> {save_path}")


# ── Training ─────────────────────────────────────────────────────────────────

def train(args):
    from transformers import AutoTokenizer, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ─── Step 1: Load or download HingBERT ─────────────────────────────────
    if SAVE_DIR.exists() and (SAVE_DIR / "config.json").exists():
        print(f"Loading existing HinglishBERT from {SAVE_DIR}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tokenizer = AutoTokenizer.from_pretrained(str(SAVE_DIR), local_files_only=True)
            encoder = AutoModel.from_pretrained(str(SAVE_DIR), local_files_only=True)
    else:
        print(f"Downloading HingBERT from HuggingFace: {HF_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        encoder = AutoModel.from_pretrained(HF_MODEL_ID)
        # Save immediately so future runs are offline-capable
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(SAVE_DIR)
        encoder.save_pretrained(SAVE_DIR)
        print(f"  [OK] Saved base model to {SAVE_DIR}")

    hidden_size = encoder.config.hidden_size

    # ─── Step 2: Build classifier ──────────────────────────────────────────
    from src.models.hinglish_bert import HinglishBertClassifier

    model = HinglishBertClassifier(encoder, hidden_size, num_labels=NUM_LABELS, dropout=args.dropout)
    model.to(device)

    # ─── Step 3: Load data ─────────────────────────────────────────────────
    if not CLEAN_CSV.exists():
        print(f"  [!] {CLEAN_CSV} not found. Running data loader first...")
        from src.data.load_data import main as load_main
        load_main()

    X, y = load_data(CLEAN_CSV, sample=args.sample)
    print(f"  Dataset: {len(X)} samples")
    print(f"  Label distribution: {dict(Counter(y))}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    train_ds = HateSpeechDataset(X_train, y_train, tokenizer, max_length=args.max_length)
    val_ds = HateSpeechDataset(X_val, y_val, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ─── Step 4: Loss + Optimizer ──────────────────────────────────────────
    class_weights = compute_class_weights(y_train, NUM_LABELS, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Phase 1: Freeze encoder, train head only
    for param in model.encoder.parameters():
        param.requires_grad = False

    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    optimizer = torch.optim.AdamW(head_params, lr=args.lr * 10, weight_decay=1e-2)

    warmup_epochs = min(args.warmup_epochs, args.epochs // 2)
    print(f"\n{'='*60}")
    print(f"  Phase 1: Head warm-up ({warmup_epochs} epochs, encoder frozen)")
    print(f"{'='*60}")

    for epoch in range(warmup_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(ids, attention_mask=mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate_accuracy(model, val_loader, device)
        print(f"  Epoch {epoch+1}/{warmup_epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2%}")

    # Phase 2: Unfreeze encoder, fine-tune everything
    for param in model.encoder.parameters():
        param.requires_grad = True

    all_params = [
        {"params": model.encoder.parameters(), "lr": args.lr},
        {"params": model.classifier.parameters(), "lr": args.lr * 5},
    ]
    optimizer = torch.optim.AdamW(all_params, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)

    finetune_epochs = args.epochs - warmup_epochs
    print(f"\n{'='*60}")
    print(f"  Phase 2: Full fine-tuning ({finetune_epochs} epochs)")
    print(f"{'='*60}")

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(finetune_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(ids, attention_mask=mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate_accuracy(model, val_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {warmup_epochs + epoch + 1}/{args.epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2%} | LR: {lr_now:.2e}")

        # Early stopping / best model save
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            from src.models.hinglish_bert import save_model
            save_model(model, tokenizer, SAVE_DIR)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  [STOP] Early stopping triggered (patience={args.patience})")
                break

    # ─── Step 5: Final evaluation ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Final Evaluation")
    print(f"{'='*60}")

    # Reload best model
    from src.models.hinglish_bert import load_model as load_hinglish
    model, tokenizer = load_hinglish(SAVE_DIR, device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(ids, attention_mask=mask)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Classification report
    target_names = [LABEL_NAMES[i] for i in range(NUM_LABELS)]
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    report_dict = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True, digits=4)

    print(report)

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"  [OK] Classification report -> {REPORT_JSON}")

    # Confusion matrix
    save_confusion_matrix(all_labels, all_preds, CONFUSION_PNG)

    print(f"\n  [DONE] Training complete! Best validation accuracy: {best_val_acc:.2%}")
    print(f"  Model saved at: {SAVE_DIR}")


def evaluate_accuracy(model, loader, device):
    """Compute accuracy on a DataLoader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(ids, attention_mask=mask)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HinglishBERT for hate speech detection")
    parser.add_argument("--epochs", type=int, default=12, help="Total training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Epochs to train head only (encoder frozen)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Base learning rate for encoder")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--max_length", type=int, default=128, help="Max token length")
    parser.add_argument("--sample", type=int, default=0, help="Use N samples for quick dev runs")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    args = parser.parse_args()
    train(args)
