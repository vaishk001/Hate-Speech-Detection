# src/models/hinglish_bert.py
"""
HinglishBERT — Fine-tunable BERT classifier for Hinglish hate speech detection.

Uses L3Cube-Pune's HingBERT (or any BERT-family model) as the backbone encoder
with a classification head for 3-class (Normal / Offensive / Hate) prediction.

Architecture:
    [HingBERT encoder] → mean-pool → dropout → linear(hidden→128) → ReLU
                       → dropout → linear(128→3) → softmax
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


class HinglishBertClassifier(nn.Module):
    """
    BERT-based 3-class classifier with a lightweight MLP head.
    """

    def __init__(self, encoder, hidden_size: int, num_labels: int = 3, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels),
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden = outputs.last_hidden_state  # (B, S, D)

        # Mean-pool over non-padding tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
        else:
            pooled = last_hidden.mean(dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


# ── Model loading / saving utilities ─────────────────────────────────────────

MODEL_DIR = Path("models/transformer/hinglish_bert")
WEIGHTS_PATH = MODEL_DIR / "classifier_head.pth"


def save_model(model: HinglishBertClassifier, tokenizer, save_dir: Path = MODEL_DIR):
    """Save the full model: encoder + tokenizer + classifier head."""
    save_dir.mkdir(parents=True, exist_ok=True)
    # Save encoder + tokenizer via HF API
    model.encoder.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    # Save classifier head separately
    torch.save(model.classifier.state_dict(), save_dir / "classifier_head.pth")
    print(f"[OK] Saved HinglishBERT model to {save_dir}")


def load_model(model_dir: Path = MODEL_DIR, device: str = "cpu") -> Tuple[HinglishBertClassifier, Any]:
    """Load the full HinglishBERT classifier (encoder + head) and tokenizer."""
    import warnings
    from transformers import AutoTokenizer, AutoModel

    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"HinglishBERT dir not found: {model_dir}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        encoder = AutoModel.from_pretrained(str(model_dir), local_files_only=True)

    hidden_size = encoder.config.hidden_size
    classifier = HinglishBertClassifier(encoder, hidden_size, num_labels=3)

    head_path = model_dir / "classifier_head.pth"
    if head_path.exists():
        state = torch.load(head_path, map_location=device, weights_only=True)
        classifier.classifier.load_state_dict(state)

    classifier.to(device)
    classifier.eval()
    return classifier, tokenizer


# ── Inference helper ─────────────────────────────────────────────────────────

_cached_model = None
_cached_tokenizer = None


def predict_text(
    texts: List[str],
    model_dir: Path = MODEL_DIR,
    device: str = "cpu",
    max_length: int = 128,
    batch_size: int = 16,
) -> List[Tuple[int, float]]:
    """
    Predict labels for a list of texts.

    Returns:
        List of (label, confidence) tuples.
    """
    global _cached_model, _cached_tokenizer

    if _cached_model is None:
        _cached_model, _cached_tokenizer = load_model(model_dir, device)

    model = _cached_model
    tokenizer = _cached_tokenizer
    model.eval()

    results: List[Tuple[int, float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc)
            probs = torch.softmax(logits, dim=-1)
            labels = probs.argmax(dim=-1)
            scores = probs.max(dim=-1).values

        for lbl, sc in zip(labels.cpu().tolist(), scores.cpu().tolist()):
            results.append((int(lbl), float(sc)))

    return results
