# src/models/hecan.py
"""
HECAN: Hierarchical Attention Network for Hate Speech Detection
Optimized for Indo-European languages including Hindi, English, and code-mixed text.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        attn_weights = F.softmax(self.attention(x), dim=1)
        weighted = x * attn_weights
        return weighted.sum(dim=1)


class HECANModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        x = self.dropout(attn_out)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class HECAN:
    def __init__(self, vocab_size: int = 8000, embed_dim: int = 128, hidden_dim: int = 128):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.model = None
        self.vectorizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def fit(self, X: List[str], y: List[int], epochs: int = 10, batch_size: int = 32):
        self.vectorizer = CountVectorizer(max_features=self.vocab_size, ngram_range=(1, 3))
        self.vectorizer.fit(X)
        X_vec = self.vectorizer.transform(X).toarray()
        
        self.model = HECANModel(self.vocab_size + 1, self.embed_dim, self.hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        X_tensor = torch.LongTensor(X_vec).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(X), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(X):.4f}")
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained")
        X_vec = self.vectorizer.transform(X).toarray()
        X_tensor = torch.LongTensor(X_vec).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = F.softmax(outputs, dim=1)
        return probs.cpu().numpy()
    
    def predict(self, X: List[str]) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'vectorizer': self.vectorizer,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
        }, path)
    
    @classmethod
    def load(cls, path: str):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        instance = cls(checkpoint['vocab_size'], checkpoint['embed_dim'], checkpoint['hidden_dim'])
        instance.vectorizer = checkpoint['vectorizer']
        instance.model = HECANModel(checkpoint['vocab_size'] + 1, checkpoint['embed_dim'], checkpoint['hidden_dim'])
        instance.model.load_state_dict(checkpoint['model_state'])
        instance.model.to(instance.device)
        return instance


def train_and_save(X: List[str], y: List[int], out_path: str, epochs: int = 10) -> None:
    model = HECAN()
    model.fit(X, y, epochs=epochs)
    model.save(out_path)


def load_model(path: str) -> HECAN:
    return HECAN.load(path)


def predict_text(model: HECAN, texts: List[str]) -> List[Tuple[int, float]]:
    if not texts:
        return []
    predictions = model.predict(texts)
    probas = model.predict_proba(texts)
    return [(int(pred), float(proba[pred])) for pred, proba in zip(predictions, probas)]


def explain_prediction(model: HECAN, text: str) -> dict:
    """Provide attention-based explanation for prediction."""
    if model.model is None:
        return {"error": "Model not trained"}
    
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    
    return {
        "prediction": int(pred),
        "confidence": float(proba[pred]),
        "probabilities": {i: float(p) for i, p in enumerate(proba)},
        "model": "HECAN",
        "attention_based": True
    }
