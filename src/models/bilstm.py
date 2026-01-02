# src/models/bilstm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class BiLSTMNetwork(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, num_layers: int = 2, num_classes: int = 3, dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(x)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        x = self.dropout(hidden)
        return self.fc(x)


class BiLSTMModel:
    def __init__(self, vocab_size: int = 5000, embed_dim: int = 64, hidden_dim: int = 64, num_layers: int = 1):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model = None
        self.vectorizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X: List[str], y: List[int], epochs: int = 10, batch_size: int = 32):
        self.vectorizer = CountVectorizer(max_features=self.vocab_size, ngram_range=(1, 2))
        self.vectorizer.fit(X)
        X_vec = self.vectorizer.transform(X).toarray()
        
        self.model = BiLSTMNetwork(self.vocab_size + 1, self.embed_dim, self.hidden_dim, self.num_layers, num_classes=3).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        X_tensor = torch.LongTensor(X_vec).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        self.model.train()
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

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
            'num_layers': self.num_layers,
        }, path)

    @classmethod
    def load(cls, path: str):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        instance = cls(checkpoint['vocab_size'], checkpoint['embed_dim'], checkpoint['hidden_dim'], checkpoint['num_layers'])
        instance.vectorizer = checkpoint['vectorizer']
        instance.model = BiLSTMNetwork(checkpoint['vocab_size'] + 1, checkpoint['embed_dim'], checkpoint['hidden_dim'], checkpoint['num_layers'], num_classes=3)
        instance.model.load_state_dict(checkpoint['model_state'])
        instance.model.to(instance.device)
        return instance


def train_and_save(X: List[str], y: List[int], out_path: str, epochs: int = 10) -> None:
    model = BiLSTMModel()
    model.fit(X, y, epochs=epochs)
    model.save(out_path)


def load_model(path: str) -> BiLSTMModel:
    return BiLSTMModel.load(path)


def predict_text(model: BiLSTMModel, texts: List[str]) -> List[Tuple[int, float]]:
    if not texts:
        return []
    predictions = model.predict(texts)
    probas = model.predict_proba(texts)
    return [(int(pred), float(proba[pred])) for pred, proba in zip(predictions, probas)]
