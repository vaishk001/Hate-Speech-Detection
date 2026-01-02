# src/models/transformer.py
from typing import List
import numpy as np
from pathlib import Path
from src.utils.config import TRANSFORMER_LOCAL, DISTILBERT_LOCAL
from src.utils.logger import get_logger

logger = get_logger(__name__)

_tokenizer = None
_model = None
_distilbert_tokenizer = None
_distilbert_model = None

def load_transformer(local_dir: str | Path = None):
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    from transformers import AutoTokenizer, AutoModel
    path = str(local_dir or TRANSFORMER_LOCAL)
    logger.info("Loading transformer tokenizer/model from %s", path)
    _tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    _model = AutoModel.from_pretrained(path, local_files_only=True)
    _model.eval()
    return _tokenizer, _model

def load_distilbert(local_dir: str | Path = None):
    global _distilbert_tokenizer, _distilbert_model
    if _distilbert_tokenizer is not None and _distilbert_model is not None:
        return _distilbert_tokenizer, _distilbert_model

    from transformers import AutoTokenizer, AutoModel
    path = str(local_dir or DISTILBERT_LOCAL)
    logger.info("Loading DistilBERT tokenizer/model from %s", path)
    _distilbert_tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    _distilbert_model = AutoModel.from_pretrained(path, local_files_only=True)
    _distilbert_model.eval()
    return _distilbert_tokenizer, _distilbert_model

def embed_texts(texts: List[str], batch_size: int = 8, max_length: int = 128) -> np.ndarray:
    """
    Return mean-pooled embeddings for each text. Uses CPU by default.
    """
    tokenizer, model = load_transformer()
    import torch

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
            # out[0] is last hidden states: (batch, seq_len, dim)
            last_hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
            masked = last_hidden * mask
            summed = masked.sum(dim=1)
            lens = mask.sum(dim=1).clamp(min=1)
            mean_pooled = (summed / lens).cpu().numpy()
            embeddings.append(mean_pooled)
    if embeddings:
        return np.vstack(embeddings)
    return np.zeros((0, model.config.hidden_size))
