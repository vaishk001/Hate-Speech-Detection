# src/utils/embed_cache.py
"""Caching layer for text embeddings."""
from pathlib import Path
from typing import Dict, Optional, Any

import joblib
import numpy as np


class EmbedCache:
    """Simple in-memory + disk cache for embeddings."""
    
    def __init__(self, cache_path: str = 'models/transformer/embeddings.joblib'):
        """Initialize cache.
        
        Args:
            cache_path: Path to save/load embeddings.
        """
        self.path = Path(cache_path)
        self._cache: Dict[str, np.ndarray] = {}
        self._loaded = False
    
    def exists(self) -> bool:
        """Check if cache file exists on disk."""
        return self.path.exists()
    
    def load(self) -> Dict[str, np.ndarray]:
        """Load embeddings from disk."""
        if self.exists():
            self._cache = joblib.load(self.path)
            self._loaded = True
        return self._cache
    
    def save(self) -> None:
        """Save embeddings to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._cache, self.path)
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get embedding by key."""
        if not self._loaded:
            self.load()
        return self._cache.get(key)
    
    def set(self, key: str, embedding: np.ndarray) -> None:
        """Store embedding in cache."""
        self._cache[key] = embedding
    
    def batch_get(self, keys: list) -> Dict[str, np.ndarray]:
        """Get multiple embeddings."""
        if not self._loaded:
            self.load()
        return {k: self._cache[k] for k in keys if k in self._cache}
    
    def batch_set(self, embeddings: Dict[str, np.ndarray]) -> None:
        """Store multiple embeddings."""
        self._cache.update(embeddings)
    
    def clear(self) -> None:
        """Clear in-memory cache."""
        self._cache.clear()
        self._loaded = False
    
    def size(self) -> int:
        """Return number of cached embeddings."""
        return len(self._cache)
