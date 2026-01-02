# src/data/normalize.py
"""Text normalization with stopword removal and lowercasing."""
import re
from typing import Tuple

from src.data.preprocess import preprocess_text

# Minimal stopword sets for English and Hindi
ENGLISH_STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that',
    'the', 'to', 'was', 'will', 'with', 'you', 'your', 'i', 'me', 'my'
}

HINDI_STOPWORDS = {
    'है', 'हैं', 'का', 'की', 'को', 'में', 'से', 'और', 'या', 'तो',
    'यह', 'वह', 'जो', 'कि', 'न', 'नहीं', 'हो', 'होगा', 'होंगे'
}


def normalize_text(text: str, remove_stopwords: bool = False) -> Tuple[str, str]:
    """Normalize text with optional stopword removal.
    
    Args:
        text: Input text string.
        remove_stopwords: If True, remove common stopwords.
    
    Returns:
        Tuple of (normalized_text, language_code).
    """
    cleaned, lang = preprocess_text(text)
    
    if remove_stopwords:
        tokens = cleaned.split()
        if lang == 'en':
            tokens = [t for t in tokens if t.lower() not in ENGLISH_STOPWORDS]
        elif lang == 'hi':
            tokens = [t for t in tokens if t not in HINDI_STOPWORDS]
        cleaned = ' '.join(tokens)
    
    return cleaned, lang
