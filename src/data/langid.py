# src/data/langid.py
"""Language identification for Hindi, English, and code-mixed text."""
from typing import Literal

Lang = Literal["en", "hi", "hi-en", "unknown"]


def detect_language(text: str) -> Lang:
    """Detect language: English, Hindi, code-mixed (hi-en), or unknown.
    
    Uses character ranges and heuristics:
    - Devanagari (U+0900-U+097F): Hindi
    - Latin + Devanagari: Code-mixed (hi-en)
    - Latin only: English
    
    Args:
        text: Input text string.
    
    Returns:
        Language code: 'en', 'hi', 'hi-en', or 'unknown'.
    """
    if not text or not text.strip():
        return "unknown"
    
    # Count character types
    has_devanagari = any('\u0900' <= ch <= '\u097F' for ch in text)
    has_latin = any(ch.isascii() and ch.isalpha() for ch in text)
    has_digit = any(ch.isdigit() for ch in text)
    
    if has_devanagari and has_latin:
        return "hi-en"
    elif has_devanagari:
        return "hi"
    elif has_latin or has_digit:
        return "en"
    
    return "unknown"
