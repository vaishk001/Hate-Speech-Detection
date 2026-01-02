# src/data/preprocess.py
"""
Text preprocessing utilities for KRIXION Hate Speech Detection.

Provides:
- preprocess_text(text: str) -> Tuple[str, str]
  Cleans text and detects language (hi, hi-en, en).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Tuple

# Simple emoji-to-text mapping (extend as needed)
EMOJI_MAP = {
    '😊': 'smile',
    '😂': 'laugh',
    '😡': 'angry',
    '😢': 'sad',
    '❤️': 'heart',
    '👍': 'thumbs_up',
    '👎': 'thumbs_down',
    '🔥': 'fire',
    '💀': 'skull',
    '🤬': 'swear',
}

# Romanized Hindi tokens (common in code-mixed hi-en text)
ROMANIZED_HINDI_TOKENS = {
    'tum', 'tumhe', 'tumhara', 'bhai', 'kya', 'nahi', 'hai', 'hain',
    'tera', 'mera', 'yaar', 'abhi', 'kuch', 'kab', 'kaun', 'aise',
    'thoda', 'bahut', 'kyun', 'matlab', 'achha', 'theek', 'chal', 'kar',
    'raha', 'rahe', 'karna', 'mat', 'bol', 'dekho', 'bhaiya',
}

# Regex helpers
_URL_RE = re.compile(r'https?://\S+|www\.\S+', flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r'\b\S+@\S+\.\w+\b', flags=re.IGNORECASE)
_MENTION_RE = re.compile(r'[@]\w+')
_HASHTAG_RE = re.compile(r'[#]\w+')
_MULTI_PUNCT_RE = re.compile(r'([!?.,;:])\1+')
_WHITESPACE_RE = re.compile(r'\s+')
_WORD_RE = re.compile(r'\b\w+\b', flags=re.UNICODE)


def _has_devanagari(text: str) -> bool:
    """Check if text contains any Devanagari script characters."""
    return bool(re.search(r'[\u0900-\u097F]', text))


def _detect_language(text: str) -> str:
    """Detect language tag based on script and token heuristics.

    Returns:
        'hi'     - Devanagari present (Hindi)
        'hi-en'  - romanized Hindi tokens found (code-mixed)
        'en'     - likely English / Latin-only
        'unknown' - empty or undecidable
    """
    if not text:
        return 'unknown'

    if _has_devanagari(text):
        return 'hi'

    tokens = {t.lower() for t in _WORD_RE.findall(text)}
    if tokens & ROMANIZED_HINDI_TOKENS:
        return 'hi-en'

    # if contains any latin letters, mark en, else unknown
    if any('a' <= ch.lower() <= 'z' for ch in text):
        return 'en'
    return 'unknown'


def _remove_urls_emails_mentions(text: str) -> str:
    """Remove URLs, emails, mentions and hashtags."""
    text = _URL_RE.sub(' ', text)
    text = _EMAIL_RE.sub(' ', text)
    text = _MENTION_RE.sub(' ', text)
    text = _HASHTAG_RE.sub(' ', text)
    return text


def _normalize_emojis(text: str) -> str:
    """Replace known emojis with plain text labels."""
    for emoji, label in EMOJI_MAP.items():
        text = text.replace(emoji, f' {label} ')
    return text


def _remove_extra_punctuation(text: str) -> str:
    """Remove/reduce repeated punctuation (e.g., '!!!' -> '!')."""
    text = _MULTI_PUNCT_RE.sub(r'\1', text)
    return text


def _collapse_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space and trim."""
    return _WHITESPACE_RE.sub(' ', text).strip()


def _strip_surrounding_quotes(text: str) -> str:
    """Strip surrounding matching quotes (single/double/backtick) if present."""
    text = text.strip()
    if len(text) >= 2:
        if (text[0] == text[-1]) and text[0] in ('"', "'", '`'):
            text = text[1:-1].strip()
    return text


def _lower_ascii_only(text: str) -> str:
    """
    Lowercase ASCII letters only so Devanagari/Hindi text remains intact.
    """
    # replace only A-Z characters to lowercase equivalents
    return re.sub(r'[A-Z]', lambda m: m.group(0).lower(), text)


def preprocess_text(text: str) -> Tuple[str, str]:
    """Clean and detect language for input text.

    Steps:
    1. Unicode normalize (NFKC)
    2. Remove URLs/emails/mentions/hashtags
    3. Normalize emojis to text labels
    4. Reduce repeated punctuation
    5. Collapse whitespace
    6. Strip surrounding quotes/backticks
    7. Lowercase ASCII letters only
    8. Detect language tag
    """
    if text is None:
        return ('', 'unknown')

    # ensure string and normalize unicode forms
    txt = str(text)
    txt = unicodedata.normalize('NFKC', txt)

    # remove noisy tokens
    txt = _remove_urls_emails_mentions(txt)
    txt = _normalize_emojis(txt)
    txt = _remove_extra_punctuation(txt)
    txt = _collapse_whitespace(txt)
    txt = _strip_surrounding_quotes(txt)
    txt = _lower_ascii_only(txt)

    lang = _detect_language(txt)
    return (txt, lang)


# quick demo when run directly
if __name__ == '__main__':
    samples = [
        'Hello world! Check this out: https://example.com 😊',
        'तुम बहुत अच्छे हो',  # Devanagari (Hindi)
        'bhai tum kya kar rahe ho yaar?',  # Romanized Hindi code-mix
        'This is a normal English sentence.',
        '"Quoted text!!!"',
        '😡😡 This is so annoying!!!',
        'Contact me: me@example.com #shoutout @user',
    ]

    for s in samples:
        cleaned, lang = preprocess_text(s)
        print(f"[{lang:5}] {s[:60]:60} -> {cleaned}")
