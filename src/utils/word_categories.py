import json
from pathlib import Path
from src.utils.hate_words import HATE_WORDS, OFFENSIVE_WORDS

CATEGORIES_FILE = Path("data/word_categories.json")

def _load_categories():
    """Load custom word categories from file."""
    if CATEGORIES_FILE.exists():
        try:
            with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "hate": set(data.get("hate", [])),
                    "offensive": set(data.get("offensive", [])),
                    "normal": set(data.get("normal", []))
                }
        except:
            return {"hate": set(HATE_WORDS), "offensive": set(OFFENSIVE_WORDS), "normal": set()}
    return {"hate": set(HATE_WORDS), "offensive": set(OFFENSIVE_WORDS), "normal": set()}

def _save_categories(categories):
    """Save word categories to file."""
    CATEGORIES_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "hate": list(categories.get("hate", set())),
        "offensive": list(categories.get("offensive", set())),
        "normal": list(categories.get("normal", set()))
    }
    with open(CATEGORIES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

_CATEGORIES = _load_categories()

def update_word_category(word: str, category: str):
    """Update a word's category (hate, offensive, or normal)."""
    global _CATEGORIES
    word_lower = word.lower().strip()
    
    _CATEGORIES["hate"].discard(word_lower)
    _CATEGORIES["offensive"].discard(word_lower)
    _CATEGORIES["normal"].discard(word_lower)
    
    if category == "hate":
        _CATEGORIES["hate"].add(word_lower)
    elif category == "offensive":
        _CATEGORIES["offensive"].add(word_lower)
    elif category == "normal":
        _CATEGORIES["normal"].add(word_lower)
    
    _save_categories(_CATEGORIES)
    _CATEGORIES = _load_categories()

def detect_words_with_categories(text: str):
    """Detect words and return with their current categories."""
    global _CATEGORIES
    
    text_lower = text.lower()
    words = text_lower.split()
    
    found_hate = []
    found_offensive = []
    
    for word in words:
        clean_word = ''.join(c for c in word if c.isalnum())
        if clean_word in _CATEGORIES["normal"]:
            continue
        elif clean_word in _CATEGORIES["hate"]:
            found_hate.append(word)
        elif clean_word in _CATEGORIES["offensive"]:
            found_offensive.append(word)
    
    return {
        "hate_words": found_hate,
        "offensive_words": found_offensive,
        "has_hate": len(found_hate) > 0,
        "has_offensive": len(found_offensive) > 0,
        "severity": "hate" if found_hate else ("offensive" if found_offensive else "normal")
    }

def get_word_category(word: str):
    """Get the current category of a word."""
    global _CATEGORIES
    word_lower = word.lower().strip()
    if word_lower in _CATEGORIES["normal"]:
        return "normal"
    elif word_lower in _CATEGORIES["hate"]:
        return "hate"
    elif word_lower in _CATEGORIES["offensive"]:
        return "offensive"
    return "normal"

def reload_categories():
    """Reload categories from file."""
    global _CATEGORIES
    _CATEGORIES = _load_categories()
