# src/utils/sarcasm_context.py
"""
Sarcasm & Context Analyser for Hate Speech Detection.

Provides:
- detect_sarcasm(text)  → sarcasm signal + confidence
- analyse_context(text) → contextual risk signals
- apply_sarcasm_context_override(text, label, score) → adjusted (label, score, meta)

Strategy (rule-based, no extra model dependency):
  1. Lexical sarcasm cues  – "oh great", "wow amazing", "sure buddy", etc.
  2. Punctuation patterns  – excessive "!", "?!", ellipsis "..."
  3. Contrast patterns     – positive word + negative subject, e.g. "love how stupid"
  4. Irony markers        – "yeah right", "totally normal", "because obviously"
  5. Quotation wrapping    – 'great "friend"'  (scare quotes)
  6. Context signals       – personal attack context, group-targeting context,
                             self-deprecation context (reduces label)
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Lexical cue banks
# ---------------------------------------------------------------------------

# Sarcasm phrases that flip positive surface → negative intent
SARCASM_PHRASES = [
    r"\boh\s+(?:great|wonderful|brilliant|fantastic|perfect|sure|yes|yeah|wow)\b",
    r"\bwow\s+(?:so|very|really|super|totally|just)\b",
    r"\bjust\s+(?:great|wonderful|perfect|what\s+i\s+needed)\b",
    r"\btotally\s+(?:normal|fine|great|cool|not|makes\s+sense)\b",
    r"\byeah\s+(?:right|sure|ok|okay|of\s+course|no)\b",
    r"\bsure\s+(?:buddy|pal|man|bro|dude|thing|whatever)\b",
    r"\bbecause\s+(?:obviously|clearly|that\s+makes\s+sense|sure)\b",
    r"\bgreat\s+(?:idea|job|work|thinking|move|plan)\b",
    r"\bso\s+(?:helpful|smart|clever|bright|brilliant|kind)\b",
    r"\bwhat\s+a\s+(?:surprise|shock|genius|brilliant\s+idea)\b",
    r"\bi\s+(?:love|just\s+love|totally\s+love)\s+how\b",
    r"\bthank\s+you\s+so\s+much\s+for\b",  # dripping sarcasm
    r"\breally\s+(?:helpful|smart|kind|nice|great|good)\b",
    r"\bnot\s+like\b",  # "not like you care"
    r"\bnot\s+at\s+all\b",
    r"\bas\s+if\b",
    # Hinglish Sarcasm
    r"\bwaah\s+(?:bhai|kya\s+baat\s+hai|yaar|ji)\b",
    r"\bbahut\s+(?:badhiya|achha|khoob)\b",
    r"\bbade\s+(?:aaye|log)\b",
    r"\bkya\s+(?:kehney|baat\s+hai)\b",
]

# Positive words used sarcastically BEFORE negative targets
CONTRAST_POSITIVE = [
    "love", "like", "enjoy", "appreciate", "admire",
    "great", "brilliant", "wonderful", "amazing", "fantastic",
    "perfect", "excellent", "outstanding",
]
CONTRAST_NEGATIVE_TARGETS = [
    "stupid", "idiot", "moron", "dumb", "fool", "loser",
    "pathetic", "useless", "worthless", "trash", "garbage",
    "hate", "awful", "terrible", "horrible", "disgusting",
]

# Irony markers
IRONY_MARKERS = [
    r"\bnot\b.{0,20}\bactually\b",
    r"\bsupposedly\b",
    r"\bapparently\b",
    r"\bso[-\s]called\b",
    r"\bpretend(?:ing)?\b",
    r"\bquote[-\s]unquote\b",
    r"\bair\s+quotes?\b",
    r"\bwink\s+wink\b",
    r"\bwink\b",
    r"🙄",   # eye-roll emoji
    r"😏",   # smirk emoji
    r"/s\b",  # Reddit sarcasm tag
]

# Scare-quote pattern: word wrapped in double quotes inside a sentence
_SCARE_QUOTE_RE = re.compile(r'\b\w+\s+"[^"]{1,30}"\s*\w*')

# Punctuation pattern signals
_EXCL_RE = re.compile(r"!{2,}")          # more than one !
_INTERROBANG_RE = re.compile(r"[?!]{2,}")  # ?! or !?
_ELLIPSIS_RE = re.compile(r"\.{3,}")     # ...
_ALL_CAPS_WORD_RE = re.compile(r"\b[A-Z]{3,}\b")  # SHOUT words

# ---------------------------------------------------------------------------
# Context signal banks
# ---------------------------------------------------------------------------

# Words that indicate personal attack context
PERSONAL_ATTACK_CONTEXT = [
    r"\byou\s+(?:are|were|look|sound|act|seem)\b",
    r"\byou'?re\b",
    r"\byour\s+\w+\s+(?:is|are|sucks?|stinks?)\b",
    r"\bpeople\s+like\s+you\b",
    r"\bsomeone\s+like\s+you\b",
]

# Words that indicate group-targeting context (raises severity)
GROUP_TARGET_CONTEXT = [
    r"\ball\s+\w+\s+(?:are|should|must|deserve|need)\b",
    r"\bevery\s+\w+\s+(?:is|are|should)\b",
    r"\bthose\s+(?:people|guys|folks|types)\b",
    r"\btheir\s+kind\b",
    r"\byou\s+(?:people|all|guys)\b",
    r"\bthey\s+all\b",
    # Hinglish Group Targets
    r"\bin\s+logo\s+ko\b",
    r"\baise\s+logo\s+ko\b",
    r"\bwoh\s+log\b",
]

# Self-deprecation context (reduces label upward pressure)
SELF_DEPRECATION_CONTEXT = [
    r"\bi\s+(?:am|was|feel|felt)\s+(?:such\s+a\s+)?(?:idiot|fool|moron|stupid|dumb)\b",
    r"\bstupid\s+me\b",
    r"\bmy\s+(?:fault|mistake|bad|issue|problem)\b",
    r"\bi\s+(?:messed|screwed|messed)\s+up\b",
]

# Humour/joke context (may reduce label)
HUMOUR_CONTEXT = [
    r"\bjust\s+kidding\b",
    r"\bjk\b",
    r"\blol\b",
    r"\bhaha\b",
    r"\bha\s+ha\b",
    r"😂",
    r"😄",
    r"🤣",
    r"\bjoking\b",
    r"\bin\s+jest\b",
    r"\bfor\s+fun\b",
]


def _compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_SARCASM_PATTERNS = _compile_patterns(SARCASM_PHRASES)
_IRONY_PATTERNS = _compile_patterns(IRONY_MARKERS)
_PERSONAL_ATTACK_PATTERNS = _compile_patterns(PERSONAL_ATTACK_CONTEXT)
_GROUP_TARGET_PATTERNS = _compile_patterns(GROUP_TARGET_CONTEXT)
_SELF_DEP_PATTERNS = _compile_patterns(SELF_DEPRECATION_CONTEXT)
_HUMOUR_PATTERNS = _compile_patterns(HUMOUR_CONTEXT)


# ---------------------------------------------------------------------------
# Core detectors
# ---------------------------------------------------------------------------

def detect_sarcasm(text: str) -> Dict:
    """
    Analyse text for sarcasm signals.

    Returns:
        {
          "is_sarcastic": bool,
          "confidence": float,       # 0.0 – 1.0
          "signals": list[str],      # human-readable triggers
          "sarcasm_score": float,    # raw accumulation
        }
    """
    t = text.lower()
    signals = []
    score = 0.0

    # 1. Lexical sarcasm phrases
    for pat in _SARCASM_PATTERNS:
        if pat.search(t):
            signals.append(f"sarcasm phrase: '{pat.pattern}'")
            score += 0.25

    # 2. Irony markers
    for pat in _IRONY_PATTERNS:
        if pat.search(text):  # keep original case for emoji/symbols
            signals.append(f"irony marker: '{pat.pattern}'")
            score += 0.20

    # 3. Contrast pattern – positive word near negative target
    words = t.split()
    for i, w in enumerate(words):
        if w in CONTRAST_POSITIVE:
            window = " ".join(words[i:i+6])
            for neg in CONTRAST_NEGATIVE_TARGETS:
                if neg in window:
                    signals.append(f"contrast: '{w}' + '{neg}'")
                    score += 0.30
                    break

    # 4. Scare quotes
    if _SCARE_QUOTE_RE.search(text):
        signals.append("scare quotes detected")
        score += 0.15

    # 5. Punctuation patterns
    if _EXCL_RE.search(text):
        signals.append("excessive exclamation marks")
        score += 0.10
    if _INTERROBANG_RE.search(text):
        signals.append("interrobang (?! or !?)")
        score += 0.10
    if _ELLIPSIS_RE.search(text):
        signals.append("ellipsis (trailing ...)")
        score += 0.08
    caps_words = _ALL_CAPS_WORD_RE.findall(text)
    if len(caps_words) >= 2:
        signals.append(f"shouting caps: {caps_words[:3]}")
        score += 0.12

    # Clamp score to [0, 1]
    score = min(score, 1.0)
    is_sarcastic = score >= 0.30

    return {
        "is_sarcastic": is_sarcastic,
        "confidence": round(score, 3),
        "signals": list(dict.fromkeys(signals)),  # deduplicate, preserve order
        "sarcasm_score": round(score, 3),
    }


def analyse_context(text: str) -> Dict:
    """
    Analyse contextual risk signals in text.

    Returns:
        {
          "personal_attack": bool,
          "group_targeting": bool,
          "self_deprecation": bool,
          "humour_context": bool,
          "risk_level": str,      # "low" | "medium" | "high"
          "context_signals": list[str],
        }
    """
    t = text.lower()
    ctx_signals = []

    personal = any(pat.search(t) for pat in _PERSONAL_ATTACK_PATTERNS)
    group = any(pat.search(t) for pat in _GROUP_TARGET_PATTERNS)
    self_dep = any(pat.search(t) for pat in _SELF_DEP_PATTERNS)
    humour = any(pat.search(text) for pat in _HUMOUR_PATTERNS)  # emojis need orig case

    if personal:
        ctx_signals.append("personal attack context")
    if group:
        ctx_signals.append("group-targeting context")
    if self_dep:
        ctx_signals.append("self-deprecation context")
    if humour:
        ctx_signals.append("humour / joking context")

    # Determine risk level
    if group:
        risk = "high"
    elif personal and not self_dep:
        risk = "medium"
    elif humour or self_dep:
        risk = "low"
    else:
        risk = "medium" if personal else "low"

    return {
        "personal_attack": personal,
        "group_targeting": group,
        "self_deprecation": self_dep,
        "humour_context": humour,
        "risk_level": risk,
        "context_signals": ctx_signals,
    }


# ---------------------------------------------------------------------------
# Override integration
# ---------------------------------------------------------------------------

def apply_sarcasm_context_override(
    text: str,
    label: int,
    score: float,
) -> Tuple[int, float, Dict]:
    """
    Adjust predicted label and score based on sarcasm & context analysis.

    Rules:
      - Sarcasm detected + label==0 (Normal) → upgrade to 1 (Offensive)
        Sarcasm can mask offensive intent behind positive words.
      - Group-targeting context + label<=1   → upgrade to max(label, 1)
        with a small score boost.
      - Self-deprecation context + label==1  → downgrade to 0 (Normal)
        as the speaker is insulting themselves, not others.
      - Humour context + label==1 (not group)→ keep label but reduce score.
      - Sarcasm + group-targeting            → upgrade to 2 (Hate), high confidence.

    Returns:
        (adjusted_label, adjusted_score, meta_dict)
    """
    sarcasm = detect_sarcasm(text)
    context = analyse_context(text)

    adjusted_label = label
    adjusted_score = score
    notes = []

    # Rule 1: Sarcasm + group-targeting → Hate
    if sarcasm["is_sarcastic"] and context["group_targeting"]:
        if adjusted_label < 2:
            adjusted_label = 2
            adjusted_score = max(adjusted_score, 0.80)
            notes.append("Sarcasm + group-targeting upgraded to Hate")

    # Rule 2: Sarcasm + normal prediction → Offensive
    elif sarcasm["is_sarcastic"] and adjusted_label == 0:
        adjusted_label = 1
        adjusted_score = max(adjusted_score, 0.65)
        notes.append("Sarcasm detected: upgraded Normal → Offensive")

    # Rule 3: Group-targeting context raises floor
    if context["group_targeting"] and adjusted_label < 1:
        adjusted_label = 1
        adjusted_score = max(adjusted_score, 0.60)
        notes.append("Group-targeting context: raised to Offensive")

    # Rule 4: Self-deprecation downgrades Offensive → Normal
    if context["self_deprecation"] and adjusted_label == 1 and not context["group_targeting"]:
        adjusted_label = 0
        adjusted_score = max(adjusted_score, 0.55)
        notes.append("Self-deprecation: downgraded Offensive → Normal")

    # Rule 5: Humour context softens Offensive (score reduction)
    if context["humour_context"] and adjusted_label == 1 and not context["group_targeting"]:
        adjusted_score = max(adjusted_score * 0.80, 0.45)
        notes.append("Humour/joking context: confidence reduced")

    meta = {
        "sarcasm": sarcasm,
        "context": context,
        "override_notes": notes,
        "original_label": label,
        "original_score": round(score, 3),
        "adjusted_label": adjusted_label,
        "adjusted_score": round(adjusted_score, 3),
    }

    return adjusted_label, round(adjusted_score, 3), meta
