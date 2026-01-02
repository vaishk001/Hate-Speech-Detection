# src/utils/hate_words.py
"""
Comprehensive hate word dictionary for detection.
"""

HATE_WORDS = {
    # Extreme hate speech
    "kill", "murder", "die", "death", "terrorist", "terrorism", "bomb", "shoot", "stab",
    "genocide", "exterminate", "eliminate", "destroy", "annihilate",
    
    # Racial slurs and discrimination
    "racist", "racism", "nigger", "nigga", "chink", "gook", "spic", "wetback", "kike",
    "raghead", "towelhead", "cracker", "whitey", "blackie",
    
    # Religious hate
    "infidel", "kafir", "heathen", "jihad", "crusade",
    
    # Sexual/Gender hate
    "rape", "rapist", "whore", "slut", "bitch", "cunt", "pussy", "dick", "cock",
    "faggot", "fag", "dyke", "tranny", "shemale",
    
    # Severe insults
    "hate", "hatred", "despise", "loathe", "detest",
    "scum", "filth", "vermin", "parasite", "disease", "cancer", "plague",
    "subhuman", "inferior", "worthless", "useless", "pathetic",
    
    # Violence and threats
    "attack", "assault", "beat", "punch", "kick", "hurt", "harm", "torture",
    "threat", "threaten", "violence", "violent", "abuse", "abusive",
    
    # Derogatory terms
    "idiot", "stupid", "moron", "imbecile", "retard", "retarded", "dumb", "dumbass",
    "fool", "foolish", "loser", "failure", "trash", "garbage", "waste",
    "ugly", "disgusting", "repulsive", "revolting", "hideous",
    
    # Bullying terms
    "bully", "bullying", "harass", "harassment", "intimidate", "intimidation",
    
    # Profanity (context-dependent)
    "fuck", "fucking", "fucked", "fucker", "shit", "shitty", "damn", "damned",
    "hell", "bastard", "asshole", "ass", "crap", "piss",
    
    # Hindi/Hinglish hate words
    "pagal", "bevakoof", "gadha", "kutta", "kutti", "saala", "saali",
    "harami", "haramzada", "kamina", "kamine", "badtameez", "ganda",
    "chutiya", "madarchod", "behenchod", "bhosdike", "randi", "bhosdi",
    
    # Code-mixed variations
    "mc", "bc", "mf", "sob", "pos", "stfu", "gtfo", "kys",
}

OFFENSIVE_WORDS = {
    # Mild profanity
    "suck", "sucks", "sucked", "worst", "terrible", "horrible", "awful",
    "annoying", "irritating", "obnoxious",
    
    # Mild insults
    "jerk", "creep", "weirdo", "freak", "nerd", "geek",
    "lame", "boring", "dull", "liar", "fake", "phony",
    
    # Dismissive terms
    "whatever", "shut up", "shutup", "get lost", "go away",
    "leave me alone", "buzz off",
}

def get_hate_words():
    """Return set of hate words."""
    return HATE_WORDS

def get_offensive_words():
    """Return set of offensive words."""
    return OFFENSIVE_WORDS

def get_all_flagged_words():
    """Return combined set of hate and offensive words."""
    return HATE_WORDS | OFFENSIVE_WORDS

def detect_hate_words(text: str):
    """Detect hate words in text and return list of found words with severity."""
    text_lower = text.lower()
    words = text_lower.split()
    
    found_hate = []
    found_offensive = []
    
    for word in words:
        # Remove punctuation for matching
        clean_word = ''.join(c for c in word if c.isalnum())
        if clean_word in HATE_WORDS:
            found_hate.append(word)
        elif clean_word in OFFENSIVE_WORDS:
            found_offensive.append(word)
    
    return {
        "hate_words": found_hate,
        "offensive_words": found_offensive,
        "has_hate": len(found_hate) > 0,
        "has_offensive": len(found_offensive) > 0,
        "severity": "hate" if found_hate else ("offensive" if found_offensive else "normal")
    }
