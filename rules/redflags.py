SENSATIONAL_PATTERNS = [
    r"\b(urgent|breaking|shocking)\b",
    r"\b(they don't want you to know|hidden truth)\b",
    r"(! ){3,}",  # Multiple exclamation marks
    r"[A-Z]{10,}"  # All-caps phrases
]

UNRELIABLE_DOMAINS = [
    "infowars.com",
    "naturalnews.com",
    "beforeitsnews.com"
]

CLICKBAIT_PHRASES = [
    "you won't believe",
    "what happened next",
    "doctors hate this"
]