import re
import numpy as np


def feature_vector(text):
    """Return bounded style features so one large-scale feature cannot dominate."""
    words = re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)
    sentences = [part for part in re.split(r"[.!?]+", text) if part.strip()]
    chars = max(len(text), 1)
    
    sentence_lengths = [len(s.split()) for s in sentences] if sentences else [0]
    sent_len_var = float(np.std(sentence_lengths)) if sentences else 0.0

    stop_words = {"the", "and", "to", "of", "a", "in", "that", "is", "it", "for", "with", "as", "on", "was", "at"}
    stop_freq = sum(1 for w in words if w.casefold() in stop_words) / max(len(words), 1)

    raw_features = [
        (sum(len(word) for word in words) / len(words) / 10) if words else 0,
        (len(words) / len(sentences) / 40) if sentences else 0,
        (len(set(word.casefold() for word in words)) / len(words)) if words else 0,
        sum(text.count(mark) for mark in ".,;:!?") / chars / 0.1,
        sum(char.isupper() for char in text) / chars / 0.25,
        sum(char.isdigit() for char in text) / chars / 0.2,
        text.count(",") / chars / 0.05,
        sent_len_var / 20, # AI tends to have low variance (e.g. 5-10), humans have higher
        stop_freq / 0.5,   # Stopword density
    ]
    return [min(max(value, 0), 1) for value in raw_features]


def feature_names():
    return [
        "Avg Word Length",
        "Avg Sentence Length",
        "Vocabulary Richness",
        "Punctuation Freq",
        "Uppercase Freq",
        "Digit Freq",
        "Comma Freq",
        "Sentence Length Variance",
        "Function Word Density",
    ]
