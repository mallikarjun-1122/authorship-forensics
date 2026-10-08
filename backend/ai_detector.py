import re
import numpy as np

INDICATORS = (
    "furthermore", "moreover", "therefore", "in conclusion", "it is important to note",
    "delve", "tapestry", "multifaceted", "testament", "realm", "underscore", "crucial",
    "navigate", "nuance", "beacon", "foster", "align", "unwavering", "intricate", 
    "landscape", "in summary", "ultimately", "undeniably", "embark", "robust", 
    "paramount", "symbiosis", "intersection", "pivotal", "profound", "meticulous",
    "seamless", "resilience", "catalyst", "holistic", "dynamic", "paradigm",
    "in today's digital age", "rapidly evolving", "ever-changing", "shed light on",
    "it is worth noting", "a testament to", "the realm of", "not merely", "but rather",
    "intricacies", "nuances", "synergy", "fostering", "empowering", "comprehensive"
)


def detect_ai_text(text: str) -> dict:
    """Calculate AI score using phrase flags and burstiness (sentence variance)."""
    if not text or not text.strip():
        return {
            "score": 0,
            "indicators": [],
            "burstiness": 0.0,
            "burst_penalty": 0,
            "matched_count": 0,
        }

    lower_text = text.casefold()
    matches = []
    total_hits = 0

    for phrase in INDICATORS:
        found = len(re.findall(r"\b" + re.escape(phrase) + r"\b", lower_text))
        if found > 0:
            matches.append(phrase)
            total_hits += found

    word_count = len(text.split())
    if word_count == 0:
        return {
            "score": 0,
            "indicators": [],
            "burstiness": 0.0,
            "burst_penalty": 0,
            "matched_count": 0,
        }

    # Phrase score (max 70 points, scaled smoothly)
    frequency_per_1000 = (total_hits / word_count) * 1000
    phrase_score = min(int(frequency_per_1000 * 3.5), 70) if total_hits > 0 else 0

    # Burstiness (Variance in sentence length)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_lengths = [len(s.split()) for s in sentences]

    if len(sentences) > 1:
        burstiness = float(np.std(sentence_lengths))
        # Humans usually have burstiness > 8.0, AI often has < 6.0
        if burstiness < 4.0:
            burst_penalty = 30
        elif burstiness < 7.0:
            burst_penalty = 15
        else:
            burst_penalty = 0
    else:
        burstiness = 0.0
        burst_penalty = 0  # Cannot penalize for variance on a single sentence

    total_score = min(phrase_score + burst_penalty, 100)

    return {
        "score": total_score,
        "indicators": matches,
        "matched_count": total_hits,
        "burstiness": round(burstiness, 2),
        "burst_penalty": burst_penalty,
    }
