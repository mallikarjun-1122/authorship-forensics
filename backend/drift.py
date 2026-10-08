import re
import numpy as np

from backend.stylometry import feature_vector
from backend.preprocess import clean_text


def detect_style_drift(text: str) -> list[dict]:
    """Detect stylistic drift across paragraphs using leave-one-out feature comparison."""
    if not text or not text.strip():
        return []

    cleaned = clean_text(text)
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]

    # If the user only has single line breaks instead of blank lines, fallback to line breaks
    if len(raw_paragraphs) <= 1:
        lines = [l.strip() for l in cleaned.split("\n") if len(l.strip().split()) >= 10]
        if len(lines) >= 2:
            raw_paragraphs = lines

    # Merge very small snippets (< 12 words) only if there are enough remaining paragraphs
    paragraphs = []
    current_para = []
    for p in raw_paragraphs:
        current_para.append(p)
        word_count = sum(len(x.split()) for x in current_para)
        # Keep distinct if at least 12 words, or if this is one of only two paragraphs
        if word_count >= 12 or len(raw_paragraphs) <= 2:
            paragraphs.append("\n\n".join(current_para))
            current_para = []

    if current_para:
        if paragraphs:
            paragraphs[-1] += "\n\n" + "\n\n".join(current_para)
        else:
            paragraphs.append("\n\n".join(current_para))

    vectors = [np.asarray(feature_vector(part), dtype=float) for part in paragraphs]
    results = []

    for idx, vector in enumerate(vectors):
        if len(vectors) < 2:
            score = None
        else:
            # Leave this paragraph out of its comparison baseline
            baseline = np.mean([other for pos, other in enumerate(vectors) if pos != idx], axis=0)
            score = round((1.0 - float(np.mean(np.abs(vector - baseline)))) * 100, 2)
        
        words = len(paragraphs[idx].split())
        results.append({
            "paragraph": idx + 1,
            "similarity": score,
            "words": words
        })

    # Calculate drift outlier flags
    valid_scores = [r["similarity"] for r in results if r["similarity"] is not None]
    if len(valid_scores) > 2:
        mean_sim = float(np.mean(valid_scores))
        std_sim = float(np.std(valid_scores))
        threshold = mean_sim - 1.5 * std_sim if std_sim > 0.5 else mean_sim - 5.0
        for r in results:
            if r["similarity"] is not None and r["similarity"] < threshold:
                r["flag"] = "⚠️ High Drift"
            else:
                r["flag"] = "Normal"
    elif len(valid_scores) == 2:
        for r in results:
            r["flag"] = "Normal" if r["similarity"] is not None else "N/A"
    else:
        for r in results:
            r["flag"] = "N/A (Single Paragraph)"

    return results
