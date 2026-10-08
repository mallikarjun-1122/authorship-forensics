import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.stylometry import feature_vector
from backend.preprocess import clean_text


def verify_author(known_docs: list[str], unknown_doc: str) -> dict:
    """Return both basic stylometric features and advanced n-gram cosine similarity."""
    if not known_docs:
        raise ValueError("Upload at least one known-author document.")

    known_cleaned = [clean_text(doc) for doc in known_docs if clean_text(doc)]
    unknown_cleaned = clean_text(unknown_doc)

    if not known_cleaned:
        raise ValueError("Known documents must contain readable text.")
    if not unknown_cleaned:
        raise ValueError("The document to analyze is empty.")

    # 1. Basic Stylometric Feature Verification
    vectors = np.asarray([feature_vector(doc) for doc in known_cleaned], dtype=float)
    profile = np.mean(vectors, axis=0)
    unknown_vector = np.asarray(feature_vector(unknown_cleaned), dtype=float)
    mean_difference = float(np.mean(np.abs(profile - unknown_vector)))
    basic_score = round(max(0.0, (1.0 - mean_difference) * 100), 2)

    # 2. Advanced Technique: TF-IDF N-Gram Cosine Similarity
    # In stylometry, function words are critical style fingerprints, so stop_words are preserved.
    explainable_ai = {"shared": [], "missing": []}
    try:
        vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            max_features=3000,
            lowercase=True,
            stop_words=None  # Preserving function words to capture authentic style signatures
        )
        # Fit on all known documents
        known_tfidf = vectorizer.fit_transform(known_cleaned)
        unknown_tfidf = vectorizer.transform([unknown_cleaned])

        # Compute cosine similarity
        similarities = cosine_similarity(unknown_tfidf, known_tfidf)
        advanced_score = round(float(np.mean(similarities)) * 100, 2)

        # Explainable AI (XAI)
        mean_known = np.asarray(known_tfidf.mean(axis=0)).flatten()
        unknown_vec = np.asarray(unknown_tfidf.toarray()).flatten()
        feat_names = vectorizer.get_feature_names_out()

        # Shared impactful patterns
        contribution = mean_known * unknown_vec
        top_sim_idx = contribution.argsort()[-10:][::-1]
        shared_ngrams = [feat_names[i] for i in top_sim_idx if contribution[i] > 0]

        # Author patterns missing in analyzed document
        missing_idx = mean_known.argsort()[-50:][::-1]
        missing_ngrams = [feat_names[i] for i in missing_idx if unknown_vec[i] == 0][:10]

        explainable_ai = {
            "shared": shared_ngrams,
            "missing": missing_ngrams
        }
    except Exception:
        # Fallback if vocabulary cannot be formed
        advanced_score = basic_score

    return {
        "basic_score": basic_score,
        "advanced_score": advanced_score,
        "profile_vector": profile.tolist(),
        "unknown_vector": unknown_vector.tolist(),
        "xai": explainable_ai
    }
