# =============================================================================
# ZONE CLASSIFICATION
# =============================================================================
#
# TF-IDF + DF band-pass zone classification.
#
# Classifies terms into three zones using both TF-IDF score thresholds
# and document frequency constraints:
#   - Too Common: df > 0.2N (ubiquitous terms)
#   - Goldilocks: TFIDF >= Q_0.95 AND 3 <= df <= 0.2N (discriminators)
#   - Too Rare: df < 3 (noise, typos, single-use terms)
#
# Used by both the pure Python and scikit-learn engines.
#
# =============================================================================

from __future__ import annotations


def classify_zones(
    all_scored: list[dict],
    top_k: int = 10,
    chunk_count: int = 0,
) -> dict:
    """Classify terms into Too Common, Goldilocks, and Too Rare zones.

    Uses a TF-IDF score threshold combined with a document frequency
    band-pass filter:
      - Too Common: df > 0.2N (appears in more than 20% of chunks)
      - Goldilocks: TFIDF >= 95th percentile AND 3 <= df <= 0.2N
      - Too Rare: df < 3 (appears in fewer than 3 chunks)

    Within each zone, terms are sorted by TF-IDF score descending and
    limited to top_k entries.

    Args:
        all_scored: List of dicts, each with keys: term, score, df, idf.
        top_k: Max terms per zone. Default 10.
        chunk_count: Total number of chunks/documents (N). Used for the
            0.2N upper DF bound. When 0, derived from max DF in the data.

    Returns:
        Dict with keys too_common, goldilocks, too_rare, each containing
        a list of {term, score, df, idf} dicts.
    """
    if not all_scored:
        return {"too_common": [], "goldilocks": [], "too_rare": []}

    # Determine N (total chunks/documents)
    n = chunk_count if chunk_count > 0 else max(t["df"] for t in all_scored)

    # DF bounds
    df_upper = max(3, int(n * 0.2))  # too-common threshold
    df_lower = 3                      # too-rare threshold

    # For small corpora where 0.2N < 3, relax lower bound
    if df_upper <= df_lower:
        df_lower = 2
        df_upper = max(df_lower + 1, df_upper)

    # TF-IDF score threshold: 95th percentile
    scores = sorted((t["score"] for t in all_scored), reverse=True)
    p95_index = max(0, int(len(scores) * 0.05) - 1)
    tfidf_threshold = scores[p95_index]

    # Classify
    too_common = []
    too_rare = []
    goldilocks = []

    for t in all_scored:
        df = t["df"]
        if df > df_upper:
            too_common.append(t)
        elif df < df_lower:
            too_rare.append(t)
        elif t["score"] >= tfidf_threshold:
            goldilocks.append(t)

    # Sort and limit
    too_common.sort(key=lambda x: x["score"], reverse=True)
    goldilocks.sort(key=lambda x: x["score"], reverse=True)
    too_rare.sort(key=lambda x: x["df"])

    return {
        "too_common": too_common[:top_k],
        "goldilocks": goldilocks[:top_k],
        "too_rare": too_rare[:top_k],
    }
