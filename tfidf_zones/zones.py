# =============================================================================
# ZONE CLASSIFICATION
# =============================================================================
#
# Percentile-based zone classification for TF-IDF terms.
#
# Classifies terms into three zones based on document frequency (DF) rank:
#   - Too Common: top 10% by DF (highest DF values)
#   - Goldilocks: 45th-55th percentile by DF, fanned out from median
#   - Too Rare: bottom 10% by DF (lowest DF values)
#
# Used by both the pure Python and scikit-learn engines.
#
# =============================================================================

from __future__ import annotations


def classify_zones(
    all_scored: list[dict],
    top_k: int = 10,
) -> dict:
    """Classify terms into Too Common, Goldilocks, and Too Rare zones.

    Uses DF-rank percentiles to bucket terms:
      - Too Common: top 10% by DF rank (highest DF values)
      - Goldilocks: 45th-55th percentile by DF rank (narrow band around median)
      - Too Rare: bottom 10% by DF rank (lowest DF values)

    Within each zone, terms are sorted by TF-IDF score descending and
    limited to top_k entries.

    Goldilocks fans out from the median so the first term returned is the
    one closest to the center of the DF distribution, then alternates
    above/below.

    Args:
        all_scored: List of dicts, each with keys: term, score, df, idf.
        top_k: Max terms per zone. Default 10.

    Returns:
        Dict with keys too_common, goldilocks, too_rare, each containing
        a list of {term, score, df, idf} dicts.
    """
    if not all_scored:
        return {"too_common": [], "goldilocks": [], "too_rare": []}

    # Sort all terms by DF descending (highest DF first)
    sorted_by_df = sorted(all_scored, key=lambda t: t["df"], reverse=True)
    n = len(sorted_by_df)

    # Percentile boundaries (by DF rank position)
    top_10_cutoff = int(n * 0.10)
    gold_start = int(n * 0.45)
    gold_end = int(n * 0.55)
    bottom_10_start = int(n * 0.90)

    # Too common: top 10% by DF, sorted by score descending
    too_common = sorted(
        sorted_by_df[:top_10_cutoff],
        key=lambda x: x["score"],
        reverse=True,
    )

    # Too rare: bottom 10% by DF, sorted by DF ascending (rarest first)
    too_rare = sorted(
        sorted_by_df[bottom_10_start:],
        key=lambda x: x["df"],
    )

    # Goldilocks: fan out from median DF, sort by score for presentation
    gold_candidates = sorted_by_df[gold_start:gold_end]
    median = len(gold_candidates) // 2
    fanned = []
    for offset in range(len(gold_candidates)):
        below = median + offset
        above = median - offset - 1
        if below < len(gold_candidates):
            fanned.append(gold_candidates[below])
        if above >= 0 and above != below:
            fanned.append(gold_candidates[above])
        if len(fanned) >= top_k:
            break
    goldilocks = sorted(
        fanned[:top_k],
        key=lambda x: x["score"],
        reverse=True,
    )

    return {
        "too_common": too_common[:top_k],
        "goldilocks": goldilocks,
        "too_rare": too_rare[:top_k],
    }
