# =============================================================================
# PUBLIC API
# =============================================================================
#
# Programmatic interface to TF-IDF Zone Analysis.
#
# Usage:
#   from tfidf_zones import analyze, analyze_docs, to_csv
#
#   result = analyze("the cat sat on the mat ...", ngram=2, top_k=50)
#   result = analyze_docs(["doc one ...", "doc two ..."], ngram=2, top_k=50)
#   csv_str = to_csv(result)
#
# =============================================================================

from __future__ import annotations

import csv
import io
import time

from tfidf_zones.runner import AnalysisResult
from tfidf_zones.tfidf_engine import EngineResult
from tfidf_zones.zones import classify_zones


def _apply_filters(
    engine_result: EngineResult,
    zones: dict,
    min_df: int | None,
    min_tf: int | None,
    top_k: int,
) -> tuple[EngineResult, dict]:
    """Apply min_df/min_tf filters and re-classify zones.

    Returns the mutated engine_result and new zones dict.
    """
    if min_df is None and min_tf is None:
        return engine_result, zones

    filtered = engine_result.all_scored
    if min_df is not None:
        filtered = [t for t in filtered if t["df"] >= min_df]
    if min_tf is not None:
        filtered = [t for t in filtered if t.get("tf", 0) >= min_tf]

    engine_result.all_scored = filtered
    engine_result.top_terms = sorted(filtered, key=lambda x: x["score"], reverse=True)[:top_k]
    zones = classify_zones(filtered, top_k=top_k, chunk_count=engine_result.chunk_count)
    return engine_result, zones


def _build_zone_lookup(all_scored: list[dict], chunk_count: int) -> dict[str, int]:
    """Classify every term into a zone: 1=too_common, 2=goldilocks, 3=too_rare."""
    if not all_scored:
        return {}

    n = chunk_count if chunk_count > 0 else max(t["df"] for t in all_scored)
    df_upper = max(3, int(n * 0.2))
    df_lower = 3
    if df_upper <= df_lower:
        df_lower = 2
        df_upper = max(df_lower + 1, df_upper)

    scores = sorted((t["score"] for t in all_scored), reverse=True)
    p95_index = max(0, int(len(scores) * 0.05) - 1)
    tfidf_threshold = scores[p95_index]

    lookup: dict[str, int] = {}
    for t in all_scored:
        df = t["df"]
        if df > df_upper:
            lookup[t["term"]] = 1
        elif df < df_lower:
            lookup[t["term"]] = 3
        elif t["score"] >= tfidf_threshold:
            lookup[t["term"]] = 2
    return lookup


def analyze(
    text: str,
    engine: str = "pure",
    ngram: int = 1,
    chunk_size: int = 2000,
    top_k: int = 10,
    wordnet: bool = False,
    no_ngram_stopwords: bool = False,
    min_df: int | None = None,
    min_tf: int | None = None,
) -> AnalysisResult:
    """Run TF-IDF zone analysis on a text string.

    Args:
        text: Input text to analyze.
        engine: "pure" or "scikit".
        ngram: N-gram level (1-6).
        chunk_size: Tokens per chunk.
        top_k: Terms per zone.
        wordnet: Only recognized English words participate.
        no_ngram_stopwords: Discard n-grams with stop/function words.
        min_df: Remove terms with DF below this value.
        min_tf: Remove terms with TF below this value.

    Returns:
        AnalysisResult with engine output and zone classification.
    """
    start = time.perf_counter()

    if engine == "scikit":
        from tfidf_zones.scikit_engine import run
    else:
        from tfidf_zones.tfidf_engine import run

    engine_result = run(
        text,
        ngram=ngram,
        chunk_size=chunk_size,
        top_k=top_k,
        wordnet=wordnet,
        no_ngram_stopwords=no_ngram_stopwords,
    )
    zones = classify_zones(engine_result.all_scored, top_k=top_k, chunk_count=engine_result.chunk_count)
    engine_result, zones = _apply_filters(engine_result, zones, min_df, min_tf, top_k)

    elapsed = time.perf_counter() - start

    return AnalysisResult(
        filename="<text>",
        text_length=len(text),
        engine_name=engine,
        engine_result=engine_result,
        zones=zones,
        elapsed=elapsed,
        chunk_size=chunk_size,
    )


def analyze_docs(
    docs: list[str],
    engine: str = "pure",
    ngram: int = 1,
    top_k: int = 10,
    wordnet: bool = False,
    no_ngram_stopwords: bool = False,
    min_df: int | None = None,
    min_tf: int | None = None,
) -> AnalysisResult:
    """Run TF-IDF zone analysis on a list of documents.

    Each string in docs is treated as a separate document (no chunking).

    Args:
        docs: List of document strings.
        engine: "pure" or "scikit".
        ngram: N-gram level (1-6).
        top_k: Terms per zone.
        wordnet: Only recognized English words participate.
        no_ngram_stopwords: Discard n-grams with stop/function words.
        min_df: Remove terms with DF below this value.
        min_tf: Remove terms with TF below this value.

    Returns:
        AnalysisResult with engine output and zone classification.
    """
    start = time.perf_counter()

    if engine == "scikit":
        from tfidf_zones.scikit_engine import run_docs as _run_docs
    else:
        from tfidf_zones.tfidf_engine import run_docs as _run_docs

    engine_result = _run_docs(
        docs,
        ngram=ngram,
        top_k=top_k,
        wordnet=wordnet,
        no_ngram_stopwords=no_ngram_stopwords,
    )
    zones = classify_zones(engine_result.all_scored, top_k=top_k, chunk_count=engine_result.chunk_count)
    engine_result, zones = _apply_filters(engine_result, zones, min_df, min_tf, top_k)

    elapsed = time.perf_counter() - start

    return AnalysisResult(
        filename="<docs>",
        text_length=sum(len(d) for d in docs),
        engine_name=engine,
        engine_result=engine_result,
        zones=zones,
        elapsed=elapsed,
        chunk_size=0,
        file_count=len(docs),
    )


def to_csv(result: AnalysisResult) -> str:
    """Convert an AnalysisResult to CSV string.

    Returns CSV content with columns:
    term, tf, df, idf, tfidf, tf_pct, tf_cum_norm, zone
    """
    all_scored = result.engine_result.all_scored
    total_tokens = result.engine_result.total_tokens

    zone_lookup = _build_zone_lookup(all_scored, result.engine_result.chunk_count)

    tf_sorted = sorted(all_scored, key=lambda x: x.get("tf", 0), reverse=True)
    cum_lookup: dict[str, tuple[float, int]] = {}
    running = 0.0
    for entry in tf_sorted:
        tf_pct = entry.get("tf", 0) / total_tokens if total_tokens > 0 else 0.0
        running += tf_pct
        tf_cum_norm = max(1, round(running * 100)) if total_tokens > 0 else 0
        cum_lookup[entry["term"]] = (tf_pct, tf_cum_norm)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["term", "tf", "df", "idf", "tfidf", "tf_pct", "tf_cum_norm", "zone"])
    for entry in all_scored:
        tf_pct, tf_cum_norm = cum_lookup[entry["term"]]
        zone = zone_lookup.get(entry["term"], "")
        writer.writerow([
            entry["term"],
            entry.get("tf", 0),
            entry["df"],
            entry["idf"],
            entry["score"],
            tf_pct,
            tf_cum_norm,
            zone,
        ])
    return buf.getvalue()
