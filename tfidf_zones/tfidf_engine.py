# =============================================================================
# TF-IDF COMPUTATION ENGINE (PURE PYTHON)
# =============================================================================
#
# Pure Python TF-IDF implementation — no scikit-learn or numpy dependency.
#
# Matches scikit-learn's TfidfVectorizer defaults:
#   - smooth_idf=True: log((1+N)/(1+DF)) + 1
#   - L2 normalization available as a separate utility (not applied in pipeline)
#   - Sublinear TF OFF by default (raw TF, not 1+log(TF))
#
# =============================================================================

from __future__ import annotations

import logging
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass

from tfidf_zones.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

# N-gram level labels used in API responses
NGRAM_LABELS = {
    1: "unigrams",
    2: "bigrams",
    3: "trigrams",
    4: "fourgrams",
    5: "fivegrams",
    6: "skipgrams",
}

MIN_CHUNK_SIZE = 100


@dataclass
class EngineResult:
    """Shared result contract for both TF-IDF engines."""

    ngram_type: str
    top_terms: list[dict]
    all_scored: list[dict]
    total_unique_terms: int
    total_items: int
    total_tokens: int
    chunk_count: int
    df_stats: dict


# =============================================================================
# CORE TF-IDF FUNCTIONS
# =============================================================================


def compute_tf(tokens: list[str]) -> dict[str, float]:
    """Compute term frequency: count(t) / total_tokens."""
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()}


def compute_idf(chunks: list[list[str]]) -> tuple[dict[str, float], dict[str, int]]:
    """Compute inverse document frequency across chunks.

    Uses scikit-learn smooth_idf=True default — dampens but never zeros
    ubiquitous terms, preserving signal for stylometric analysis where
    even common words carry weight.

    Formula: log((1 + N) / (1 + DF)) + 1

    Returns:
        Tuple of (idf_scores, df_counts) where df_counts maps each term
        to the number of chunks it appears in.
    """
    n = len(chunks)
    if n == 0:
        return {}, {}
    df: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        for term in set(chunk):
            df[term] += 1
    idf = {
        term: math.log((1 + n) / (1 + freq)) + 1
        for term, freq in df.items()
    }
    return idf, dict(df)


def scale_tf_by_idf(tf: dict[str, float], idf: dict[str, float]) -> dict[str, float]:
    """Multiply TF scores by IDF weights."""
    return {term: tf_val * idf.get(term, 0.0) for term, tf_val in tf.items()}


# =============================================================================
# N-GRAM GENERATION
# =============================================================================


def generate_ngrams(tokens: list[str], n: int) -> list[str]:
    """Generate n-grams by sliding a window of size n over tokens."""
    if n < 1 or n > len(tokens):
        logger.warning("n-gram size %d exceeds token count %d; returning empty", n, len(tokens))
        return []
    return ["_".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def generate_skipgrams(tokens: list[str], skip: int = 1) -> list[str]:
    """Generate bigram skip-grams with a gap of `skip` tokens between pairs."""
    if len(tokens) < 2 + skip:
        logger.warning("Token count %d too small for skip=%d; returning empty", len(tokens), skip)
        return []
    return [
        f"{tokens[i]}_{tokens[i + 1 + skip]}"
        for i in range(len(tokens) - 1 - skip)
    ]


# =============================================================================
# CHUNKING
# =============================================================================


def chunk_tokens(tokens: list[str], chunk_size: int = 2000) -> list[list[str]]:
    """Split tokens into approximately equal chunks for internal corpus creation.

    Why chunking is necessary:
        Without chunking, a single document has DF=1 for every term and IDF
        becomes constant — TF-IDF collapses to just TF. Chunking creates
        sub-documents so IDF can discriminate between locally frequent and
        globally frequent terms.

    Chunking strategy:
        1. Divide tokens into N full chunks of chunk_size.
        2. If the remainder is >= 90% of chunk_size, keep it as a final chunk.
        3. If the remainder is < 90% (a "runt chunk"), drop it and redistribute
           all tokens evenly across N chunks.
        4. If the document is shorter than chunk_size, return it as a single chunk.
    """
    if chunk_size < MIN_CHUNK_SIZE:
        raise ValueError(f"chunk_size must be >= {MIN_CHUNK_SIZE}, got {chunk_size}")

    if not tokens:
        return []

    total = len(tokens)

    if total <= chunk_size:
        return [tokens]

    n_full = total // chunk_size
    remainder = total % chunk_size
    threshold = chunk_size * 0.9

    if remainder == 0:
        n_chunks = n_full
    elif remainder >= threshold:
        n_chunks = n_full + 1
    else:
        n_chunks = n_full

    base_size = total // n_chunks
    extra = total % n_chunks

    chunks = []
    start = 0
    for i in range(n_chunks):
        end = start + base_size + (1 if i < extra else 0)
        chunks.append(tokens[start:end])
        start = end

    return chunks


# =============================================================================
# AGGREGATION
# =============================================================================


def aggregate_tfidf(
    chunks: list[list[str]], idf: dict[str, float],
) -> tuple[dict[str, float], dict[str, int]]:
    """Compute TF-IDF per chunk and average raw scores across chunks.

    Returns:
        Tuple of (tfidf_scores, raw_counts) where raw_counts maps each term
        to its total occurrence count across all chunks.
    """
    if not chunks:
        return {}, {}
    merged: dict[str, float] = defaultdict(float)
    raw_counts: dict[str, int] = Counter()
    for chunk in chunks:
        tf = compute_tf(chunk)
        raw_scores = scale_tf_by_idf(tf, idf)
        for term, score in raw_scores.items():
            merged[term] += score
        raw_counts.update(chunk)
    n = len(chunks)
    return (
        {term: score / n for term, score in merged.items()},
        dict(raw_counts),
    )


# =============================================================================
# TOP-K EXTRACTION
# =============================================================================


def top_k_terms(
    scores: dict[str, float],
    k: int = 50,
    df: dict[str, int] | None = None,
    idf: dict[str, float] | None = None,
) -> list[dict]:
    """Extract top K terms by TF-IDF score, descending."""
    if k < 1:
        raise ValueError(f"top_k must be >= 1, got {k}")
    sorted_terms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for term, score in sorted_terms[:k]:
        entry = {"term": term, "score": score}
        if df is not None:
            entry["df"] = df.get(term, 0)
        if idf is not None:
            entry["idf"] = idf.get(term, 0.0)
        result.append(entry)
    return result


# =============================================================================
# DF STATS
# =============================================================================


def compute_df_stats(df: dict[str, int]) -> dict:
    """Compute mean, median, and mode of document frequency values."""
    if not df:
        return {"mean": 0.0, "median": 0.0, "mode": 0}
    values = list(df.values())
    return {
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "mode": statistics.mode(values),
    }


# =============================================================================
# UNIFIED PIPELINE
# =============================================================================


def tfidf_compute(
    tokens: list[str],
    ngram: int = 1,
    chunk_size: int = 2000,
    top_k: int = 50,
    no_ngram_stopwords: bool = False,
) -> dict:
    """Compute TF-IDF for a specific n-gram level.

    Args:
        tokens: List of word tokens from the tokenizer.
        ngram: N-gram level (1-6). Default 1 (unigrams).
        chunk_size: Target chunk size for IDF computation. Must be >= 100.
        top_k: Number of top terms to return. Must be >= 1.

    Returns:
        Dict with keys: ngram_type, top_terms, all_scored, total_unique_terms,
        total_items, total_tokens, chunk_count, df_stats.
    """
    if ngram < 1 or ngram > 6:
        raise ValueError(f"ngram must be 1-6, got {ngram}")
    if chunk_size < MIN_CHUNK_SIZE:
        raise ValueError(f"chunk_size must be >= {MIN_CHUNK_SIZE}, got {chunk_size}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    if not tokens:
        return {
            "ngram_type": NGRAM_LABELS[ngram],
            "top_terms": [],
            "all_scored": [],
            "total_unique_terms": 0,
            "total_items": 0,
            "total_tokens": 0,
            "chunk_count": 0,
            "df_stats": {"mean": 0.0, "median": 0.0, "mode": 0},
        }

    # Generate items for the requested n-gram level
    if ngram == 1:
        items = tokens
    elif ngram == 6:
        items = generate_skipgrams(tokens)
    else:
        items = generate_ngrams(tokens, ngram)

    if no_ngram_stopwords and ngram >= 2:
        from tfidf_zones.word_lists import filter_ngrams
        items = filter_ngrams(items)

    if not items:
        return {
            "ngram_type": NGRAM_LABELS[ngram],
            "top_terms": [],
            "all_scored": [],
            "total_unique_terms": 0,
            "total_items": 0,
            "total_tokens": len(tokens),
            "chunk_count": 0,
            "df_stats": {"mean": 0.0, "median": 0.0, "mode": 0},
        }

    chunks = chunk_tokens(items, chunk_size)
    idf, df = compute_idf(chunks)
    scores, raw_counts = aggregate_tfidf(chunks, idf)

    # Build all_scored list (every term with score/tf/df/idf)
    all_scored = sorted(
        [
            {
                "term": term,
                "score": score,
                "tf": raw_counts.get(term, 0),
                "df": df.get(term, 0),
                "idf": idf.get(term, 0.0),
            }
            for term, score in scores.items()
        ],
        key=lambda x: x["score"],
        reverse=True,
    )

    return {
        "ngram_type": NGRAM_LABELS[ngram],
        "top_terms": all_scored[:top_k],
        "all_scored": all_scored,
        "total_unique_terms": len(scores),
        "total_items": len(items),
        "total_tokens": len(tokens),
        "chunk_count": len(chunks),
        "df_stats": compute_df_stats(df),
    }


# =============================================================================
# PUBLIC API
# =============================================================================


def run(text: str, ngram: int = 1, chunk_size: int = 2000, top_k: int = 50, wordnet: bool = False, no_ngram_stopwords: bool = False) -> EngineResult:
    """Run the pure Python TF-IDF engine on raw text.

    Tokenizes text using the pystylometry Tokenizer, then computes TF-IDF.
    """
    tokenizer = Tokenizer(
        lowercase=True,
        strip_punctuation=True,
        strip_numbers=True,
        min_length=2,
        wordnet_filter=wordnet,
    )
    tokens = tokenizer.tokenize(text)
    result = tfidf_compute(tokens, ngram=ngram, chunk_size=chunk_size, top_k=top_k, no_ngram_stopwords=no_ngram_stopwords)

    return EngineResult(
        ngram_type=result["ngram_type"],
        top_terms=result["top_terms"],
        all_scored=result["all_scored"],
        total_unique_terms=result["total_unique_terms"],
        total_items=result["total_items"],
        total_tokens=result["total_tokens"],
        chunk_count=result["chunk_count"],
        df_stats=result["df_stats"],
    )


def run_docs(docs: list[str], ngram: int = 1, top_k: int = 50, wordnet: bool = False, no_ngram_stopwords: bool = False) -> EngineResult:
    """Run TF-IDF where each document string is one corpus document (no chunking).

    Each entry in docs is tokenized independently and treated as a single
    document for IDF computation.
    """
    if ngram < 1 or ngram > 6:
        raise ValueError(f"ngram must be 1-6, got {ngram}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    tokenizer = Tokenizer(
        lowercase=True,
        strip_punctuation=True,
        strip_numbers=True,
        min_length=2,
        wordnet_filter=wordnet,
    )

    _filter = None
    if no_ngram_stopwords and ngram >= 2:
        from tfidf_zones.word_lists import filter_ngrams
        _filter = filter_ngrams

    # Tokenize each document separately and generate ngrams
    doc_chunks: list[list[str]] = []
    total_tokens = 0
    total_items = 0
    for doc in docs:
        tokens = tokenizer.tokenize(doc)
        total_tokens += len(tokens)
        if ngram == 1:
            items = tokens
        elif ngram == 6:
            items = generate_skipgrams(tokens)
        else:
            items = generate_ngrams(tokens, ngram)
        if _filter is not None:
            items = _filter(items)
        total_items += len(items)
        if items:
            doc_chunks.append(items)

    if not doc_chunks:
        return EngineResult(
            ngram_type=NGRAM_LABELS[ngram],
            top_terms=[],
            all_scored=[],
            total_unique_terms=0,
            total_items=0,
            total_tokens=total_tokens,
            chunk_count=0,
            df_stats={"mean": 0.0, "median": 0.0, "mode": 0},
        )

    idf, df = compute_idf(doc_chunks)
    scores, raw_counts = aggregate_tfidf(doc_chunks, idf)

    all_scored = sorted(
        [
            {
                "term": term,
                "score": score,
                "tf": raw_counts.get(term, 0),
                "df": df.get(term, 0),
                "idf": idf.get(term, 0.0),
            }
            for term, score in scores.items()
        ],
        key=lambda x: x["score"],
        reverse=True,
    )

    return EngineResult(
        ngram_type=NGRAM_LABELS[ngram],
        top_terms=all_scored[:top_k],
        all_scored=all_scored,
        total_unique_terms=len(scores),
        total_items=total_items,
        total_tokens=total_tokens,
        chunk_count=len(doc_chunks),
        df_stats=compute_df_stats(df),
    )
