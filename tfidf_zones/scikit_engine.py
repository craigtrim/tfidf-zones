# =============================================================================
# TF-IDF COMPUTATION ENGINE (SCIKIT-LEARN)
# =============================================================================
#
# TF-IDF implementation using scikit-learn's TfidfVectorizer.
#
# Uses the same pystylometry tokenizer as the pure Python engine — no stop
# word removal. Function words are preserved for stylometric analysis.
#
# Key differences from the pure engine:
#   - L2 normalization applied by default (sklearn default)
#   - Chunking splits raw text by word count (not pre-tokenized)
#   - sklearn handles the TF-IDF math internally
#
# =============================================================================

from __future__ import annotations

import logging
import statistics
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer

from tfidf_zones.tfidf_engine import EngineResult, NGRAM_LABELS, MIN_CHUNK_SIZE
from tfidf_zones.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

# Shared tokenizer instance for all analyzer functions
_tokenizer = Tokenizer(
    lowercase=True,
    strip_punctuation=True,
    strip_numbers=True,
    min_length=2,
)


# =============================================================================
# CHUNKING
# =============================================================================


def chunk_text(text: str, chunk_size: int = 2000) -> list[str]:
    """Split text into chunks of approximately chunk_size words.

    Each chunk is a raw text string (not pre-tokenized) so that
    TfidfVectorizer can apply its own analyzer to each chunk independently.
    """
    words = text.split()
    total = len(words)

    if total <= chunk_size:
        return [text]

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
        chunks.append(" ".join(words[start:end]))
        start = end

    return chunks


# =============================================================================
# CUSTOM ANALYZERS
# =============================================================================


def custom_analyzer(text: str) -> list[str]:
    """Custom analyzer for TfidfVectorizer using the pystylometry tokenizer."""
    return _tokenizer.tokenize(text)


def make_ngram_analyzer(n: int):
    """Factory that returns an analyzer producing n-grams of size n."""
    def analyzer(text: str) -> list[str]:
        tokens = _tokenizer.tokenize(text)
        if len(tokens) < n:
            return []
        return ["_".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return analyzer


def skipgram_analyzer(text: str) -> list[str]:
    """Analyzer that produces bigrams with 1 word skipped (skip-1-grams)."""
    tokens = _tokenizer.tokenize(text)
    if len(tokens) < 3:
        return []
    return [f"{tokens[i]}_{tokens[i + 2]}" for i in range(len(tokens) - 2)]


# =============================================================================
# PUBLIC API
# =============================================================================


def run(text: str, ngram: int = 1, chunk_size: int = 2000, top_k: int = 50) -> EngineResult:
    """Run the scikit-learn TF-IDF engine on raw text.

    Tokenizes text using the pystylometry Tokenizer (same as the pure engine),
    then uses sklearn's TfidfVectorizer for TF-IDF computation.
    """
    if ngram < 1 or ngram > 6:
        raise ValueError(f"ngram must be 1-6, got {ngram}")
    if chunk_size < MIN_CHUNK_SIZE:
        raise ValueError(f"chunk_size must be >= {MIN_CHUNK_SIZE}, got {chunk_size}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    # Count total tokens before chunking
    all_tokens = _tokenizer.tokenize(text)
    total_tokens = len(all_tokens)

    if total_tokens == 0:
        return EngineResult(
            ngram_type=NGRAM_LABELS[ngram],
            top_terms=[],
            all_scored=[],
            total_unique_terms=0,
            total_items=0,
            total_tokens=0,
            chunk_count=0,
            df_stats={"mean": 0.0, "median": 0.0, "mode": 0},
        )

    # Chunk the text into sub-documents
    chunks = chunk_text(text, chunk_size)
    n_chunks = len(chunks)

    logger.info(
        "Processing %d characters, %d tokens, %d chunks (ngram=%d)",
        len(text), total_tokens, n_chunks, ngram,
    )

    # Configure TfidfVectorizer with the appropriate analyzer
    if ngram == 6:
        vectorizer = TfidfVectorizer(analyzer=skipgram_analyzer, min_df=1)
    elif ngram == 1:
        vectorizer = TfidfVectorizer(analyzer=custom_analyzer, min_df=1)
    else:
        vectorizer = TfidfVectorizer(analyzer=make_ngram_analyzer(ngram), min_df=1)

    tfidf_matrix = vectorizer.fit_transform(chunks)
    feature_names = vectorizer.get_feature_names_out()

    # Compute average TF-IDF score across all chunks for each term
    mean_scores = tfidf_matrix.mean(axis=0).A1

    # Get document frequency for each term
    doc_freq = (tfidf_matrix > 0).sum(axis=0).A1.astype(int)

    # Compute IDF values (sklearn stores them internally)
    idf_values = vectorizer.idf_

    # Build scored term list (all terms)
    all_scored = []
    for i in range(len(feature_names)):
        if mean_scores[i] > 0:
            all_scored.append({
                "term": feature_names[i],
                "score": round(float(mean_scores[i]), 6),
                "df": int(doc_freq[i]),
                "idf": round(float(idf_values[i]), 6),
            })

    all_scored.sort(key=lambda x: x["score"], reverse=True)

    # Compute DF stats for output parity with pure engine
    df_values = [int(v) for v in doc_freq if v > 0]
    if df_values:
        df_stats = {
            "mean": round(statistics.mean(df_values), 2),
            "median": round(statistics.median(df_values), 2),
            "mode": statistics.mode(df_values),
        }
    else:
        df_stats = {"mean": 0.0, "median": 0.0, "mode": 0}

    return EngineResult(
        ngram_type=NGRAM_LABELS[ngram],
        top_terms=all_scored[:top_k],
        all_scored=all_scored,
        total_unique_terms=len(feature_names),
        total_items=len(all_scored),
        total_tokens=total_tokens,
        chunk_count=n_chunks,
        df_stats=df_stats,
    )
