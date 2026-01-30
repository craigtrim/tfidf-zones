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

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from tfidf_zones.tfidf_engine import EngineResult, NGRAM_LABELS, MIN_CHUNK_SIZE
from tfidf_zones.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

# Shared tokenizer instance for all analyzer functions (default, no wordnet)
_tokenizer = Tokenizer(
    lowercase=True,
    strip_punctuation=True,
    strip_numbers=True,
    min_length=2,
)


def _make_tokenizer(wordnet: bool = False) -> Tokenizer:
    """Create a tokenizer, optionally with WordNet filtering."""
    return Tokenizer(
        lowercase=True,
        strip_punctuation=True,
        strip_numbers=True,
        min_length=2,
        wordnet_filter=wordnet,
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


def _make_custom_analyzer(tokenizer: Tokenizer):
    """Factory for a custom unigram analyzer using the given tokenizer."""
    def analyzer(text: str) -> list[str]:
        return tokenizer.tokenize(text)
    return analyzer


def _make_ngram_analyzer(n: int, tokenizer: Tokenizer, ngram_filter=None):
    """Factory for an n-gram analyzer using the given tokenizer."""
    def analyzer(text: str) -> list[str]:
        tokens = tokenizer.tokenize(text)
        if len(tokens) < n:
            return []
        ngrams = ["_".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        if ngram_filter is not None:
            ngrams = ngram_filter(ngrams)
        return ngrams
    return analyzer


def _make_skipgram_analyzer(tokenizer: Tokenizer, ngram_filter=None):
    """Factory for a skipgram analyzer using the given tokenizer."""
    def analyzer(text: str) -> list[str]:
        tokens = tokenizer.tokenize(text)
        if len(tokens) < 3:
            return []
        ngrams = [f"{tokens[i]}_{tokens[i + 2]}" for i in range(len(tokens) - 2)]
        if ngram_filter is not None:
            ngrams = ngram_filter(ngrams)
        return ngrams
    return analyzer


# Default analyzers (no wordnet filtering)
custom_analyzer = _make_custom_analyzer(_tokenizer)
make_ngram_analyzer = lambda n: _make_ngram_analyzer(n, _tokenizer)
skipgram_analyzer = _make_skipgram_analyzer(_tokenizer)


# =============================================================================
# PUBLIC API
# =============================================================================


def run(text: str, ngram: int = 1, chunk_size: int = 2000, top_k: int = 50, wordnet: bool = False, no_ngram_stopwords: bool = False) -> EngineResult:
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

    # Optional n-gram stop word filter
    _filter = None
    if no_ngram_stopwords and ngram >= 2:
        from tfidf_zones.word_lists import filter_ngrams
        _filter = filter_ngrams

    # Select tokenizer and analyzers
    if wordnet:
        tok = _make_tokenizer(wordnet=True)
        _uni = _make_custom_analyzer(tok)
        _ngram_fn = lambda n: _make_ngram_analyzer(n, tok, ngram_filter=_filter)
        _skip = _make_skipgram_analyzer(tok, ngram_filter=_filter)
    else:
        tok = _tokenizer
        _uni = custom_analyzer
        if _filter is not None:
            _ngram_fn = lambda n: _make_ngram_analyzer(n, _tokenizer, ngram_filter=_filter)
            _skip = _make_skipgram_analyzer(_tokenizer, ngram_filter=_filter)
        else:
            _ngram_fn = make_ngram_analyzer
            _skip = skipgram_analyzer

    # Count total tokens before chunking
    all_tokens = tok.tokenize(text)
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
        vectorizer = TfidfVectorizer(analyzer=_skip, min_df=1)
    elif ngram == 1:
        vectorizer = TfidfVectorizer(analyzer=_uni, min_df=1)
    else:
        vectorizer = TfidfVectorizer(analyzer=_ngram_fn(ngram), min_df=1)

    tfidf_matrix = vectorizer.fit_transform(chunks)
    feature_names = vectorizer.get_feature_names_out()

    # Compute average TF-IDF score across all chunks for each term
    mean_scores = tfidf_matrix.mean(axis=0).A1

    # Get document frequency for each term
    doc_freq = (tfidf_matrix > 0).sum(axis=0).A1.astype(int)

    # Compute IDF values (sklearn stores them internally)
    idf_values = vectorizer.idf_

    # Compute raw term counts using CountVectorizer
    if ngram == 6:
        count_vec = CountVectorizer(analyzer=_skip, min_df=1)
    elif ngram == 1:
        count_vec = CountVectorizer(analyzer=_uni, min_df=1)
    else:
        count_vec = CountVectorizer(analyzer=_ngram_fn(ngram), min_df=1)
    count_matrix = count_vec.fit_transform(chunks)
    # Sum raw counts across all chunks for each term
    import numpy as np
    total_counts = np.asarray(count_matrix.sum(axis=0)).flatten()
    count_features = count_vec.get_feature_names_out()
    tf_lookup = {count_features[i]: int(total_counts[i]) for i in range(len(count_features))}

    # Build scored term list (all terms)
    all_scored = []
    for i in range(len(feature_names)):
        if mean_scores[i] > 0:
            all_scored.append({
                "term": feature_names[i],
                "score": float(mean_scores[i]),
                "tf": tf_lookup.get(feature_names[i], 0),
                "df": int(doc_freq[i]),
                "idf": float(idf_values[i]),
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


def run_docs(docs: list[str], ngram: int = 1, top_k: int = 50, wordnet: bool = False, no_ngram_stopwords: bool = False) -> EngineResult:
    """Run TF-IDF where each document string is one corpus document (no chunking).

    Each entry in docs is passed directly to TfidfVectorizer as a separate
    document for IDF computation.
    """
    import numpy as np

    if ngram < 1 or ngram > 6:
        raise ValueError(f"ngram must be 1-6, got {ngram}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    # Optional n-gram stop word filter
    _filter = None
    if no_ngram_stopwords and ngram >= 2:
        from tfidf_zones.word_lists import filter_ngrams
        _filter = filter_ngrams

    # Select tokenizer and analyzers
    if wordnet:
        tok = _make_tokenizer(wordnet=True)
        _uni = _make_custom_analyzer(tok)
        _ngram_fn = lambda n: _make_ngram_analyzer(n, tok, ngram_filter=_filter)
        _skip = _make_skipgram_analyzer(tok, ngram_filter=_filter)
    else:
        tok = _tokenizer
        _uni = custom_analyzer
        if _filter is not None:
            _ngram_fn = lambda n: _make_ngram_analyzer(n, _tokenizer, ngram_filter=_filter)
            _skip = _make_skipgram_analyzer(_tokenizer, ngram_filter=_filter)
        else:
            _ngram_fn = make_ngram_analyzer
            _skip = skipgram_analyzer

    # Count total tokens across all docs
    total_tokens = sum(len(tok.tokenize(doc)) for doc in docs)

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

    n_docs = len(docs)

    # Configure TfidfVectorizer with the appropriate analyzer
    if ngram == 6:
        vectorizer = TfidfVectorizer(analyzer=_skip, min_df=1)
    elif ngram == 1:
        vectorizer = TfidfVectorizer(analyzer=_uni, min_df=1)
    else:
        vectorizer = TfidfVectorizer(analyzer=_ngram_fn(ngram), min_df=1)

    tfidf_matrix = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()

    mean_scores = tfidf_matrix.mean(axis=0).A1
    doc_freq = (tfidf_matrix > 0).sum(axis=0).A1.astype(int)
    idf_values = vectorizer.idf_

    # Raw term counts
    if ngram == 6:
        count_vec = CountVectorizer(analyzer=_skip, min_df=1)
    elif ngram == 1:
        count_vec = CountVectorizer(analyzer=_uni, min_df=1)
    else:
        count_vec = CountVectorizer(analyzer=_ngram_fn(ngram), min_df=1)
    count_matrix = count_vec.fit_transform(docs)
    total_counts = np.asarray(count_matrix.sum(axis=0)).flatten()
    count_features = count_vec.get_feature_names_out()
    tf_lookup = {count_features[i]: int(total_counts[i]) for i in range(len(count_features))}

    all_scored = []
    for i in range(len(feature_names)):
        if mean_scores[i] > 0:
            all_scored.append({
                "term": feature_names[i],
                "score": float(mean_scores[i]),
                "tf": tf_lookup.get(feature_names[i], 0),
                "df": int(doc_freq[i]),
                "idf": float(idf_values[i]),
            })

    all_scored.sort(key=lambda x: x["score"], reverse=True)

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
        chunk_count=n_docs,
        df_stats=df_stats,
    )
