# =============================================================================
# TESTS: --no-ngram-stopwords filtering
# =============================================================================

from __future__ import annotations

import pytest

from tfidf_zones.word_lists import FUNCTION_WORDS, NGRAM_STOP_WORDS, STOP_WORDS, filter_ngrams


# ── filter_ngrams unit tests ────────────────────────────────────────────────


class TestFilterNgrams:
    """Unit tests for the filter_ngrams function."""

    def test_removes_bigram_with_function_word(self):
        ngrams = ["of_the", "crimson_velvet", "in_a"]
        result = filter_ngrams(ngrams)
        assert result == ["crimson_velvet"]

    def test_removes_trigram_with_stop_word(self):
        ngrams = ["big_dark_night", "run_the_show"]
        result = filter_ngrams(ngrams)
        # "big" and "run" are in STOP_WORDS; "the" is in FUNCTION_WORDS
        assert "run_the_show" not in result

    def test_preserves_content_ngrams(self):
        ngrams = ["neural_network", "quantum_computing", "crimson_velvet"]
        result = filter_ngrams(ngrams)
        assert result == ngrams

    def test_empty_input(self):
        assert filter_ngrams([]) == []

    def test_all_filtered(self):
        ngrams = ["of_the", "in_a", "to_the"]
        result = filter_ngrams(ngrams)
        assert result == []

    def test_skipgram_filtered(self):
        # Skip-grams also use underscore separator
        ngrams = ["the_house", "castle_dragon"]
        result = filter_ngrams(ngrams)
        assert result == ["castle_dragon"]

    def test_custom_stop_words(self):
        custom = frozenset(["foo", "bar"])
        ngrams = ["foo_baz", "baz_qux", "bar_quux"]
        result = filter_ngrams(ngrams, stop_words=custom)
        assert result == ["baz_qux"]


# ── Word list sanity checks ─────────────────────────────────────────────────


class TestWordLists:
    """Sanity checks for the word list sets."""

    def test_function_words_not_empty(self):
        assert len(FUNCTION_WORDS) > 100

    def test_stop_words_not_empty(self):
        assert len(STOP_WORDS) > 100

    def test_ngram_stop_words_is_union(self):
        assert NGRAM_STOP_WORDS == FUNCTION_WORDS | STOP_WORDS

    def test_disjoint(self):
        assert len(FUNCTION_WORDS & STOP_WORDS) == 0

    def test_common_function_words_present(self):
        for word in ["the", "a", "an", "of", "in", "to", "and", "or", "but", "is", "was"]:
            assert word in FUNCTION_WORDS, f"{word!r} missing from FUNCTION_WORDS"


# ── Engine integration tests ────────────────────────────────────────────────


class TestPureEngineStopwords:
    """Test that the pure engine respects no_ngram_stopwords."""

    def test_bigram_filtering(self):
        from tfidf_zones.tfidf_engine import run
        text = "the cat sat on the mat and the dog lay on the rug " * 50
        result_no_filter = run(text, ngram=2, chunk_size=200, top_k=50)
        result_filtered = run(text, ngram=2, chunk_size=200, top_k=50, no_ngram_stopwords=True)

        no_filter_terms = {t["term"] for t in result_no_filter.all_scored}
        filtered_terms = {t["term"] for t in result_filtered.all_scored}

        # "on_the" should be in unfiltered but not filtered
        assert "on_the" in no_filter_terms
        assert "on_the" not in filtered_terms

        # Filtered set should be smaller
        assert len(filtered_terms) < len(no_filter_terms)

    def test_unigram_unaffected(self):
        from tfidf_zones.tfidf_engine import run
        text = "the cat sat on the mat " * 50
        result_no_filter = run(text, ngram=1, chunk_size=200, top_k=50)
        result_filtered = run(text, ngram=1, chunk_size=200, top_k=50, no_ngram_stopwords=True)

        # Unigrams should be identical regardless of flag
        assert len(result_no_filter.all_scored) == len(result_filtered.all_scored)

    def test_run_docs_filtering(self):
        from tfidf_zones.tfidf_engine import run_docs
        docs = [
            "the cat sat on the mat and the dog lay on the rug",
            "a quick brown fox jumped over the lazy dog near the fence",
            "the old man walked through the dark forest beside the river",
        ]
        result_no_filter = run_docs(docs, ngram=2, top_k=50)
        result_filtered = run_docs(docs, ngram=2, top_k=50, no_ngram_stopwords=True)

        no_filter_terms = {t["term"] for t in result_no_filter.all_scored}
        filtered_terms = {t["term"] for t in result_filtered.all_scored}

        assert len(filtered_terms) < len(no_filter_terms)


class TestScikitEngineStopwords:
    """Test that the scikit engine respects no_ngram_stopwords."""

    # Use text with content-word bigrams that survive filtering
    _TEXT = (
        "the crimson velvet curtain hung beside the marble fireplace "
        "and the crystal chandelier sparkled above the mahogany table "
        "while the porcelain vase rested on the granite countertop "
    ) * 30

    def test_bigram_filtering(self):
        from tfidf_zones.scikit_engine import run
        result_no_filter = run(self._TEXT, ngram=2, chunk_size=500, top_k=50)
        result_filtered = run(self._TEXT, ngram=2, chunk_size=500, top_k=50, no_ngram_stopwords=True)

        no_filter_terms = {t["term"] for t in result_no_filter.all_scored}
        filtered_terms = {t["term"] for t in result_filtered.all_scored}

        # Function-word bigrams should be removed
        assert "the_crimson" in no_filter_terms or "the_marble" in no_filter_terms
        # Content bigrams should survive
        assert "crimson_velvet" in filtered_terms
        assert len(filtered_terms) < len(no_filter_terms)

    def test_unigram_unaffected(self):
        from tfidf_zones.scikit_engine import run
        text = "the cat sat on the mat " * 50
        result_no_filter = run(text, ngram=1, chunk_size=200, top_k=50)
        result_filtered = run(text, ngram=1, chunk_size=200, top_k=50, no_ngram_stopwords=True)

        assert len(result_no_filter.all_scored) == len(result_filtered.all_scored)

    def test_run_docs_filtering(self):
        from tfidf_zones.scikit_engine import run_docs
        docs = [
            "the crimson velvet curtain hung beside the marble fireplace and the crystal chandelier",
            "the porcelain vase rested on the granite countertop near the mahogany table",
            "the emerald brooch sparkled beneath the sapphire pendant inside the velvet cushion",
        ]
        result_no_filter = run_docs(docs, ngram=2, top_k=50)
        result_filtered = run_docs(docs, ngram=2, top_k=50, no_ngram_stopwords=True)

        no_filter_terms = {t["term"] for t in result_no_filter.all_scored}
        filtered_terms = {t["term"] for t in result_filtered.all_scored}

        assert len(filtered_terms) < len(no_filter_terms)
