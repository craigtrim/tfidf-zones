import pytest

from tfidf_zones.scikit_engine import chunk_text, run
from tfidf_zones.tfidf_engine import EngineResult


class TestChunkText:
    def test_short_text(self):
        text = "hello world"
        chunks = chunk_text(text, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_exact_division(self):
        words = [f"word{i}" for i in range(4000)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=2000)
        assert len(chunks) == 2

    def test_runt_redistribution(self):
        words = [f"word{i}" for i in range(4010)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=2000)
        assert len(chunks) == 2
        # All words should be present across chunks
        total_words = sum(len(c.split()) for c in chunks)
        assert total_words == 4010

    def test_large_remainder_kept(self):
        words = [f"word{i}" for i in range(3900)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=2000)
        assert len(chunks) == 2


class TestRun:
    def test_returns_engine_result(self):
        text = "The cat sat on the mat. " * 200
        result = run(text, ngram=1, chunk_size=500, top_k=5)
        assert isinstance(result, EngineResult)
        assert result.ngram_type == "unigrams"
        assert len(result.top_terms) <= 5
        assert result.total_tokens > 0
        assert result.chunk_count > 0
        assert len(result.all_scored) > 0
        assert "mean" in result.df_stats

    def test_bigrams(self):
        text = "The cat sat on the mat by the door. " * 200
        result = run(text, ngram=2, chunk_size=500, top_k=5)
        assert result.ngram_type == "bigrams"
        assert len(result.top_terms) > 0

    def test_skipgrams(self):
        text = "The cat sat on the mat by the door. " * 200
        result = run(text, ngram=6, chunk_size=500, top_k=5)
        assert result.ngram_type == "skipgrams"
        assert len(result.top_terms) > 0

    def test_empty_text(self):
        result = run("", ngram=1)
        assert result.total_tokens == 0
        assert result.top_terms == []

    def test_invalid_ngram(self):
        with pytest.raises(ValueError):
            run("hello world", ngram=7)

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError):
            run("hello world", chunk_size=50)

    def test_scores_have_required_keys(self):
        text = "Alice went through the looking glass into wonderland. " * 200
        result = run(text, ngram=1, chunk_size=500, top_k=5)
        for entry in result.top_terms:
            assert "term" in entry
            assert "tf" in entry
            assert "score" in entry
            assert "df" in entry
            assert "idf" in entry
