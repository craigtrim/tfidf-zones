import pytest

from tfidf_zones.tfidf_engine import (
    EngineResult,
    aggregate_tfidf,
    chunk_tokens,
    compute_df_stats,
    compute_idf,
    compute_tf,
    generate_ngrams,
    generate_skipgrams,
    run,
    scale_tf_by_idf,
    tfidf_compute,
    top_k_terms,
)


class TestComputeTf:
    def test_basic(self):
        tf = compute_tf(["a", "b", "a", "c"])
        assert tf["a"] == pytest.approx(0.5)
        assert tf["b"] == pytest.approx(0.25)
        assert tf["c"] == pytest.approx(0.25)

    def test_empty(self):
        assert compute_tf([]) == {}

    def test_single_token(self):
        tf = compute_tf(["hello"])
        assert tf["hello"] == pytest.approx(1.0)


class TestComputeIdf:
    def test_basic(self):
        chunks = [["a", "b"], ["a", "c"], ["b", "c"]]
        idf, df = compute_idf(chunks)
        assert df["a"] == 2
        assert df["b"] == 2
        assert df["c"] == 2
        # IDF should be > 1 for all terms
        assert all(v > 1.0 for v in idf.values())

    def test_ubiquitous_term(self):
        chunks = [["the", "cat"], ["the", "dog"], ["the", "bird"]]
        idf, df = compute_idf(chunks)
        assert df["the"] == 3
        # With smooth_idf, ubiquitous term gets IDF = log(4/4) + 1 = 1.0
        assert idf["the"] == pytest.approx(1.0)

    def test_empty(self):
        idf, df = compute_idf([])
        assert idf == {}
        assert df == {}


class TestScaleTfByIdf:
    def test_basic(self):
        tf = {"a": 0.5, "b": 0.25}
        idf = {"a": 2.0, "b": 1.0}
        result = scale_tf_by_idf(tf, idf)
        assert result["a"] == pytest.approx(1.0)
        assert result["b"] == pytest.approx(0.25)


class TestGenerateNgrams:
    def test_bigrams(self):
        tokens = ["the", "cat", "sat"]
        ngrams = generate_ngrams(tokens, 2)
        assert ngrams == ["the_cat", "cat_sat"]

    def test_trigrams(self):
        tokens = ["the", "cat", "sat", "down"]
        ngrams = generate_ngrams(tokens, 3)
        assert ngrams == ["the_cat_sat", "cat_sat_down"]

    def test_too_few_tokens(self):
        assert generate_ngrams(["a"], 2) == []

    def test_unigrams(self):
        tokens = ["a", "b", "c"]
        assert generate_ngrams(tokens, 1) == ["a", "b", "c"]


class TestGenerateSkipgrams:
    def test_basic(self):
        tokens = ["the", "big", "cat", "sat"]
        sgrams = generate_skipgrams(tokens, skip=1)
        assert "the_cat" in sgrams
        assert "big_sat" in sgrams
        assert len(sgrams) == 2

    def test_too_few_tokens(self):
        assert generate_skipgrams(["a", "b"], skip=1) == []


class TestChunkTokens:
    def test_single_chunk(self):
        tokens = list(range(50))
        chunks = chunk_tokens([str(t) for t in tokens], chunk_size=100)
        assert len(chunks) == 1

    def test_exact_division(self):
        tokens = [str(i) for i in range(4000)]
        chunks = chunk_tokens(tokens, chunk_size=2000)
        assert len(chunks) == 2
        assert all(len(c) == 2000 for c in chunks)

    def test_runt_redistribution(self):
        # 4010 tokens / 2000 = 2 full + 10 runt -> redistribute to 2 chunks
        tokens = [str(i) for i in range(4010)]
        chunks = chunk_tokens(tokens, chunk_size=2000)
        assert len(chunks) == 2
        assert sum(len(c) for c in chunks) == 4010

    def test_large_remainder_kept(self):
        # 3900 tokens / 2000 = 1 full + 1900 remainder (95% >= 90%) -> 2 chunks
        tokens = [str(i) for i in range(3900)]
        chunks = chunk_tokens(tokens, chunk_size=2000)
        assert len(chunks) == 2

    def test_empty(self):
        assert chunk_tokens([], chunk_size=2000) == []

    def test_min_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size must be >= 100"):
            chunk_tokens(["a"], chunk_size=50)


class TestTopKTerms:
    def test_basic(self):
        scores = {"a": 0.5, "b": 0.3, "c": 0.1}
        result = top_k_terms(scores, k=2)
        assert len(result) == 2
        assert result[0]["term"] == "a"
        assert result[1]["term"] == "b"

    def test_with_df_idf(self):
        scores = {"a": 0.5}
        df = {"a": 3}
        idf = {"a": 1.5}
        result = top_k_terms(scores, k=1, df=df, idf=idf)
        assert result[0]["df"] == 3
        assert result[0]["idf"] == 1.5

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            top_k_terms({"a": 1.0}, k=0)


class TestComputeDfStats:
    def test_basic(self):
        df = {"a": 1, "b": 2, "c": 3}
        stats = compute_df_stats(df)
        assert stats["mean"] == 2.0
        assert stats["median"] == 2.0
        assert stats["mode"] == 1  # all unique, mode picks first

    def test_empty(self):
        stats = compute_df_stats({})
        assert stats["mean"] == 0.0


class TestTfidfCompute:
    def test_unigrams(self):
        tokens = ["the", "cat", "sat", "on", "the", "mat"] * 500
        result = tfidf_compute(tokens, ngram=1, chunk_size=500, top_k=5)
        assert result["ngram_type"] == "unigrams"
        assert len(result["top_terms"]) <= 5
        assert result["total_tokens"] == len(tokens)
        assert result["chunk_count"] > 0
        assert len(result["all_scored"]) > 0

    def test_bigrams(self):
        tokens = ["the", "cat", "sat", "on", "the", "mat"] * 500
        result = tfidf_compute(tokens, ngram=2, chunk_size=500, top_k=5)
        assert result["ngram_type"] == "bigrams"
        assert all("_" in t["term"] for t in result["top_terms"])

    def test_skipgrams(self):
        tokens = ["the", "cat", "sat", "on", "the", "mat"] * 500
        result = tfidf_compute(tokens, ngram=6, chunk_size=500, top_k=5)
        assert result["ngram_type"] == "skipgrams"

    def test_empty_tokens(self):
        result = tfidf_compute([], ngram=1)
        assert result["top_terms"] == []
        assert result["total_tokens"] == 0

    def test_invalid_ngram(self):
        with pytest.raises(ValueError):
            tfidf_compute(["a"], ngram=7)


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

    def test_empty_text(self):
        result = run("", ngram=1)
        assert result.total_tokens == 0
        assert result.top_terms == []
