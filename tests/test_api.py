# =============================================================================
# TESTS: Public API (analyze, analyze_docs, to_csv)
# =============================================================================

from __future__ import annotations

import csv
import io

import pytest

from tfidf_zones.api import analyze, analyze_docs, to_csv


_TEXT = "the cat sat on the mat and the dog lay on the rug " * 50


class TestAnalyze:
    """Tests for analyze() — single text input."""

    def test_returns_analysis_result(self):
        result = analyze(_TEXT, chunk_size=200, top_k=10)
        assert result.engine_result is not None
        assert result.zones is not None
        assert result.engine_name == "pure"
        assert result.filename == "<text>"

    def test_all_scored_populated(self):
        result = analyze(_TEXT, chunk_size=200, top_k=10)
        assert len(result.engine_result.all_scored) > 0

    def test_scikit_engine(self):
        result = analyze(_TEXT, engine="scikit", chunk_size=200, top_k=10)
        assert result.engine_name == "scikit"
        assert len(result.engine_result.all_scored) > 0

    def test_bigram(self):
        result = analyze(_TEXT, ngram=2, chunk_size=200, top_k=10)
        assert result.engine_result.ngram_type == "bigrams"
        terms = {t["term"] for t in result.engine_result.all_scored}
        assert any("_" in t for t in terms)

    def test_min_df_filter(self):
        result_no = analyze(_TEXT, chunk_size=200, top_k=50)
        result_yes = analyze(_TEXT, chunk_size=200, top_k=50, min_df=2)
        all_df = [t["df"] for t in result_yes.engine_result.all_scored]
        assert all(df >= 2 for df in all_df)
        assert len(result_yes.engine_result.all_scored) <= len(result_no.engine_result.all_scored)

    def test_min_tf_filter(self):
        result = analyze(_TEXT, chunk_size=200, top_k=50, min_tf=5)
        all_tf = [t.get("tf", 0) for t in result.engine_result.all_scored]
        assert all(tf >= 5 for tf in all_tf)

    def test_text_length(self):
        result = analyze(_TEXT, chunk_size=200, top_k=10)
        assert result.text_length == len(_TEXT)

    def test_elapsed_positive(self):
        result = analyze(_TEXT, chunk_size=200, top_k=10)
        assert result.elapsed > 0


class TestAnalyzeDocs:
    """Tests for analyze_docs() — multi-document input."""

    _DOCS = [
        "the cat sat on the mat and the dog lay on the rug",
        "a quick brown fox jumped over the lazy dog near the fence",
        "the old man walked through the dark forest beside the river",
    ]

    def test_returns_analysis_result(self):
        result = analyze_docs(self._DOCS, top_k=10)
        assert result.engine_result is not None
        assert result.zones is not None
        assert result.engine_name == "pure"
        assert result.filename == "<docs>"
        assert result.file_count == 3

    def test_scikit_engine(self):
        result = analyze_docs(self._DOCS, engine="scikit", top_k=10)
        assert result.engine_name == "scikit"
        assert len(result.engine_result.all_scored) > 0

    def test_chunk_size_zero(self):
        result = analyze_docs(self._DOCS, top_k=10)
        assert result.chunk_size == 0

    def test_bigram(self):
        result = analyze_docs(self._DOCS, ngram=2, top_k=10)
        terms = {t["term"] for t in result.engine_result.all_scored}
        assert any("_" in t for t in terms)

    def test_min_df_filter(self):
        result = analyze_docs(self._DOCS, top_k=50, min_df=2)
        all_df = [t["df"] for t in result.engine_result.all_scored]
        assert all(df >= 2 for df in all_df)


class TestToCsv:
    """Tests for to_csv() — CSV string generation."""

    def test_returns_string(self):
        result = analyze(_TEXT, chunk_size=200, top_k=10)
        csv_str = to_csv(result)
        assert isinstance(csv_str, str)
        assert len(csv_str) > 0

    def test_header_columns(self):
        result = analyze(_TEXT, chunk_size=200, top_k=10)
        csv_str = to_csv(result)
        reader = csv.reader(io.StringIO(csv_str))
        header = next(reader)
        assert header == ["term", "tf", "df", "idf", "tfidf", "tf_pct", "tf_cum_norm", "zone"]

    def test_row_count_matches_all_scored(self):
        result = analyze(_TEXT, chunk_size=200, top_k=50)
        csv_str = to_csv(result)
        reader = csv.reader(io.StringIO(csv_str))
        next(reader)  # skip header
        rows = list(reader)
        assert len(rows) == len(result.engine_result.all_scored)

    def test_tf_pct_values(self):
        result = analyze(_TEXT, chunk_size=200, top_k=50)
        csv_str = to_csv(result)
        reader = csv.DictReader(io.StringIO(csv_str))
        for row in reader:
            tf_pct = float(row["tf_pct"])
            assert 0.0 <= tf_pct <= 1.0

    def test_zone_values(self):
        result = analyze(_TEXT, chunk_size=200, top_k=50)
        csv_str = to_csv(result)
        reader = csv.DictReader(io.StringIO(csv_str))
        for row in reader:
            assert row["zone"] in ("", "1", "2", "3")

    def test_empty_result(self):
        from tfidf_zones.runner import AnalysisResult
        from tfidf_zones.tfidf_engine import EngineResult

        er = EngineResult(
            ngram_type="unigrams",
            top_terms=[],
            all_scored=[],
            total_unique_terms=0,
            total_items=0,
            total_tokens=0,
            chunk_count=1,
            df_stats={"mean": 0, "median": 0, "mode": 0},
        )
        result = AnalysisResult(
            filename="<text>",
            text_length=0,
            engine_name="pure",
            engine_result=er,
            zones={"too_common": [], "goldilocks": [], "too_rare": []},
            elapsed=0.0,
            chunk_size=200,
        )
        csv_str = to_csv(result)
        reader = csv.reader(io.StringIO(csv_str))
        header = next(reader)
        rows = list(reader)
        assert len(rows) == 0
        assert "term" in header
