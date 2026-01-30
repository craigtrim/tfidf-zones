import csv
import os
import tempfile

import pytest

from tfidf_zones.tfidf_engine import EngineResult


class _FakeResult:
    """Minimal result object matching what _write_csv expects."""

    def __init__(self, all_scored, total_tokens):
        self.engine_result = EngineResult(
            ngram_type="unigrams",
            top_terms=all_scored[:5],
            all_scored=all_scored,
            total_unique_terms=len(all_scored),
            total_items=len(all_scored),
            total_tokens=total_tokens,
            chunk_count=1,
            df_stats={"mean": 1.0, "median": 1.0, "mode": 1},
        )


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestWriteCsv:
    def _write(self, result) -> list[dict]:
        from tfidf_zones.cli import _write_csv

        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            _write_csv(result, path)
            return _read_csv(path)
        finally:
            os.unlink(path)

    def test_header_contains_cumulative_columns(self):
        scored = [
            {"term": "the", "score": 0.5, "tf": 100, "df": 1, "idf": 1.0},
        ]
        result = _FakeResult(scored, total_tokens=100)
        rows = self._write(result)
        assert "tf_pct" in rows[0]
        assert "tf_cum_norm" in rows[0]

    def test_tf_pct_computation(self):
        scored = [
            {"term": "the", "score": 0.5, "tf": 60, "df": 1, "idf": 1.0},
            {"term": "of", "score": 0.3, "tf": 40, "df": 1, "idf": 1.0},
        ]
        result = _FakeResult(scored, total_tokens=100)
        rows = self._write(result)
        # Rows are in TF-IDF descending order
        the_row = next(r for r in rows if r["term"] == "the")
        of_row = next(r for r in rows if r["term"] == "of")
        assert float(the_row["tf_pct"]) == pytest.approx(0.6)
        assert float(of_row["tf_pct"]) == pytest.approx(0.4)

    def test_tf_cum_norm_matches_running_sum(self):
        scored = [
            {"term": "the", "score": 0.5, "tf": 50, "df": 1, "idf": 1.0},
            {"term": "of", "score": 0.4, "tf": 30, "df": 1, "idf": 1.0},
            {"term": "and", "score": 0.3, "tf": 20, "df": 1, "idf": 1.0},
        ]
        result = _FakeResult(scored, total_tokens=100)
        rows = self._write(result)
        # TF-sorted order: the(50), of(30), and(20)
        # tf_cum_norm: 50, 80, 100
        the_row = next(r for r in rows if r["term"] == "the")
        of_row = next(r for r in rows if r["term"] == "of")
        and_row = next(r for r in rows if r["term"] == "and")
        assert int(the_row["tf_cum_norm"]) == 50
        assert int(of_row["tf_cum_norm"]) == 80
        assert int(and_row["tf_cum_norm"]) == 100

    def test_tf_cum_norm_scaled_1_to_100(self):
        scored = [
            {"term": "the", "score": 0.5, "tf": 50, "df": 1, "idf": 1.0},
            {"term": "of", "score": 0.4, "tf": 30, "df": 1, "idf": 1.0},
            {"term": "and", "score": 0.3, "tf": 20, "df": 1, "idf": 1.0},
        ]
        result = _FakeResult(scored, total_tokens=100)
        rows = self._write(result)
        the_row = next(r for r in rows if r["term"] == "the")
        of_row = next(r for r in rows if r["term"] == "of")
        and_row = next(r for r in rows if r["term"] == "and")
        assert int(the_row["tf_cum_norm"]) == 50
        assert int(of_row["tf_cum_norm"]) == 80
        assert int(and_row["tf_cum_norm"]) == 100

    def test_tf_cum_norm_minimum_is_1(self):
        # A very rare term with tiny tf_pct should still get tf_cum_norm >= 1
        scored = [
            {"term": "rare", "score": 0.001, "tf": 1, "df": 1, "idf": 4.0},
        ]
        result = _FakeResult(scored, total_tokens=1000000)
        rows = self._write(result)
        assert int(rows[0]["tf_cum_norm"]) >= 1

    def test_csv_sort_order_is_tfidf_descending(self):
        scored = [
            {"term": "alpha", "score": 0.9, "tf": 10, "df": 1, "idf": 2.0},
            {"term": "beta", "score": 0.5, "tf": 50, "df": 1, "idf": 1.0},
            {"term": "gamma", "score": 0.1, "tf": 5, "df": 1, "idf": 1.0},
        ]
        result = _FakeResult(scored, total_tokens=65)
        rows = self._write(result)
        # CSV should preserve TF-IDF descending order (alpha, beta, gamma)
        assert rows[0]["term"] == "alpha"
        assert rows[1]["term"] == "beta"
        assert rows[2]["term"] == "gamma"

    def test_zero_tokens_produces_zero_values(self):
        scored = []
        result = _FakeResult(scored, total_tokens=0)
        rows = self._write(result)
        assert len(rows) == 0
