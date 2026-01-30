from tfidf_zones.zones import classify_zones


def _make_scored(n=100, chunk_count=20):
    """Generate a scored list with varying DF and score values.

    Creates n terms where:
      - DF ranges from n down to 1 (high to low)
      - Score ranges from 1.0 down (high to low)
      - chunk_count controls the corpus size for DF bounds
    """
    return [
        {"term": f"term{i}", "score": round(1.0 - i * 0.005, 6), "df": n - i, "idf": round(1.0 + i * 0.01, 6)}
        for i in range(n)
    ]


class TestClassifyZones:
    def test_basic_structure(self):
        scored = _make_scored(100, chunk_count=100)
        zones = classify_zones(scored, top_k=10, chunk_count=100)
        assert "too_common" in zones
        assert "goldilocks" in zones
        assert "too_rare" in zones

    def test_zone_sizes(self):
        scored = _make_scored(100, chunk_count=100)
        zones = classify_zones(scored, top_k=10, chunk_count=100)
        assert len(zones["too_common"]) <= 10
        assert len(zones["goldilocks"]) <= 10
        assert len(zones["too_rare"]) <= 10

    def test_too_common_has_high_df(self):
        """Too common terms should have df > 0.2N."""
        scored = _make_scored(100, chunk_count=100)
        zones = classify_zones(scored, top_k=10, chunk_count=100)
        threshold = int(100 * 0.2)  # 20
        for t in zones["too_common"]:
            assert t["df"] > threshold, f"too_common term {t['term']} has df={t['df']} <= {threshold}"

    def test_too_rare_has_low_df(self):
        """Too rare terms should have df < 3."""
        scored = _make_scored(100, chunk_count=100)
        zones = classify_zones(scored, top_k=10, chunk_count=100)
        for t in zones["too_rare"]:
            assert t["df"] < 3, f"too_rare term {t['term']} has df={t['df']} >= 3"

    def test_goldilocks_within_df_band(self):
        """Goldilocks terms should have 3 <= df <= 0.2N."""
        scored = _make_scored(100, chunk_count=100)
        zones = classify_zones(scored, top_k=10, chunk_count=100)
        df_upper = int(100 * 0.2)
        for t in zones["goldilocks"]:
            assert t["df"] >= 3, f"goldilocks term {t['term']} has df={t['df']} < 3"
            assert t["df"] <= df_upper, f"goldilocks term {t['term']} has df={t['df']} > {df_upper}"

    def test_goldilocks_has_high_tfidf(self):
        """Goldilocks terms should have TF-IDF >= 95th percentile."""
        scored = _make_scored(100, chunk_count=100)
        zones = classify_zones(scored, top_k=10, chunk_count=100)
        if zones["goldilocks"]:
            # Compute the 95th percentile threshold
            all_scores = sorted((t["score"] for t in scored), reverse=True)
            p95_index = max(0, int(len(all_scores) * 0.05) - 1)
            threshold = all_scores[p95_index]
            for t in zones["goldilocks"]:
                assert t["score"] >= threshold

    def test_empty_input(self):
        zones = classify_zones([], top_k=10)
        assert zones == {"too_common": [], "goldilocks": [], "too_rare": []}

    def test_small_input(self):
        scored = _make_scored(5, chunk_count=5)
        zones = classify_zones(scored, top_k=3, chunk_count=5)
        assert isinstance(zones["too_common"], list)
        assert isinstance(zones["goldilocks"], list)
        assert isinstance(zones["too_rare"], list)

    def test_top_k_limits(self):
        scored = _make_scored(200, chunk_count=200)
        zones = classify_zones(scored, top_k=3, chunk_count=200)
        assert len(zones["too_common"]) <= 3
        assert len(zones["goldilocks"]) <= 3
        assert len(zones["too_rare"]) <= 3

    def test_entry_keys(self):
        scored = _make_scored(100, chunk_count=100)
        zones = classify_zones(scored, top_k=5, chunk_count=100)
        for zone_name in ["too_common", "goldilocks", "too_rare"]:
            for entry in zones[zone_name]:
                assert "term" in entry
                assert "score" in entry
                assert "df" in entry
                assert "idf" in entry

    def test_too_common_sorted_by_score_desc(self):
        scored = _make_scored(100, chunk_count=100)
        zones = classify_zones(scored, top_k=10, chunk_count=100)
        tc = zones["too_common"]
        if len(tc) > 1:
            scores = [t["score"] for t in tc]
            assert scores == sorted(scores, reverse=True)

    def test_goldilocks_sorted_by_score_desc(self):
        scored = _make_scored(100, chunk_count=100)
        zones = classify_zones(scored, top_k=10, chunk_count=100)
        gl = zones["goldilocks"]
        if len(gl) > 1:
            scores = [t["score"] for t in gl]
            assert scores == sorted(scores, reverse=True)

    def test_chunk_count_zero_uses_max_df(self):
        """When chunk_count=0, N should be derived from max DF."""
        scored = [
            {"term": "a", "score": 0.9, "df": 50, "idf": 1.0},
            {"term": "b", "score": 0.8, "df": 5, "idf": 2.0},
            {"term": "c", "score": 0.1, "df": 1, "idf": 3.0},
        ]
        zones = classify_zones(scored, top_k=10, chunk_count=0)
        # N=50 (max df), 0.2*50=10, so df=50 > 10 → too_common
        assert any(t["term"] == "a" for t in zones["too_common"])

    def test_small_corpus_relaxes_bounds(self):
        """With very few chunks, DF bounds should relax gracefully."""
        scored = [
            {"term": "common", "score": 0.5, "df": 5, "idf": 1.0},
            {"term": "mid", "score": 0.9, "df": 3, "idf": 2.0},
            {"term": "rare", "score": 0.1, "df": 1, "idf": 3.0},
        ]
        # chunk_count=10 → 0.2*10=2, which is < 3, triggers relaxation
        zones = classify_zones(scored, top_k=10, chunk_count=10)
        # Should not crash; rare term with df=1 should be too_rare
        assert any(t["term"] == "rare" for t in zones["too_rare"])
