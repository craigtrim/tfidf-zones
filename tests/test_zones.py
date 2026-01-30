from tfidf_zones.zones import classify_zones


def _make_scored(n=100):
    """Generate a scored list with varying DF values."""
    return [
        {"term": f"term{i}", "score": round(1.0 - i * 0.005, 6), "df": n - i, "idf": round(1.0 + i * 0.01, 6)}
        for i in range(n)
    ]


class TestClassifyZones:
    def test_basic_structure(self):
        scored = _make_scored(100)
        zones = classify_zones(scored, top_k=10)
        assert "too_common" in zones
        assert "goldilocks" in zones
        assert "too_rare" in zones

    def test_zone_sizes(self):
        scored = _make_scored(100)
        zones = classify_zones(scored, top_k=10)
        assert len(zones["too_common"]) <= 10
        assert len(zones["goldilocks"]) <= 10
        assert len(zones["too_rare"]) <= 10

    def test_too_common_has_high_df(self):
        scored = _make_scored(100)
        zones = classify_zones(scored, top_k=10)
        if zones["too_common"]:
            max_common_df = max(t["df"] for t in zones["too_common"])
            # Too common terms should have the highest DF values
            all_dfs = [t["df"] for t in scored]
            median_df = sorted(all_dfs)[len(all_dfs) // 2]
            assert max_common_df > median_df

    def test_too_rare_has_low_df(self):
        scored = _make_scored(100)
        zones = classify_zones(scored, top_k=10)
        if zones["too_rare"]:
            min_rare_df = min(t["df"] for t in zones["too_rare"])
            all_dfs = [t["df"] for t in scored]
            median_df = sorted(all_dfs)[len(all_dfs) // 2]
            assert min_rare_df < median_df

    def test_empty_input(self):
        zones = classify_zones([], top_k=10)
        assert zones == {"too_common": [], "goldilocks": [], "too_rare": []}

    def test_small_input(self):
        scored = _make_scored(5)
        zones = classify_zones(scored, top_k=3)
        # Should not crash on small input
        assert isinstance(zones["too_common"], list)
        assert isinstance(zones["goldilocks"], list)
        assert isinstance(zones["too_rare"], list)

    def test_top_k_limits(self):
        scored = _make_scored(200)
        zones = classify_zones(scored, top_k=3)
        assert len(zones["too_common"]) <= 3
        assert len(zones["goldilocks"]) <= 3
        assert len(zones["too_rare"]) <= 3

    def test_entry_keys(self):
        scored = _make_scored(100)
        zones = classify_zones(scored, top_k=5)
        for zone_name in ["too_common", "goldilocks", "too_rare"]:
            for entry in zones[zone_name]:
                assert "term" in entry
                assert "score" in entry
                assert "df" in entry
                assert "idf" in entry

    def test_too_common_sorted_by_score_desc(self):
        scored = _make_scored(100)
        zones = classify_zones(scored, top_k=10)
        tc = zones["too_common"]
        if len(tc) > 1:
            scores = [t["score"] for t in tc]
            assert scores == sorted(scores, reverse=True)

    def test_goldilocks_sorted_by_score_desc(self):
        scored = _make_scored(100)
        zones = classify_zones(scored, top_k=10)
        gl = zones["goldilocks"]
        if len(gl) > 1:
            scores = [t["score"] for t in gl]
            assert scores == sorted(scores, reverse=True)
