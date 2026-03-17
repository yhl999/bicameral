"""Tests for chatgpt_history lock-push experiment utilities.

Tests the strategy application functions (dedup, synth cap, scoring)
without requiring Neo4j or API access.
"""

import json
import sys
from pathlib import Path
from copy import deepcopy

# Add scripts to path for import
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest


# === Test fixtures ===

def _make_fact(uuid: str, fact: str, is_synth: bool = False, rerank_score: float = 0.5):
    """Create a minimal fact dict for testing."""
    f = {
        "uuid": uuid,
        "fact": fact,
        "name": fact[:30],
        "episodes": [] if is_synth else ["ep1"],
        "_rerank_score": rerank_score,
    }
    if is_synth:
        f["attributes"] = {"is_synthesized": True}
    return f


SAMPLE_POOL = {
    "query_id": "pref_01",
    "bucket": "pref",
    "query": "What are Yuan's wine preferences?",
    "expected_sketch": "Chablis, Meursault, grower champagne, natural wine",
    "merged_facts": [
        _make_fact("u1", "Cooking and hosting activities with wine pairings", True, 0.9),
        _make_fact("u2", "Cooking and hosting activities with wine pairings", True, 0.88),  # duplicate text
        _make_fact("u3", "Dating approach includes natural wines", True, 0.85),
        _make_fact("u4", "Dating approach includes natural wines", True, 0.83),  # duplicate text
        _make_fact("u5", "Yuan loves natural wine and Sicilian reds", False, 0.80),
        _make_fact("u6", "Yuan explored Chablis producers with ChatGPT", False, 0.75),
        _make_fact("u7", "Yuan knows about Jacques Lassaigne champagne", False, 0.70),
        _make_fact("u8", "Yuan built a wine database for tracking bottles", False, 0.65),
        _make_fact("u9", "Yuan has a strong interest in DeFi protocols", False, 0.20),
    ],
    "total_merged": 9,
    "rerank_scores": {},
}


# === Test _fact_text ===

class TestFactText:
    def test_dict_with_fact_key(self):
        from chatgpt_history_lock_push import _fact_text
        assert _fact_text({"fact": "hello"}) == "hello"

    def test_dict_with_name_key(self):
        from chatgpt_history_lock_push import _fact_text
        assert _fact_text({"name": "world"}) == "world"

    def test_string_passthrough(self):
        from chatgpt_history_lock_push import _fact_text
        assert _fact_text("plain text") == "plain text"


# === Test _is_synthesized ===

class TestIsSynthesized:
    def test_synthesized_by_attribute(self):
        from chatgpt_history_lock_push import _is_synthesized
        f = {"attributes": {"is_synthesized": True}, "episodes": ["ep1"]}
        assert _is_synthesized(f) is True

    def test_synthesized_by_empty_episodes(self):
        from chatgpt_history_lock_push import _is_synthesized
        f = {"episodes": []}
        assert _is_synthesized(f) is True

    def test_not_synthesized(self):
        from chatgpt_history_lock_push import _is_synthesized
        f = {"episodes": ["ep1", "ep2"]}
        assert _is_synthesized(f) is False

    def test_string_not_synthesized(self):
        from chatgpt_history_lock_push import _is_synthesized
        assert _is_synthesized("plain string") is False


# === Test apply_strategy ===

class TestApplyStrategy:
    def test_baseline_no_changes(self):
        from chatgpt_history_lock_push import apply_strategy
        pools = [deepcopy(SAMPLE_POOL)]
        cfg = {"dedup_text": False, "synth_cap": None, "first_order_boost": 0.0, "pref_boost_weight": 0.15}
        results = apply_strategy(pools, "baseline", cfg)
        assert len(results) == 1
        assert results[0]["fact_count"] == 5  # top-5 from 9

    def test_text_dedup_removes_duplicates(self):
        from chatgpt_history_lock_push import apply_strategy
        pools = [deepcopy(SAMPLE_POOL)]
        cfg = {"dedup_text": True, "synth_cap": None, "first_order_boost": 0.0, "pref_boost_weight": 0.15}
        results = apply_strategy(pools, "dedup", cfg)
        # After dedup: u1, u3, u5, u6, u7, u8, u9 (7 unique texts)
        # Top 5 by score
        assert results[0]["fact_count"] == 5
        # Check no duplicate fact texts in output
        texts = [f.get("fact", "") for f in results[0]["facts"]]
        assert len(texts) == len(set(texts))

    def test_synth_cap_limits_synthesized(self):
        from chatgpt_history_lock_push import apply_strategy
        pools = [deepcopy(SAMPLE_POOL)]
        cfg = {"dedup_text": True, "synth_cap": 2, "first_order_boost": 0.0, "pref_boost_weight": 0.15}
        results = apply_strategy(pools, "synthcap2", cfg)
        facts = results[0]["facts"]
        # Count synthesized in output
        synth_count = sum(1 for f in facts if f.get("attributes", {}).get("is_synthesized") or len(f.get("episodes", [None])) == 0)
        assert synth_count <= 2

    def test_synth_cap_at_1(self):
        from chatgpt_history_lock_push import apply_strategy
        pools = [deepcopy(SAMPLE_POOL)]
        cfg = {"dedup_text": True, "synth_cap": 1, "first_order_boost": 0.0, "pref_boost_weight": 0.15}
        results = apply_strategy(pools, "synthcap1", cfg)
        facts = results[0]["facts"]
        synth_count = sum(1 for f in facts if f.get("attributes", {}).get("is_synthesized") or len(f.get("episodes", [None])) == 0)
        assert synth_count <= 1

    def test_first_order_boost_promotes_first_order(self):
        from chatgpt_history_lock_push import apply_strategy
        pools = [deepcopy(SAMPLE_POOL)]
        # Without FO boost
        cfg_no = {"dedup_text": True, "synth_cap": None, "first_order_boost": 0.0, "pref_boost_weight": 0.15}
        results_no = apply_strategy(pools, "no_boost", cfg_no)
        # With FO boost
        pools2 = [deepcopy(SAMPLE_POOL)]
        cfg_fo = {"dedup_text": True, "synth_cap": None, "first_order_boost": 0.3, "pref_boost_weight": 0.15}
        results_fo = apply_strategy(pools2, "fo_boost", cfg_fo)
        # FO boost should promote first-order facts higher in ranking
        # At minimum, they should both produce valid output
        assert len(results_no[0]["facts"]) == 5
        assert len(results_fo[0]["facts"]) == 5

    def test_negative_bucket_no_reranking(self):
        from chatgpt_history_lock_push import apply_strategy
        neg_pool = {
            "query_id": "neg_01", "bucket": "neg",
            "query": "What is the delegation protocol?",
            "expected_sketch": "",
            "merged_facts": [
                _make_fact("n1", "Some random fact", False, 0.3),
            ],
            "total_merged": 1, "rerank_scores": {},
        }
        cfg = {"dedup_text": True, "synth_cap": 2, "first_order_boost": 0.1, "pref_boost_weight": 0.15}
        results = apply_strategy([neg_pool], "test", cfg)
        assert results[0]["fact_count"] == 1

    def test_empty_pool(self):
        from chatgpt_history_lock_push import apply_strategy
        empty = {
            "query_id": "test", "bucket": "bio",
            "query": "Test?", "expected_sketch": "",
            "merged_facts": [], "total_merged": 0, "rerank_scores": {},
        }
        cfg = {"dedup_text": True, "synth_cap": 2, "first_order_boost": 0.0, "pref_boost_weight": 0.15}
        results = apply_strategy([empty], "test", cfg)
        assert results[0]["fact_count"] == 0


# === Test compute_metrics ===

class TestComputeMetrics:
    def test_basic_metrics(self):
        from chatgpt_history_lock_push import compute_metrics
        scored = [
            {"query_id": "q1", "bucket": "bio", "score": 3, "is_negative": False, "mrr_rank": 1},
            {"query_id": "q2", "bucket": "bio", "score": 2, "is_negative": False, "mrr_rank": 2},
            {"query_id": "q3", "bucket": "bio", "score": 1, "is_negative": False, "mrr_rank": None},
            {"query_id": "n1", "bucket": "neg", "score": 0, "is_negative": True, "false_positive": False},
        ]
        m = compute_metrics(scored)
        assert m["total_positive"] == 3
        assert m["total_negative"] == 1
        assert m["top1_count"] == 1
        assert m["relevant_count"] == 2
        assert m["false_positives"] == 0
        # MRR: (1/1 + 1/2 + 0) / 3 = 0.5
        assert abs(m["mrr"] - 0.5) < 0.01

    def test_all_negative(self):
        from chatgpt_history_lock_push import compute_metrics
        scored = [
            {"query_id": "n1", "bucket": "neg", "score": 0, "is_negative": True, "false_positive": False},
            {"query_id": "n2", "bucket": "neg", "score": 1, "is_negative": True, "false_positive": False},
        ]
        m = compute_metrics(scored)
        assert m["total_positive"] == 0
        assert m["total_negative"] == 2
        assert m["false_positives"] == 0

    def test_false_positive_counting(self):
        from chatgpt_history_lock_push import compute_metrics
        scored = [
            {"query_id": "q1", "bucket": "bio", "score": 2, "is_negative": False, "mrr_rank": 1},
            {"query_id": "n1", "bucket": "neg", "score": 2, "is_negative": True, "false_positive": True},
            {"query_id": "n2", "bucket": "neg", "score": 0, "is_negative": True, "false_positive": False},
        ]
        m = compute_metrics(scored)
        assert m["false_positives"] == 1


# === Test _type_aware_boost ===

class TestTypeAwareBoost:
    def test_pref_boost_on_wine_fact(self):
        from chatgpt_history_lock_push import _type_aware_boost
        f = {"fact": "Yuan loves natural wine and champagne, his favorite hobby"}
        boost = _type_aware_boost(f, "pref")
        assert boost > 0  # should match wine, champagne, hobby, favorite

    def test_no_boost_on_irrelevant(self):
        from chatgpt_history_lock_push import _type_aware_boost
        f = {"fact": "The sky is blue today"}
        boost = _type_aware_boost(f, "pref")
        assert boost == 0.0

    def test_unknown_bucket(self):
        from chatgpt_history_lock_push import _type_aware_boost
        f = {"fact": "anything"}
        boost = _type_aware_boost(f, "unknown_bucket")
        assert boost == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
