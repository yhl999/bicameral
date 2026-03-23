"""
Tests for Phase 2B: RRF Softening + Candidate Diversity.

Covers:
- _HYBRID_RRF_K value (should be 60.0)
- _candidate_fact_text extraction
- _text_similarity (Jaccard token overlap)
- _dedup_candidates (identical, similar, dissimilar)
- _apply_mmr_diversity (relevance-only, diversity-only, balanced)
- rrf_merge_hybrid with diversity pipeline
- No regression, no duplicates in top-10
"""
from __future__ import annotations

import hashlib
import os
import sys
from typing import Any
from unittest.mock import patch

import pytest

# Ensure the MCP server source is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.typed_retrieval_service import (
    _HYBRID_RRF_K,
    _candidate_fact_text,
    _text_similarity,
    _tokenize,
    _dedup_candidates,
    _apply_mmr_diversity,
    rrf_merge_hybrid,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_candidate(
    fact: str,
    score: float = 0.1,
    *,
    name: str = "",
    uuid: str | None = None,
    source: str = "graph",
) -> dict[str, Any]:
    """Build a synthetic candidate with a hybrid score."""
    return {
        "uuid": uuid or hashlib.md5(fact.encode(), usedforsecurity=False).hexdigest()[:8],
        "name": name or fact[:30],
        "fact": fact,
        "_source": source,
        "_hybrid_score": score,
        "_original": {},
    }


def _make_graph_fact(
    name: str,
    fact: str = "",
    *,
    uuid: str | None = None,
) -> dict[str, Any]:
    return {
        "uuid": uuid or hashlib.md5(name.encode(), usedforsecurity=False).hexdigest()[:8],
        "name": name,
        "fact": fact or f"Fact about {name}",
        "_source": "graph",
    }


def _make_typed_state(
    subject: str,
    predicate: str = "is",
    value: str = "something",
    *,
    object_id: str | None = None,
    bucket: str = "",
) -> dict[str, Any]:
    return {
        "object_id": object_id or hashlib.md5(subject.encode(), usedforsecurity=False).hexdigest()[:8],
        "subject": subject,
        "predicate": predicate,
        "value": value,
        "bucket": bucket,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Section 1: RRF Constant
# ══════════════════════════════════════════════════════════════════════════════

class TestRRFConstant:
    """Verify the RRF smoothing constant is set to 60.0."""

    def test_rrf_k_is_60(self):
        assert _HYBRID_RRF_K == 60.0

    def test_rrf_k_softens_position_bias(self):
        """With k=60, rank 1 vs rank 2 difference should be <5%."""
        score_rank1 = 1.0 / (60.0 + 1)
        score_rank2 = 1.0 / (60.0 + 2)
        pct_diff = (score_rank1 - score_rank2) / score_rank1
        assert pct_diff < 0.05  # less than 5% difference


# ══════════════════════════════════════════════════════════════════════════════
# Section 2: _candidate_fact_text
# ══════════════════════════════════════════════════════════════════════════════

class TestCandidateFactText:
    """Tests for _candidate_fact_text helper."""

    def test_extracts_fact(self):
        c = _make_candidate("This is a test fact about sessions")
        text = _candidate_fact_text(c)
        assert "This is a test fact about sessions" in text

    def test_includes_name(self):
        c = _make_candidate("some fact", name="important name")
        text = _candidate_fact_text(c)
        assert "important name" in text

    def test_includes_original_fields(self):
        c = _make_candidate("fact")
        c["_original"] = {
            "subject": "Yuan",
            "predicate": "likes",
            "value": "sushi",
        }
        text = _candidate_fact_text(c)
        assert "Yuan" in text
        assert "likes" in text
        assert "sushi" in text

    def test_empty_candidate_returns_uuid(self):
        c = {"uuid": "abc123", "fact": "", "name": "", "_original": {}}
        text = _candidate_fact_text(c)
        assert text == "abc123"

    def test_deduplicates_name_and_fact(self):
        """If name == fact, don't repeat."""
        c = _make_candidate("same text", name="same text")
        text = _candidate_fact_text(c)
        # "same text" should appear only once
        assert text.count("same text") == 1


# ══════════════════════════════════════════════════════════════════════════════
# Section 3: _text_similarity (Jaccard)
# ══════════════════════════════════════════════════════════════════════════════

class TestTextSimilarity:
    """Tests for Jaccard token-overlap similarity."""

    def test_identical_strings(self):
        assert _text_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self):
        sim = _text_similarity("alpha beta gamma", "delta epsilon zeta")
        assert sim == 0.0

    def test_partial_overlap(self):
        sim = _text_similarity("hello world foo", "hello world bar")
        # Overlap: {hello, world}, Union: {hello, world, foo, bar}
        assert abs(sim - 0.5) < 0.01

    def test_empty_strings(self):
        assert _text_similarity("", "") == 0.0
        assert _text_similarity("hello", "") == 0.0
        assert _text_similarity("", "hello") == 0.0

    def test_case_insensitive(self):
        assert _text_similarity("Hello World", "hello world") == 1.0

    def test_near_duplicate_high_similarity(self):
        a = "Sessions Main experiment v7 shows memory architecture"
        b = "Sessions Main experiment v7 variant A shows memory architecture"
        sim = _text_similarity(a, b)
        assert sim > 0.75

    def test_tokenize_basic(self):
        tokens = _tokenize("Hello, World! 123")
        assert tokens == {"hello", "world", "123"}


# ══════════════════════════════════════════════════════════════════════════════
# Section 4: _dedup_candidates
# ══════════════════════════════════════════════════════════════════════════════

class TestDedupCandidates:
    """Tests for near-duplicate removal."""

    def test_identical_candidates_dedup_to_one(self):
        c1 = _make_candidate("Exact same fact about memory architecture", score=0.9, uuid="a1")
        c2 = _make_candidate("Exact same fact about memory architecture", score=0.8, uuid="a2")
        result = _dedup_candidates([c1, c2], threshold=0.85)
        assert len(result) == 1
        assert result[0]["_hybrid_score"] == 0.9  # keeps highest-scored

    def test_similar_candidates_deduped(self):
        c1 = _make_candidate(
            "Sessions Main experiment v7 memory architecture",
            score=0.88, uuid="b1",
            name="Sessions Main experiment v7 memory architecture",
        )
        c2 = _make_candidate(
            "Sessions Main experiment v7 memory architecture results",
            score=0.85, uuid="b2",
            name="Sessions Main experiment v7 memory architecture results",
        )
        # Jaccard overlap between these is high (>0.65)
        result = _dedup_candidates([c1, c2], threshold=0.65)
        assert len(result) == 1

    def test_dissimilar_candidates_kept(self):
        c1 = _make_candidate("Yuan likes Japanese food for dinner", score=0.9, uuid="c1")
        c2 = _make_candidate("The RRF merge architecture uses k=60", score=0.8, uuid="c2")
        c3 = _make_candidate("Deploy pipeline runs every 30 minutes", score=0.7, uuid="c3")
        result = _dedup_candidates([c1, c2, c3], threshold=0.85)
        assert len(result) == 3

    def test_preserves_order(self):
        """Highest-scored first, then next unique, etc."""
        candidates = [
            _make_candidate("Alpha fact unique content", score=0.9, uuid="d1"),
            _make_candidate("Beta fact different content", score=0.8, uuid="d2"),
            _make_candidate("Gamma fact another topic", score=0.7, uuid="d3"),
        ]
        result = _dedup_candidates(candidates, threshold=0.85)
        scores = [r["_hybrid_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_input(self):
        assert _dedup_candidates([], threshold=0.85) == []

    def test_single_candidate(self):
        c = _make_candidate("only one", score=0.5)
        result = _dedup_candidates([c], threshold=0.85)
        assert len(result) == 1

    def test_three_variants_dedup_to_one(self):
        """Three near-identical variants → keep highest only."""
        base = "Sessions Main experiment v7 shows AI memory"
        c1 = _make_candidate(base, score=0.88, uuid="e1")
        c2 = _make_candidate(base + " architecture", score=0.85, uuid="e2")
        c3 = _make_candidate(base + " architecture results", score=0.82, uuid="e3")
        result = _dedup_candidates([c1, c2, c3], threshold=0.70)
        assert len(result) == 1
        assert result[0]["_hybrid_score"] == 0.88

    def test_threshold_above_one_keeps_all(self):
        """threshold > 1.0 → nothing reaches it, all candidates kept."""
        c1 = _make_candidate("fact A", score=0.9, uuid="f1")
        c2 = _make_candidate("fact A", score=0.8, uuid="f2")
        result = _dedup_candidates([c1, c2], threshold=1.01)
        assert len(result) == 2  # nothing reaches threshold of 1.01


# ══════════════════════════════════════════════════════════════════════════════
# Section 5: _apply_mmr_diversity
# ══════════════════════════════════════════════════════════════════════════════

class TestApplyMmrDiversity:
    """Tests for MMR re-ranking."""

    def _diverse_candidates(self) -> list[dict[str, Any]]:
        """Build a set of candidates with varying similarity."""
        return [
            _make_candidate("Memory architecture uses graph and typed retrieval", score=0.10, uuid="m1"),
            _make_candidate("Memory architecture uses graph typed retrieval system", score=0.09, uuid="m2"),
            _make_candidate("Yuan prefers Japanese cuisine for dinner", score=0.08, uuid="m3"),
            _make_candidate("Deploy pipeline runs every thirty minutes on cron", score=0.07, uuid="m4"),
            _make_candidate("RRF merge with k=60 softens position bias", score=0.06, uuid="m5"),
        ]

    def test_relevance_only(self):
        """mmr_weight=0 → pure relevance, same order as input."""
        candidates = self._diverse_candidates()
        result = _apply_mmr_diversity(candidates, mmr_weight=0.0, max_items=5)
        scores = [r["_hybrid_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_diversity_promotes_dissimilar(self):
        """mmr_weight=1.0 → max diversity; similar items should be separated."""
        candidates = self._diverse_candidates()
        result = _apply_mmr_diversity(candidates, mmr_weight=1.0, max_items=5)
        # m1 and m2 are very similar — with full diversity weight,
        # m2 should not be selected right after m1
        uuids = [r.get("uuid") for r in result]
        if "m1" in uuids and "m2" in uuids:
            idx1 = uuids.index("m1")
            idx2 = uuids.index("m2")
            assert abs(idx1 - idx2) > 1  # not adjacent

    def test_balanced_mmr(self):
        """mmr_weight=0.3 → balanced, all items present."""
        candidates = self._diverse_candidates()
        result = _apply_mmr_diversity(candidates, mmr_weight=0.3, max_items=5)
        assert len(result) == 5

    def test_max_items_cap(self):
        candidates = self._diverse_candidates()
        result = _apply_mmr_diversity(candidates, mmr_weight=0.3, max_items=3)
        assert len(result) == 3

    def test_empty_input(self):
        assert _apply_mmr_diversity([], mmr_weight=0.3) == []

    def test_single_candidate(self):
        c = _make_candidate("only one", score=0.5)
        result = _apply_mmr_diversity([c], mmr_weight=0.3, max_items=5)
        assert len(result) == 1

    def test_first_selected_is_highest_score(self):
        """MMR always picks the highest-relevance item first."""
        candidates = self._diverse_candidates()
        result = _apply_mmr_diversity(candidates, mmr_weight=0.3, max_items=5)
        assert result[0]["_hybrid_score"] == max(c["_hybrid_score"] for c in candidates)


# ══════════════════════════════════════════════════════════════════════════════
# Section 6: Integration — rrf_merge_hybrid with diversity
# ══════════════════════════════════════════════════════════════════════════════

class TestRrfMergeWithDiversity:
    """Integration tests for the full RRF + dedup + MMR pipeline."""

    def test_no_duplicates_in_top_10(self):
        """After dedup+MMR, no near-duplicates should appear in top-10."""
        # Create graph facts with duplicates
        graph_facts = [
            _make_graph_fact("experiment v7 memory architecture", uuid="g1"),
            _make_graph_fact("experiment v7 memory architecture variant", uuid="g2"),
            _make_graph_fact("experiment v7 memory architecture results", uuid="g3"),
            _make_graph_fact("Yuan favorite cuisine Japanese", uuid="g4"),
            _make_graph_fact("deploy pipeline cron schedule", uuid="g5"),
            _make_graph_fact("RRF merge fusion scoring", uuid="g6"),
        ]
        typed_state = [
            _make_typed_state("scheduling preference", bucket="persona"),
            _make_typed_state("wine selection taste", bucket="preference"),
        ]
        results = rrf_merge_hybrid(
            graph_facts=graph_facts,
            typed_state=typed_state,
            typed_procedures=[],
            max_facts=10,
            apply_diversity=True,
        )
        # Check no two items in results have similarity ≥ 0.85
        for i, a in enumerate(results):
            for j, b in enumerate(results):
                if i >= j:
                    continue
                sim = _text_similarity(
                    _candidate_fact_text(a), _candidate_fact_text(b)
                )
                assert sim < 0.85, (
                    f"Near-duplicates in top-10: items {i} and {j} "
                    f"have similarity {sim:.2f}"
                )

    def test_diversity_disabled_no_dedup(self):
        """apply_diversity=False → raw RRF, no dedup/MMR."""
        graph_facts = [
            _make_graph_fact("same fact about memory", uuid="h1"),
            _make_graph_fact("same fact about memory", uuid="h2"),
        ]
        results = rrf_merge_hybrid(
            graph_facts=graph_facts,
            typed_state=[],
            typed_procedures=[],
            max_facts=10,
            apply_diversity=False,
        )
        # Both should be present (no dedup)
        assert len(results) == 2

    def test_no_regression_basic_merge(self):
        """Basic merge without intent or diversity still works."""
        graph_facts = [
            _make_graph_fact("fact A", uuid="r1"),
            _make_graph_fact("fact B", uuid="r2"),
        ]
        typed_state = [
            _make_typed_state("state C", object_id="r3"),
        ]
        results = rrf_merge_hybrid(
            graph_facts=graph_facts,
            typed_state=typed_state,
            typed_procedures=[],
            max_facts=10,
            query_intent=None,
            apply_diversity=False,
        )
        assert len(results) == 3
        assert all(r["_hybrid_score"] > 0 for r in results)
        # Score-descending order
        scores = [r["_hybrid_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_max_facts_respected_with_diversity(self):
        graph_facts = [_make_graph_fact(f"unique fact {i}", uuid=f"u{i}") for i in range(15)]
        results = rrf_merge_hybrid(
            graph_facts=graph_facts,
            typed_state=[],
            typed_procedures=[],
            max_facts=10,
            apply_diversity=True,
        )
        assert len(results) <= 10

    def test_empty_inputs(self):
        results = rrf_merge_hybrid(
            graph_facts=[],
            typed_state=[],
            typed_procedures=[],
            max_facts=10,
            apply_diversity=True,
        )
        assert results == []

    def test_max_facts_zero(self):
        results = rrf_merge_hybrid(
            graph_facts=[_make_graph_fact("test")],
            typed_state=[],
            typed_procedures=[],
            max_facts=0,
        )
        assert results == []


# ══════════════════════════════════════════════════════════════════════════════
# Section 7: Env var overrides
# ══════════════════════════════════════════════════════════════════════════════

class TestEnvVarOverrides:
    """Test that env var overrides are respected."""

    def test_dedup_threshold_env(self):
        """BICAMERAL_DEDUP_THRESHOLD should override default."""
        with patch.dict(os.environ, {"BICAMERAL_DEDUP_THRESHOLD": "0.50"}):
            # Re-import to pick up new env value
            import importlib
            import services.typed_retrieval_service as mod
            importlib.reload(mod)
            assert mod._DEDUP_THRESHOLD == 0.50
            # Restore
            importlib.reload(mod)

    def test_mmr_weight_env(self):
        """BICAMERAL_MMR_WEIGHT_DIVERSITY should override default."""
        with patch.dict(os.environ, {"BICAMERAL_MMR_WEIGHT_DIVERSITY": "0.70"}):
            import importlib
            import services.typed_retrieval_service as mod
            importlib.reload(mod)
            assert mod._MMR_WEIGHT_DIVERSITY == 0.70
            # Restore
            importlib.reload(mod)
