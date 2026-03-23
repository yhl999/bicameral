"""
Tests for Phase 2A: Query-Intent Lane Filter.

Covers:
- QueryIntentClassifier: caching, intent classification on 20+ queries, fallback
- _lane_label: graph + typed candidate lane assignment
- Suppression logic in rrf_merge_hybrid
- No-regression without intent param
"""
from __future__ import annotations

import hashlib
import sys
import os
import time
from typing import Any
from unittest.mock import patch

import pytest

# Ensure the MCP server source is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.typed_retrieval_service import (
    QueryIntentClassifier,
    SUPPRESSION_MATRIX,
    _lane_label,
    rrf_merge_hybrid,
    HybridRetrievalService,
    _query_intent_classifier,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_graph_fact(
    name: str,
    fact: str = "",
    *,
    uuid: str | None = None,
    entity_type: str = "",
    tags: list[str] | None = None,
    group_id: str = "",
) -> dict[str, Any]:
    """Build a synthetic graph-edge fact."""
    return {
        "uuid": uuid or hashlib.md5(name.encode(), usedforsecurity=False).hexdigest()[:8],
        "name": name,
        "fact": fact or f"Fact about {name}",
        "_source": "graph",
        "entity_type": entity_type,
        "tags": tags or [],
        "group_id": group_id,
    }


def _make_typed_state(
    subject: str,
    predicate: str = "is",
    value: str = "something",
    *,
    object_id: str | None = None,
    bucket: str = "",
    object_type: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Build a synthetic typed state item (ChangeLedger format)."""
    return {
        "object_id": object_id or hashlib.md5(subject.encode(), usedforsecurity=False).hexdigest()[:8],
        "subject": subject,
        "predicate": predicate,
        "value": value,
        "bucket": bucket,
        "object_type": object_type,
        "tags": tags or [],
    }


def _make_typed_procedure(
    name: str,
    trigger: str = "manual",
    *,
    object_id: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Build a synthetic typed procedure item."""
    return {
        "object_id": object_id or hashlib.md5(name.encode(), usedforsecurity=False).hexdigest()[:8],
        "name": name,
        "trigger": trigger,
        "object_type": "procedure",
        "tags": tags or [],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Section 1: QueryIntentClassifier
# ══════════════════════════════════════════════════════════════════════════════

class TestQueryIntentClassifier:
    """Unit tests for QueryIntentClassifier."""

    def setup_method(self):
        self.classifier = QueryIntentClassifier()

    # ── Caching ───────────────────────────────────────────────────────────

    def test_cache_hit_returns_same_result(self):
        """Same query → same cached result without re-classification."""
        result1 = self.classifier.classify("What are Yuan's favorite cuisines?")
        result2 = self.classifier.classify("What are Yuan's favorite cuisines?")
        assert result1 == result2

    def test_cache_stores_entry(self):
        """Verify the internal cache is populated."""
        self.classifier.classify("test query alpha")
        key = hashlib.sha256(b"test query alpha").hexdigest()
        assert key in self.classifier._cache

    def test_cache_clear(self):
        """clear_cache() empties internal state."""
        self.classifier.classify("some query")
        assert len(self.classifier._cache) > 0
        self.classifier.clear_cache()
        assert len(self.classifier._cache) == 0

    def test_cache_ttl_expiry(self):
        """Expired entries are re-classified."""
        self.classifier.classify("ttl test query")
        key = hashlib.sha256(b"ttl test query").hexdigest()
        # Manually expire the entry
        intent, _ = self.classifier._cache[key]
        self.classifier._cache[key] = (intent, time.monotonic() - 7200)
        # Re-classify — should go through uncached path
        result = self.classifier.classify("ttl test query")
        _, ts = self.classifier._cache[key]
        assert time.monotonic() - ts < 5  # freshly cached

    # ── Intent classification (20+ example queries) ───────────────────────

    @pytest.mark.parametrize("query,expected", [
        # persona
        ("What are Yuan's favorite cuisines?", "persona"),
        ("Tell me about Yuan's scheduling preference", "persona"),
        ("What is his communication style?", "persona"),
        ("What are the meeting defaults?", "persona"),
        ("Who is Yuan?", "persona"),
        ("What is Yuan's background?", "persona"),
        ("What are his working hours?", "persona"),
        # preference
        ("What's his opinion on React vs Vue?", "preference"),
        ("Does Yuan like sushi?", "preference"),
        ("What's the best restaurant for dinner?", "preference"),
        ("Any wine recommendation for tonight?", "preference"),
        ("What cuisine does Yuan enjoy most?", "preference"),
        # decision
        ("Why did we pick FalkorDB over Neo4j?", "decision"),
        ("What were the trade-offs for the hybrid approach?", "decision"),
        ("What alternative was considered for RRF?", "decision"),
        ("Rationale behind the k=60 constant?", "decision"),
        # operational
        ("How does the interrupt vs batch update work?", "operational"),
        ("What's the cron schedule for audits?", "operational"),
        ("Describe the heartbeat workflow", "operational"),
        ("Show me the deployment procedure", "operational"),
        # technical
        ("How does the RRF merge function work?", "technical"),
        ("What is the hybrid retrieval architecture?", "technical"),
        ("Explain the vector embedding pipeline", "technical"),
        ("What's the latency benchmark for graph retrieval?", "technical"),
        ("Show me the implementation of typed retrieval", "technical"),
        # generic (no clear match)
        ("Hello there!", "generic"),
        ("Tell me something interesting", "generic"),
        ("What time is it?", "generic"),
    ])
    def test_intent_classification(self, query: str, expected: str):
        assert self.classifier.classify(query) == expected

    def test_fallback_to_generic(self):
        """Completely ambiguous query → generic."""
        assert self.classifier.classify("xyzzy plugh nothing") == "generic"

    def test_empty_query_is_generic(self):
        assert self.classifier.classify("") == "generic"


# ══════════════════════════════════════════════════════════════════════════════
# Section 2: _lane_label
# ══════════════════════════════════════════════════════════════════════════════

class TestLaneLabel:
    """Unit tests for _lane_label helper."""

    # ── Typed candidates ──────────────────────────────────────────────────

    def test_typed_state_persona_bucket(self):
        candidate = {
            "_source": "typed_state",
            "_original": {"bucket": "persona", "object_type": "identity"},
        }
        assert _lane_label(candidate) == "persona"

    def test_typed_state_preference_bucket(self):
        candidate = {
            "_source": "typed_state",
            "_original": {"bucket": "preference"},
        }
        assert _lane_label(candidate) == "preference"

    def test_typed_state_operational_bucket(self):
        candidate = {
            "_source": "typed_state",
            "_original": {"bucket": "operational"},
        }
        assert _lane_label(candidate) == "operational"

    def test_typed_state_decision_bucket(self):
        candidate = {
            "_source": "typed_state",
            "_original": {"bucket": "decision"},
        }
        assert _lane_label(candidate) == "decision"

    def test_typed_procedure_decision_tag(self):
        candidate = {
            "_source": "typed_procedure",
            "_original": {"object_type": "procedure", "tags": ["decision_framework"]},
        }
        assert _lane_label(candidate) == "decision"

    def test_typed_state_engineering_tag(self):
        candidate = {
            "_source": "typed_state",
            "_original": {"bucket": "", "tags": ["engineering"]},
        }
        assert _lane_label(candidate) == "engineering"

    def test_typed_state_subject_preference_heuristic(self):
        candidate = {
            "_source": "typed_state",
            "_original": {"bucket": "", "subject": "Yuan's favorite wine"},
        }
        assert _lane_label(candidate) == "preference"

    def test_typed_state_subject_schedule_heuristic(self):
        candidate = {
            "_source": "typed_state",
            "_original": {"bucket": "", "subject": "daily schedule routine"},
        }
        assert _lane_label(candidate) == "persona"

    def test_typed_state_generic_fallback(self):
        candidate = {
            "_source": "typed_state",
            "_original": {"bucket": "", "subject": "random thing"},
        }
        assert _lane_label(candidate) == "generic"

    # ── Graph candidates ──────────────────────────────────────────────────

    def test_graph_incident_entity_type(self):
        candidate = {
            "_source": "graph",
            "entity_type": "incident",
            "name": "prod outage",
            "fact": "something broke",
        }
        assert _lane_label(candidate) == "incident"

    def test_graph_engineering_keyword_in_name(self):
        candidate = {
            "_source": "graph",
            "name": "feature flag wiring",
            "fact": "ensuring feature flags are fully wired",
        }
        assert _lane_label(candidate) == "engineering"

    def test_graph_engineering_tag(self):
        candidate = {
            "_source": "graph",
            "name": "some fact",
            "fact": "details about it",
            "tags": ["architecture"],
        }
        assert _lane_label(candidate) == "engineering"

    def test_graph_persona_keyword(self):
        candidate = {
            "_source": "graph",
            "name": "favorite cuisine",
            "fact": "yuan likes japanese food",
        }
        assert _lane_label(candidate) == "persona"

    def test_graph_decision_keyword(self):
        candidate = {
            "_source": "graph",
            "name": "decision on DB choice",
            "fact": "chose FalkorDB for performance",
        }
        assert _lane_label(candidate) == "decision"

    def test_graph_generic_fallback(self):
        candidate = {
            "_source": "graph",
            "name": "some random thing",
            "fact": "unrelated content",
        }
        assert _lane_label(candidate) == "generic"

    # ── Unknown source ────────────────────────────────────────────────────

    def test_unknown_source_generic(self):
        candidate = {"_source": "unknown"}
        assert _lane_label(candidate) == "generic"

    def test_no_source_generic(self):
        candidate = {}
        assert _lane_label(candidate) == "generic"


# ══════════════════════════════════════════════════════════════════════════════
# Section 3: Suppression Matrix
# ══════════════════════════════════════════════════════════════════════════════

class TestSuppressionMatrix:
    """Verify suppression matrix structure and content."""

    def test_all_intents_present(self):
        expected_intents = {
            "persona", "preference", "operational", "technical",
            "decision", "engineering", "incident", "generic",
        }
        assert set(SUPPRESSION_MATRIX.keys()) == expected_intents

    def test_persona_suppresses_engineering_and_technical(self):
        assert "engineering" in SUPPRESSION_MATRIX["persona"]
        assert "technical" in SUPPRESSION_MATRIX["persona"]

    def test_preference_suppresses_engineering_incident_operational(self):
        assert "engineering" in SUPPRESSION_MATRIX["preference"]
        assert "incident" in SUPPRESSION_MATRIX["preference"]
        assert "operational" in SUPPRESSION_MATRIX["preference"]

    def test_technical_suppresses_persona_preference(self):
        assert "persona" in SUPPRESSION_MATRIX["technical"]
        assert "preference" in SUPPRESSION_MATRIX["technical"]

    def test_decision_suppresses_persona_preference(self):
        assert "persona" in SUPPRESSION_MATRIX["decision"]
        assert "preference" in SUPPRESSION_MATRIX["decision"]

    def test_operational_suppresses_nothing(self):
        assert SUPPRESSION_MATRIX["operational"] == []

    def test_generic_suppresses_nothing(self):
        assert SUPPRESSION_MATRIX["generic"] == []


# ══════════════════════════════════════════════════════════════════════════════
# Section 4: rrf_merge_hybrid with suppression
# ══════════════════════════════════════════════════════════════════════════════

class TestRrfMergeHybridSuppression:
    """Test lane suppression in rrf_merge_hybrid."""

    def _build_mixed_pool(self):
        """Build a candidate pool with engineering, persona, and generic items."""
        graph_facts = [
            _make_graph_fact("feature flag wiring", "ensuring feature flags are wired"),
            _make_graph_fact("runtime version mismatch", "mismatched claims about version"),
            _make_graph_fact("favorite cuisine", "yuan likes japanese food"),
            _make_graph_fact("random context", "some generic information"),
        ]
        typed_state = [
            _make_typed_state("Yuan's scheduling preference", bucket="persona"),
            _make_typed_state("deploy pipeline", bucket="operational"),
            _make_typed_state("wine taste", bucket="preference"),
        ]
        typed_procedures = [
            _make_typed_procedure("rollback procedure", tags=["ops"]),
        ]
        return graph_facts, typed_state, typed_procedures

    def test_persona_intent_suppresses_engineering(self):
        """Persona query → engineering facts get zero score."""
        graph_facts, typed_state, typed_procedures = self._build_mixed_pool()
        results = rrf_merge_hybrid(
            graph_facts=graph_facts,
            typed_state=typed_state,
            typed_procedures=typed_procedures,
            max_facts=10,
            query_intent="persona",
            apply_diversity=False,
        )
        # Engineering facts should be at the bottom (zero score)
        for r in results:
            lane = _lane_label(r)
            if lane == "engineering":
                assert r["_hybrid_score"] == 0.0

    def test_technical_intent_suppresses_persona_preference(self):
        """Technical query → persona and preference candidates get zero score."""
        graph_facts, typed_state, typed_procedures = self._build_mixed_pool()
        results = rrf_merge_hybrid(
            graph_facts=graph_facts,
            typed_state=typed_state,
            typed_procedures=typed_procedures,
            max_facts=10,
            query_intent="technical",
            apply_diversity=False,
        )
        for r in results:
            lane = _lane_label(r)
            if lane in ("persona", "preference"):
                assert r["_hybrid_score"] == 0.0

    def test_generic_intent_no_suppression(self):
        """Generic intent → all candidates keep their scores."""
        graph_facts, typed_state, typed_procedures = self._build_mixed_pool()
        results = rrf_merge_hybrid(
            graph_facts=graph_facts,
            typed_state=typed_state,
            typed_procedures=typed_procedures,
            max_facts=10,
            query_intent="generic",
            apply_diversity=False,
        )
        assert all(r["_hybrid_score"] > 0 for r in results)

    def test_no_intent_param_no_suppression(self):
        """No query_intent → backwards-compatible, no suppression."""
        graph_facts, typed_state, typed_procedures = self._build_mixed_pool()
        results = rrf_merge_hybrid(
            graph_facts=graph_facts,
            typed_state=typed_state,
            typed_procedures=typed_procedures,
            max_facts=10,
            query_intent=None,
            apply_diversity=False,
        )
        assert all(r["_hybrid_score"] > 0 for r in results)

    def test_suppressed_candidates_sink_to_bottom(self):
        """Zero-scored candidates sort after positive-scored ones."""
        graph_facts, typed_state, typed_procedures = self._build_mixed_pool()
        results = rrf_merge_hybrid(
            graph_facts=graph_facts,
            typed_state=typed_state,
            typed_procedures=typed_procedures,
            max_facts=10,
            query_intent="persona",
            apply_diversity=False,
        )
        # Find the first zero-scored result
        scores = [r["_hybrid_score"] for r in results]
        positive_scores = [s for s in scores if s > 0]
        zero_scores = [s for s in scores if s == 0]
        # All positives should come before zeros
        if positive_scores and zero_scores:
            last_positive_idx = max(i for i, s in enumerate(scores) if s > 0)
            first_zero_idx = min(i for i, s in enumerate(scores) if s == 0)
            assert last_positive_idx < first_zero_idx

    def test_max_facts_contract_preserved(self):
        """Output length never exceeds max_facts."""
        graph_facts, typed_state, typed_procedures = self._build_mixed_pool()
        results = rrf_merge_hybrid(
            graph_facts=graph_facts,
            typed_state=typed_state,
            typed_procedures=typed_procedures,
            max_facts=3,
            query_intent="persona",
            apply_diversity=False,
        )
        assert len(results) <= 3


# ══════════════════════════════════════════════════════════════════════════════
# Section 5: Integration — HybridRetrievalService.merge with intent
# ══════════════════════════════════════════════════════════════════════════════

class TestHybridServiceMergeWithIntent:
    """Test that merge() integrates query-intent classification."""

    def test_merge_with_query_classifies_intent(self):
        """merge(query=...) should classify and apply suppression."""
        service = HybridRetrievalService()
        graph_facts = [
            _make_graph_fact("feature flag wiring", "ensuring feature flags are wired"),
            _make_graph_fact("favorite cuisine", "yuan likes japanese food"),
        ]
        typed_results = {
            "state": [_make_typed_state("Yuan's preference", bucket="persona")],
            "procedures": [],
        }
        # Persona query — should suppress engineering
        results = service.merge(
            graph_facts=graph_facts,
            typed_results=typed_results,
            max_facts=10,
            query="What are Yuan's favorite cuisines?",
            apply_diversity=False,
        )
        # The engineering fact should be zero-scored
        for r in results:
            if _lane_label(r) == "engineering":
                assert r["_hybrid_score"] == 0.0

    def test_merge_without_query_no_suppression(self):
        """merge(query=None) → no classification, backwards-compatible."""
        service = HybridRetrievalService()
        graph_facts = [
            _make_graph_fact("feature flag wiring", "ensuring feature flags are wired"),
        ]
        typed_results = {"state": [], "procedures": []}
        results = service.merge(
            graph_facts=graph_facts,
            typed_results=typed_results,
            max_facts=10,
            apply_diversity=False,
        )
        assert all(r["_hybrid_score"] > 0 for r in results)

    def test_merge_uses_module_singleton_classifier(self):
        """Verify merge() uses the module-level classifier singleton."""
        _query_intent_classifier.clear_cache()
        service = HybridRetrievalService()
        service.merge(
            graph_facts=[_make_graph_fact("test")],
            typed_results={"state": [], "procedures": []},
            max_facts=10,
            query="What are Yuan's favorite cuisines?",
            apply_diversity=False,
        )
        # After calling merge, the singleton cache should have an entry
        key = hashlib.sha256(b"What are Yuan's favorite cuisines?").hexdigest()
        assert key in _query_intent_classifier._cache
        _query_intent_classifier.clear_cache()
