"""Phase 2A — Query-Intent Lane Filter tests.

Covers:
- QueryIntentClassifier: classification accuracy, caching behaviour, fallback
- _lane_label(): lane detection for graph + typed candidates
- rrf_merge_hybrid(): Phase 2A suppression logic
- HybridRetrievalService.merge(): intent classification wiring

All tests are self-contained (no DB, no network).
"""
from __future__ import annotations

import pathlib
import sys
import time
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Import the service under test
# ---------------------------------------------------------------------------
_svc_pkg = str(
    pathlib.Path(__file__).parent.parent / "src" / "services"
)
if _svc_pkg not in sys.path:
    sys.path.insert(0, _svc_pkg)

from mcp_server.src.services.typed_retrieval_service import (  # noqa: E402
    SUPPRESSION_MATRIX,
    HybridRetrievalService,
    QueryIntentClassifier,
    _lane_label,
    rrf_merge_hybrid,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _graph_fact(
    uuid: str,
    name: str = "",
    entity_type: str = "",
    tags: list[str] | None = None,
    fact: str = "",
) -> dict[str, Any]:
    return {
        "uuid": uuid,
        "name": name,
        "entity_type": entity_type,
        "tags": tags or [],
        "fact": fact,
        "_source": "graph",
    }


def _typed_state(
    object_id: str,
    subject: str = "",
    predicate: str = "",
    value: str = "",
    fact_type: str = "preference",
    bucket: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "object_id": object_id,
        "object_type": "state_fact",
        "subject": subject,
        "predicate": predicate,
        "value": value,
        "fact_type": fact_type,
        "bucket": bucket,
        "tags": tags or [],
    }


def _typed_procedure(
    object_id: str,
    name: str = "",
    trigger: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "object_id": object_id,
        "object_type": "procedure",
        "name": name,
        "trigger": trigger,
        "tags": tags or [],
    }


# ===========================================================================
# QueryIntentClassifier tests
# ===========================================================================

class TestQueryIntentClassifier:
    """Tests for the rule-based query intent classifier."""

    def setup_method(self):
        self.clf = QueryIntentClassifier()

    # ── Persona intent ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("query", [
        "What are Yuan's scheduling preferences?",
        "What is Yuan's favorite cuisine?",
        "Tell me about Yuan's morning workout routine",
        "What is Yuan's communication style?",
        "What are Yuan's working hours?",
        "Yuan's calendar availability defaults",
        "Yuan's timezone preference",
    ])
    def test_persona_queries(self, query: str):
        assert self.clf.classify(query) == "persona", f"Expected persona for: {query!r}"

    # ── Preference intent ───────────────────────────────────────────────────

    @pytest.mark.parametrize("query", [
        "What is Yuan's opinion on DeFi protocols?",
        "What does Yuan think about async frameworks?",
        "What's Yuan's taste in wine?",
        "Yuan's favorite food",
        "Best restaurant recommendations for Yuan",
    ])
    def test_preference_queries(self, query: str):
        result = self.clf.classify(query)
        assert result in ("preference", "persona"), (
            f"Expected preference/persona for: {query!r}, got {result!r}"
        )

    # ── Decision intent ─────────────────────────────────────────────────────

    @pytest.mark.parametrize("query", [
        "Why did we switch to RRF fusion?",
        "What were the trade-offs of Option A vs Option B?",
        "What alternatives were considered for the memory architecture?",
        "What was the rationale for choosing Neo4j?",
    ])
    def test_decision_queries(self, query: str):
        assert self.clf.classify(query) == "decision", (
            f"Expected decision for: {query!r}"
        )

    # ── Technical intent ────────────────────────────────────────────────────

    @pytest.mark.parametrize("query", [
        "How does the RRF merge implementation work?",
        "What is the hybrid retrieval architecture?",
        "What are the GraphitiMCP API endpoints?",
        "How does the embedding indexing work?",
        "Explain the TypedRetrievalService implementation",
    ])
    def test_technical_queries(self, query: str):
        assert self.clf.classify(query) == "technical", (
            f"Expected technical for: {query!r}"
        )

    # ── Operational intent ──────────────────────────────────────────────────

    @pytest.mark.parametrize("query", [
        "Interrupt vs batch update rules?",
        "What does the communication guard do?",
        "How does the heartbeat cron job work?",
        "What is the deployment runbook?",
    ])
    def test_operational_queries(self, query: str):
        assert self.clf.classify(query) == "operational", (
            f"Expected operational for: {query!r}"
        )

    # ── Generic fallback ────────────────────────────────────────────────────

    @pytest.mark.parametrize("query", [
        "Xyzzy 123 foobar",
        "Random text that matches no patterns at all",
        "  ",  # whitespace only
    ])
    def test_generic_fallback(self, query: str):
        result = self.clf.classify(query)
        assert result == "generic", f"Expected generic for: {query!r}"

    def test_empty_query(self):
        assert self.clf.classify("") == "generic"

    # ── Caching ─────────────────────────────────────────────────────────────

    def test_cache_hit_returns_same_result(self):
        query = "Why did we pick RRF over BM25?"
        r1 = self.clf.classify(query)
        r2 = self.clf.classify(query)
        assert r1 == r2

    def test_cache_stores_result(self):
        query = "Yuan's favorite cuisine"
        self.clf.clear_cache()
        assert len(self.clf._cache) == 0
        self.clf.classify(query)
        assert len(self.clf._cache) == 1

    def test_clear_cache_empties_store(self):
        self.clf.classify("Yuan's scheduling preferences")
        self.clf.classify("Why did we change the schema?")
        assert len(self.clf._cache) == 2
        self.clf.clear_cache()
        assert len(self.clf._cache) == 0

    def test_expired_cache_reclassifies(self, monkeypatch):
        """Expired cache entries should be re-classified."""
        query = "Yuan's favorite food"
        # Artificially set very short TTL by patching the cached timestamp
        self.clf.classify(query)
        key = list(self.clf._cache.keys())[0]

        # Expire the entry by backdating the timestamp
        intent, ts = self.clf._cache[key]
        self.clf._cache[key] = (intent, ts - 7200.0)  # 2 hours ago

        # Should re-classify (same result expected)
        result = self.clf.classify(query)
        assert result in ("persona", "preference", "generic")  # any valid intent
        # Cache should now have fresh timestamp
        _, new_ts = self.clf._cache[key]
        assert new_ts > ts - 3600  # fresher than the expired entry

    def test_different_queries_different_cache_entries(self):
        self.clf.clear_cache()
        self.clf.classify("Yuan's favorite cuisine")
        self.clf.classify("Why did we adopt RRF?")
        assert len(self.clf._cache) == 2


# ===========================================================================
# SUPPRESSION_MATRIX structural tests
# ===========================================================================

class TestSuppressionMatrix:

    def test_all_intent_keys_present(self):
        expected = {"persona", "preference", "operational", "technical",
                    "decision", "engineering", "incident", "generic"}
        assert set(SUPPRESSION_MATRIX.keys()) == expected

    def test_persona_suppresses_engineering_and_technical(self):
        assert "engineering" in SUPPRESSION_MATRIX["persona"]
        assert "technical" in SUPPRESSION_MATRIX["persona"]

    def test_preference_suppresses_engineering_incident_operational(self):
        assert "engineering" in SUPPRESSION_MATRIX["preference"]
        assert "incident" in SUPPRESSION_MATRIX["preference"]
        assert "operational" in SUPPRESSION_MATRIX["preference"]

    def test_technical_suppresses_persona_and_preference(self):
        assert "persona" in SUPPRESSION_MATRIX["technical"]
        assert "preference" in SUPPRESSION_MATRIX["technical"]

    def test_decision_suppresses_persona_and_preference(self):
        assert "persona" in SUPPRESSION_MATRIX["decision"]
        assert "preference" in SUPPRESSION_MATRIX["decision"]

    def test_generic_suppresses_nothing(self):
        assert SUPPRESSION_MATRIX["generic"] == []

    def test_operational_suppresses_nothing(self):
        assert SUPPRESSION_MATRIX["operational"] == []


# ===========================================================================
# _lane_label() tests
# ===========================================================================

class TestLaneLabel:

    # ── Graph candidates ─────────────────────────────────────────────────────

    def test_graph_incident_entity_type(self):
        cand = _graph_fact("g1", entity_type="incident")
        assert _lane_label(cand) == "incident"

    def test_graph_incident_tag(self):
        cand = _graph_fact("g2", tags=["incident"])
        assert _lane_label(cand) == "incident"

    def test_graph_engineering_tag(self):
        cand = _graph_fact("g3", tags=["engineering"])
        assert _lane_label(cand) == "engineering"

    def test_graph_engineering_fact_keyword(self):
        cand = _graph_fact("g4", fact="Ensuring feature flag is fully wired in production")
        assert _lane_label(cand) == "engineering"

    def test_graph_persona_keyword_in_fact(self):
        cand = _graph_fact("g5", fact="Yuan's favorite restaurant preference is Japanese")
        assert _lane_label(cand) == "persona"

    def test_graph_generic_fallback(self):
        cand = _graph_fact("g6", name="some random fact", fact="no special keywords")
        assert _lane_label(cand) == "generic"

    # ── Typed state candidates ────────────────────────────────────────────────

    def test_typed_state_preference_fact_type(self):
        item = _typed_state("ts1", fact_type="preference")
        cand = {
            "_source": "typed_state",
            "_object_type": "state_fact",
            "uuid": "ts1",
            "fact": "Yuan prefers espresso",
            "_original": item,
        }
        assert _lane_label(cand) == "preference"

    def test_typed_state_decision_fact_type(self):
        item = _typed_state("ts2", fact_type="decision")
        cand = {
            "_source": "typed_state",
            "_object_type": "state_fact",
            "uuid": "ts2",
            "fact": "Decision to adopt Neo4j",
            "_original": item,
        }
        assert _lane_label(cand) == "decision"

    def test_typed_state_operational_rule_fact_type(self):
        item = _typed_state("ts3", fact_type="operational_rule")
        cand = {
            "_source": "typed_state",
            "_object_type": "state_fact",
            "uuid": "ts3",
            "fact": "Interrupt vs batch update rule",
            "_original": item,
        }
        assert _lane_label(cand) == "operational"

    def test_typed_state_bucket_persona(self):
        item = _typed_state("ts4", fact_type="", bucket="persona")
        cand = {
            "_source": "typed_state",
            "_object_type": "state_fact",
            "uuid": "ts4",
            "fact": "identity info",
            "_original": item,
        }
        assert _lane_label(cand) == "persona"

    def test_typed_state_engineering_tag(self):
        item = _typed_state("ts5", fact_type="", tags=["engineering"])
        cand = {
            "_source": "typed_state",
            "_object_type": "state_fact",
            "uuid": "ts5",
            "fact": "some technical fact",
            "_original": item,
        }
        assert _lane_label(cand) == "engineering"

    # ── Typed procedure candidates ─────────────────────────────────────────────

    def test_typed_procedure_is_operational(self):
        item = _typed_procedure("tp1", name="Deploy rollback procedure")
        cand = {
            "_source": "typed_procedure",
            "_object_type": "procedure",
            "uuid": "tp1",
            "fact": "Deploy rollback procedure",
            "_original": item,
        }
        assert _lane_label(cand) == "operational"

    # ── Unknown source ────────────────────────────────────────────────────────

    def test_unknown_source_generic(self):
        cand = {"_source": "unknown_source", "uuid": "u1"}
        assert _lane_label(cand) == "generic"


# ===========================================================================
# rrf_merge_hybrid() — Phase 2A suppression tests
# ===========================================================================

class TestRrfMergeHybridSuppression:
    """Tests for the Phase 2A lane suppression logic in rrf_merge_hybrid."""

    def _make_engineering_graph_fact(self, uuid: str) -> dict[str, Any]:
        return _graph_fact(uuid, tags=["engineering"], fact="feature flag wiring details")

    def _make_preference_typed_state(self, oid: str) -> dict[str, Any]:
        return _typed_state(oid, subject="Yuan", predicate="favorite cuisine", value="Japanese",
                            fact_type="preference")

    def test_no_intent_no_suppression(self):
        """Without query_intent, all candidates are returned normally."""
        graph = [self._make_engineering_graph_fact("g1")]
        typed_state = [self._make_preference_typed_state("ts1")]

        result = rrf_merge_hybrid(
            graph_facts=graph,
            typed_state=typed_state,
            typed_procedures=[],
            max_facts=10,
            query_intent=None,
            apply_diversity=False,
        )
        uuids = [r.get("uuid", "") for r in result]
        assert "g1" in uuids or any("g1" in str(r) for r in result)

    def test_persona_query_suppresses_engineering(self):
        """A persona query should zero-score engineering candidates."""
        graph = [self._make_engineering_graph_fact("g_eng")]
        typed_state = [self._make_preference_typed_state("ts_pref")]

        result = rrf_merge_hybrid(
            graph_facts=graph,
            typed_state=typed_state,
            typed_procedures=[],
            max_facts=10,
            query_intent="persona",
            apply_diversity=False,
        )

        # The engineering graph fact should have _hybrid_score == 0.0
        for item in result:
            source = item.get("_source")
            if source == "graph":
                assert item["_hybrid_score"] == 0.0, (
                    f"Engineering graph fact should be zero-scored for persona query, "
                    f"got {item['_hybrid_score']}"
                )

    def test_persona_suppressed_candidates_sink_to_bottom(self):
        """Suppressed candidates should have score 0 and sort below unsuppressed ones."""
        graph = [self._make_engineering_graph_fact("g_eng")]
        typed_state = [self._make_preference_typed_state("ts_pref")]

        result = rrf_merge_hybrid(
            graph_facts=graph,
            typed_state=typed_state,
            typed_procedures=[],
            max_facts=10,
            query_intent="persona",
            apply_diversity=False,
        )
        # The preference typed candidate should appear before (or equal to) engineering
        scores = [(r.get("_source"), r.get("_hybrid_score", 0.0)) for r in result]
        eng_scores = [s for src, s in scores if src == "graph"]
        typed_scores = [s for src, s in scores if src == "typed_state"]
        if eng_scores and typed_scores:
            assert max(typed_scores) >= max(eng_scores), (
                f"Typed preference should score >= suppressed engineering, "
                f"typed={typed_scores}, eng={eng_scores}"
            )

    def test_technical_query_suppresses_preference(self):
        """A technical query should suppress preference-typed candidates."""
        typed_state = [self._make_preference_typed_state("ts_pref")]

        result = rrf_merge_hybrid(
            graph_facts=[],
            typed_state=typed_state,
            typed_procedures=[],
            max_facts=10,
            query_intent="technical",
            apply_diversity=False,
        )
        for item in result:
            if item.get("_source") == "typed_state":
                assert item["_hybrid_score"] == 0.0

    def test_generic_query_no_suppression(self):
        """Generic intent should suppress nothing — full pool returned."""
        graph = [self._make_engineering_graph_fact("g_eng")]
        typed_state = [self._make_preference_typed_state("ts_pref")]

        result = rrf_merge_hybrid(
            graph_facts=graph,
            typed_state=typed_state,
            typed_procedures=[],
            max_facts=10,
            query_intent="generic",
            apply_diversity=False,
        )
        # Both should have positive scores
        for item in result:
            assert item["_hybrid_score"] > 0.0, (
                f"Generic intent should not suppress {item.get('_source')}"
            )

    def test_max_facts_contract_preserved_with_suppression(self):
        """Suppression should never cause the list to exceed max_facts."""
        graph = [self._make_engineering_graph_fact(f"g{i}") for i in range(5)]
        typed_state = [self._make_preference_typed_state(f"ts{i}") for i in range(5)]

        result = rrf_merge_hybrid(
            graph_facts=graph,
            typed_state=typed_state,
            typed_procedures=[],
            max_facts=6,
            query_intent="persona",
            apply_diversity=False,
        )
        assert len(result) <= 6

    def test_suppression_does_not_delete_candidates(self):
        """Suppression zeroes scores but keeps candidate in pool (bottom) — max_facts contract."""
        graph = [self._make_engineering_graph_fact(f"g{i}") for i in range(3)]
        typed_state = [self._make_preference_typed_state(f"ts{i}") for i in range(3)]

        result = rrf_merge_hybrid(
            graph_facts=graph,
            typed_state=typed_state,
            typed_procedures=[],
            max_facts=10,  # large enough to see all candidates
            query_intent="persona",
            apply_diversity=False,
        )
        # All 6 items should be present (zeroed engineering don't disappear)
        assert len(result) == 6

    def test_empty_inputs_with_intent(self):
        """No candidates + intent should return empty list safely."""
        result = rrf_merge_hybrid(
            graph_facts=[],
            typed_state=[],
            typed_procedures=[],
            max_facts=10,
            query_intent="persona",
            apply_diversity=False,
        )
        assert result == []

    def test_max_facts_zero(self):
        graph = [self._make_engineering_graph_fact("g1")]
        result = rrf_merge_hybrid(
            graph_facts=graph,
            typed_state=[],
            typed_procedures=[],
            max_facts=0,
            query_intent="persona",
            apply_diversity=False,
        )
        assert result == []


# ===========================================================================
# HybridRetrievalService.merge() — intent wiring tests
# ===========================================================================

class TestHybridRetrievalServiceMerge:
    """Tests that merge() correctly wires the intent classifier."""

    def _make_svc(self) -> HybridRetrievalService:
        svc = HybridRetrievalService.__new__(HybridRetrievalService)
        svc.om_projection_service = None
        svc._typed_service = None
        return svc

    def _typed_results(
        self,
        state: list[dict] | None = None,
        procedures: list[dict] | None = None,
    ) -> dict:
        return {
            "state": state or [],
            "procedures": procedures or [],
            "counts": {"state": 0, "procedures": 0},
        }

    def test_merge_without_query_uses_no_intent(self):
        """merge() with no query should not crash and should return results."""
        svc = self._make_svc()
        graph = [_graph_fact("g1")]
        result = svc.merge(
            graph_facts=graph,
            typed_results=self._typed_results(),
            max_facts=5,
            apply_diversity=False,
        )
        assert isinstance(result, list)
        assert len(result) == 1

    def test_merge_with_persona_query_suppresses_engineering(self):
        """merge() with a persona query should suppress engineering candidates."""
        svc = self._make_svc()
        graph = [_graph_fact("g1", tags=["engineering"], fact="feature flag wiring")]
        typed_state = [_typed_state("ts1", fact_type="preference",
                                    subject="Yuan", predicate="favorite", value="espresso")]

        result = svc.merge(
            graph_facts=graph,
            typed_results=self._typed_results(state=typed_state),
            max_facts=10,
            query="What are Yuan's scheduling preferences?",
            apply_diversity=False,
        )
        # Engineering graph fact should be zero-scored
        for item in result:
            if item.get("_source") == "graph":
                assert item["_hybrid_score"] == 0.0

    def test_merge_with_technical_query_suppresses_preference(self):
        """merge() with technical query should suppress preference typed items."""
        svc = self._make_svc()
        typed_state = [_typed_state("ts1", fact_type="preference",
                                    subject="Yuan", predicate="likes", value="wine")]

        result = svc.merge(
            graph_facts=[],
            typed_results=self._typed_results(state=typed_state),
            max_facts=5,
            query="How does the hybrid retrieval architecture work?",
            apply_diversity=False,
        )
        for item in result:
            assert item["_hybrid_score"] == 0.0

    def test_merge_returns_list_of_dicts(self):
        svc = self._make_svc()
        graph = [_graph_fact("g1"), _graph_fact("g2")]
        result = svc.merge(
            graph_facts=graph,
            typed_results=self._typed_results(),
            max_facts=10,
            query="generic query",
            apply_diversity=False,
        )
        assert all(isinstance(r, dict) for r in result)

    def test_merge_respects_max_facts(self):
        svc = self._make_svc()
        graph = [_graph_fact(f"g{i}") for i in range(20)]
        result = svc.merge(
            graph_facts=graph,
            typed_results=self._typed_results(),
            max_facts=5,
            apply_diversity=False,
        )
        assert len(result) <= 5
