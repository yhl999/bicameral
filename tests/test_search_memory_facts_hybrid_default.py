"""Tests for the Hybrid Default Surface Implementation (v1).

Covers:
- Default path uses hybrid retrieval + reranking (not graph-only stub)
- Explicit retrieval_mode='hybrid' gives the same behaviour
- Explicit retrieval_mode='graph' still returns graph-only FactSearchResponse
- Explicit retrieval_mode='typed' still routes to typed-ledger path
- Empty-result behaviour: both sources empty → deterministic empty hybrid response
- Partial-result cases: graph empty / typed empty
- Hybrid response shape is deterministic and testable
- No contract regressions (graph + typed paths unchanged)
- HybridRetrievalService.merge() and rrf_merge_hybrid() unit tests
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
from types import SimpleNamespace
from typing import Any

from tests.helpers_mcp_import import load_graphiti_mcp_server

# Also import the service module directly for unit tests.
# Make the service importable without a live Neo4j connection.
_svc_path = str(
    pathlib.Path(__file__).parent.parent / "mcp_server" / "src" / "services"
)
if _svc_path not in sys.path:
    sys.path.insert(0, _svc_path)

from mcp_server.src.services.typed_retrieval_service import (  # noqa: E402
    HybridRetrievalService,
    rrf_merge_hybrid,
)

server = load_graphiti_mcp_server()


def _run(coro):
    return asyncio.run(coro)


# ── Shared fixtures ───────────────────────────────────────────────────────────


def _test_config():
    """Minimal config: Neo4j, no OM groups in scope, no authorized restriction."""
    return SimpleNamespace(
        database=SimpleNamespace(provider="neo4j"),
        graphiti=SimpleNamespace(
            group_id="s1_sessions_main",
            lane_aliases={
                "sessions_main": ["s1_sessions_main"],
                "observational_memory": ["s1_observational_memory"],
            },
            authorized_group_ids=[],
        ),
    )


def _fake_typed_payload(state=None, procedures=None):
    state = state or []
    procedures = procedures or []
    return {
        "message": "Typed memory retrieved successfully",
        "retrieval_mode": "typed",
        "result_format": "typed",
        "query_mode": "all",
        "state": state,
        "episodes": [],
        "procedures": procedures,
        "evidence": [],
        "counts": {
            "state": len(state),
            "episodes": 0,
            "procedures": len(procedures),
            "evidence": 0,
        },
        "filters_applied": {"object_types": ["state_fact", "procedure"], "metadata_filters": {}},
        "limits_applied": {},
    }


class _FakeHybridRetrievalService:
    """Test double for HybridRetrievalService.

    Captures calls and returns configurable payloads without touching the
    change ledger or any real services.
    """

    def __init__(self, typed_payload: dict | None = None, merged_override: list | None = None):
        self.typed_payload = typed_payload or _fake_typed_payload()
        self.merged_override = merged_override  # if set, merge() returns this list
        self.get_typed_calls: list[dict] = []
        self.merge_calls: list[dict] = []

    async def get_typed_candidates(self, **kwargs: Any) -> dict[str, Any]:
        self.get_typed_calls.append(kwargs)
        return self.typed_payload

    def merge(
        self,
        *,
        graph_facts: list[dict],
        typed_results: dict,
        max_facts: int,
    ) -> list[dict[str, Any]]:
        self.merge_calls.append(
            {"graph_facts": graph_facts, "typed_results": typed_results, "max_facts": max_facts}
        )
        if self.merged_override is not None:
            return self.merged_override[:max_facts]
        # Simple deterministic merge: graph first, then typed state, then procedures.
        state = typed_results.get("state", []) or []
        procedures = typed_results.get("procedures", []) or []
        combined = (
            [{"_source": "graph", **f} for f in graph_facts]
            + [{"_source": "typed_state", **s} for s in state]
            + [{"_source": "typed_procedure", **p} for p in procedures]
        )
        return combined[:max_facts]


class _FakeSearchResults:
    """Fake graphiti search results (no real Neo4j edges)."""

    def __init__(self, edges=None):
        self.edges = edges or []
        self.nodes = []


class _FakeGraphitiClient:
    def __init__(self, edges=None):
        self._edges = edges or []

    async def search_(self, **kwargs: Any) -> _FakeSearchResults:
        return _FakeSearchResults(self._edges)


class _FakeGraphitiService:
    """Minimal fake that passes the ``is None`` check and returns no edges."""

    def __init__(self, edges=None):
        self._edges = edges or []

    async def get_client_for_group(self, group_id: str) -> _FakeGraphitiClient:
        return _FakeGraphitiClient(self._edges)


class _FakeSearchService:
    """Minimal fake for the module-level search_service dependency."""

    def includes_observational_memory(self, group_ids: list[str]) -> bool:
        return False


# ── Helper: run a hybrid search call with standard mocks ─────────────────────


def _run_hybrid(
    query: str = "test query",
    retrieval_mode: str | None = None,
    group_ids: list[str] | None = None,
    max_facts: int = 10,
    fake_typed_payload: dict | None = None,
    fake_merged_override: list | None = None,
    **kwargs: Any,
) -> dict:
    """Run search_memory_facts with hybrid mocks, return the raw response."""
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    original_graphiti = server.graphiti_service
    original_hybrid_cls = server.HybridRetrievalService
    original_search_service = server.search_service

    fake_hybrid = _FakeHybridRetrievalService(
        typed_payload=fake_typed_payload,
        merged_override=fake_merged_override,
    )
    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False
        server.graphiti_service = _FakeGraphitiService()
        server.HybridRetrievalService = lambda **_kw: fake_hybrid
        server.search_service = _FakeSearchService()

        return _run(
            server.search_memory_facts(
                query=query,
                retrieval_mode=retrieval_mode,
                group_ids=group_ids or ["s1_sessions_main"],
                max_facts=max_facts,
                ctx=None,
                **kwargs,
            )
        ), fake_hybrid
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.graphiti_service = original_graphiti
        server.HybridRetrievalService = original_hybrid_cls
        server.search_service = original_search_service


# ─────────────────────────────────────────────────────────────────────────────
# §1  Default path is hybrid (not graph-only stub)
# ─────────────────────────────────────────────────────────────────────────────


def test_default_path_returns_hybrid_response_shape():
    """No retrieval_mode supplied → default 'hybrid' → hybrid response dict."""
    typed_payload = _fake_typed_payload(
        state=[{"object_id": "sf_01", "subject": "user", "predicate": "likes", "value": "tea"}]
    )
    response, fake_hybrid = _run_hybrid(fake_typed_payload=typed_payload)

    assert isinstance(response, dict), f"Expected dict, got {type(response)}"
    assert response.get("retrieval_mode") == "hybrid"
    assert "facts" in response
    assert "typed_candidates" in response
    assert "merged_results" in response
    assert "result_count" in response


def test_default_path_calls_typed_candidates():
    """Default hybrid path must call get_typed_candidates (typed retrieval is used)."""
    typed_payload = _fake_typed_payload(
        state=[{"object_id": "sf_01", "subject": "user", "predicate": "likes", "value": "tea"}]
    )
    response, fake_hybrid = _run_hybrid(fake_typed_payload=typed_payload)

    assert len(fake_hybrid.get_typed_calls) == 1, (
        "Expected exactly one get_typed_candidates call in default hybrid path"
    )


def test_default_path_calls_merge():
    """Default hybrid path must call merge to combine graph + typed results."""
    typed_payload = _fake_typed_payload(
        state=[{"object_id": "sf_01", "subject": "user", "predicate": "likes", "value": "tea"}]
    )
    response, fake_hybrid = _run_hybrid(fake_typed_payload=typed_payload)

    assert len(fake_hybrid.merge_calls) == 1, (
        "Expected exactly one merge() call in default hybrid path"
    )


def test_default_path_typed_candidates_present_in_response():
    """Typed state facts from the typed path appear in typed_candidates of the response."""
    state_item = {
        "object_id": "sf_01",
        "object_type": "state_fact",
        "subject": "user",
        "predicate": "likes",
        "value": "tea",
    }
    typed_payload = _fake_typed_payload(state=[state_item])
    response, _ = _run_hybrid(fake_typed_payload=typed_payload)

    typed_cands = response.get("typed_candidates", {})
    assert typed_cands.get("counts", {}).get("state", 0) == 1, (
        f"Expected 1 typed state candidate, got: {typed_cands}"
    )
    assert len(typed_cands.get("state", [])) == 1


def test_default_path_result_count_matches_merged_list():
    """result_count must equal len(merged_results)."""
    typed_payload = _fake_typed_payload(
        state=[{"object_id": "sf_01"}, {"object_id": "sf_02"}]
    )
    response, _ = _run_hybrid(fake_typed_payload=typed_payload)

    assert response["result_count"] == len(response["merged_results"]), (
        "result_count must equal len(merged_results)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# §2  Explicit retrieval_mode='hybrid' behaves identically to default
# ─────────────────────────────────────────────────────────────────────────────


def test_explicit_hybrid_returns_same_shape_as_default():
    """retrieval_mode='hybrid' explicit → identical response shape to default."""
    typed_payload = _fake_typed_payload(
        state=[{"object_id": "sf_01"}]
    )
    response, fake_hybrid = _run_hybrid(
        retrieval_mode="hybrid", fake_typed_payload=typed_payload
    )

    assert response.get("retrieval_mode") == "hybrid"
    assert "merged_results" in response
    assert len(fake_hybrid.get_typed_calls) == 1


# ─────────────────────────────────────────────────────────────────────────────
# §3  Explicit graph path is unchanged (no typed candidates, FactSearchResponse)
# ─────────────────────────────────────────────────────────────────────────────


def test_explicit_graph_does_not_call_hybrid_service():
    """retrieval_mode='graph' must NOT invoke HybridRetrievalService."""
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    original_graphiti = server.graphiti_service
    original_hybrid_cls = server.HybridRetrievalService
    original_search_service = server.search_service

    fake_hybrid = _FakeHybridRetrievalService()
    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False
        server.graphiti_service = _FakeGraphitiService()
        server.HybridRetrievalService = lambda **_kw: fake_hybrid
        server.search_service = _FakeSearchService()

        _run(
            server.search_memory_facts(
                query="test",
                retrieval_mode="graph",
                group_ids=["s1_sessions_main"],
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.graphiti_service = original_graphiti
        server.HybridRetrievalService = original_hybrid_cls
        server.search_service = original_search_service

    assert fake_hybrid.get_typed_calls == [], (
        "Graph path must not call get_typed_candidates"
    )
    assert fake_hybrid.merge_calls == [], (
        "Graph path must not call merge()"
    )


def test_explicit_graph_returns_fact_search_response_shape():
    """retrieval_mode='graph' with no edges → FactSearchResponse empty shape."""
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    original_graphiti = server.graphiti_service
    original_search_service = server.search_service

    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False
        server.graphiti_service = _FakeGraphitiService()  # returns no edges
        server.search_service = _FakeSearchService()

        response = _run(
            server.search_memory_facts(
                query="test",
                retrieval_mode="graph",
                group_ids=["s1_sessions_main"],
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.graphiti_service = original_graphiti
        server.search_service = original_search_service

    # Empty graph → FactSearchResponse
    assert "facts" in response or "error" in response, (
        f"Unexpected graph response shape: {response!r}"
    )
    # Must NOT have hybrid-specific keys
    assert "merged_results" not in response
    assert "typed_candidates" not in response


def test_explicit_graph_with_null_graphiti_returns_initialized_error():
    """retrieval_mode='graph', graphiti_service=None → deterministic error."""
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    original_graphiti = server.graphiti_service
    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False
        server.graphiti_service = None

        response = _run(
            server.search_memory_facts(
                query="test",
                retrieval_mode="graph",
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.graphiti_service = original_graphiti

    assert response == {"error": "Graphiti service not initialized"}


# ─────────────────────────────────────────────────────────────────────────────
# §4  Explicit typed path is unchanged
# ─────────────────────────────────────────────────────────────────────────────


def test_explicit_typed_routes_to_typed_service():
    """retrieval_mode='typed' still routes to TypedRetrievalService (unchanged)."""
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    original_service_cls = server.TypedRetrievalService

    payload = {
        "message": "Typed memory retrieved successfully",
        "retrieval_mode": "typed",
        "result_format": "typed",
        "query_mode": "current",
        "state": [],
        "episodes": [],
        "procedures": [],
        "evidence": [],
        "counts": {"state": 0, "episodes": 0, "procedures": 0, "evidence": 0},
        "filters_applied": {},
        "limits_applied": {},
    }

    class _FakeTyped:
        async def search(self, **kw):
            return payload

    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False
        server.TypedRetrievalService = lambda **_kw: _FakeTyped()

        response = _run(
            server.search_memory_facts(
                query="current state",
                retrieval_mode="typed",
                group_ids=["s1_sessions_main"],
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.TypedRetrievalService = original_service_cls

    assert response == payload


# ─────────────────────────────────────────────────────────────────────────────
# §5  Empty-result behaviour
# ─────────────────────────────────────────────────────────────────────────────


def test_hybrid_both_empty_returns_deterministic_empty_response():
    """Both graph and typed empty → deterministic empty hybrid response (not error)."""
    empty_payload = _fake_typed_payload()  # no state, no procedures
    response, _ = _run_hybrid(
        fake_typed_payload=empty_payload,
        fake_merged_override=[],  # merge returns empty
    )

    assert isinstance(response, dict)
    assert response.get("retrieval_mode") == "hybrid"
    assert response.get("merged_results") == []
    assert response.get("result_count") == 0
    assert response.get("facts") == []
    typed_cands = response.get("typed_candidates", {})
    assert typed_cands.get("state", []) == []
    assert typed_cands.get("procedures", []) == []


def test_hybrid_both_empty_message_indicates_no_results():
    """Empty hybrid result has a 'No relevant memory found' message."""
    response, _ = _run_hybrid(
        fake_typed_payload=_fake_typed_payload(),
        fake_merged_override=[],
    )
    assert "No relevant" in response.get("message", ""), (
        f"Unexpected message: {response.get('message')!r}"
    )


def test_hybrid_graph_empty_typed_has_results():
    """Graph empty but typed has state facts → merged_results contains typed items."""
    state_item = {
        "object_id": "sf_01",
        "object_type": "state_fact",
        "subject": "user",
        "predicate": "prefers",
        "value": "dark roast",
        "is_current": True,
    }
    typed_payload = _fake_typed_payload(state=[state_item])
    response, _ = _run_hybrid(fake_typed_payload=typed_payload)

    # Graph was empty (FakeGraphitiService returns no edges); typed has items.
    assert response.get("result_count", 0) >= 1, (
        "Expected at least one result when typed has candidates"
    )
    assert any(
        r.get("_source") in ("typed_state", "typed_procedure")
        for r in response.get("merged_results", [])
    ), "merged_results should contain typed items when graph is empty"


def test_hybrid_empty_response_has_all_required_keys():
    """Empty hybrid response must include all required top-level keys."""
    response, _ = _run_hybrid(
        fake_typed_payload=_fake_typed_payload(),
        fake_merged_override=[],
    )
    required = {"message", "retrieval_mode", "facts", "typed_candidates", "merged_results", "result_count"}
    missing = required - set(response.keys())
    assert not missing, f"Missing keys in empty hybrid response: {missing}"


# ─────────────────────────────────────────────────────────────────────────────
# §6  Contract regressions: hybrid with graphiti_service=None still errors
# ─────────────────────────────────────────────────────────────────────────────


def test_hybrid_with_null_graphiti_returns_initialized_error():
    """Default hybrid + graphiti_service=None → 'Graphiti service not initialized'."""
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    original_graphiti = server.graphiti_service
    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False
        server.graphiti_service = None

        response = _run(
            server.search_memory_facts(
                query="test",
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.graphiti_service = original_graphiti

    assert response == {"error": "Graphiti service not initialized"}


def test_explicit_hybrid_with_null_graphiti_returns_initialized_error():
    """Explicit hybrid + graphiti_service=None → same error (fail-closed)."""
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    original_graphiti = server.graphiti_service
    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False
        server.graphiti_service = None

        response = _run(
            server.search_memory_facts(
                query="test",
                retrieval_mode="hybrid",
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.graphiti_service = original_graphiti

    assert response == {"error": "Graphiti service not initialized"}


def test_valid_retrieval_modes_still_contains_hybrid():
    """VALID_RETRIEVAL_MODES must still contain 'hybrid'."""
    assert "hybrid" in server.VALID_RETRIEVAL_MODES


def test_invalid_retrieval_mode_still_returns_error():
    """Invalid retrieval_mode still returns a deterministic error (unchanged contract)."""
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    try:
        server._SEARCH_RATE_LIMIT_ENABLED = False
        response = _run(
            server.search_memory_facts(
                query="test",
                retrieval_mode="totally_invalid",
                ctx=None,
            )
        )
    finally:
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit

    assert "error" in response
    assert "retrieval_mode must be one of" in response["error"]


# ─────────────────────────────────────────────────────────────────────────────
# §7  HybridRetrievalService unit tests (typed_retrieval_service.py)
# ─────────────────────────────────────────────────────────────────────────────


def test_rrf_merge_hybrid_empty_inputs():
    """All empty sources → empty merged list."""
    result = rrf_merge_hybrid(
        graph_facts=[],
        typed_state=[],
        typed_procedures=[],
        max_facts=10,
    )
    assert result == []


def test_rrf_merge_hybrid_graph_only():
    """Graph facts only → all items present with _source='graph'."""
    facts = [{"uuid": "f1", "fact": "user likes coffee"}, {"uuid": "f2", "fact": "user is admin"}]
    result = rrf_merge_hybrid(
        graph_facts=facts,
        typed_state=[],
        typed_procedures=[],
        max_facts=10,
    )
    assert len(result) == 2
    assert all(r.get("_source") == "graph" for r in result)


def test_rrf_merge_hybrid_typed_only():
    """Typed candidates only → items present with typed_state source."""
    state = [{"object_id": "sf_01", "subject": "u", "predicate": "p", "value": "v"}]
    result = rrf_merge_hybrid(
        graph_facts=[],
        typed_state=state,
        typed_procedures=[],
        max_facts=10,
    )
    assert len(result) == 1
    assert result[0].get("_source") == "typed_state"


def test_rrf_merge_hybrid_max_facts_respected():
    """max_facts caps the merged list length."""
    facts = [{"uuid": f"f{i}", "fact": f"fact {i}"} for i in range(10)]
    state = [{"object_id": f"sf{i}"} for i in range(10)]
    result = rrf_merge_hybrid(
        graph_facts=facts,
        typed_state=state,
        typed_procedures=[],
        max_facts=5,
    )
    assert len(result) == 5


def test_rrf_merge_hybrid_scores_are_annotated():
    """Every item in the merged result has a _hybrid_score float."""
    facts = [{"uuid": "f1", "fact": "x"}]
    state = [{"object_id": "sf1"}]
    result = rrf_merge_hybrid(
        graph_facts=facts,
        typed_state=state,
        typed_procedures=[],
        max_facts=10,
    )
    for item in result:
        assert "_hybrid_score" in item, f"Missing _hybrid_score on {item!r}"
        assert isinstance(item["_hybrid_score"], float)


def test_rrf_merge_hybrid_graph_ranked_higher_than_typed():
    """Graph item at rank-1 should outscore typed item at rank-1 (graph weight > typed weight)."""
    facts = [{"uuid": "gf1", "fact": "top graph fact"}]
    state = [{"object_id": "sf1", "subject": "a", "predicate": "b", "value": "c"}]
    result = rrf_merge_hybrid(
        graph_facts=facts,
        typed_state=state,
        typed_procedures=[],
        max_facts=10,
    )
    assert len(result) == 2
    # Graph item should be first (higher weight).
    assert result[0].get("_source") == "graph", (
        f"Expected graph item first, got _source={result[0].get('_source')!r}"
    )


def test_rrf_merge_hybrid_max_facts_zero():
    """max_facts=0 → empty list (no division by zero)."""
    result = rrf_merge_hybrid(
        graph_facts=[{"uuid": "f1"}],
        typed_state=[{"object_id": "sf1"}],
        typed_procedures=[],
        max_facts=0,
    )
    assert result == []


def test_rrf_merge_hybrid_procedure_source_label():
    """Procedure items are labeled 'typed_procedure'."""
    procs = [{"object_id": "pr1", "name": "reset password", "trigger": "user forgets"}]
    result = rrf_merge_hybrid(
        graph_facts=[],
        typed_state=[],
        typed_procedures=procs,
        max_facts=10,
    )
    assert len(result) == 1
    assert result[0].get("_source") == "typed_procedure"


def test_hybrid_retrieval_service_merge_delegates_to_rrf():
    """HybridRetrievalService.merge() produces a list from rrf_merge_hybrid."""
    svc = HybridRetrievalService.__new__(HybridRetrievalService)
    # Manually set _typed_service to a sentinel (merge doesn't use it).
    svc.om_projection_service = None
    svc._typed_service = None

    graph_facts = [{"uuid": "g1", "fact": "graph fact"}]
    typed_results = {
        "state": [{"object_id": "sf1", "subject": "u", "predicate": "p", "value": "v"}],
        "procedures": [],
    }
    merged = svc.merge(graph_facts=graph_facts, typed_results=typed_results, max_facts=5)

    assert isinstance(merged, list)
    assert len(merged) == 2
    sources = {item.get("_source") for item in merged}
    assert "graph" in sources
    assert "typed_state" in sources


# ─────────────────────────────────────────────────────────────────────────────
# §8  Hybrid query propagation (typed candidates receive correct args)
# ─────────────────────────────────────────────────────────────────────────────


def test_hybrid_typed_candidates_receive_query_and_group_ids():
    """get_typed_candidates must be called with the correct query and group scope."""
    typed_payload = _fake_typed_payload(state=[{"object_id": "sf_01"}])
    _response, fake_hybrid = _run_hybrid(
        query="my specific query",
        group_ids=["s1_sessions_main"],
        fake_typed_payload=typed_payload,
    )

    assert len(fake_hybrid.get_typed_calls) == 1
    call = fake_hybrid.get_typed_calls[0]
    assert call.get("query") == "my specific query"
    assert call.get("effective_group_ids") == ["s1_sessions_main"]


def test_hybrid_max_facts_forwarded_to_typed_candidates():
    """max_facts is forwarded to get_typed_candidates as max_candidates."""
    typed_payload = _fake_typed_payload()
    _response, fake_hybrid = _run_hybrid(
        max_facts=7,
        fake_typed_payload=typed_payload,
        fake_merged_override=[],
    )

    assert len(fake_hybrid.get_typed_calls) == 1
    call = fake_hybrid.get_typed_calls[0]
    assert call.get("max_candidates") == 7


# ─────────────────────────────────────────────────────────────────────────────
# §9  Hybrid retrieval_mode key is present on ALL hybrid responses
# ─────────────────────────────────────────────────────────────────────────────


def test_hybrid_response_always_has_retrieval_mode_key():
    """Every non-error hybrid response must carry retrieval_mode='hybrid'."""
    for state_count in (0, 1, 3):
        state = [{"object_id": f"sf{i}"} for i in range(state_count)]
        typed_payload = _fake_typed_payload(state=state)
        merged = [{"_source": "typed_state", **s} for s in state]
        response, _ = _run_hybrid(
            fake_typed_payload=typed_payload,
            fake_merged_override=merged if merged else [],
        )
        assert response.get("retrieval_mode") == "hybrid", (
            f"state_count={state_count}: expected retrieval_mode='hybrid', got {response!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# §10  P0: Lane-scope intersection (metadata_filters must be intersected)
# ─────────────────────────────────────────────────────────────────────────────


def test_hybrid_metadata_filters_intersected_with_group_ids():
    """P0: metadata_filters passed to get_typed_candidates must already be
    intersected with the effective_group_ids lane scope, not passed raw."""
    typed_payload = _fake_typed_payload(state=[{"object_id": "sf_01"}])
    _response, fake_hybrid = _run_hybrid(
        query="test",
        group_ids=["s1_sessions_main"],
        fake_typed_payload=typed_payload,
        metadata_filters=None,
    )
    assert len(fake_hybrid.get_typed_calls) == 1
    call = fake_hybrid.get_typed_calls[0]
    # After intersection, metadata_filters must carry the lane scope in source_lane.
    forwarded = call.get("metadata_filters") or {}
    assert "source_lane" in forwarded, (
        f"P0: metadata_filters should contain source_lane after intersection; got {forwarded!r}"
    )
    source_lane = forwarded["source_lane"]
    # source_lane must reference the effective group_ids
    lane_values = source_lane.get("in", []) if isinstance(source_lane, dict) else []
    assert "s1_sessions_main" in lane_values, (
        f"P0: effective group_id 's1_sessions_main' missing from intersected source_lane; "
        f"got {source_lane!r}"
    )


def test_hybrid_invalid_source_lane_filter_returns_error():
    """P0: invalid metadata_filters.source_lane shape returns an ErrorResponse
    (not an unhandled exception), matching typed-only path behaviour."""
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    original_graphiti = server.graphiti_service
    original_hybrid_cls = server.HybridRetrievalService
    original_search_service = server.search_service

    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False
        server.graphiti_service = _FakeGraphitiService()
        server.HybridRetrievalService = lambda **_kw: _FakeHybridRetrievalService()
        server.search_service = _FakeSearchService()

        response = _run(
            server.search_memory_facts(
                query="test",
                group_ids=["s1_sessions_main"],
                metadata_filters={"source_lane": {"bad_key": "x"}},  # invalid shape
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.graphiti_service = original_graphiti
        server.HybridRetrievalService = original_hybrid_cls
        server.search_service = original_search_service

    assert "error" in response, f"Expected error for invalid source_lane shape; got {response!r}"


def test_hybrid_caller_source_lane_is_intersected_not_replaced():
    """P0: caller-supplied source_lane is intersected with (not replaced by) group scope."""
    typed_payload = _fake_typed_payload(state=[{"object_id": "sf_01"}])

    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    original_graphiti = server.graphiti_service
    original_search_service = server.search_service
    original_hybrid_cls = server.HybridRetrievalService

    fake_hybrid = _FakeHybridRetrievalService(typed_payload=typed_payload)
    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False
        server.graphiti_service = _FakeGraphitiService()
        server.search_service = _FakeSearchService()
        server.HybridRetrievalService = lambda **_kw: fake_hybrid

        _response = _run(
            server.search_memory_facts(
                query="test",
                group_ids=["s1_sessions_main"],
                # Caller requests a lane that is IN scope → should survive intersection.
                metadata_filters={"source_lane": "s1_sessions_main"},
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.graphiti_service = original_graphiti
        server.search_service = original_search_service
        server.HybridRetrievalService = original_hybrid_cls

    assert fake_hybrid.get_typed_calls, "Expected get_typed_candidates to be called"
    call = fake_hybrid.get_typed_calls[0]
    forwarded = call.get("metadata_filters") or {}
    source_lane = forwarded.get("source_lane")
    # Intersection of ['s1_sessions_main'] ∩ ['s1_sessions_main'] = ['s1_sessions_main']
    lane_values = source_lane.get("in", []) if isinstance(source_lane, dict) else []
    assert "s1_sessions_main" in lane_values, (
        f"P0: matching lane should survive intersection; got source_lane={source_lane!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# §11  P1: history_mode and current_only forwarding
# ─────────────────────────────────────────────────────────────────────────────


def test_hybrid_history_mode_forwarded_to_typed_candidates():
    """P1: history_mode is forwarded to get_typed_candidates (not hardcoded 'auto')."""
    typed_payload = _fake_typed_payload(state=[{"object_id": "sf_01"}])
    _response, fake_hybrid = _run_hybrid(
        fake_typed_payload=typed_payload,
        history_mode="current",
    )
    assert len(fake_hybrid.get_typed_calls) == 1
    call = fake_hybrid.get_typed_calls[0]
    assert call.get("history_mode") == "current", (
        f"P1: expected history_mode='current' forwarded to typed candidates; got {call!r}"
    )


def test_hybrid_current_only_forwarded_to_typed_candidates():
    """P1: current_only=True is forwarded to get_typed_candidates."""
    typed_payload = _fake_typed_payload(state=[{"object_id": "sf_01"}])
    _response, fake_hybrid = _run_hybrid(
        fake_typed_payload=typed_payload,
        current_only=True,
    )
    assert len(fake_hybrid.get_typed_calls) == 1
    call = fake_hybrid.get_typed_calls[0]
    assert call.get("current_only") is True, (
        f"P1: expected current_only=True forwarded to typed candidates; got {call!r}"
    )


def test_hybrid_history_mode_all_forwarded():
    """P1: history_mode='all' (non-default) is forwarded correctly."""
    typed_payload = _fake_typed_payload(state=[{"object_id": "sf_01"}])
    _response, fake_hybrid = _run_hybrid(
        fake_typed_payload=typed_payload,
        history_mode="all",
    )
    call = fake_hybrid.get_typed_calls[0]
    assert call.get("history_mode") == "all", (
        f"P1: expected history_mode='all'; got {call!r}"
    )


def test_hybrid_current_only_false_forwarded():
    """P1: current_only=False (explicit falsy) is forwarded and not swallowed."""
    typed_payload = _fake_typed_payload(state=[{"object_id": "sf_01"}])
    _response, fake_hybrid = _run_hybrid(
        fake_typed_payload=typed_payload,
        current_only=False,
    )
    call = fake_hybrid.get_typed_calls[0]
    assert call.get("current_only") is False, (
        f"P1: expected current_only=False; got {call!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# §12  P1: candidate_rows parity — hybrid response includes candidate_rows
#          when OM facts produce them (non-regression)
# ─────────────────────────────────────────────────────────────────────────────


def test_hybrid_response_no_candidate_rows_when_om_absent():
    """When OM lane is not in scope, candidate_rows is absent from the hybrid response."""
    typed_payload = _fake_typed_payload(state=[{"object_id": "sf_01"}])
    response, _ = _run_hybrid(fake_typed_payload=typed_payload)
    # _FakeSearchService returns includes_observational_memory=False → no OM facts
    # → candidate_rows must be absent (no spurious empty list).
    assert "candidate_rows" not in response, (
        f"P1: candidate_rows should be absent when OM is not in scope; got {response!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# §13  Hybrid fallback observability — diagnostics when typed retrieval fails
#
# Goal: when get_typed_candidates() raises, the response must include a
# 'diagnostics' key with typed_retrieval_failed=True, fallback='graph_only',
# and the error string.  The path must degrade gracefully (not crash) and still
# return a well-formed hybrid response (or empty hybrid response if graph is
# also empty).
# ─────────────────────────────────────────────────────────────────────────────


class _FakeHybridRetrievalServiceWithFailure(_FakeHybridRetrievalService):
    """Like _FakeHybridRetrievalService but get_typed_candidates raises."""

    def __init__(self, error_msg: str = "simulated typed retrieval failure", **kwargs):
        super().__init__(**kwargs)
        self._error_msg = error_msg

    async def get_typed_candidates(self, **kwargs: Any) -> dict[str, Any]:
        self.get_typed_calls.append(kwargs)
        raise RuntimeError(self._error_msg)


def _run_hybrid_with_failure(
    error_msg: str = "simulated typed retrieval failure",
    fake_graph_edges=None,
    **kwargs: Any,
) -> tuple[dict, Any]:
    """Run search_memory_facts with a typed-retrieval failure injected."""
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    original_graphiti = server.graphiti_service
    original_hybrid_cls = server.HybridRetrievalService
    original_search_service = server.search_service

    fake_hybrid = _FakeHybridRetrievalServiceWithFailure(error_msg=error_msg)
    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False
        server.graphiti_service = _FakeGraphitiService(edges=fake_graph_edges or [])
        server.HybridRetrievalService = lambda **_kw: fake_hybrid
        server.search_service = _FakeSearchService()

        result = _run(
            server.search_memory_facts(
                query="test query",
                group_ids=["s1_sessions_main"],
                ctx=None,
                **kwargs,
            )
        )
        return result, fake_hybrid
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.graphiti_service = original_graphiti
        server.HybridRetrievalService = original_hybrid_cls
        server.search_service = original_search_service


def test_hybrid_fallback_diagnostics_present_when_typed_retrieval_fails():
    """When typed retrieval raises, 'diagnostics' key must appear in the response."""
    response, _ = _run_hybrid_with_failure()
    assert "diagnostics" in response, (
        f"Expected 'diagnostics' key in response when typed retrieval fails; got {response!r}"
    )


def test_hybrid_fallback_diagnostics_typed_retrieval_failed_flag():
    """diagnostics.typed_retrieval_failed must be True when typed retrieval raises."""
    response, _ = _run_hybrid_with_failure()
    diag = response.get("diagnostics", {})
    assert diag.get("typed_retrieval_failed") is True, (
        f"Expected diagnostics.typed_retrieval_failed=True; got {diag!r}"
    )


def test_hybrid_fallback_diagnostics_fallback_value():
    """diagnostics.fallback must be 'graph_only' when typed retrieval degrades."""
    response, _ = _run_hybrid_with_failure()
    diag = response.get("diagnostics", {})
    assert diag.get("fallback") == "graph_only", (
        f"Expected diagnostics.fallback='graph_only'; got {diag!r}"
    )


def test_hybrid_fallback_diagnostics_error_string_captured():
    """diagnostics.error must contain the exception message string."""
    err_msg = "neo4j connection refused"
    response, _ = _run_hybrid_with_failure(error_msg=err_msg)
    diag = response.get("diagnostics", {})
    assert err_msg in diag.get("error", ""), (
        f"Expected error string '{err_msg}' in diagnostics.error; got {diag!r}"
    )


def test_hybrid_fallback_still_returns_retrieval_mode_hybrid():
    """Degraded hybrid path must still return retrieval_mode='hybrid' (not 'graph')."""
    response, _ = _run_hybrid_with_failure()
    assert response.get("retrieval_mode") == "hybrid", (
        f"Expected retrieval_mode='hybrid' even on fallback; got {response!r}"
    )


def test_hybrid_fallback_returns_well_formed_response():
    """Degraded hybrid response must still have all required top-level keys."""
    response, _ = _run_hybrid_with_failure()
    required = {"retrieval_mode", "facts", "typed_candidates", "merged_results", "result_count"}
    missing = required - set(response.keys())
    assert not missing, (
        f"Degraded hybrid response missing required keys: {missing}; got {list(response.keys())!r}"
    )


def test_hybrid_fallback_typed_candidates_empty_on_failure():
    """When typed retrieval fails, typed_candidates must reflect empty state/procedures."""
    response, _ = _run_hybrid_with_failure()
    typed_cands = response.get("typed_candidates", {})
    assert typed_cands.get("state", None) == [], (
        f"Expected typed_candidates.state=[] on failure; got {typed_cands!r}"
    )
    assert typed_cands.get("procedures", None) == [], (
        f"Expected typed_candidates.procedures=[] on failure; got {typed_cands!r}"
    )


def test_hybrid_clean_run_has_no_diagnostics_key():
    """On a successful (non-degraded) hybrid run, 'diagnostics' must NOT be present."""
    typed_payload = _fake_typed_payload(
        state=[{"object_id": "sf_01", "subject": "user", "predicate": "likes", "value": "tea"}]
    )
    response, _ = _run_hybrid(fake_typed_payload=typed_payload)
    assert "diagnostics" not in response, (
        f"'diagnostics' should be absent on clean hybrid run; got keys: {list(response.keys())!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# §14  RRF k=60 calibration — regression guard
# ─────────────────────────────────────────────────────────────────────────────


def test_rrf_k_constant_is_60():
    """_HYBRID_RRF_K must be 60.0 (experiment-calibrated; do not revert to 1.0)."""
    svc_path = str(pathlib.Path(__file__).parent.parent / "mcp_server" / "src" / "services")
    if svc_path not in sys.path:
        sys.path.insert(0, svc_path)

    import mcp_server.src.services.typed_retrieval_service as trs_mod
    assert trs_mod._HYBRID_RRF_K == 60.0, (
        f"_HYBRID_RRF_K must be 60.0 (experiment-calibrated); got {trs_mod._HYBRID_RRF_K}"
    )


def test_rrf_k60_softens_rank_gap_vs_k1():
    """k=60 produces a smaller rank-score gap than k=1 between rank-1 and rank-2 items.

    This verifies that the calibration change (k=1 → k=60) has the intended
    behaviour: softer rank differentiation, more forgiving for mid-rank candidates.
    """
    facts_k60 = rrf_merge_hybrid(
        graph_facts=[
            {"uuid": "f1", "fact": "top"},
            {"uuid": "f2", "fact": "second"},
        ],
        typed_state=[],
        typed_procedures=[],
        max_facts=10,
        rrf_k=60.0,
    )
    facts_k1 = rrf_merge_hybrid(
        graph_facts=[
            {"uuid": "f1", "fact": "top"},
            {"uuid": "f2", "fact": "second"},
        ],
        typed_state=[],
        typed_procedures=[],
        max_facts=10,
        rrf_k=1.0,
    )
    score_gap_k60 = facts_k60[0]["_hybrid_score"] - facts_k60[1]["_hybrid_score"]
    score_gap_k1 = facts_k1[0]["_hybrid_score"] - facts_k1[1]["_hybrid_score"]
    assert score_gap_k60 < score_gap_k1, (
        f"k=60 should produce a smaller rank gap ({score_gap_k60:.6f}) "
        f"than k=1 ({score_gap_k1:.6f})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# §15  P3 regression guards (surgical follow-up pass)
# ─────────────────────────────────────────────────────────────────────────────


# ── P3-2: max_evidence forwarding ────────────────────────────────────────────


def test_hybrid_max_evidence_forwarded_to_typed_candidates():
    """max_evidence supplied by the caller must be forwarded to get_typed_candidates.

    Previously hardcoded to 20 inside get_typed_candidates regardless of the
    caller-supplied value.
    """
    typed_payload = _fake_typed_payload(
        state=[{"object_id": "sf_01", "subject": "x", "predicate": "y", "value": "z"}]
    )
    response, fake_hybrid = _run_hybrid(fake_typed_payload=typed_payload, max_evidence=7)
    assert len(fake_hybrid.get_typed_calls) == 1
    call_kwargs = fake_hybrid.get_typed_calls[0]
    assert "max_evidence" in call_kwargs, (
        "get_typed_candidates must receive max_evidence keyword argument"
    )
    assert call_kwargs["max_evidence"] == 7, (
        f"Expected max_evidence=7 forwarded to typed candidates, got {call_kwargs.get('max_evidence')!r}"
    )


def test_hybrid_max_evidence_default_20_forwarded():
    """Default max_evidence=20 must still reach get_typed_candidates unchanged."""
    typed_payload = _fake_typed_payload()
    response, fake_hybrid = _run_hybrid(fake_typed_payload=typed_payload)
    assert len(fake_hybrid.get_typed_calls) == 1
    call_kwargs = fake_hybrid.get_typed_calls[0]
    # Default is 20; it must be forwarded (not swallowed).
    assert "max_evidence" in call_kwargs, (
        "max_evidence must always be forwarded even at its default value"
    )
    assert call_kwargs["max_evidence"] == 20, (
        f"Default max_evidence should be 20, got {call_kwargs.get('max_evidence')!r}"
    )


def test_get_typed_candidates_accepts_max_evidence_param():
    """HybridRetrievalService.get_typed_candidates signature includes max_evidence."""
    import inspect
    sig = inspect.signature(HybridRetrievalService.get_typed_candidates)
    assert "max_evidence" in sig.parameters, (
        "get_typed_candidates must declare a max_evidence parameter"
    )
    # Default should be 20 (backward-compatible for direct callers).
    assert sig.parameters["max_evidence"].default == 20, (
        "max_evidence default must be 20 for backward compatibility"
    )


# ── P3-3: dedup collision safety on missing object_id ────────────────────────


def test_typed_item_id_unique_for_items_without_object_id():
    """Two distinct ID-less typed objects must produce different dedup keys.

    Previously both mapped to 'typed:<bucket>:unknown', causing the second
    to silently overwrite the first in the RRF registry.
    """
    from mcp_server.src.services.typed_retrieval_service import _typed_item_id

    item_a = {"subject": "user", "predicate": "likes", "value": "coffee"}
    item_b = {"subject": "user", "predicate": "dislikes", "value": "tea"}

    key_a = _typed_item_id(item_a, bucket="state")
    key_b = _typed_item_id(item_b, bucket="state")

    assert key_a != key_b, (
        f"Distinct ID-less items must have distinct dedup keys; "
        f"both mapped to {key_a!r}"
    )
    # Both must still use the expected prefix convention.
    assert key_a.startswith("typed:state:"), f"Unexpected prefix: {key_a!r}"
    assert key_b.startswith("typed:state:"), f"Unexpected prefix: {key_b!r}"


def test_typed_item_id_identical_items_produce_same_key():
    """Two identical ID-less objects must produce the same key (deterministic hash)."""
    from mcp_server.src.services.typed_retrieval_service import _typed_item_id

    item = {"subject": "user", "predicate": "likes", "value": "coffee"}
    key_1 = _typed_item_id(item, bucket="state")
    key_2 = _typed_item_id(item, bucket="state")
    assert key_1 == key_2, "Same item must hash to the same key on repeated calls"


def test_typed_item_id_with_object_id_still_uses_id():
    """When object_id is present the fast-path (no hashing) is used."""
    from mcp_server.src.services.typed_retrieval_service import _typed_item_id

    item = {"object_id": "obj-123", "subject": "user", "predicate": "likes", "value": "coffee"}
    key = _typed_item_id(item, bucket="state")
    assert key == "typed:state:obj-123", f"Unexpected key with object_id: {key!r}"


def test_rrf_merge_hybrid_preserves_distinct_id_less_items():
    """RRF merge must not collapse distinct ID-less typed items into a single entry."""
    item_a = {"subject": "user", "predicate": "likes", "value": "coffee"}
    item_b = {"subject": "user", "predicate": "dislikes", "value": "tea"}

    merged = rrf_merge_hybrid(
        graph_facts=[],
        typed_state=[item_a, item_b],
        typed_procedures=[],
        max_facts=10,
    )
    assert len(merged) == 2, (
        f"Both distinct ID-less items must appear in merged output; got {len(merged)}: {merged!r}"
    )


# ── P3-1: object_types in hybrid mode emits a warning ────────────────────────


def test_hybrid_object_types_emits_warning(caplog):
    """object_types supplied to hybrid mode must trigger a logger.warning."""
    import logging

    typed_payload = _fake_typed_payload(
        state=[{"object_id": "sf_01", "subject": "x", "predicate": "y", "value": "z"}]
    )
    with caplog.at_level(logging.WARNING):
        response, _ = _run_hybrid(
            fake_typed_payload=typed_payload,
            object_types=["state"],
        )

    warning_texts = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("object_types" in txt for txt in warning_texts), (
        f"Expected a warning mentioning 'object_types' when hybrid mode receives "
        f"object_types=; got warnings: {warning_texts!r}"
    )


def test_hybrid_no_object_types_no_warning(caplog):
    """No object_types supplied → no warning about object_types filtering."""
    import logging

    typed_payload = _fake_typed_payload(
        state=[{"object_id": "sf_01"}]
    )
    with caplog.at_level(logging.WARNING):
        response, _ = _run_hybrid(fake_typed_payload=typed_payload)

    object_type_warnings = [
        r.message for r in caplog.records
        if r.levelno == logging.WARNING and "object_types" in r.message
    ]
    assert not object_type_warnings, (
        f"No object_types warning expected when none supplied; got: {object_type_warnings!r}"
    )
