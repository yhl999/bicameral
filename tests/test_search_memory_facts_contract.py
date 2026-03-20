"""Contract tests for search_memory_facts retrieval_mode surface selector (v1).

Covers:
- FR-1: retrieval_mode is the top-level surface selector
- FR-2: valid values are 'graph', 'typed', 'hybrid'
- FR-3: default is 'hybrid' (not graph-only)
- FR-4: search_mode is graph-internal only; typed path rejects non-hybrid values
- FR-5: center_node_uuid is graph-only; typed path rejects it
- FR-6: result_format='typed' is a deprecated backward-compat alias
- FR-7: invalid retrieval_mode yields a deterministic ErrorResponse

Routing discrimination strategy:
- Graph vs hybrid: provide a working fake graphiti_service so the code passes
  the null-check and diverges at the graph/hybrid split.  Graph responses have
  FactSearchResponse shape (message + facts, no merged_results).  Hybrid
  responses carry retrieval_mode='hybrid' + merged_results + typed_candidates.
- graphiti_service=None tests are reserved for null-guard contract proofs where
  both paths legitimately produce the same error.
"""

import asyncio
from types import SimpleNamespace
from typing import Any

from tests.helpers_mcp_import import load_graphiti_mcp_server

server = load_graphiti_mcp_server()


def _run(coro):
    return asyncio.run(coro)


def _test_config():
    """Minimal config with no authorized_group_ids restriction (open scope)."""
    return SimpleNamespace(
        database=SimpleNamespace(provider='neo4j'),
        graphiti=SimpleNamespace(
            group_id='s1_sessions_main',
            lane_aliases={
                'sessions_main': ['s1_sessions_main'],
                'observational_memory': ['s1_observational_memory'],
            },
            authorized_group_ids=[],
        ),
    )


def _fake_typed_payload():
    return {
        'message': 'Typed memory retrieved successfully',
        'query_mode': 'current',
        'state': [],
        'episodes': [],
        'procedures': [],
        'evidence': [],
        'counts': {'state': 0, 'episodes': 0, 'procedures': 0, 'evidence': 0},
    }


class _FakeTypedRetrievalService:
    def __init__(self, payload=None):
        self.payload = payload or _fake_typed_payload()
        self.calls = []

    async def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


# ── Minimal fakes for hybrid-path discrimination ──────────────────────────────
# These allow the code to pass the graphiti_service null-check and reach the
# graph/hybrid routing split, so contract tests can distinguish which path ran.


class _FakeSearchResults:
    def __init__(self, edges=None):
        self.edges = edges or []
        self.nodes = []


class _FakeGraphitiClient:
    async def search_(self, **kwargs: Any) -> _FakeSearchResults:
        return _FakeSearchResults()


class _FakeGraphitiService:
    async def get_client_for_group(self, group_id: str) -> _FakeGraphitiClient:
        return _FakeGraphitiClient()


class _FakeSearchService:
    def includes_observational_memory(self, group_ids: list[str]) -> bool:
        return False


class _FakeHybridRetrievalService:
    """Returns empty typed candidates and passes merge through."""

    def __init__(self, **_kwargs: Any):
        self.get_typed_calls: list[dict] = []

    async def get_typed_candidates(self, **kwargs: Any) -> dict[str, Any]:
        self.get_typed_calls.append(kwargs)
        return {
            'state': [],
            'procedures': [],
            'counts': {'state': 0, 'procedures': 0},
        }

    def merge(
        self,
        *,
        graph_facts: list[dict],
        typed_results: dict,
        max_facts: int,
    ) -> list[dict[str, Any]]:
        return graph_facts[:max_facts]


# ---------------------------------------------------------------------------
# FR-3: Default surface is 'hybrid'
# ---------------------------------------------------------------------------


def test_default_retrieval_mode_routes_to_hybrid_path():
    """retrieval_mode not supplied → default 'hybrid' → hybrid response shape.

    Discriminating: graph-only would return FactSearchResponse (message + facts,
    no merged_results/retrieval_mode key).  Hybrid returns retrieval_mode='hybrid'
    + merged_results + typed_candidates.
    """
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
                query='test query',
                group_ids=['s1_sessions_main'],
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.graphiti_service = original_graphiti
        server.HybridRetrievalService = original_hybrid_cls
        server.search_service = original_search_service

    # Must be a hybrid response, not a graph FactSearchResponse.
    assert isinstance(response, dict)
    assert response.get('retrieval_mode') == 'hybrid', (
        f"Default should route to hybrid, got: {response!r}"
    )
    assert 'merged_results' in response, (
        f"Hybrid response must have merged_results; got keys: {list(response.keys())!r}"
    )


# ---------------------------------------------------------------------------
# FR-2 + FR-3: Explicit retrieval_mode values
# ---------------------------------------------------------------------------


def test_retrieval_mode_graph_routes_to_graph_path():
    """retrieval_mode='graph' explicitly selects the graph-edge path."""
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    original_graphiti = server.graphiti_service
    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False
        server.graphiti_service = None

        response = _run(
            server.search_memory_facts(
                query='test',
                retrieval_mode='graph',
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.graphiti_service = original_graphiti

    assert response == {'error': 'Graphiti service not initialized'}


def test_retrieval_mode_hybrid_explicit_routes_to_hybrid_path():
    """retrieval_mode='hybrid' explicit → hybrid response shape (not graph-only).

    Discriminating: asserts retrieval_mode='hybrid' + merged_results, which
    the graph-only FactSearchResponse path does not produce.
    """
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
                query='test',
                retrieval_mode='hybrid',
                group_ids=['s1_sessions_main'],
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.graphiti_service = original_graphiti
        server.HybridRetrievalService = original_hybrid_cls
        server.search_service = original_search_service

    assert isinstance(response, dict)
    assert response.get('retrieval_mode') == 'hybrid', (
        f"Explicit hybrid should produce hybrid response, got: {response!r}"
    )
    assert 'merged_results' in response


def test_retrieval_mode_typed_routes_to_typed_path():
    """retrieval_mode='typed' selects the typed-ledger path."""
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    original_service_cls = server.TypedRetrievalService

    payload = _fake_typed_payload()
    fake_service = _FakeTypedRetrievalService(payload)

    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False
        server.TypedRetrievalService = lambda **kwargs: fake_service

        response = _run(
            server.search_memory_facts(
                query='current state',
                retrieval_mode='typed',
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.TypedRetrievalService = original_service_cls

    assert response == payload
    assert len(fake_service.calls) == 1


# ---------------------------------------------------------------------------
# FR-7: Invalid retrieval_mode → deterministic error
# ---------------------------------------------------------------------------


def test_invalid_retrieval_mode_returns_deterministic_error():
    """retrieval_mode with an unrecognized value returns an explicit ErrorResponse."""
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    try:
        server._SEARCH_RATE_LIMIT_ENABLED = False
        response = _run(
            server.search_memory_facts(
                query='test',
                retrieval_mode='unknown_surface',
                ctx=None,
            )
        )
    finally:
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit

    assert 'error' in response
    assert "retrieval_mode must be one of:" in response['error']
    # All three valid values must appear in the error message.
    assert 'graph' in response['error']
    assert 'typed' in response['error']
    assert 'hybrid' in response['error']


def test_invalid_retrieval_mode_is_stable_across_calls():
    """FR-7: Same invalid value always produces identical error text."""
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    try:
        server._SEARCH_RATE_LIMIT_ENABLED = False
        r1 = _run(server.search_memory_facts(query='q', retrieval_mode='bad', ctx=None))
        r2 = _run(server.search_memory_facts(query='q', retrieval_mode='bad', ctx=None))
    finally:
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit

    assert r1 == r2


# ---------------------------------------------------------------------------
# FR-4: search_mode is graph-internal only
# ---------------------------------------------------------------------------


def test_search_mode_rejected_for_retrieval_mode_typed():
    """search_mode must not be used as a surface selector for retrieval_mode='typed'."""
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False

        response = _run(
            server.search_memory_facts(
                query='test',
                retrieval_mode='typed',
                search_mode='semantic',
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit

    assert 'error' in response
    assert 'search_mode' in response['error']


def test_search_mode_keyword_rejected_for_retrieval_mode_typed():
    """search_mode='keyword' is also rejected for typed surface."""
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False

        response = _run(
            server.search_memory_facts(
                query='test',
                retrieval_mode='typed',
                search_mode='keyword',
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit

    assert 'error' in response
    assert 'search_mode' in response['error']


def test_search_mode_accepted_for_retrieval_mode_graph():
    """search_mode is graph-internal; it must be accepted for retrieval_mode='graph'."""
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    original_graphiti = server.graphiti_service
    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False
        server.graphiti_service = None

        response = _run(
            server.search_memory_facts(
                query='test',
                retrieval_mode='graph',
                search_mode='semantic',
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.graphiti_service = original_graphiti

    # Routing reached graph path (not rejected at search_mode validation).
    # graphiti_service = None → deterministic graph-path error.
    assert response == {'error': 'Graphiti service not initialized'}


# ---------------------------------------------------------------------------
# FR-5: center_node_uuid is graph-only
# ---------------------------------------------------------------------------


def test_center_node_uuid_rejected_for_retrieval_mode_typed():
    """center_node_uuid is not meaningful for the typed surface."""
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False

        response = _run(
            server.search_memory_facts(
                query='test',
                retrieval_mode='typed',
                center_node_uuid='some-node-uuid',
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit

    assert 'error' in response
    assert 'center_node_uuid' in response['error']


# ---------------------------------------------------------------------------
# FR-6: Backward-compat alias for legacy callers
# ---------------------------------------------------------------------------


def test_legacy_result_format_typed_aliases_to_typed_surface():
    """result_format='typed' without retrieval_mode routes to typed path (compat alias)."""
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    original_service_cls = server.TypedRetrievalService

    payload = _fake_typed_payload()
    fake_service = _FakeTypedRetrievalService(payload)

    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False
        server.TypedRetrievalService = lambda **kwargs: fake_service

        response = _run(
            server.search_memory_facts(
                query='current state',
                result_format='typed',
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.TypedRetrievalService = original_service_cls

    assert response == payload
    assert len(fake_service.calls) == 1


def test_legacy_result_format_facts_aliases_to_hybrid_path():
    """result_format='facts' (default) without retrieval_mode routes to hybrid path.

    Contract: result_format='facts' resolves to retrieval_mode='hybrid' (the
    default surface).  Discriminating: asserts hybrid response shape, which the
    graph-only FactSearchResponse path cannot produce.
    """
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
                query='test',
                result_format='facts',
                group_ids=['s1_sessions_main'],
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.graphiti_service = original_graphiti
        server.HybridRetrievalService = original_hybrid_cls
        server.search_service = original_search_service

    # result_format='facts' now routes to hybrid (the default surface).
    assert isinstance(response, dict)
    assert response.get('retrieval_mode') == 'hybrid', (
        f"result_format='facts' should route to hybrid, got: {response!r}"
    )
    assert 'merged_results' in response


def test_invalid_legacy_result_format_returns_error():
    """result_format with unknown value (no retrieval_mode) returns a clear error."""
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    try:
        server._SEARCH_RATE_LIMIT_ENABLED = False
        response = _run(
            server.search_memory_facts(
                query='test',
                result_format='banana',
                ctx=None,
            )
        )
    finally:
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit

    assert response == {'error': "result_format must be one of: 'facts', 'typed'"}


# ---------------------------------------------------------------------------
# Priority: retrieval_mode wins when both parameters are supplied
# ---------------------------------------------------------------------------


def test_retrieval_mode_wins_over_result_format_typed_compat():
    """When retrieval_mode='graph' and result_format='typed', graph wins."""
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    original_graphiti = server.graphiti_service
    original_service_cls = server.TypedRetrievalService

    fake_service = _FakeTypedRetrievalService()

    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False
        server.graphiti_service = None
        server.TypedRetrievalService = lambda **kwargs: fake_service

        response = _run(
            server.search_memory_facts(
                query='test',
                retrieval_mode='graph',
                result_format='typed',  # should be ignored since retrieval_mode wins
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.graphiti_service = original_graphiti
        server.TypedRetrievalService = original_service_cls

    # Typed service was NOT called; routing went to graph path.
    assert fake_service.calls == []
    assert response == {'error': 'Graphiti service not initialized'}


def test_retrieval_mode_typed_wins_over_result_format_facts():
    """When retrieval_mode='typed' and result_format='facts', typed wins."""
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    original_service_cls = server.TypedRetrievalService

    payload = _fake_typed_payload()
    fake_service = _FakeTypedRetrievalService(payload)

    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False
        server.TypedRetrievalService = lambda **kwargs: fake_service

        response = _run(
            server.search_memory_facts(
                query='test',
                retrieval_mode='typed',
                result_format='facts',  # should be ignored since retrieval_mode wins
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.TypedRetrievalService = original_service_cls

    assert response == payload
    assert len(fake_service.calls) == 1


# ---------------------------------------------------------------------------
# US-002: search_mode semantics documented via contract
# ---------------------------------------------------------------------------


def test_valid_retrieval_modes_constant_contains_exactly_three_values():
    """VALID_RETRIEVAL_MODES set has exactly the three documented values."""
    assert server.VALID_RETRIEVAL_MODES == {'graph', 'typed', 'hybrid'}


def test_valid_search_modes_constant_unchanged():
    """VALID_SEARCH_MODES (graph-internal) remains hybrid|semantic|keyword."""
    assert server.VALID_SEARCH_MODES == {'hybrid', 'semantic', 'keyword'}


def test_retrieval_mode_and_search_mode_are_orthogonal():
    """search_mode is graph-internal; retrieval_mode is the surface selector.
    They must remain independent parameters: setting both for graph is valid."""
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    original_graphiti = server.graphiti_service
    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False
        server.graphiti_service = None

        # Both parameters accepted together for the graph surface.
        response = _run(
            server.search_memory_facts(
                query='test',
                retrieval_mode='graph',
                search_mode='keyword',
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.graphiti_service = original_graphiti

    # Reached graph path without parameter conflict errors.
    assert response == {'error': 'Graphiti service not initialized'}


# ---------------------------------------------------------------------------
# P2 fix: error strings now reference canonical retrieval_mode, not result_format
# ---------------------------------------------------------------------------

def test_search_mode_error_references_retrieval_mode_not_result_format():
    """Error for typed+search_mode must say retrieval_mode='typed', not result_format='typed'."""
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False

        response = _run(
            server.search_memory_facts(
                query='test',
                retrieval_mode='typed',
                search_mode='semantic',
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit

    assert 'error' in response
    assert "retrieval_mode='typed'" in response['error'], (
        f"Error should reference retrieval_mode='typed', got: {response['error']!r}"
    )
    assert "result_format='typed'" not in response['error'], (
        f"Error should not reference deprecated result_format, got: {response['error']!r}"
    )


def test_center_node_uuid_error_references_retrieval_mode_not_result_format():
    """Error for typed+center_node_uuid must say retrieval_mode='typed', not result_format='typed'."""
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False

        response = _run(
            server.search_memory_facts(
                query='test',
                retrieval_mode='typed',
                center_node_uuid='some-node',
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit

    assert 'error' in response
    assert "retrieval_mode='typed'" in response['error'], (
        f"Error should reference retrieval_mode='typed', got: {response['error']!r}"
    )
    assert "result_format='typed'" not in response['error'], (
        f"Error should not reference deprecated result_format, got: {response['error']!r}"
    )
