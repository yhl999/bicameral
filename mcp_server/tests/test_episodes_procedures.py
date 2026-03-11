from __future__ import annotations

import sys
import types

import pytest

# Lightweight stubs so tests can run without the full Graphiti runtime.
if 'graphiti_core' not in sys.modules:
    graphiti_core = types.ModuleType('graphiti_core')
    graphiti_core.Graphiti = type('Graphiti', (), {})

    graphiti_edges = types.ModuleType('graphiti_core.edges')
    graphiti_edges.EntityEdge = type('EntityEdge', (), {})

    class _EpisodeType:
        memory = 'memory'

    graphiti_nodes = types.ModuleType('graphiti_core.nodes')
    graphiti_nodes.EpisodeType = _EpisodeType
    graphiti_nodes.EpisodicNode = type('EpisodicNode', (), {})

    search_filters_module = types.ModuleType('graphiti_core.search.search_filters')
    search_filters_module.SearchFilters = type('SearchFilters', (), {})

    maintenance_graph_data = types.ModuleType('graphiti_core.utils.maintenance.graph_data_operations')
    maintenance_graph_data.clear_data = lambda *args, **kwargs: None

    graphiti_utils = types.ModuleType('graphiti_core.utils')
    graphiti_utils.__path__ = []
    maintenance_module = types.ModuleType('graphiti_core.utils.maintenance')
    maintenance_module.__path__ = []
    search_module = types.ModuleType('graphiti_core.search')
    search_module.__path__ = []

    sys.modules['graphiti_core'] = graphiti_core
    sys.modules['graphiti_core.edges'] = graphiti_edges
    sys.modules['graphiti_core.nodes'] = graphiti_nodes
    sys.modules['graphiti_core.search'] = search_module
    sys.modules['graphiti_core.search.search_filters'] = search_filters_module
    sys.modules['graphiti_core.utils'] = graphiti_utils
    sys.modules['graphiti_core.utils.maintenance'] = maintenance_module
    sys.modules['graphiti_core.utils.maintenance.graph_data_operations'] = maintenance_graph_data

if 'mcp' not in sys.modules:
    mcp = types.ModuleType('mcp')
    server_mod = types.ModuleType('mcp.server')
    server_mod.__path__ = []

    auth_module = types.ModuleType('mcp.server.auth')
    auth_module.__path__ = []
    middleware_module = types.ModuleType('mcp.server.auth.middleware')
    middleware_module.__path__ = []
    auth_context_module = types.ModuleType('mcp.server.auth.middleware.auth_context')
    auth_context_module.get_access_token = lambda: None

    class _Context:
        client_id: str | None = None

    class _FastMCP:
        def __init__(self, *args, **kwargs):
            self.title = args[0] if args else 'fastmcp'

        def _decorate(self, *args):
            if args and callable(args[0]) and len(args) == 1:
                return args[0]

            def decorator(func):
                return func

            return decorator

        def tool(self, *args, **kwargs):
            return self._decorate(*args)

        def custom_route(self, *args, **kwargs):
            return self._decorate(*args)

    fastmcp_module = types.ModuleType('mcp.server.fastmcp')
    fastmcp_module.Context = _Context
    fastmcp_module.FastMCP = _FastMCP

    sys.modules['mcp'] = mcp
    sys.modules['mcp.server'] = server_mod
    sys.modules['mcp.server.auth'] = auth_module
    sys.modules['mcp.server.auth.middleware'] = middleware_module
    sys.modules['mcp.server.auth.middleware.auth_context'] = auth_context_module
    sys.modules['mcp.server.fastmcp'] = fastmcp_module

if 'mcp_server.src.config.schema' not in sys.modules:
    schema = types.ModuleType('mcp_server.src.config.schema')

    class GraphitiConfig:
        def __init__(self, *args, **kwargs):
            self.graphiti = types.SimpleNamespace(group_id='default')
            self.database = types.SimpleNamespace(provider='neo4j')

    class ServerConfig:
        def __init__(self, *args, **kwargs):
            pass

    schema.GraphitiConfig = GraphitiConfig
    schema.ServerConfig = ServerConfig
    sys.modules['mcp_server.src.config.schema'] = schema
    sys.modules['config.schema'] = schema

if 'mcp_server.src.services.factories' not in sys.modules:
    factories = types.ModuleType('mcp_server.src.services.factories')

    class DatabaseDriverFactory:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def create(*args, **kwargs):
            return None

    class EmbedderFactory:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def create(*args, **kwargs):
            return None

    class LLMClientFactory:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def create(*args, **kwargs):
            return None

    factories.DatabaseDriverFactory = DatabaseDriverFactory
    factories.EmbedderFactory = EmbedderFactory
    factories.LLMClientFactory = LLMClientFactory
    sys.modules['mcp_server.src.services.factories'] = factories
    sys.modules['services.factories'] = factories

if 'mcp_server.src.services.om_group_scope' not in sys.modules:
    scope = types.ModuleType('mcp_server.src.services.om_group_scope')
    scope.is_om_native_only_scope = lambda *args, **kwargs: False
    scope.requires_strict_om_native_only_scope = lambda *args, **kwargs: False
    scope.is_om_group_scope = lambda *args, **kwargs: True
    scope.is_om_type_scope = lambda *args, **kwargs: False
    scope.is_om_type_only_scope = lambda *args, **kwargs: False
    sys.modules['mcp_server.src.services.om_group_scope'] = scope
    sys.modules['services.om_group_scope'] = scope

if 'mcp_server.src.services.om_typed_projection' not in sys.modules:
    proj = types.ModuleType('mcp_server.src.services.om_typed_projection')

    class OMTypedProjectionService:
        def __init__(self, *args, **kwargs):
            pass

    proj.OMTypedProjectionService = OMTypedProjectionService
    sys.modules['mcp_server.src.services.om_typed_projection'] = proj
    sys.modules['services.om_typed_projection'] = proj

if 'mcp_server.src.services.ontology_registry' not in sys.modules:
    registry = types.ModuleType('mcp_server.src.services.ontology_registry')
    registry.OntologyRegistry = object
    sys.modules['mcp_server.src.services.ontology_registry'] = registry
    sys.modules['services.ontology_registry'] = registry

if 'mcp_server.src.services.queue_service' not in sys.modules:
    queue_service = types.ModuleType('mcp_server.src.services.queue_service')
    queue_service.QueueService = object
    queue_service.build_om_candidate_rows = lambda rows: []
    sys.modules['mcp_server.src.services.queue_service'] = queue_service
    sys.modules['services.queue_service'] = queue_service

if 'mcp_server.src.services.search_service' not in sys.modules:
    search_service_mod = types.ModuleType('mcp_server.src.services.search_service')
    search_service_mod.DEFAULT_OM_GROUP_ID = 'om'

    class SearchService:
        def __init__(self, *args, **kwargs):
            pass

        @property
        def om_projection(self):
            return None

        def includes_observational_memory(self, *_args, **kwargs):
            return False

        async def search_observational_facts(self, *args, **kwargs):
            return []

        async def search_observational_nodes(self, *args, **kwargs):
            return []

        class _Neo4jService:
            async def verify_om_fulltext_index_shape(self, *args, **kwargs):
                return None

        neo4j_service = _Neo4jService()

    search_service_mod.SearchService = SearchService
    sys.modules['mcp_server.src.services.search_service'] = search_service_mod
    sys.modules['services.search_service'] = search_service_mod

if 'mcp_server.src.services.typed_retrieval' not in sys.modules:
    typed_retrieval = types.ModuleType('mcp_server.src.services.typed_retrieval')

    class TypedRetrievalService:
        def __init__(self, *args, **kwargs):
            pass

        async def search(self, *args, **kwargs):
            return {'state': [], 'episodes': [], 'procedures': []}

    typed_retrieval.TypedRetrievalService = TypedRetrievalService
    sys.modules['mcp_server.src.services.typed_retrieval'] = typed_retrieval
    sys.modules['services.typed_retrieval'] = typed_retrieval

if 'mcp_server.src.utils.formatting' not in sys.modules:
    formatting = types.ModuleType('mcp_server.src.utils.formatting')

    def format_fact_result(*_args, **_kwargs):
        return {}

    formatting.format_fact_result = format_fact_result
    sys.modules['mcp_server.src.utils.formatting'] = formatting
    sys.modules['utils.formatting'] = formatting

if 'mcp_server.src.utils.rate_limiter' not in sys.modules:
    limiter = types.ModuleType('mcp_server.src.utils.rate_limiter')

    class SlidingWindowRateLimiter:
        def __init__(self, *_args, **_kwargs):
            pass

        async def is_allowed(self, *_args, **_kwargs):
            return True

    limiter.SlidingWindowRateLimiter = SlidingWindowRateLimiter
    sys.modules['mcp_server.src.utils.rate_limiter'] = limiter
    sys.modules['utils.rate_limiter'] = limiter

from mcp_server.src import graphiti_mcp_server as server
from mcp_server.src.models.typed_memory import Episode, EvidenceRef, Procedure
from mcp_server.src.services.change_ledger import ChangeLedger


def _evidence_ref(tag: str) -> EvidenceRef:
    return EvidenceRef.from_legacy_ref({'source_key': 'unit-test', 'evidence_id': tag})


def _episode(
    *,
    object_id: str,
    root_id: str,
    title: str,
    summary: str,
    created_at: str,
    started_at: str | None = None,
    ended_at: str | None = None,
) -> Episode:
    return Episode.model_validate(
        {
            'object_id': object_id,
            'root_id': root_id,
            'object_type': 'episode',
            'title': title,
            'summary': summary,
            'policy_scope': 'private',
            'visibility_scope': 'private',
            'created_at': created_at,
            'started_at': started_at,
            'ended_at': ended_at,
            'evidence_refs': [_evidence_ref(object_id)],
        }
    )


def _procedure(
    *,
    object_id: str,
    root_id: str,
    name: str,
    trigger: str,
    steps: list[str],
    expected_outcome: str,
    success_count: int = 0,
) -> Procedure:
    return Procedure.model_validate(
        {
            'object_id': object_id,
            'root_id': root_id,
            'object_type': 'procedure',
            'name': name,
            'trigger': trigger,
            'steps': steps,
            'expected_outcome': expected_outcome,
            'policy_scope': 'private',
            'visibility_scope': 'private',
            'success_count': success_count,
            'evidence_refs': [_evidence_ref(object_id)],
        }
    )


@pytest.fixture
def ledger(monkeypatch, tmp_path):
    temp_ledger = ChangeLedger(tmp_path / 'change_ledger.db')
    monkeypatch.setattr(server, 'change_ledger', temp_ledger)
    monkeypatch.setattr(server, 'procedure_service', None)
    return temp_ledger


@pytest.mark.anyio
async def test_search_episodes_returns_empty_with_no_data(ledger):
    assert await server.search_episodes('anything') == []
    assert await server.search_procedures('anything') == []


@pytest.mark.anyio
async def test_search_episodes_query_and_time_range_filters(ledger):
    ledger.append_event(
        'assert',
        payload=_episode(
            object_id='ep-1',
            root_id='root-1',
            title='Self-audit run',
            summary='Captured system health',
            created_at='2026-01-01T09:00:00Z',
            started_at='2026-01-01T09:00:00Z',
        ),
        root_id='root-1',
        recorded_at='2026-01-01T09:00:00Z',
    )
    ledger.append_event(
        'assert',
        payload=_episode(
            object_id='ep-2',
            root_id='root-2',
            title='Deployment window',
            summary='Manual rollback notes',
            created_at='2026-01-02T09:00:00Z',
            started_at='2026-01-02T09:00:00Z',
        ),
        root_id='root-2',
        recorded_at='2026-01-02T09:00:00Z',
    )

    results = await server.search_episodes('self-audit')
    assert len(results) == 1
    assert results[0].title == 'Self-audit run'

    results_after = await server.search_episodes('self-audit', time_range={'start': '2026-01-02T00:00:00Z'})
    assert results_after == []


@pytest.mark.anyio
async def test_search_episodes_include_history_when_requested(ledger):
    ledger.append_event(
        'assert',
        payload=_episode(
            object_id='ep-chain-1',
            root_id='root-chain',
            title='Incident',
            summary='first',
            created_at='2026-01-01T10:00:00Z',
            started_at='2026-01-01T10:00:00Z',
        ),
        root_id='root-chain',
        recorded_at='2026-01-01T10:00:00Z',
    )
    ledger.append_event(
        'refine',
        payload=_episode(
            object_id='ep-chain-2',
            root_id='root-chain',
            title='Incident',
            summary='second',
            created_at='2026-01-01T11:00:00Z',
            started_at='2026-01-01T11:00:00Z',
        ),
        target_object_id='ep-chain-1',
        root_id='root-chain',
        recorded_at='2026-01-01T11:00:00Z',
    )

    # Current-only excludes superseded history entries.
    current_only = await server.search_episodes('incident', include_history=False)
    assert len(current_only) == 1
    assert current_only[0].summary == 'second'

    # include_history returns full lineage.
    with_history = await server.search_episodes('', include_history=True)
    assert len(with_history) == 2


@pytest.mark.anyio
async def test_get_episode_returns_error_for_unknown_id(ledger):
    result = await server.get_episode('nonexistent')
    assert 'error' in result
    assert result['error'].startswith('not_found')


@pytest.mark.anyio
async def test_search_procedures_filters_default_and_include_all(ledger):
    ledger.append_event(
        'assert',
        payload=_procedure(
            object_id='proc-proposed',
            root_id='proc-proposed-root',
            name='Check Funds',
            trigger='low funds',
            steps=['inspect wallet'],
            expected_outcome='insight',
            success_count=5,
        ),
        root_id='proc-proposed-root',
        recorded_at='2026-01-01T10:00:00Z',
    )

    ledger.append_event(
        'assert',
        payload=_procedure(
            object_id='proc-promoted',
            root_id='proc-promoted-root',
            name='Urgent Escalation',
            trigger='urgent',
            steps=['page oncall'],
            expected_outcome='closure',
            success_count=1,
        ),
        root_id='proc-promoted-root',
        recorded_at='2026-01-01T10:00:00Z',
    )
    ledger.append_event(
        'promote',
        object_id='proc-promoted',
        root_id='proc-promoted-root',
        recorded_at='2026-01-01T10:05:00Z',
    )

    default = await server.search_procedures('urgent')
    assert len(default) == 1
    assert default[0].name == 'Urgent Escalation'

    all_results = await server.search_procedures('', include_all=True)
    assert len(all_results) == 2


@pytest.mark.anyio
async def test_search_procedures_sorts_by_success_count_desc(ledger):
    first = _procedure(
        object_id='p-low',
        root_id='p-low-root',
        name='A',
        trigger='x',
        steps=['a'],
        expected_outcome='a',
        success_count=1,
    )
    second = _procedure(
        object_id='p-high',
        root_id='p-high-root',
        name='B',
        trigger='x',
        steps=['b'],
        expected_outcome='b',
        success_count=9,
    )
    ledger.append_event('assert', payload=first, root_id='p-low-root')
    ledger.append_event('promote', object_id='p-low', root_id='p-low-root')
    ledger.append_event('assert', payload=second, root_id='p-high-root')
    ledger.append_event('promote', object_id='p-high', root_id='p-high-root')

    ranked = await server.search_procedures('x', include_all=True)
    assert ranked[0].object_id == 'p-high'
    assert ranked[1].object_id == 'p-low'


@pytest.mark.anyio
async def test_get_procedure_returns_error_for_unknown_id(ledger):
    result = await server.get_procedure('missing-procedure')
    assert 'error' in result
    assert result['error'].startswith('not_found')


@pytest.mark.anyio
async def test_get_procedure_returns_current_by_id(ledger):
    proc = _procedure(
        object_id='proc-hit',
        root_id='proc-hit-root',
        name='Recovery',
        trigger='red',
        steps=['stabilize'],
        expected_outcome='ok',
    )
    ledger.append_event('assert', payload=proc, root_id='proc-hit-root')
    ledger.append_event('promote', object_id='proc-hit', root_id='proc-hit-root')

    result = await server.get_procedure('proc-hit')
    assert not isinstance(result, dict) or 'error' not in result
    assert result.object_id == 'proc-hit'

