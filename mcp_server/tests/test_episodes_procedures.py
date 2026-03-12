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
    source_lane: str = 'default',
    started_at: str | None = None,
    ended_at: str | None = None,
    policy_scope: str = 'private',
    visibility_scope: str = 'private',
    annotations: list[str] | None = None,
    history_meta: dict[str, object] | None = None,
) -> Episode:
    return Episode.model_validate(
        {
            'object_id': object_id,
            'root_id': root_id,
            'object_type': 'episode',
            'title': title,
            'summary': summary,
            'source_lane': source_lane,
            'policy_scope': policy_scope,
            'visibility_scope': visibility_scope,
            'annotations': annotations or [],
            'history_meta': history_meta or {},
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
    source_lane: str = 'default',
    policy_scope: str = 'private',
    visibility_scope: str = 'private',
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
            'source_lane': source_lane,
            'policy_scope': policy_scope,
            'visibility_scope': visibility_scope,
            'success_count': success_count,
            'evidence_refs': [_evidence_ref(object_id)],
        }
    )


@pytest.fixture
def ledger(monkeypatch, tmp_path):
    temp_ledger = ChangeLedger(tmp_path / 'change_ledger.db')
    test_config = types.SimpleNamespace(
        graphiti=types.SimpleNamespace(group_id='default', lane_aliases=None),
        database=types.SimpleNamespace(provider='neo4j'),
    )
    monkeypatch.setattr(server, 'config', test_config)
    monkeypatch.setattr(server, 'change_ledger', temp_ledger)
    monkeypatch.setattr(server, 'procedure_service', None)
    return temp_ledger


@pytest.mark.anyio
async def test_search_tools_return_wrapped_empty_results(ledger):
    episodes = await server.search_episodes('anything')
    procedures = await server.search_procedures('anything')

    assert episodes['episodes'] == []
    assert episodes['total'] == 0
    assert episodes['limit'] == 10
    assert episodes['offset'] == 0
    assert episodes['has_more'] is False

    assert procedures['procedures'] == []
    assert procedures['total'] == 0
    assert procedures['has_more'] is False


@pytest.mark.anyio
async def test_search_episodes_scopes_filters_provisional_and_paginates(ledger):
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
    ledger.append_event(
        'assert',
        payload=_episode(
            object_id='ep-other-lane',
            root_id='root-other-lane',
            title='Wrong lane',
            summary='Should not leak',
            created_at='2026-01-03T09:00:00Z',
            source_lane='other',
            started_at='2026-01-03T09:00:00Z',
        ),
        root_id='root-other-lane',
        recorded_at='2026-01-03T09:00:00Z',
    )
    ledger.append_event(
        'assert',
        payload=_episode(
            object_id='ep-provisional',
            root_id='root-provisional',
            title='Observed preference',
            summary='Should stay out of exec5 surface',
            created_at='2026-01-04T09:00:00Z',
            started_at='2026-01-04T09:00:00Z',
            policy_scope='observational',
            visibility_scope='owner',
            annotations=['provisional', 'unpromoted'],
            history_meta={'derivation_level': 'provisional', 'promotion_status': 'unpromoted'},
        ),
        root_id='root-provisional',
        recorded_at='2026-01-04T09:00:00Z',
    )

    page_one = await server.search_episodes('', limit=1)
    assert [item['object_id'] for item in page_one['episodes']] == ['ep-2']
    assert page_one['total'] == 2
    assert page_one['has_more'] is True
    assert page_one['next_offset'] == 1

    page_two = await server.search_episodes('', limit=1, offset=1)
    assert [item['object_id'] for item in page_two['episodes']] == ['ep-1']
    assert page_two['has_more'] is False


@pytest.mark.anyio
async def test_search_episodes_time_range_uses_interval_overlap_for_open_ended_episodes(ledger):
    ledger.append_event(
        'assert',
        payload=_episode(
            object_id='ep-open',
            root_id='root-open',
            title='Long incident',
            summary='Still ongoing',
            created_at='2026-01-01T00:00:00Z',
            started_at='2026-01-01T00:00:00Z',
            ended_at=None,
        ),
        root_id='root-open',
        recorded_at='2026-01-01T00:00:00Z',
    )
    ledger.append_event(
        'assert',
        payload=_episode(
            object_id='ep-future',
            root_id='root-future',
            title='Future event',
            summary='Starts after the queried end bound',
            created_at='2026-01-03T00:00:00Z',
            started_at='2026-01-03T00:00:00Z',
            ended_at=None,
        ),
        root_id='root-future',
        recorded_at='2026-01-03T00:00:00Z',
    )

    after_start = await server.search_episodes(
        '',
        time_range={'start': '2026-01-02T00:00:00Z'},
    )
    assert [item['object_id'] for item in after_start['episodes']] == ['ep-future', 'ep-open']

    before_end = await server.search_episodes(
        '',
        time_range={'end': '2026-01-02T12:00:00Z'},
    )
    assert [item['object_id'] for item in before_end['episodes']] == ['ep-open']


@pytest.mark.anyio
async def test_search_episodes_include_history_and_validate_time_range(ledger):
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

    current_only = await server.search_episodes('incident', include_history=False)
    assert [item['object_id'] for item in current_only['episodes']] == ['ep-chain-2']

    with_history = await server.search_episodes('incident', include_history=True)
    assert [item['object_id'] for item in with_history['episodes']] == ['ep-chain-2', 'ep-chain-1']

    invalid = await server.search_episodes(
        'incident',
        time_range={'start': '2026-01-02T00:00:00Z', 'end': '2026-01-01T00:00:00Z'},
    )
    assert invalid['error'] == 'validation_error'


@pytest.mark.anyio
async def test_get_episode_supports_root_lookup_and_respects_scope(ledger):
    ledger.append_event(
        'assert',
        payload=_episode(
            object_id='ep-current',
            root_id='root-current',
            title='Scoped episode',
            summary='visible',
            created_at='2026-01-01T09:00:00Z',
            started_at='2026-01-01T09:00:00Z',
        ),
        root_id='root-current',
        recorded_at='2026-01-01T09:00:00Z',
    )
    ledger.append_event(
        'assert',
        payload=_episode(
            object_id='ep-other',
            root_id='root-other',
            title='Other lane',
            summary='hidden',
            created_at='2026-01-02T09:00:00Z',
            source_lane='other',
            started_at='2026-01-02T09:00:00Z',
        ),
        root_id='root-other',
        recorded_at='2026-01-02T09:00:00Z',
    )

    result = await server.get_episode('root-current')
    assert not isinstance(result, dict)
    assert result.object_id == 'ep-current'

    hidden = await server.get_episode('ep-other')
    assert hidden['error'] == 'not_found'


@pytest.mark.anyio
async def test_search_requires_single_explicit_group_scope(ledger):
    result = await server.search_episodes('', group_ids=[])
    assert result['error'] == 'validation_error'
    assert 'exactly one scoped group_id' in result['message']


@pytest.mark.anyio
async def test_search_procedures_filters_default_include_all_and_paginates(ledger):
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
    ledger.append_event(
        'assert',
        payload=_procedure(
            object_id='proc-other',
            root_id='proc-other-root',
            name='Wrong lane',
            trigger='urgent',
            steps=['nope'],
            expected_outcome='hidden',
            source_lane='other',
            success_count=99,
        ),
        root_id='proc-other-root',
        recorded_at='2026-01-01T10:00:00Z',
    )
    ledger.append_event(
        'promote',
        object_id='proc-other',
        root_id='proc-other-root',
        recorded_at='2026-01-01T10:05:00Z',
    )

    default = await server.search_procedures('urgent')
    assert [item['object_id'] for item in default['procedures']] == ['proc-promoted']
    assert default['total'] == 1

    all_results = await server.search_procedures('', include_all=True, limit=1)
    assert [item['object_id'] for item in all_results['procedures']] == ['proc-proposed']
    assert all_results['total'] == 2
    assert all_results['has_more'] is True

    second_page = await server.search_procedures('', include_all=True, limit=1, offset=1)
    assert [item['object_id'] for item in second_page['procedures']] == ['proc-promoted']


@pytest.mark.anyio
async def test_get_procedure_supports_trigger_lookup_and_hides_non_promoted(ledger):
    ledger.append_event(
        'assert',
        payload=_procedure(
            object_id='proc-hit',
            root_id='proc-hit-root',
            name='Recovery',
            trigger='recover cluster',
            steps=['stabilize'],
            expected_outcome='ok',
        ),
        root_id='proc-hit-root',
    )
    ledger.append_event('promote', object_id='proc-hit', root_id='proc-hit-root')
    ledger.append_event(
        'assert',
        payload=_procedure(
            object_id='proc-proposed-hidden',
            root_id='proc-proposed-hidden-root',
            name='Draft plan',
            trigger='recover cluster draft',
            steps=['draft'],
            expected_outcome='maybe',
        ),
        root_id='proc-proposed-hidden-root',
    )

    by_trigger = await server.get_procedure('recover cluster')
    assert not isinstance(by_trigger, dict)
    assert by_trigger.object_id == 'proc-hit'

    hidden = await server.get_procedure('proc-proposed-hidden')
    assert hidden['error'] == 'not_found'


@pytest.mark.anyio
async def test_get_procedure_does_not_flatten_operational_failures(monkeypatch, ledger):
    class _ExplodingEvolution:
        def resolve_current(self, _identifier):
            raise RuntimeError('sqlite locked')

    class _FakeProcedureService:
        evolution = _ExplodingEvolution()

        def list_current_procedures(self, include_proposed=False):
            return []

    monkeypatch.setattr(server, '_procedure_service', lambda: _FakeProcedureService())

    with pytest.raises(RuntimeError, match='sqlite locked'):
        await server.get_procedure('proc-hit')
