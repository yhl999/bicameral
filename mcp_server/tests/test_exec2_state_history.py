from __future__ import annotations

import os
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

    search_service_mod.SearchService = SearchService
    sys.modules['mcp_server.src.services.search_service'] = search_service_mod
    sys.modules['services.search_service'] = search_service_mod

if 'mcp_server.src.services.typed_retrieval' not in sys.modules:
    typed_retrieval = types.ModuleType('mcp_server.src.services.typed_retrieval')

    class TypedRetrievalService:
        def __init__(self, *args, **kwargs):
            pass

        async def search(self, *args, **kwargs):
            return {'facts': [], 'episodes': [], 'procedures': []}

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
from mcp_server.src.models.typed_memory import EvidenceRef, StateFact
from mcp_server.src.services.change_ledger import ChangeLedger


def _evidence_ref(tag: str) -> EvidenceRef:
    return EvidenceRef.from_legacy_ref(
        {
            'source_key': 'unit-test',
            'evidence_id': tag,
        }
    )


def _state_fact(
    *,
    object_id: str,
    root_id: str,
    subject: str,
    predicate: str,
    value: str,
    created_at: str,
) -> StateFact:
    return StateFact.model_validate(
        {
            'object_id': object_id,
            'root_id': root_id,
            'object_type': 'state_fact',
            'fact_type': 'preference',
            'subject': subject,
            'predicate': predicate,
            'value': value,
            'policy_scope': 'private',
            'visibility_scope': 'private',
            'created_at': created_at,
            'evidence_refs': [_evidence_ref(object_id)],
        }
    )


@pytest.fixture
def ledger(monkeypatch, tmp_path):
    temp_ledger = ChangeLedger(tmp_path / 'change_ledger.db')
    monkeypatch.setattr(server, 'change_ledger', temp_ledger)
    return temp_ledger


@pytest.mark.anyio
async def test_get_current_state_prefers_predicate_when_subject_is_ambiguous(ledger):
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='f1',
            root_id='r-state-user',
            subject='UI',
            predicate='theme',
            value='dark',
            created_at='2026-01-01T10:00:00Z',
        ),
        root_id='r-state-user',
        recorded_at='2026-01-01T10:00:00Z',
    )
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='f2',
            root_id='r-display-user',
            subject='UI',
            predicate='font',
            value='serif',
            created_at='2026-01-01T10:00:00Z',
        ),
        root_id='r-display-user',
        recorded_at='2026-01-01T10:00:00Z',
    )

    result = await server.get_current_state('UI')
    assert 'error' in result
    assert 'ambiguous_subject' in result['error']


@pytest.mark.anyio
async def test_get_current_state_returns_not_found_when_no_state_for_subject(ledger):
    result = await server.get_current_state('missing')
    assert 'error' in result
    assert 'not_found' in result['error']


@pytest.mark.anyio
async def test_get_current_state_returns_most_recent_matching_predicate(ledger):
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='theme-1',
            root_id='r-theme',
            subject='UI',
            predicate='theme',
            value='light',
            created_at='2026-01-01T10:00:00Z',
        ),
        root_id='r-theme',
        recorded_at='2026-01-01T10:00:00Z',
    )
    ledger.append_event(
        'refine',
        payload=_state_fact(
            object_id='theme-2',
            root_id='r-theme',
            subject='UI',
            predicate='theme',
            value='dark',
            created_at='2026-01-01T11:00:00Z',
        ),
        target_object_id='theme-1',
        root_id='r-theme',
        recorded_at='2026-01-01T11:00:00Z',
    )

    result = await server.get_current_state('UI', predicate='theme')
    assert 'error' not in result
    assert result.value == 'dark'
    assert result.predicate == 'theme'


@pytest.mark.anyio
async def test_get_history_returns_full_statused_chain_in_order(ledger):
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='chain-1',
            root_id='r-chain',
            subject='Workspace',
            predicate='theme',
            value='light',
            created_at='2026-01-01T10:00:00Z',
        ),
        root_id='r-chain',
        recorded_at='2026-01-01T10:00:00Z',
    )
    ledger.append_event(
        'refine',
        payload=_state_fact(
            object_id='chain-2',
            root_id='r-chain',
            subject='Workspace',
            predicate='theme',
            value='dark',
            created_at='2026-01-01T11:00:00Z',
        ),
        target_object_id='chain-1',
        root_id='r-chain',
        recorded_at='2026-01-01T11:00:00Z',
    )
    ledger.append_event(
        'refine',
        payload=_state_fact(
            object_id='chain-3',
            root_id='r-chain',
            subject='Workspace',
            predicate='theme',
            value='system',
            created_at='2026-01-01T12:00:00Z',
        ),
        target_object_id='chain-2',
        root_id='r-chain',
        recorded_at='2026-01-01T12:00:00Z',
    )

    # Add a different predicate so predicate-agnostic history still includes all subject facts.
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='contrast',
            root_id='r-contrast',
            subject='Workspace',
            predicate='contrast',
            value='high',
            created_at='2026-01-01T09:00:00Z',
        ),
        root_id='r-contrast',
        recorded_at='2026-01-01T09:00:00Z',
    )

    by_predicate = await server.get_history('Workspace', predicate='theme')
    assert len(by_predicate) == 3
    assert [event.value for event in by_predicate] == ['light', 'dark', 'system']
    assert by_predicate[-1].status == 'active'
    assert by_predicate[0].status == 'superseded'

    by_subject = await server.get_history('Workspace')
    assert len(by_subject) == 4


@pytest.mark.anyio
async def test_get_history_returns_empty_list_when_no_records(ledger):
    assert await server.get_history('never-seen') == []
