import asyncio
from types import SimpleNamespace

from tests.helpers_mcp_import import load_graphiti_mcp_server

server = load_graphiti_mcp_server()


def _run(coro):
    return asyncio.run(coro)


def _test_config():
    return SimpleNamespace(
        database=SimpleNamespace(provider='neo4j'),
        graphiti=SimpleNamespace(
            group_id='s1_sessions_main',
            lane_aliases={
                'sessions_main': ['s1_sessions_main'],
                'observational_memory': ['s1_observational_memory'],
            },
        ),
    )


class _FakeTypedRetrievalService:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


def test_search_memory_facts_typed_mode_reuses_lane_scope_and_skips_graphiti_dependency():
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    original_graphiti_service = server.graphiti_service
    original_service_cls = server.TypedRetrievalService

    payload = {
        'message': 'Typed memory retrieved successfully',
        'query_mode': 'current',
        'state': [{'object_id': 'fact_1'}],
        'episodes': [],
        'procedures': [],
        'evidence': [],
        'counts': {'state': 1, 'episodes': 0, 'procedures': 0, 'evidence': 0},
    }
    fake_service = _FakeTypedRetrievalService(payload)

    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False
        server.graphiti_service = None
        server.TypedRetrievalService = lambda: fake_service

        response = _run(
            server.search_memory_facts(
                query='current coffee preference',
                group_ids=None,
                lane_alias=['sessions_main'],
                search_mode='hybrid',
                max_facts=3,
                center_node_uuid=None,
                result_format='typed',
                object_types=['state'],
                metadata_filters={'policy_scope': {'eq': 'private'}},
                history_mode='current',
                current_only=None,
                max_results=None,
                max_evidence=7,
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.graphiti_service = original_graphiti_service
        server.TypedRetrievalService = original_service_cls

    assert response == payload
    assert len(fake_service.calls) == 1
    call = fake_service.calls[0]
    assert call['query'] == 'current coffee preference'
    assert call['object_types'] == ['state']
    assert call['metadata_filters'] == {
        'policy_scope': {'eq': 'private'},
        'source_lane': {'in': ['s1_sessions_main']},
    }
    assert call['history_mode'] == 'current'
    assert call['current_only'] is None
    assert call['max_results'] == 3
    assert call['max_evidence'] == 7


def test_search_memory_facts_typed_mode_intersects_caller_source_lane_filter_with_effective_scope():
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    original_service_cls = server.TypedRetrievalService

    fake_service = _FakeTypedRetrievalService(
        {
            'message': 'Typed memory retrieved successfully',
            'query_mode': 'all',
            'state': [],
            'episodes': [],
            'procedures': [],
            'evidence': [],
            'counts': {'state': 0, 'episodes': 0, 'procedures': 0, 'evidence': 0},
        }
    )

    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False
        server.TypedRetrievalService = lambda: fake_service

        _run(
            server.search_memory_facts(
                query='coffee',
                group_ids=['s1_sessions_main', 's1_observational_memory'],
                result_format='typed',
                max_facts=5,
                object_types=['state'],
                metadata_filters={
                    'policy_scope': {'eq': 'private'},
                    'source_lane': {'eq': 's1_sessions_main'},
                },
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.TypedRetrievalService = original_service_cls

    assert len(fake_service.calls) == 1
    call = fake_service.calls[0]
    assert call['metadata_filters']['policy_scope'] == {'eq': 'private'}
    assert call['metadata_filters']['source_lane'] == {'in': ['s1_sessions_main']}


def test_search_memory_facts_typed_mode_honors_explicit_max_results():
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    original_service_cls = server.TypedRetrievalService

    fake_service = _FakeTypedRetrievalService(
        {
            'message': 'Typed memory retrieved successfully',
            'query_mode': 'all',
            'state': [],
            'episodes': [],
            'procedures': [],
            'evidence': [],
            'counts': {'state': 0, 'episodes': 0, 'procedures': 0, 'evidence': 0},
        }
    )

    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False
        server.TypedRetrievalService = lambda: fake_service

        _run(
            server.search_memory_facts(
                query='launch procedure',
                result_format='typed',
                max_facts=2,
                max_results=9,
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.TypedRetrievalService = original_service_cls

    assert len(fake_service.calls) == 1
    assert fake_service.calls[0]['max_results'] == 9


def test_search_memory_facts_typed_mode_rejects_non_hybrid_search_mode():
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False
        response = _run(
            server.search_memory_facts(
                query='coffee',
                result_format='typed',
                search_mode='semantic',
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit

    assert response == {
        'error': "search_mode is not supported for result_format='typed'; use 'hybrid' or omit it"
    }


def test_search_memory_facts_typed_mode_rejects_center_node_uuid():
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False
        response = _run(
            server.search_memory_facts(
                query='coffee',
                result_format='typed',
                center_node_uuid='node-123',
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit

    assert response == {'error': "center_node_uuid is not supported for result_format='typed'"}


def test_search_memory_facts_typed_mode_caps_requested_limits_before_service_call():
    original_config = server.config
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    original_service_cls = server.TypedRetrievalService

    fake_service = _FakeTypedRetrievalService(
        {
            'message': 'Typed memory retrieved successfully',
            'query_mode': 'all',
            'state': [],
            'episodes': [],
            'procedures': [],
            'evidence': [],
            'counts': {'state': 0, 'episodes': 0, 'procedures': 0, 'evidence': 0},
        }
    )

    try:
        server.config = _test_config()
        server._SEARCH_RATE_LIMIT_ENABLED = False
        server.TypedRetrievalService = lambda: fake_service

        _run(
            server.search_memory_facts(
                query='coffee',
                result_format='typed',
                max_results=999,
                max_evidence=999,
                ctx=None,
            )
        )
    finally:
        server.config = original_config
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit
        server.TypedRetrievalService = original_service_cls

    assert fake_service.calls[0]['max_results'] == 200
    assert fake_service.calls[0]['max_evidence'] == 200


def test_search_memory_facts_rejects_unknown_result_format():
    original_rate_limit = server._SEARCH_RATE_LIMIT_ENABLED
    try:
        server._SEARCH_RATE_LIMIT_ENABLED = False
        response = _run(
            server.search_memory_facts(
                query='coffee',
                result_format='banana',
                ctx=None,
            )
        )
    finally:
        server._SEARCH_RATE_LIMIT_ENABLED = original_rate_limit

    assert response == {'error': "result_format must be one of: 'facts', 'typed'"}
