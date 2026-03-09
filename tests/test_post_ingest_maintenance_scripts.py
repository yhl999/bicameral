from __future__ import annotations

import asyncio
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

if 'graph_driver' not in sys.modules:
    graph_driver = types.ModuleType('graph_driver')
    graph_driver.add_backend_args = lambda parser: parser
    graph_driver.get_graph_client = lambda *args, **kwargs: None
    sys.modules['graph_driver'] = graph_driver


dedupe_nodes = importlib.import_module('scripts.dedupe_nodes')
repair_timeline = importlib.import_module('scripts.repair_timeline')


class FakeClient:
    def __init__(self, query_results: list[list[list[object]]] | None = None):
        self.query_results = list(query_results or [])
        self.query_calls: list[tuple[str, dict | None]] = []
        self.transaction_calls: list[list[tuple[str, dict]]] = []
        self.closed = False

    async def query(self, cypher: str, params: dict | None = None):
        self.query_calls.append((cypher, params))
        result_set = self.query_results.pop(0) if self.query_results else []
        return SimpleNamespace(result_set=result_set)

    async def run_in_transaction(self, queries: list[tuple[str, dict]]):
        self.transaction_calls.append(queries)

    async def close(self):
        self.closed = True


def test_bucket_duplicate_records_merges_generic_into_typed_bucket() -> None:
    records = [
        {
            'uuid': 'generic-older',
            'created_at': '2026-03-09T10:00:00Z',
            'labels': ['Entity'],
        },
        {
            'uuid': 'typed-newer',
            'created_at': '2026-03-09T10:05:00Z',
            'labels': ['Entity', 'Person'],
        },
    ]

    buckets = dedupe_nodes.bucket_duplicate_records(records, 's1_sessions_main')

    assert len(buckets) == 1
    assert [row['uuid'] for row in buckets[0]] == ['generic-older', 'typed-newer']


def test_bucket_duplicate_records_keeps_bridge_merge_shapes_apart() -> None:
    records = [
        {
            'uuid': 'person-node',
            'created_at': '2026-03-09T10:00:00Z',
            'labels': ['Entity', 'Person'],
        },
        {
            'uuid': 'bridge-node',
            'created_at': '2026-03-09T10:01:00Z',
            'labels': ['Entity', 'Person', 'Organization'],
        },
        {
            'uuid': 'org-node',
            'created_at': '2026-03-09T10:02:00Z',
            'labels': ['Entity', 'Organization'],
        },
    ]

    buckets = dedupe_nodes.bucket_duplicate_records(records, 's1_sessions_main')

    assert buckets == []


def test_build_merged_entity_payload_preserves_typed_metadata() -> None:
    records = [
        {
            'uuid': 'winner',
            'labels': ['Entity', 'Person'],
            'properties': {
                'uuid': 'winner',
                'name': 'Ada',
                'group_id': 's1_sessions_main',
                'created_at': '2026-03-09T09:00:00Z',
                'summary': '',
                'attributes': {'role': 'founder'},
                'aliases': ['Ada'],
                'name_embedding': None,
            },
        },
        {
            'uuid': 'loser',
            'labels': ['Entity', 'Person'],
            'properties': {
                'uuid': 'loser',
                'name': 'Ada',
                'group_id': 's1_sessions_main',
                'created_at': '2026-03-09T09:05:00Z',
                'summary': 'Ada founded Example Labs.',
                'attributes': {'industry': 'AI'},
                'aliases': ['A. Lovelace'],
                'custom_score': 42,
                'name_embedding': [0.1, 0.2, 0.3],
            },
        },
    ]

    payload = dedupe_nodes.build_merged_entity_payload(
        records,
        winner_uuid='winner',
        group_id='s1_sessions_main',
    )

    assert payload['winner_props']['group_id'] == 's1_sessions_main'
    assert payload['winner_props']['created_at'] == '2026-03-09T09:00:00Z'
    assert payload['winner_props']['summary'] == 'Ada founded Example Labs.'
    assert payload['winner_props']['attributes'] == {'role': 'founder', 'industry': 'AI'}
    assert payload['winner_props']['aliases'] == ['Ada', 'A. Lovelace']
    assert payload['winner_props']['custom_score'] == 42
    assert payload['name_embedding'] == [0.1, 0.2, 0.3]


def test_merge_bucket_scopes_destructive_queries_by_group_id() -> None:
    fake_client = FakeClient(
        query_results=[
            [
                [
                    'winner',
                    ['Entity', 'Person'],
                    {
                        'uuid': 'winner',
                        'name': 'Ada',
                        'group_id': 's1_sessions_main',
                        'created_at': '2026-03-09T09:00:00Z',
                        'summary': '',
                    },
                ],
                [
                    'loser',
                    ['Entity', 'Person'],
                    {
                        'uuid': 'loser',
                        'name': 'Ada',
                        'group_id': 's1_sessions_main',
                        'created_at': '2026-03-09T09:05:00Z',
                        'summary': 'Ada founded Example Labs.',
                        'name_embedding': [0.1, 0.2],
                    },
                ],
            ]
        ]
    )
    bucket = {
        'name': 'Ada',
        'nodes': [
            {'uuid': 'winner', 'created_at': '2026-03-09T09:00:00Z', 'labels': ['Entity', 'Person']},
            {'uuid': 'loser', 'created_at': '2026-03-09T09:05:00Z', 'labels': ['Entity', 'Person']},
        ],
    }

    merged = asyncio.run(dedupe_nodes._merge_bucket(fake_client, 'falkordb', 's1_sessions_main', bucket))

    assert merged == 1
    assert fake_client.transaction_calls, 'expected transactional merge queries to be issued'
    queries = fake_client.transaction_calls[0]
    combined = '\n'.join(query for query, _ in queries)
    assert 'winner.group_id = $group_id' in combined
    assert 'loser.group_id = $group_id' in combined
    assert 'e.group_id = $group_id' in combined
    assert 'c.group_id = $group_id' in combined
    assert 'target.group_id = $group_id' in combined
    assert 'source.group_id = $group_id' in combined
    assert 'loser.uuid IN $loser_uuids' in combined


def test_sort_episodes_for_timeline_prefers_valid_at_with_created_at_fallback() -> None:
    episodes = [
        {'uuid': 'ep-created-only', 'valid_at': None, 'created_at': '2026-03-09T09:00:00Z'},
        {'uuid': 'ep-valid-later', 'valid_at': '2026-03-09T09:30:00Z', 'created_at': '2026-03-09T09:40:00Z'},
        {'uuid': 'ep-valid-earlier', 'valid_at': '2026-03-09T08:30:00Z', 'created_at': '2026-03-09T08:35:00Z'},
    ]

    ordered = repair_timeline.sort_episodes_for_timeline(episodes)

    assert [episode['uuid'] for episode in ordered] == [
        'ep-valid-earlier',
        'ep-created-only',
        'ep-valid-later',
    ]


def test_build_timeline_pairs_assigns_group_and_deterministic_uuid() -> None:
    episodes = [
        {'uuid': 'ep-1', 'created_at': '2026-03-09T09:00:00Z'},
        {'uuid': 'ep-2', 'created_at': '2026-03-09T09:05:00Z'},
        {'uuid': 'ep-3', 'created_at': '2026-03-09T09:10:00Z'},
    ]

    pairs = repair_timeline.build_timeline_pairs(episodes, 's1_sessions_main')

    assert pairs == [
        {
            'prev': 'ep-1',
            'curr': 'ep-2',
            'uuid': 'timeline:ep-1->ep-2',
            'group_id': 's1_sessions_main',
            'created_at': '2026-03-09T09:00:00Z',
        },
        {
            'prev': 'ep-2',
            'curr': 'ep-3',
            'uuid': 'timeline:ep-2->ep-3',
            'group_id': 's1_sessions_main',
            'created_at': '2026-03-09T09:05:00Z',
        },
    ]


def test_repair_timeline_scopes_falkordb_queries_and_uses_valid_at() -> None:
    fake_client = FakeClient(
        query_results=[
            [],
            [
                ['ep-2', '2026-03-09T09:30:00Z', '2026-03-09T09:31:00Z'],
                ['ep-1', None, '2026-03-09T09:00:00Z'],
            ],
            [],
        ]
    )

    async def fake_get_graph_client(*args, **kwargs):
        return fake_client

    original = repair_timeline.get_graph_client
    repair_timeline.get_graph_client = fake_get_graph_client
    try:
        asyncio.run(repair_timeline.repair_timeline('falkordb', 'localhost', 6379, 's1_sessions_main'))
    finally:
        repair_timeline.get_graph_client = original

    assert len(fake_client.query_calls) == 3

    delete_query, delete_params = fake_client.query_calls[0]
    assert 'e1.group_id = $group_id AND e2.group_id = $group_id' in delete_query
    assert delete_params == {'group_id': 's1_sessions_main'}

    fetch_query, fetch_params = fake_client.query_calls[1]
    assert 'RETURN e.uuid, e.valid_at, e.created_at' in fetch_query
    assert 'coalesce(e.valid_at, e.created_at)' in fetch_query
    assert fetch_params == {'group_id': 's1_sessions_main'}

    link_query, link_params = fake_client.query_calls[2]
    assert 'WHERE a.group_id = $group_id AND b.group_id = $group_id' in link_query
    assert link_params['prev'] == 'ep-1'
    assert link_params['curr'] == 'ep-2'
    assert link_params['group_id'] == 's1_sessions_main'
