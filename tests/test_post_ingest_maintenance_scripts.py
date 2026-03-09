from __future__ import annotations

import asyncio
import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def _make_merge_row(
    uuid: str,
    labels: list[str],
    *,
    name: str,
    created_at: str,
    **properties: object,
) -> list[object]:
    return [
        uuid,
        labels,
        {
            'uuid': uuid,
            'name': name,
            'group_id': 's1_sessions_main',
            'created_at': created_at,
            **properties,
        },
    ]


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


def test_bucket_duplicate_records_refuses_ambiguous_generic_only_merges() -> None:
    records = [
        {
            'uuid': 'generic-1',
            'created_at': '2026-03-09T10:00:00Z',
            'labels': ['Entity'],
        },
        {
            'uuid': 'generic-2',
            'created_at': '2026-03-09T10:01:00Z',
            'labels': ['Entity'],
        },
        {
            'uuid': 'person-1',
            'created_at': '2026-03-09T10:02:00Z',
            'labels': ['Entity', 'Person'],
        },
        {
            'uuid': 'org-1',
            'created_at': '2026-03-09T10:03:00Z',
            'labels': ['Entity', 'Organization'],
        },
    ]

    buckets = dedupe_nodes.bucket_duplicate_records(records, 's1_sessions_main')

    assert buckets == []


def test_build_merged_entity_payload_preserves_typed_metadata_and_conflicts() -> None:
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
                'attributes': {'role': 'founder', 'status': 'active'},
                'aliases': ['Ada'],
                'custom_score': 41,
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
                'attributes': {'industry': 'AI', 'status': 'inactive'},
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
    assert payload['winner_props']['attributes'] == {
        'role': 'founder',
        'status': 'active',
        'industry': 'AI',
    }
    assert payload['winner_props']['aliases'] == ['Ada', 'A. Lovelace']
    assert payload['winner_props']['custom_score'] == 41
    assert payload['name_embedding'] == [0.1, 0.2, 0.3]

    assert payload['winner_props']['_dedupe_conflict_keys'] == ['attributes.status', 'custom_score']
    conflict_payload = json.loads(payload['winner_props']['_dedupe_conflicts_json'])
    assert conflict_payload['custom_score'] == [41, 42]
    assert conflict_payload['attributes.status'] == ['active', 'inactive']


def test_find_duplicate_buckets_filters_blank_names() -> None:
    fake_client = FakeClient(query_results=[[]])

    buckets = asyncio.run(dedupe_nodes._find_duplicate_buckets(fake_client, 'neo4j', 's1_sessions_main'))

    assert buckets == []
    query, params = fake_client.query_calls[0]
    assert 'n.name IS NOT NULL' in query
    assert "trim(toString(n.name)) <> ''" in query
    assert params == {'group_id': 's1_sessions_main'}


def test_merge_bucket_scopes_destructive_queries_by_group_id_and_uses_plain_delete() -> None:
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
                        'external_id': 'crm:ada',
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
                        'external_id': 'crm:ada',
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
    assert 'DELETE loser' in combined
    assert 'DETACH DELETE loser' not in combined


def test_merge_bucket_rejects_blank_uuid() -> None:
    fake_client = FakeClient()
    bucket = {
        'name': 'Ada',
        'nodes': [
            {'uuid': 'winner', 'created_at': '2026-03-09T09:00:00Z', 'labels': ['Entity', 'Person']},
            {'uuid': '', 'created_at': '2026-03-09T09:05:00Z', 'labels': ['Entity', 'Person']},
        ],
    }

    with pytest.raises(ValueError, match='blank uuid'):
        asyncio.run(dedupe_nodes._merge_bucket(fake_client, 'neo4j', 's1_sessions_main', bucket))


def test_merge_bucket_surfaces_remaining_relationship_diagnostics_on_failure() -> None:
    class FailingClient(FakeClient):
        async def run_in_transaction(self, queries: list[tuple[str, dict]]):
            self.transaction_calls.append(queries)
            raise RuntimeError('delete blocked')

    fake_client = FailingClient(
        query_results=[
            [
                [
                    'winner',
                    ['Entity'],
                    {
                        'uuid': 'winner',
                        'name': 'Ada',
                        'group_id': 's1_sessions_main',
                        'created_at': '2026-03-09T09:00:00Z',
                        'summary': '',
                        'external_id': 'crm:123',
                    },
                ],
                [
                    'loser',
                    ['Entity'],
                    {
                        'uuid': 'loser',
                        'name': 'Ada',
                        'group_id': 's1_sessions_main',
                        'created_at': '2026-03-09T09:05:00Z',
                        'summary': '',
                        'external_id': 'crm:123',
                    },
                ],
            ],
            [['loser', 'RELATES_TO', 2], ['loser', 'HAS_MEMBER', 1]],
        ]
    )
    bucket = {
        'name': 'Ada',
        'nodes': [
            {'uuid': 'winner', 'created_at': '2026-03-09T09:00:00Z', 'labels': ['Entity']},
            {'uuid': 'loser', 'created_at': '2026-03-09T09:05:00Z', 'labels': ['Entity']},
        ],
    }

    with pytest.raises(RuntimeError, match='remaining_relationships') as exc:
        asyncio.run(dedupe_nodes._merge_bucket(fake_client, 'neo4j', 's1_sessions_main', bucket))

    message = str(exc.value)
    assert 'loser' in message
    assert 'RELATES_TO' in message
    assert 'HAS_MEMBER' in message
    assert 'winner_uuid' in message


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


def test_infer_episode_stream_key_normalizes_chunk_suffixes() -> None:
    assert (
        repair_timeline.infer_episode_stream_key('session chunk: sessions:abc:c12:p3 (scope=private)')
        == 'sessions:abc'
    )


def test_build_timeline_groups_partitions_by_saga_and_stream() -> None:
    episodes = [
        {
            'uuid': 'ep-saga-1',
            'created_at': '2026-03-09T09:00:00Z',
            'source_description': '',
            'saga_uuids': ['saga-1'],
        },
        {
            'uuid': 'ep-saga-2',
            'created_at': '2026-03-09T09:05:00Z',
            'source_description': '',
            'saga_uuids': ['saga-1'],
        },
        {
            'uuid': 'ep-stream-1',
            'created_at': '2026-03-09T09:10:00Z',
            'source_description': 'session chunk: sessions:xyz:c0 (scope=private)',
            'saga_uuids': [],
        },
        {
            'uuid': 'ep-stream-2',
            'created_at': '2026-03-09T09:15:00Z',
            'source_description': 'session chunk: sessions:xyz:c1:p0 (scope=private)',
            'saga_uuids': [],
        },
    ]

    groups = repair_timeline.build_timeline_groups(episodes)

    assert sorted(groups) == ['saga:saga-1', 'stream:sessions:xyz']
    assert [episode['uuid'] for episode in groups['saga:saga-1']] == ['ep-saga-1', 'ep-saga-2']
    assert [episode['uuid'] for episode in groups['stream:sessions:xyz']] == [
        'ep-stream-1',
        'ep-stream-2',
    ]


def test_build_timeline_groups_refuses_multi_stream_without_identity() -> None:
    episodes = [
        {'uuid': 'ep-1', 'created_at': '2026-03-09T09:00:00Z', 'source_description': '', 'saga_uuids': []},
        {'uuid': 'ep-2', 'created_at': '2026-03-09T09:05:00Z', 'source_description': '', 'saga_uuids': []},
    ]

    with pytest.raises(ValueError, match='cannot be proven to be a single linear stream'):
        repair_timeline.build_timeline_groups(episodes)


def test_build_timeline_groups_refuses_episode_in_multiple_sagas() -> None:
    episodes = [
        {
            'uuid': 'ep-1',
            'created_at': '2026-03-09T09:00:00Z',
            'source_description': '',
            'saga_uuids': ['saga-1', 'saga-2'],
        }
    ]

    with pytest.raises(ValueError, match='belongs to multiple sagas') as exc:
        repair_timeline.build_timeline_groups(episodes)

    assert 'ep-1' in str(exc.value)
    assert 'saga-1' in str(exc.value)
    assert 'saga-2' in str(exc.value)


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


def test_merge_bucket_refuses_same_name_same_type_without_proof() -> None:
    fake_client = FakeClient(
        query_results=[
            [
                _make_merge_row(
                    'winner',
                    ['Entity', 'Person'],
                    name='Alex Kim',
                    created_at='2026-03-09T09:00:00Z',
                ),
                _make_merge_row(
                    'loser',
                    ['Entity', 'Person'],
                    name='Alex Kim',
                    created_at='2026-03-09T09:05:00Z',
                ),
            ]
        ]
    )
    bucket = {
        'name': 'Alex Kim',
        'nodes': [
            {'uuid': 'winner', 'created_at': '2026-03-09T09:00:00Z', 'labels': ['Entity', 'Person']},
            {'uuid': 'loser', 'created_at': '2026-03-09T09:05:00Z', 'labels': ['Entity', 'Person']},
        ],
    }

    with pytest.raises(ValueError, match='same-name entities without shared identity proof') as exc:
        asyncio.run(dedupe_nodes._merge_bucket(fake_client, 'neo4j', 's1_sessions_main', bucket))

    message = str(exc.value)
    assert 'Alex Kim' in message
    assert 'winner' in message
    assert 'loser' in message
    assert 'labels' in message
    assert 'proof_fields_present' in message



def test_prepare_bucket_merge_refuses_generic_only_same_name_without_proof() -> None:
    fake_client = FakeClient(
        query_results=[
            [
                _make_merge_row(
                    'generic-older',
                    ['Entity'],
                    name='Alex Kim',
                    created_at='2026-03-09T09:00:00Z',
                ),
                _make_merge_row(
                    'generic-newer',
                    ['Entity'],
                    name='Alex Kim',
                    created_at='2026-03-09T09:05:00Z',
                ),
            ]
        ]
    )
    bucket = {
        'name': 'Alex Kim',
        'nodes': [
            {'uuid': 'generic-older', 'created_at': '2026-03-09T09:00:00Z', 'labels': ['Entity']},
            {'uuid': 'generic-newer', 'created_at': '2026-03-09T09:05:00Z', 'labels': ['Entity']},
        ],
    }

    with pytest.raises(ValueError, match='same-name entities without shared identity proof') as exc:
        asyncio.run(dedupe_nodes._prepare_bucket_merge(fake_client, 's1_sessions_main', bucket))

    message = str(exc.value)
    assert 'generic_count' in message
    assert 'typed_signatures' in message
    assert 'candidate_proofs={}' in message
    assert 'generic-older' in message
    assert 'generic-newer' in message



def test_prepare_bucket_merge_refuses_generic_plus_typed_without_shared_proof() -> None:
    fake_client = FakeClient(
        query_results=[
            [
                _make_merge_row(
                    'generic-older',
                    ['Entity'],
                    name='Alex Kim',
                    created_at='2026-03-09T09:00:00Z',
                ),
                _make_merge_row(
                    'typed-newer',
                    ['Entity', 'Person'],
                    name='Alex Kim',
                    created_at='2026-03-09T09:05:00Z',
                    external_id='crm:alex-kim',
                ),
            ]
        ]
    )
    bucket = {
        'name': 'Alex Kim',
        'nodes': [
            {'uuid': 'generic-older', 'created_at': '2026-03-09T09:00:00Z', 'labels': ['Entity']},
            {'uuid': 'typed-newer', 'created_at': '2026-03-09T09:05:00Z', 'labels': ['Entity', 'Person']},
        ],
    }

    with pytest.raises(ValueError, match='same-name entities without shared identity proof') as exc:
        asyncio.run(dedupe_nodes._prepare_bucket_merge(fake_client, 's1_sessions_main', bucket))

    message = str(exc.value)
    assert 'generic_count' in message
    assert 'Person' in message
    assert 'external_id' in message
    assert 'proof_fields_present' in message



def test_prepare_bucket_merge_allows_generic_only_with_shared_external_id() -> None:
    fake_client = FakeClient(
        query_results=[
            [
                _make_merge_row(
                    'generic-older',
                    ['Entity'],
                    name='Alex Kim',
                    created_at='2026-03-09T09:00:00Z',
                    external_id='crm:alex-kim',
                ),
                _make_merge_row(
                    'generic-newer',
                    ['Entity'],
                    name='Alex Kim',
                    created_at='2026-03-09T09:05:00Z',
                    external_id='crm:alex-kim',
                ),
            ]
        ]
    )
    bucket = {
        'name': 'Alex Kim',
        'nodes': [
            {'uuid': 'generic-older', 'created_at': '2026-03-09T09:00:00Z', 'labels': ['Entity']},
            {'uuid': 'generic-newer', 'created_at': '2026-03-09T09:05:00Z', 'labels': ['Entity']},
        ],
    }

    merge_plan = asyncio.run(dedupe_nodes._prepare_bucket_merge(fake_client, 's1_sessions_main', bucket))

    assert merge_plan is not None
    assert merge_plan['winner_uuid'] == 'generic-older'
    assert merge_plan['loser_uuids'] == ['generic-newer']
    assert merge_plan['merged_labels'] == []



def test_prepare_bucket_merge_allows_generic_plus_typed_with_shared_external_id() -> None:
    fake_client = FakeClient(
        query_results=[
            [
                _make_merge_row(
                    'generic-older',
                    ['Entity'],
                    name='Alex Kim',
                    created_at='2026-03-09T09:00:00Z',
                    external_id='crm:alex-kim',
                ),
                _make_merge_row(
                    'typed-newer',
                    ['Entity', 'Person'],
                    name='Alex Kim',
                    created_at='2026-03-09T09:05:00Z',
                    external_id='crm:alex-kim',
                ),
            ]
        ]
    )
    bucket = {
        'name': 'Alex Kim',
        'nodes': [
            {'uuid': 'generic-older', 'created_at': '2026-03-09T09:00:00Z', 'labels': ['Entity']},
            {'uuid': 'typed-newer', 'created_at': '2026-03-09T09:05:00Z', 'labels': ['Entity', 'Person']},
        ],
    }

    merge_plan = asyncio.run(dedupe_nodes._prepare_bucket_merge(fake_client, 's1_sessions_main', bucket))

    assert merge_plan is not None
    assert merge_plan['winner_uuid'] == 'generic-older'
    assert merge_plan['loser_uuids'] == ['typed-newer']
    assert merge_plan['merged_labels'] == ['Person']



def test_repair_timeline_scopes_falkordb_queries_and_uses_valid_at() -> None:
    fake_client = FakeClient(
        query_results=[
            [
                ['ep-2', '2026-03-09T09:30:00Z', '2026-03-09T09:31:00Z', 'session chunk: s:c1 (scope=private)', []],
                ['ep-1', None, '2026-03-09T09:00:00Z', 'session chunk: s:c0 (scope=private)', []],
            ],
            [],
            [],
            [['ep-1', 'ep-2', 'timeline:ep-1->ep-2']],
            [['ep-1', 'ep-2', 'timeline:ep-1->ep-2']],
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

    assert len(fake_client.query_calls) == 5

    fetch_query, fetch_params = fake_client.query_calls[0]
    assert 'RETURN e.uuid,' in fetch_query
    assert 'e.source_description' in fetch_query
    assert 'collect(DISTINCT s.uuid) AS saga_uuids' in fetch_query
    assert 'coalesce(e.valid_at, e.created_at)' in fetch_query
    assert fetch_params == {'group_id': 's1_sessions_main'}

    existing_query, existing_params = fake_client.query_calls[1]
    assert 'MATCH (a:Episodic)-[r:NEXT_EPISODE]->(b:Episodic)' in existing_query
    assert existing_params == {'group_id': 's1_sessions_main'}

    link_query, link_params = fake_client.query_calls[2]
    assert 'WHERE a.group_id = $group_id AND b.group_id = $group_id' in link_query
    assert link_params['prev'] == 'ep-1'
    assert link_params['curr'] == 'ep-2'
    assert link_params['group_id'] == 's1_sessions_main'



def test_repair_timeline_stages_new_edges_before_pruning_stale_ones() -> None:
    fake_client = FakeClient(
        query_results=[
            [
                ['ep-2', '2026-03-09T09:30:00Z', '2026-03-09T09:31:00Z', 'session chunk: s:c1 (scope=private)', []],
                ['ep-1', None, '2026-03-09T09:00:00Z', 'session chunk: s:c0 (scope=private)', []],
            ],
            [['ep-1', 'ep-2', 'legacy-edge']],
            [],
            [
                ['ep-1', 'ep-2', 'legacy-edge'],
                ['ep-1', 'ep-2', 'timeline:ep-1->ep-2'],
            ],
            [],
            [['ep-1', 'ep-2', 'timeline:ep-1->ep-2']],
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

    assert len(fake_client.query_calls) == 6
    link_query, _ = fake_client.query_calls[2]
    delete_query, delete_params = fake_client.query_calls[4]

    assert 'MERGE (a)-[e:NEXT_EPISODE {uuid: $uuid}]->(b)' in link_query
    assert 'DELETE r' in delete_query
    assert "coalesce(toString(r.uuid), '') = $edge_uuid" in delete_query
    assert delete_params == {
        'prev': 'ep-1',
        'curr': 'ep-2',
        'edge_uuid': 'legacy-edge',
        'group_id': 's1_sessions_main',
    }
    assert 'DELETE r' not in fake_client.query_calls[1][0]


def test_repair_timeline_refuses_pruning_edges_that_appeared_after_snapshot() -> None:
    fake_client = FakeClient(
        query_results=[
            [
                ['ep-1', '2026-03-09T09:00:00Z', '2026-03-09T09:00:00Z', 'session chunk: s:c0 (scope=private)', []],
                ['ep-2', '2026-03-09T09:05:00Z', '2026-03-09T09:05:00Z', 'session chunk: s:c1 (scope=private)', []],
            ],
            [],
            [],
            [
                ['ep-1', 'ep-2', 'timeline:ep-1->ep-2'],
                ['ep-2', 'ep-3', 'concurrent-edge'],
            ],
            [
                ['ep-1', 'ep-2', 'timeline:ep-1->ep-2'],
                ['ep-2', 'ep-3', 'concurrent-edge'],
            ],
        ]
    )

    async def fake_get_graph_client(*args, **kwargs):
        return fake_client

    original = repair_timeline.get_graph_client
    repair_timeline.get_graph_client = fake_get_graph_client
    try:
        with pytest.raises(RuntimeError, match='unexpected_edges'):
            asyncio.run(repair_timeline.repair_timeline('neo4j', 'localhost', 7687, 's1_sessions_main'))
    finally:
        repair_timeline.get_graph_client = original

    assert len(fake_client.query_calls) == 5
    assert all('DELETE r' not in query for query, _ in fake_client.query_calls)



def test_repair_timeline_refuses_ambiguous_groups_before_delete() -> None:
    fake_client = FakeClient(
        query_results=[
            [
                ['ep-1', None, '2026-03-09T09:00:00Z', '', []],
                ['ep-2', None, '2026-03-09T09:05:00Z', '', []],
            ]
        ]
    )

    async def fake_get_graph_client(*args, **kwargs):
        return fake_client

    original = repair_timeline.get_graph_client
    repair_timeline.get_graph_client = fake_get_graph_client
    try:
        with pytest.raises(ValueError, match='cannot be proven to be a single linear stream'):
            asyncio.run(repair_timeline.repair_timeline('neo4j', 'localhost', 6379, 's1_sessions_main'))
    finally:
        repair_timeline.get_graph_client = original

    assert len(fake_client.query_calls) == 1
    assert fake_client.closed is True
