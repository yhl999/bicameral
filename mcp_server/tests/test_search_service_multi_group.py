from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

from mcp_server.src.services.search_service import SearchService


@pytest.mark.anyio
async def test_search_observational_nodes_reranks_across_multi_group_scope():
    rows_by_group = {
        'bench_20260310_om_a': [
            {
                'uuid': 'a-low-1',
                'content': 'low ranking candidate from first group',
                'created_at': '2026-03-10T12:00:00Z',
                'group_id': 'bench_20260310_om_a',
                'status': 'open',
                'semantic_domain': 'observational_memory',
                'urgency_score': 1,
                'lexical_score': 1,
            },
            {
                'uuid': 'a-low-2',
                'content': 'second low ranking candidate from first group',
                'created_at': '2026-03-10T12:01:00Z',
                'group_id': 'bench_20260310_om_a',
                'status': 'open',
                'semantic_domain': 'observational_memory',
                'urgency_score': 1,
                'lexical_score': 2,
            },
        ],
        'bench_20260310_om_b': [
            {
                'uuid': 'b-high-1',
                'content': 'best candidate from second group',
                'created_at': '2026-03-10T12:02:00Z',
                'group_id': 'bench_20260310_om_b',
                'status': 'active',
                'semantic_domain': 'observational_memory',
                'urgency_score': 3,
                'lexical_score': 10,
            },
            {
                'uuid': 'b-high-2',
                'content': 'runner up from second group',
                'created_at': '2026-03-10T12:03:00Z',
                'group_id': 'bench_20260310_om_b',
                'status': 'active',
                'semantic_domain': 'observational_memory',
                'urgency_score': 2,
                'lexical_score': 9,
            },
        ],
    }
    neo4j_service = SimpleNamespace(
        search_om_nodes=AsyncMock(side_effect=lambda *_args, group_id, **_kwargs: rows_by_group[group_id])
    )
    service = SearchService(neo4j_service=neo4j_service)
    driver = object()
    graphiti_service = SimpleNamespace(
        config=SimpleNamespace(database=SimpleNamespace(provider='neo4j')),
        get_client=AsyncMock(return_value=SimpleNamespace(driver=driver)),
    )

    results = await service.search_observational_nodes(
        graphiti_service=graphiti_service,
        query='shared candidate',
        group_ids=['bench_20260310_om_a', 'bench_20260310_om_b'],
        max_nodes=2,
        entity_types=None,
    )

    assert [row['uuid'] for row in results] == ['b-high-1', 'b-high-2']
    neo4j_service.search_om_nodes.assert_has_awaits(
        [
            call(driver, group_id='bench_20260310_om_a', query='shared candidate', limit=2),
            call(driver, group_id='bench_20260310_om_b', query='shared candidate', limit=2),
        ]
    )


@pytest.mark.anyio
async def test_search_observational_facts_reranks_across_multi_group_scope():
    rows_by_group = {
        'bench_20260310_om_a': [
            {
                'uuid': 'a-fact-1',
                'relation_type': 'RESOLVES',
                'source_node_id': 'a-source-1',
                'target_node_id': 'a-target-1',
                'created_at': '2026-03-10T12:00:00Z',
                'group_id': 'bench_20260310_om_a',
                'source_content': 'first group low score source',
                'target_content': 'first group low score target',
                'lexical_score': 1,
            },
            {
                'uuid': 'a-fact-2',
                'relation_type': 'RESOLVES',
                'source_node_id': 'a-source-2',
                'target_node_id': 'a-target-2',
                'created_at': '2026-03-10T12:01:00Z',
                'group_id': 'bench_20260310_om_a',
                'source_content': 'first group second low source',
                'target_content': 'first group second low target',
                'lexical_score': 2,
            },
        ],
        'bench_20260310_om_b': [
            {
                'uuid': 'b-fact-1',
                'relation_type': 'RESOLVES',
                'source_node_id': 'b-source-1',
                'target_node_id': 'b-target-1',
                'created_at': '2026-03-10T12:02:00Z',
                'group_id': 'bench_20260310_om_b',
                'source_content': 'second group best source',
                'target_content': 'second group best target',
                'lexical_score': 10,
            },
            {
                'uuid': 'b-fact-2',
                'relation_type': 'RESOLVES',
                'source_node_id': 'b-source-2',
                'target_node_id': 'b-target-2',
                'created_at': '2026-03-10T12:03:00Z',
                'group_id': 'bench_20260310_om_b',
                'source_content': 'second group runner up source',
                'target_content': 'second group runner up target',
                'lexical_score': 9,
            },
        ],
    }
    neo4j_service = SimpleNamespace(
        search_om_facts=AsyncMock(side_effect=lambda *_args, group_id, **_kwargs: rows_by_group[group_id])
    )
    service = SearchService(neo4j_service=neo4j_service)
    driver = object()
    graphiti_service = SimpleNamespace(
        config=SimpleNamespace(database=SimpleNamespace(provider='neo4j')),
        get_client=AsyncMock(return_value=SimpleNamespace(driver=driver)),
    )

    results = await service.search_observational_facts(
        graphiti_service=graphiti_service,
        query='shared relation',
        group_ids=['bench_20260310_om_a', 'bench_20260310_om_b'],
        max_facts=2,
        center_node_uuid=None,
    )

    assert [row['uuid'] for row in results] == ['b-fact-1', 'b-fact-2']
    neo4j_service.search_om_facts.assert_has_awaits(
        [
            call(
                driver,
                group_id='bench_20260310_om_a',
                query='shared relation',
                limit=2,
                center_node_uuid=None,
            ),
            call(
                driver,
                group_id='bench_20260310_om_b',
                query='shared relation',
                limit=2,
                center_node_uuid=None,
            ),
        ]
    )
