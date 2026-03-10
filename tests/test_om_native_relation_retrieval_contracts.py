from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from mcp_server.src.services.neo4j_service import Neo4jService
from mcp_server.src.services.search_service import SearchService


def _run(coro):
    return asyncio.run(coro)


def test_search_om_facts_lexical_query_supports_native_relation_endpoint_fallbacks():
    driver = AsyncMock()
    driver.execute_query = AsyncMock(return_value=([], None, None))

    service = Neo4jService()
    _run(
        service.search_om_facts(
            driver,
            group_id='s1_observational_memory',
            query='latency',
            limit=5,
        )
    )

    query_text = driver.execute_query.await_args.args[0]
    assert 'MATCH (matched_node)-[rel:MOTIVATES|GENERATES|SUPERSEDES|ADDRESSES|RESOLVES]-(neighbor)' in query_text
    assert 'neighbor:OMNode' not in query_text
    assert 'coalesce(rel.source_node_id, source.node_id, source.uuid, \'\') AS source_node_id' in query_text
    assert 'coalesce(rel.target_node_id, target.node_id, target.uuid, \'\') AS target_node_id' in query_text
    assert 'properties(rel) AS relation_properties' in query_text
    assert 'rel.invalid_at AS invalid_at' in query_text


def test_search_om_facts_center_query_allows_non_omnode_relation_sources():
    driver = AsyncMock()
    driver.execute_query = AsyncMock(return_value=([], None, None))

    service = Neo4jService()
    _run(
        service.search_om_facts(
            driver,
            group_id='s1_observational_memory',
            query='',
            limit=5,
            center_node_uuid='issue_1',
        )
    )

    query_text = driver.execute_query.await_args.args[0]
    assert 'MATCH (source)-[rel:RESOLVES]->(center)' in query_text
    assert 'MATCH (source:OMNode)-[rel:RESOLVES]->(center)' not in query_text
    assert 'coalesce(source.group_id, $group_id) = $group_id' in query_text
    assert 'coalesce(target.group_id, $group_id) = $group_id' in query_text
    assert 'coalesce(rel.source_node_id, source.node_id, source.uuid' in query_text


def test_search_service_preserves_native_relation_lifecycle_fields():
    neo4j_service = SimpleNamespace(
        search_om_facts=AsyncMock(
            return_value=[
                {
                    'uuid': 'om-rel-1',
                    'relation_type': 'RESOLVES',
                    'source_node_id': 'judgment_1',
                    'target_node_id': 'issue_1',
                    'created_at': '2026-03-10T00:00:00Z',
                    'valid_at': '2026-03-10T00:00:00Z',
                    'invalid_at': '2026-03-11T00:00:00Z',
                    'group_id': 's1_observational_memory',
                    'source_content': 'Ship cache fix',
                    'target_content': 'Latency issue',
                    'relation_properties': {
                        'relation_root_id': 'omrelroot:123',
                        'lineage_parent_relation_id': 'om-rel-0',
                        'lifecycle_status': 'invalidated',
                        'transition_basis': 'convergence_resolution',
                    },
                    'lexical_score': 4.2,
                }
            ]
        )
    )
    service = SearchService(neo4j_service=neo4j_service)
    graphiti_service = SimpleNamespace(
        config=SimpleNamespace(database=SimpleNamespace(provider='neo4j')),
        get_client=AsyncMock(return_value=SimpleNamespace(driver=object())),
    )

    facts = _run(
        service.search_observational_facts(
            graphiti_service=graphiti_service,
            query='latency issue',
            group_ids=['s1_observational_memory'],
            max_facts=5,
            center_node_uuid='issue_1',
        )
    )

    assert len(facts) == 1
    fact = facts[0]
    assert fact['valid_at'] == '2026-03-10T00:00:00Z'
    assert fact['invalid_at'] == '2026-03-11T00:00:00Z'
    assert fact['source_node_uuid'] == 'judgment_1'
    assert fact['target_node_uuid'] == 'issue_1'
    assert fact['attributes']['relation_properties'] == {
        'relation_root_id': 'omrelroot:123',
        'lineage_parent_relation_id': 'om-rel-0',
        'lifecycle_status': 'invalidated',
        'transition_basis': 'convergence_resolution',
    }
