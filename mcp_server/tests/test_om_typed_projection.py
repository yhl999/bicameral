from types import SimpleNamespace

import pytest

from mcp_server.src.models.typed_memory import Episode, StateFact
from mcp_server.src.services.om_typed_projection import OMTypedProjectionService


class _FakeSearchService:
    def __init__(self, *, node_rows=None, fact_rows=None):
        self.node_rows = node_rows or []
        self.fact_rows = fact_rows or []

    def includes_observational_memory(self, group_ids):
        return True

    def _om_groups_in_scope(self, group_ids):
        return list(group_ids or ['s1_observational_memory'])

    async def search_observational_nodes(self, **_kwargs):
        return list(self.node_rows)

    async def search_observational_facts(self, **_kwargs):
        return list(self.fact_rows)


class _FakeDriver:
    def __init__(self, records=None):
        self.records = records or []
        self.calls = []

    async def execute_query(self, query, **params):
        self.calls.append({'query': query, 'params': params})
        return list(self.records), None, None


class _FakeGraphitiService:
    class config:  # noqa: D401 - tiny provider shim
        class database:
            provider = 'neo4j'

    def __init__(self, driver):
        self.driver = driver

    async def get_client(self):
        return SimpleNamespace(driver=self.driver)


@pytest.mark.anyio
async def test_history_projection_materializes_linear_supersession_lineage():
    search_service = _FakeSearchService(
        node_rows=[
            {
                'uuid': 'node_v3',
                'name': 'Current routine preference',
                'summary': 'espresso after training',
                'group_id': 's1_observational_memory',
                'created_at': '2026-03-09T00:00:00Z',
                'attributes': {
                    'status': 'active',
                    'semantic_domain': 'observational_memory',
                },
            }
        ]
    )
    driver = _FakeDriver(
        records=[
            {
                'node_id': 'node_v1',
                'uuid': 'node_v1',
                'group_id': 's1_observational_memory',
                'content': 'coffee before training',
                'created_at': '2026-03-01T00:00:00Z',
                'status': 'open',
                'semantic_domain': 'observational_memory',
                'supersedes': [],
            },
            {
                'node_id': 'node_v2',
                'uuid': 'node_v2',
                'group_id': 's1_observational_memory',
                'content': 'americano before training',
                'created_at': '2026-03-05T00:00:00Z',
                'status': 'open',
                'semantic_domain': 'observational_memory',
                'supersedes': [
                    {
                        'target_id': 'node_v1',
                        'created_at': '2026-03-05T00:00:00Z',
                        'relation_uuid': 'rel_v2_v1',
                    }
                ],
            },
            {
                'node_id': 'node_v3',
                'uuid': 'node_v3',
                'group_id': 's1_observational_memory',
                'content': 'espresso after training',
                'created_at': '2026-03-09T00:00:00Z',
                'status': 'active',
                'semantic_domain': 'observational_memory',
                'supersedes': [
                    {
                        'target_id': 'node_v2',
                        'created_at': '2026-03-09T00:00:00Z',
                        'relation_uuid': 'rel_v3_v2',
                    }
                ],
            },
        ]
    )
    service = OMTypedProjectionService(
        search_service=search_service,
        graphiti_service=_FakeGraphitiService(driver),
    )

    objects, search_overrides, limits = await service.project(
        query='what changed about the training drink',
        effective_group_ids=['s1_observational_memory'],
        object_types=set(),
        max_results=5,
        query_mode='history',
    )

    assert len(objects) == 3
    assert all(isinstance(obj, Episode) for obj in objects)

    by_id = {obj.object_id: obj for obj in objects}
    v1_id = 'om_episode:s1_observational_memory:node_v1'
    v2_id = 'om_episode:s1_observational_memory:node_v2'
    v3_id = 'om_episode:s1_observational_memory:node_v3'

    assert by_id[v1_id].root_id == v1_id
    assert by_id[v1_id].version == 1
    assert by_id[v1_id].parent_id is None
    assert by_id[v1_id].is_current is False
    assert by_id[v1_id].superseded_by == v2_id
    assert by_id[v1_id].invalid_at == '2026-03-05T00:00:00Z'

    assert by_id[v2_id].root_id == v1_id
    assert by_id[v2_id].version == 2
    assert by_id[v2_id].parent_id == v1_id
    assert by_id[v2_id].is_current is False
    assert by_id[v2_id].superseded_by == v3_id
    assert by_id[v2_id].invalid_at == '2026-03-09T00:00:00Z'

    assert by_id[v3_id].root_id == v1_id
    assert by_id[v3_id].version == 3
    assert by_id[v3_id].parent_id == v2_id
    assert by_id[v3_id].is_current is True
    assert by_id[v3_id].superseded_by is None
    assert by_id[v3_id].invalid_at is None

    assert search_overrides[v1_id].startswith('coffee before training')
    assert limits == {
        'enabled': True,
        'reason': 'projected_history',
        'groups_considered': ['s1_observational_memory'],
        'episodes_projected': 3,
        'state_projected': 0,
        'max_results': 5,
        'history_mode': True,
        'history_candidates': 1,
        'history_lineages_projected': 1,
        'history_state_projection_supported': False,
        'unsupported_object_types': [],
        'skipped_candidates': [],
    }
    assert driver.calls, 'history projection should query Neo4j for supersession lineage'


@pytest.mark.anyio
async def test_history_projection_fails_closed_for_branching_supersession_graph():
    search_service = _FakeSearchService(
        node_rows=[
            {
                'uuid': 'node_v2a',
                'name': 'Branching update',
                'summary': 'two competing successors exist',
                'group_id': 's1_observational_memory',
                'created_at': '2026-03-06T00:00:00Z',
                'attributes': {'status': 'active', 'semantic_domain': 'observational_memory'},
            }
        ]
    )
    driver = _FakeDriver(
        records=[
            {
                'node_id': 'node_v1',
                'uuid': 'node_v1',
                'group_id': 's1_observational_memory',
                'content': 'original state',
                'created_at': '2026-03-01T00:00:00Z',
                'status': 'open',
                'semantic_domain': 'observational_memory',
                'supersedes': [],
            },
            {
                'node_id': 'node_v2a',
                'uuid': 'node_v2a',
                'group_id': 's1_observational_memory',
                'content': 'first successor',
                'created_at': '2026-03-05T00:00:00Z',
                'status': 'active',
                'semantic_domain': 'observational_memory',
                'supersedes': [{'target_id': 'node_v1', 'created_at': '2026-03-05T00:00:00Z'}],
            },
            {
                'node_id': 'node_v2b',
                'uuid': 'node_v2b',
                'group_id': 's1_observational_memory',
                'content': 'second successor',
                'created_at': '2026-03-06T00:00:00Z',
                'status': 'active',
                'semantic_domain': 'observational_memory',
                'supersedes': [{'target_id': 'node_v1', 'created_at': '2026-03-06T00:00:00Z'}],
            },
        ]
    )
    service = OMTypedProjectionService(
        search_service=search_service,
        graphiti_service=_FakeGraphitiService(driver),
    )

    objects, search_overrides, limits = await service.project(
        query='what changed',
        effective_group_ids=['s1_observational_memory'],
        object_types=set(),
        max_results=5,
        query_mode='history',
    )

    assert objects == []
    assert search_overrides == {}
    assert limits['enabled'] is True
    assert limits['reason'] == 'projected_history'
    assert limits['episodes_projected'] == 0
    assert limits['history_lineages_projected'] == 0
    assert limits['skipped_candidates'] == [
        {
            'group_id': 's1_observational_memory',
            'node_id': 'node_v2a',
            'reason': 'ambiguous_supersession_graph',
        }
    ]


@pytest.mark.anyio
async def test_non_history_projection_still_surfaces_episodes_and_state():
    search_service = _FakeSearchService(
        node_rows=[
            {
                'uuid': 'node_current',
                'name': 'Current observation',
                'summary': 'still active',
                'group_id': 's1_observational_memory',
                'created_at': '2026-03-09T00:00:00Z',
                'attributes': {'status': 'active', 'semantic_domain': 'observational_memory'},
            }
        ],
        fact_rows=[
            {
                'uuid': 'rel_current',
                'name': 'RESOLVES',
                'fact': 'RESOLVES: current observation -> latency issue',
                'group_id': 's1_observational_memory',
                'source_node_uuid': 'node_current',
                'target_node_uuid': 'issue_1',
                'created_at': '2026-03-09T00:00:00Z',
                'attributes': {
                    'source_content': 'current observation',
                    'target_content': 'latency issue',
                },
            }
        ],
    )
    service = OMTypedProjectionService(
        search_service=search_service,
        graphiti_service=_FakeGraphitiService(_FakeDriver()),
    )

    objects, search_overrides, limits = await service.project(
        query='latency issue',
        effective_group_ids=['s1_observational_memory'],
        object_types=set(),
        max_results=5,
        query_mode='all',
    )

    assert len(objects) == 2
    assert any(isinstance(obj, Episode) for obj in objects)
    assert any(isinstance(obj, StateFact) for obj in objects)
    assert set(search_overrides) == {
        'om_episode:s1_observational_memory:node_current',
        'om_state:s1_observational_memory:rel_current',
    }
    assert limits == {
        'enabled': True,
        'reason': 'projected',
        'groups_considered': ['s1_observational_memory'],
        'episodes_projected': 1,
        'state_projected': 1,
        'max_results': 5,
        'history_mode': False,
    }
