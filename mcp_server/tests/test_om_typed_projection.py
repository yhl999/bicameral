from types import SimpleNamespace

import pytest

from mcp_server.src.models.typed_memory import Episode, StateFact
from mcp_server.src.services.om_typed_projection import OMTypedProjectionService


class _FakeSearchService:
    def __init__(self, *, node_rows=None, fact_rows=None, neighborhood_rows=None):
        self.node_rows = node_rows or []
        self.fact_rows = fact_rows or []
        self.neighborhood_rows = neighborhood_rows or {}

    def includes_observational_memory(self, group_ids):
        return True

    def _om_groups_in_scope(self, group_ids):
        return list(group_ids or ['s1_observational_memory'])

    async def search_observational_nodes(self, **_kwargs):
        return list(self.node_rows)

    async def search_observational_facts(self, *, center_node_uuid=None, **_kwargs):
        if center_node_uuid is not None:
            return list(self.neighborhood_rows.get(center_node_uuid, []))
        return list(self.fact_rows)


class _FakeDriver:
    def __init__(self, records_by_seed=None):
        self.records_by_seed = records_by_seed or {}
        self.calls = []

    async def execute_query(self, query, **params):
        self.calls.append({'query': query, 'params': params})
        seed = params.get('seed_node_id')
        return list(self.records_by_seed.get(seed, [])), None, None


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
        records_by_seed={
            'node_v3': [
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
        }
    )
    service = OMTypedProjectionService(
        search_service=search_service,
        graphiti_service=_FakeGraphitiService(driver),
    )

    objects, search_overrides, limits = await service.project(
        query='what changed about the training drink',
        effective_group_ids=['s1_observational_memory'],
        object_types={'episode'},
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
    assert by_id[v1_id].lifecycle_status == 'superseded'
    assert 'history_topology:linear' in by_id[v1_id].annotations

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
    assert by_id[v3_id].lifecycle_status == 'asserted'

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
        'history_relation_lineages_projected': 0,
        'history_state_projection_supported': False,
        'unsupported_object_types': [],
        'skipped_candidates': [],
        'history_topology_counts': {'linear': 1},
        'history_relation_candidates': 0,
    }
    assert driver.calls, 'history projection should query Neo4j for supersession lineage'


@pytest.mark.anyio
async def test_history_projection_derives_relation_history_and_closure_invalidation():
    search_service = _FakeSearchService(
        node_rows=[
            {
                'uuid': 'plan_v2',
                'name': 'Fix plan current',
                'summary': 'Ship the cache fix',
                'group_id': 's1_observational_memory',
                'created_at': '2026-03-05T00:00:00Z',
                'attributes': {'status': 'active', 'semantic_domain': 'observational_memory'},
            }
        ],
        fact_rows=[
            {
                'uuid': 'rel_plan_v2_issue',
                'name': 'ADDRESSES',
                'fact': 'ADDRESSES: ship cache fix -> latency issue',
                'group_id': 's1_observational_memory',
                'source_node_uuid': 'plan_v2',
                'target_node_uuid': 'issue_1',
                'created_at': '2026-03-05T00:00:00Z',
                'attributes': {
                    'source_content': 'Ship cache fix',
                    'target_content': 'Latency issue',
                },
            },
            {
                'uuid': 'rel_fix_resolves_issue',
                'name': 'RESOLVES',
                'fact': 'RESOLVES: ship cache fix -> latency issue',
                'group_id': 's1_observational_memory',
                'source_node_uuid': 'fix_1',
                'target_node_uuid': 'issue_1',
                'created_at': '2026-03-06T00:00:00Z',
                'attributes': {
                    'source_content': 'Ship cache fix',
                    'target_content': 'Latency issue',
                },
            },
        ],
        neighborhood_rows={
            'plan_v1': [
                {
                    'uuid': 'rel_plan_v1_issue',
                    'name': 'ADDRESSES',
                    'fact': 'ADDRESSES: investigate latency -> latency issue',
                    'group_id': 's1_observational_memory',
                    'source_node_uuid': 'plan_v1',
                    'target_node_uuid': 'issue_1',
                    'created_at': '2026-03-01T00:00:00Z',
                    'attributes': {
                        'source_content': 'Investigate latency spike',
                        'target_content': 'Latency issue',
                    },
                }
            ],
            'plan_v2': [
                {
                    'uuid': 'rel_plan_v2_issue',
                    'name': 'ADDRESSES',
                    'fact': 'ADDRESSES: ship cache fix -> latency issue',
                    'group_id': 's1_observational_memory',
                    'source_node_uuid': 'plan_v2',
                    'target_node_uuid': 'issue_1',
                    'created_at': '2026-03-05T00:00:00Z',
                    'attributes': {
                        'source_content': 'Ship cache fix',
                        'target_content': 'Latency issue',
                    },
                }
            ],
            'issue_1': [
                {
                    'uuid': 'rel_fix_resolves_issue',
                    'name': 'RESOLVES',
                    'fact': 'RESOLVES: ship cache fix -> latency issue',
                    'group_id': 's1_observational_memory',
                    'source_node_uuid': 'fix_1',
                    'target_node_uuid': 'issue_1',
                    'created_at': '2026-03-06T00:00:00Z',
                    'attributes': {
                        'source_content': 'Ship cache fix',
                        'target_content': 'Latency issue',
                    },
                }
            ],
            'fix_1': [
                {
                    'uuid': 'rel_fix_resolves_issue',
                    'name': 'RESOLVES',
                    'fact': 'RESOLVES: ship cache fix -> latency issue',
                    'group_id': 's1_observational_memory',
                    'source_node_uuid': 'fix_1',
                    'target_node_uuid': 'issue_1',
                    'created_at': '2026-03-06T00:00:00Z',
                    'attributes': {
                        'source_content': 'Ship cache fix',
                        'target_content': 'Latency issue',
                    },
                }
            ],
        },
    )
    driver = _FakeDriver(
        records_by_seed={
            'plan_v2': [
                {
                    'node_id': 'plan_v1',
                    'uuid': 'plan_v1',
                    'group_id': 's1_observational_memory',
                    'content': 'Investigate latency spike',
                    'created_at': '2026-03-01T00:00:00Z',
                    'status': 'open',
                    'semantic_domain': 'observational_memory',
                    'supersedes': [],
                },
                {
                    'node_id': 'plan_v2',
                    'uuid': 'plan_v2',
                    'group_id': 's1_observational_memory',
                    'content': 'Ship cache fix',
                    'created_at': '2026-03-05T00:00:00Z',
                    'status': 'active',
                    'semantic_domain': 'observational_memory',
                    'supersedes': [
                        {'target_id': 'plan_v1', 'created_at': '2026-03-05T00:00:00Z', 'relation_uuid': 'rel_plan_v2_v1'}
                    ],
                },
            ],
            'issue_1': [
                {
                    'node_id': 'issue_1',
                    'uuid': 'issue_1',
                    'group_id': 's1_observational_memory',
                    'content': 'Latency issue',
                    'created_at': '2026-03-01T00:00:00Z',
                    'status': 'open',
                    'semantic_domain': 'observational_memory',
                    'supersedes': [],
                }
            ],
            'fix_1': [
                {
                    'node_id': 'fix_1',
                    'uuid': 'fix_1',
                    'group_id': 's1_observational_memory',
                    'content': 'Ship cache fix',
                    'created_at': '2026-03-06T00:00:00Z',
                    'status': 'active',
                    'semantic_domain': 'observational_memory',
                    'supersedes': [],
                }
            ],
        }
    )
    service = OMTypedProjectionService(
        search_service=search_service,
        graphiti_service=_FakeGraphitiService(driver),
    )

    objects, _search_overrides, limits = await service.project(
        query='what changed for the latency issue',
        effective_group_ids=['s1_observational_memory'],
        object_types={'state_fact'},
        max_results=5,
        query_mode='history',
    )

    assert len(objects) == 3
    assert all(isinstance(obj, StateFact) for obj in objects)

    by_id = {obj.object_id: obj for obj in objects}
    addresses_v1 = by_id['om_state:s1_observational_memory:rel_plan_v1_issue']
    addresses_v2 = by_id['om_state:s1_observational_memory:rel_plan_v2_issue']
    resolves = by_id['om_state:s1_observational_memory:rel_fix_resolves_issue']

    assert addresses_v1.root_id == addresses_v2.root_id
    assert addresses_v1.version == 1
    assert addresses_v1.is_current is False
    assert addresses_v1.invalid_at == '2026-03-05T00:00:00Z'
    assert addresses_v1.superseded_by == addresses_v2.object_id
    assert addresses_v1.lifecycle_status == 'superseded'
    assert addresses_v1.value['om_history']['lineage_basis'] == 'om_relation_anchor_history'
    assert addresses_v1.value['om_history']['topology'] == 'linear'

    assert addresses_v2.version == 2
    assert addresses_v2.is_current is False
    assert addresses_v2.invalid_at == '2026-03-06T00:00:00Z'
    assert addresses_v2.superseded_by is None
    assert addresses_v2.lifecycle_status == 'invalidated'
    assert addresses_v2.value['om_history']['invalidation_reason'] == 'target_node_invalidated'

    assert resolves.is_current is True
    assert resolves.invalid_at is None
    assert resolves.value['om_history']['topology'] == 'singleton'

    assert limits['state_projected'] == 3
    assert limits['history_relation_lineages_projected'] == 2
    assert limits['history_state_projection_supported'] is True


@pytest.mark.anyio
async def test_history_projection_surfaces_relation_branching_without_fake_linear_chain():
    search_service = _FakeSearchService(
        node_rows=[
            {
                'uuid': 'plan_v2a',
                'name': 'Branching fix plan',
                'summary': 'two competing fixes exist',
                'group_id': 's1_observational_memory',
                'created_at': '2026-03-05T00:00:00Z',
                'attributes': {'status': 'active', 'semantic_domain': 'observational_memory'},
            }
        ],
        neighborhood_rows={
            'plan_v1': [
                {
                    'uuid': 'rel_plan_v1_issue',
                    'name': 'ADDRESSES',
                    'fact': 'ADDRESSES: investigate latency -> latency issue',
                    'group_id': 's1_observational_memory',
                    'source_node_uuid': 'plan_v1',
                    'target_node_uuid': 'issue_1',
                    'created_at': '2026-03-01T00:00:00Z',
                    'attributes': {
                        'source_content': 'Investigate latency',
                        'target_content': 'Latency issue',
                    },
                }
            ],
            'plan_v2a': [
                {
                    'uuid': 'rel_plan_v2a_issue',
                    'name': 'ADDRESSES',
                    'fact': 'ADDRESSES: ship cache fix -> latency issue',
                    'group_id': 's1_observational_memory',
                    'source_node_uuid': 'plan_v2a',
                    'target_node_uuid': 'issue_1',
                    'created_at': '2026-03-05T00:00:00Z',
                    'attributes': {
                        'source_content': 'Ship cache fix',
                        'target_content': 'Latency issue',
                    },
                }
            ],
            'plan_v2b': [
                {
                    'uuid': 'rel_plan_v2b_issue',
                    'name': 'ADDRESSES',
                    'fact': 'ADDRESSES: rewrite cache layer -> latency issue',
                    'group_id': 's1_observational_memory',
                    'source_node_uuid': 'plan_v2b',
                    'target_node_uuid': 'issue_1',
                    'created_at': '2026-03-06T00:00:00Z',
                    'attributes': {
                        'source_content': 'Rewrite cache layer',
                        'target_content': 'Latency issue',
                    },
                }
            ],
            'issue_1': [
                {
                    'uuid': 'rel_plan_v1_issue',
                    'name': 'ADDRESSES',
                    'fact': 'ADDRESSES: investigate latency -> latency issue',
                    'group_id': 's1_observational_memory',
                    'source_node_uuid': 'plan_v1',
                    'target_node_uuid': 'issue_1',
                    'created_at': '2026-03-01T00:00:00Z',
                    'attributes': {
                        'source_content': 'Investigate latency',
                        'target_content': 'Latency issue',
                    },
                },
                {
                    'uuid': 'rel_plan_v2a_issue',
                    'name': 'ADDRESSES',
                    'fact': 'ADDRESSES: ship cache fix -> latency issue',
                    'group_id': 's1_observational_memory',
                    'source_node_uuid': 'plan_v2a',
                    'target_node_uuid': 'issue_1',
                    'created_at': '2026-03-05T00:00:00Z',
                    'attributes': {
                        'source_content': 'Ship cache fix',
                        'target_content': 'Latency issue',
                    },
                },
                {
                    'uuid': 'rel_plan_v2b_issue',
                    'name': 'ADDRESSES',
                    'fact': 'ADDRESSES: rewrite cache layer -> latency issue',
                    'group_id': 's1_observational_memory',
                    'source_node_uuid': 'plan_v2b',
                    'target_node_uuid': 'issue_1',
                    'created_at': '2026-03-06T00:00:00Z',
                    'attributes': {
                        'source_content': 'Rewrite cache layer',
                        'target_content': 'Latency issue',
                    },
                },
            ],
        },
    )
    driver = _FakeDriver(
        records_by_seed={
            'plan_v2a': [
                {
                    'node_id': 'plan_v1',
                    'uuid': 'plan_v1',
                    'group_id': 's1_observational_memory',
                    'content': 'Investigate latency',
                    'created_at': '2026-03-01T00:00:00Z',
                    'status': 'open',
                    'semantic_domain': 'observational_memory',
                    'supersedes': [],
                },
                {
                    'node_id': 'plan_v2a',
                    'uuid': 'plan_v2a',
                    'group_id': 's1_observational_memory',
                    'content': 'Ship cache fix',
                    'created_at': '2026-03-05T00:00:00Z',
                    'status': 'active',
                    'semantic_domain': 'observational_memory',
                    'supersedes': [{'target_id': 'plan_v1', 'created_at': '2026-03-05T00:00:00Z', 'relation_uuid': 'rel_plan_v2a_v1'}],
                },
                {
                    'node_id': 'plan_v2b',
                    'uuid': 'plan_v2b',
                    'group_id': 's1_observational_memory',
                    'content': 'Rewrite cache layer',
                    'created_at': '2026-03-06T00:00:00Z',
                    'status': 'active',
                    'semantic_domain': 'observational_memory',
                    'supersedes': [{'target_id': 'plan_v1', 'created_at': '2026-03-06T00:00:00Z', 'relation_uuid': 'rel_plan_v2b_v1'}],
                },
            ],
            'issue_1': [
                {
                    'node_id': 'issue_1',
                    'uuid': 'issue_1',
                    'group_id': 's1_observational_memory',
                    'content': 'Latency issue',
                    'created_at': '2026-03-01T00:00:00Z',
                    'status': 'open',
                    'semantic_domain': 'observational_memory',
                    'supersedes': [],
                }
            ],
        }
    )
    service = OMTypedProjectionService(
        search_service=search_service,
        graphiti_service=_FakeGraphitiService(driver),
    )

    objects, _search_overrides, limits = await service.project(
        query='what changed for latency',
        effective_group_ids=['s1_observational_memory'],
        object_types={'state_fact'},
        max_results=5,
        query_mode='history',
    )

    by_id = {obj.object_id: obj for obj in objects}
    root = by_id['om_state:s1_observational_memory:rel_plan_v1_issue']
    first = by_id['om_state:s1_observational_memory:rel_plan_v2a_issue']
    second = by_id['om_state:s1_observational_memory:rel_plan_v2b_issue']

    assert root.value['om_history']['topology'] == 'branching'
    assert root.superseded_by is None
    assert root.history_meta['topology_flags'][0] == 'branching'
    assert first.history_meta['is_ambiguous'] is True
    assert second.history_meta['is_ambiguous'] is True
    assert first.is_current is True
    assert second.is_current is True
    assert limits['history_relation_lineages_projected'] == 1


@pytest.mark.anyio
async def test_history_projection_prefers_native_relation_invalid_at_when_available():
    search_service = _FakeSearchService(
        node_rows=[
            {
                'uuid': 'plan_v1',
                'name': 'Plan V1',
                'summary': 'legacy plan',
                'group_id': 's1_observational_memory',
                'created_at': '2026-03-01T00:00:00Z',
                'attributes': {'status': 'open', 'semantic_domain': 'observational_memory'},
            }
        ],
        fact_rows=[
            {
                'uuid': 'rel_plan_issue',
                'name': 'ADDRESSES',
                'fact': 'ADDRESSES: legacy plan -> latency issue',
                'group_id': 's1_observational_memory',
                'source_node_uuid': 'plan_v1',
                'target_node_uuid': 'issue_1',
                'created_at': '2026-03-01T00:00:00Z',
                'valid_at': '2026-03-01T00:00:00Z',
                'invalid_at': '2026-03-04T00:00:00Z',
                'attributes': {
                    'source_content': 'Legacy plan',
                    'target_content': 'Latency issue',
                    'relation_properties': {'invalid_at': '2026-03-04T00:00:00Z'},
                },
            }
        ],
        neighborhood_rows={
            'plan_v1': [
                {
                    'uuid': 'rel_plan_issue',
                    'name': 'ADDRESSES',
                    'fact': 'ADDRESSES: legacy plan -> latency issue',
                    'group_id': 's1_observational_memory',
                    'source_node_uuid': 'plan_v1',
                    'target_node_uuid': 'issue_1',
                    'created_at': '2026-03-01T00:00:00Z',
                    'valid_at': '2026-03-01T00:00:00Z',
                    'invalid_at': '2026-03-04T00:00:00Z',
                    'attributes': {
                        'source_content': 'Legacy plan',
                        'target_content': 'Latency issue',
                        'relation_properties': {'invalid_at': '2026-03-04T00:00:00Z'},
                    },
                }
            ],
            'issue_1': [
                {
                    'uuid': 'rel_plan_issue',
                    'name': 'ADDRESSES',
                    'fact': 'ADDRESSES: legacy plan -> latency issue',
                    'group_id': 's1_observational_memory',
                    'source_node_uuid': 'plan_v1',
                    'target_node_uuid': 'issue_1',
                    'created_at': '2026-03-01T00:00:00Z',
                    'valid_at': '2026-03-01T00:00:00Z',
                    'invalid_at': '2026-03-04T00:00:00Z',
                    'attributes': {
                        'source_content': 'Legacy plan',
                        'target_content': 'Latency issue',
                        'relation_properties': {'invalid_at': '2026-03-04T00:00:00Z'},
                    },
                }
            ],
        },
    )
    driver = _FakeDriver(
        records_by_seed={
            'plan_v1': [
                {
                    'node_id': 'plan_v1',
                    'uuid': 'plan_v1',
                    'group_id': 's1_observational_memory',
                    'content': 'Legacy plan',
                    'created_at': '2026-03-01T00:00:00Z',
                    'status': 'open',
                    'semantic_domain': 'observational_memory',
                    'supersedes': [],
                }
            ],
            'issue_1': [
                {
                    'node_id': 'issue_1',
                    'uuid': 'issue_1',
                    'group_id': 's1_observational_memory',
                    'content': 'Latency issue',
                    'created_at': '2026-03-01T00:00:00Z',
                    'status': 'open',
                    'semantic_domain': 'observational_memory',
                    'supersedes': [],
                }
            ],
        }
    )
    service = OMTypedProjectionService(
        search_service=search_service,
        graphiti_service=_FakeGraphitiService(driver),
    )

    objects, _search_overrides, _limits = await service.project(
        query='legacy plan',
        effective_group_ids=['s1_observational_memory'],
        object_types={'state_fact'},
        max_results=5,
        query_mode='history',
    )

    relation = next(obj for obj in objects if obj.object_id == 'om_state:s1_observational_memory:rel_plan_issue')
    assert relation.invalid_at == '2026-03-04T00:00:00Z'
    assert relation.lifecycle_status == 'invalidated'
    assert relation.history_meta['invalidation_basis'] == 'relation_edge_invalid_at'
    assert relation.value['om_history']['derivation_level'] == 'native'



@pytest.mark.anyio
async def test_history_projection_surfaces_branching_topology_without_fabrication():
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
        records_by_seed={
            'node_v2a': [
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
        }
    )
    service = OMTypedProjectionService(
        search_service=search_service,
        graphiti_service=_FakeGraphitiService(driver),
    )

    objects, _search_overrides, limits = await service.project(
        query='what changed',
        effective_group_ids=['s1_observational_memory'],
        object_types={'episode'},
        max_results=5,
        query_mode='history',
    )

    assert len(objects) == 3
    by_id = {obj.object_id: obj for obj in objects}
    root = by_id['om_episode:s1_observational_memory:node_v1']
    first = by_id['om_episode:s1_observational_memory:node_v2a']
    second = by_id['om_episode:s1_observational_memory:node_v2b']

    assert root.is_current is False
    assert root.invalid_at == '2026-03-05T00:00:00Z'
    assert root.superseded_by is None
    assert 'history_topology:branching' in root.annotations
    assert 'history_ambiguous' in root.annotations
    assert first.is_current is True
    assert second.is_current is True
    assert limits['skipped_candidates'] == []
    assert limits['history_topology_counts'] == {'branching': 1}


@pytest.mark.anyio
async def test_history_projection_surfaces_cyclic_topology_as_ambiguous_history():
    search_service = _FakeSearchService(
        node_rows=[
            {
                'uuid': 'cycle_a',
                'name': 'Cycle A',
                'summary': 'malformed cycle start',
                'group_id': 's1_observational_memory',
                'created_at': '2026-03-02T00:00:00Z',
                'attributes': {'status': 'active', 'semantic_domain': 'observational_memory'},
            }
        ]
    )
    driver = _FakeDriver(
        records_by_seed={
            'cycle_a': [
                {
                    'node_id': 'cycle_a',
                    'uuid': 'cycle_a',
                    'group_id': 's1_observational_memory',
                    'content': 'Cycle A',
                    'created_at': '2026-03-02T00:00:00Z',
                    'status': 'open',
                    'semantic_domain': 'observational_memory',
                    'supersedes': [{'target_id': 'cycle_b', 'created_at': '2026-03-02T00:00:00Z'}],
                },
                {
                    'node_id': 'cycle_b',
                    'uuid': 'cycle_b',
                    'group_id': 's1_observational_memory',
                    'content': 'Cycle B',
                    'created_at': '2026-03-03T00:00:00Z',
                    'status': 'open',
                    'semantic_domain': 'observational_memory',
                    'supersedes': [{'target_id': 'cycle_a', 'created_at': '2026-03-03T00:00:00Z'}],
                },
            ]
        }
    )
    service = OMTypedProjectionService(
        search_service=search_service,
        graphiti_service=_FakeGraphitiService(driver),
    )

    objects, _search_overrides, limits = await service.project(
        query='history of malformed cycle',
        effective_group_ids=['s1_observational_memory'],
        object_types={'episode'},
        max_results=5,
        query_mode='history',
    )

    assert len(objects) == 2
    assert all(isinstance(obj, Episode) for obj in objects)
    assert all('history_topology:cyclic' in obj.annotations for obj in objects)
    assert all('history_ambiguous' in obj.annotations for obj in objects)
    assert all(obj.is_current is False for obj in objects)
    assert limits['skipped_candidates'] == []
    assert limits['history_topology_counts'] == {'cyclic': 1}


@pytest.mark.anyio
async def test_non_history_projection_keeps_currentness_semantics_for_current_and_all_modes():
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
        neighborhood_rows={
            'node_current': [
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
            'issue_1': [
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
        },
    )
    driver = _FakeDriver(
        records_by_seed={
            'node_current': [
                {
                    'node_id': 'node_current',
                    'uuid': 'node_current',
                    'group_id': 's1_observational_memory',
                    'content': 'current observation',
                    'created_at': '2026-03-09T00:00:00Z',
                    'status': 'active',
                    'semantic_domain': 'observational_memory',
                    'supersedes': [],
                }
            ],
            'issue_1': [
                {
                    'node_id': 'issue_1',
                    'uuid': 'issue_1',
                    'group_id': 's1_observational_memory',
                    'content': 'latency issue',
                    'created_at': '2026-03-08T00:00:00Z',
                    'status': 'open',
                    'semantic_domain': 'observational_memory',
                    'supersedes': [],
                }
            ],
        }
    )
    service = OMTypedProjectionService(
        search_service=search_service,
        graphiti_service=_FakeGraphitiService(driver),
    )

    objects, search_overrides, limits = await service.project(
        query='latency issue',
        effective_group_ids=['s1_observational_memory'],
        object_types=set(),
        max_results=5,
        query_mode='all',
    )

    assert len(objects) == 3
    episodes = [obj for obj in objects if isinstance(obj, Episode)]
    state = [obj for obj in objects if isinstance(obj, StateFact)]
    issue_episode = next(obj for obj in episodes if obj.object_id.endswith(':issue_1'))
    resolve_state = next(obj for obj in state if obj.object_id.endswith(':rel_current'))

    assert issue_episode.is_current is False
    assert issue_episode.invalid_at == '2026-03-09T00:00:00Z'
    assert issue_episode.lifecycle_status == 'invalidated'
    assert resolve_state.is_current is True
    assert set(search_overrides) >= {
        'om_episode:s1_observational_memory:node_current',
        'om_episode:s1_observational_memory:issue_1',
        'om_state:s1_observational_memory:rel_current',
    }
    assert limits['reason'] == 'projected'
    assert limits['history_mode'] is False


@pytest.mark.anyio
async def test_history_projection_prefers_native_node_lifecycle_truth() -> None:
    search_service = _FakeSearchService(
        node_rows=[
            {
                'uuid': 'issue_1',
                'name': 'Latency issue',
                'summary': 'Latency issue',
                'group_id': 's1_observational_memory',
                'created_at': '2026-03-01T00:00:00Z',
                'attributes': {'status': 'closed', 'semantic_domain': 'observational_memory'},
            }
        ]
    )
    driver = _FakeDriver(
        records_by_seed={
            'issue_1': [
                {
                    'node_id': 'issue_1',
                    'uuid': 'issue_1',
                    'group_id': 's1_observational_memory',
                    'content': 'Latency issue',
                    'valid_at': '2026-03-01T00:00:00Z',
                    'created_at': '2026-03-01T00:00:00Z',
                    'invalid_at': '2026-03-07T00:00:00Z',
                    'lifecycle_status': 'invalidated',
                    'status': 'closed',
                    'previous_status': 'monitoring',
                    'transition_cause': 'closed_by_convergence',
                    'semantic_domain': 'observational_memory',
                    'lineage_root_id': 'issue_1',
                    'lineage_parent_id': '',
                    'superseded_by_node_id': '',
                    'supersedes': [],
                }
            ]
        }
    )
    service = OMTypedProjectionService(
        search_service=search_service,
        graphiti_service=_FakeGraphitiService(driver),
    )

    objects, _, limits = await service.project(
        query='latency issue',
        effective_group_ids=['s1_observational_memory'],
        object_types={'episode'},
        max_results=5,
        query_mode='history',
    )

    assert len(objects) == 1
    episode = objects[0]
    assert isinstance(episode, Episode)
    assert episode.object_id == 'om_episode:s1_observational_memory:issue_1'
    assert episode.root_id == 'om_episode:s1_observational_memory:issue_1'
    assert episode.valid_at == '2026-03-01T00:00:00Z'
    assert episode.invalid_at == '2026-03-07T00:00:00Z'
    assert episode.is_current is False
    assert episode.lifecycle_status == 'invalidated'
    assert episode.history_meta['lineage_basis'] == 'native_node_lifecycle'
    assert episode.history_meta['transition_reason'] == 'closed_by_convergence'
    assert limits['history_lineages_projected'] == 1


@pytest.mark.anyio
async def test_history_projection_prefers_native_relation_lineage_truth() -> None:
    search_service = _FakeSearchService(
        node_rows=[
            {
                'uuid': 'plan_v2',
                'name': 'Current plan',
                'summary': 'Ship cache fix with retry guard',
                'group_id': 's1_observational_memory',
                'created_at': '2026-03-05T00:00:00Z',
                'attributes': {'status': 'active', 'semantic_domain': 'observational_memory'},
            }
        ],
        fact_rows=[
            {
                'uuid': 'rel_old',
                'name': 'ADDRESSES',
                'fact': 'ADDRESSES: old plan -> latency issue',
                'group_id': 's1_observational_memory',
                'source_node_uuid': 'plan_v1',
                'target_node_uuid': 'issue_1',
                'created_at': '2026-03-02T00:00:00Z',
                'valid_at': '2026-03-02T00:00:00Z',
                'invalid_at': '2026-03-05T00:00:00Z',
                'attributes': {
                    'source_content': 'Ship cache fix',
                    'target_content': 'Latency issue',
                    'relation_properties': {
                        'relation_root_id': 'rel_old',
                        'superseded_by_relation_id': 'rel_new',
                        'lifecycle_status': 'superseded',
                        'transition_cause': 'relation_replaced_by_endpoint_supersession',
                        'transition_basis': 'native_endpoint_lineage',
                        'lineage_basis': 'endpoint_supersession',
                        'lineage_topology': 'linear',
                    },
                },
            },
            {
                'uuid': 'rel_new',
                'name': 'ADDRESSES',
                'fact': 'ADDRESSES: new plan -> latency issue',
                'group_id': 's1_observational_memory',
                'source_node_uuid': 'plan_v2',
                'target_node_uuid': 'issue_1',
                'created_at': '2026-03-05T00:00:00Z',
                'valid_at': '2026-03-05T00:00:00Z',
                'attributes': {
                    'source_content': 'Ship cache fix with retry guard',
                    'target_content': 'Latency issue',
                    'relation_properties': {
                        'relation_root_id': 'rel_old',
                        'lineage_parent_relation_id': 'rel_old',
                        'lifecycle_status': 'asserted',
                        'transition_cause': 'relation_asserted',
                        'transition_basis': 'chunk_assertion',
                        'lineage_basis': 'endpoint_supersession',
                        'lineage_topology': 'linear',
                    },
                },
            },
        ],
    )
    driver = _FakeDriver(
        records_by_seed={
            'plan_v2': [
                {
                    'node_id': 'plan_v1',
                    'uuid': 'plan_v1',
                    'group_id': 's1_observational_memory',
                    'content': 'Ship cache fix',
                    'valid_at': '2026-03-02T00:00:00Z',
                    'created_at': '2026-03-02T00:00:00Z',
                    'invalid_at': '2026-03-05T00:00:00Z',
                    'lifecycle_status': 'superseded',
                    'status': 'open',
                    'previous_status': '',
                    'transition_cause': 'superseded_by_newer_node',
                    'semantic_domain': 'observational_memory',
                    'lineage_root_id': 'plan_v1',
                    'lineage_parent_id': '',
                    'superseded_by_node_id': 'plan_v2',
                    'supersedes': [],
                },
                {
                    'node_id': 'plan_v2',
                    'uuid': 'plan_v2',
                    'group_id': 's1_observational_memory',
                    'content': 'Ship cache fix with retry guard',
                    'valid_at': '2026-03-05T00:00:00Z',
                    'created_at': '2026-03-05T00:00:00Z',
                    'invalid_at': None,
                    'lifecycle_status': 'asserted',
                    'status': 'open',
                    'previous_status': '',
                    'transition_cause': 'node_asserted',
                    'semantic_domain': 'observational_memory',
                    'lineage_root_id': 'plan_v1',
                    'lineage_parent_id': 'plan_v1',
                    'superseded_by_node_id': '',
                    'supersedes': [
                        {
                            'target_id': 'plan_v1',
                            'created_at': '2026-03-05T00:00:00Z',
                            'relation_uuid': 'rel_plan_v2_v1',
                        }
                    ],
                },
            ],
            'issue_1': [
                {
                    'node_id': 'issue_1',
                    'uuid': 'issue_1',
                    'group_id': 's1_observational_memory',
                    'content': 'Latency issue',
                    'valid_at': '2026-03-01T00:00:00Z',
                    'created_at': '2026-03-01T00:00:00Z',
                    'invalid_at': None,
                    'lifecycle_status': 'asserted',
                    'status': 'open',
                    'previous_status': '',
                    'transition_cause': 'node_asserted',
                    'semantic_domain': 'observational_memory',
                    'lineage_root_id': 'issue_1',
                    'lineage_parent_id': '',
                    'superseded_by_node_id': '',
                    'supersedes': [],
                }
            ],
        }
    )
    service = OMTypedProjectionService(
        search_service=search_service,
        graphiti_service=_FakeGraphitiService(driver),
    )

    objects, _, limits = await service.project(
        query='cache fix',
        effective_group_ids=['s1_observational_memory'],
        object_types={'state_fact'},
        max_results=5,
        query_mode='history',
    )

    assert len(objects) == 2
    assert all(isinstance(obj, StateFact) for obj in objects)
    by_id = {obj.object_id: obj for obj in objects}
    old_state = by_id['om_state:s1_observational_memory:rel_old']
    new_state = by_id['om_state:s1_observational_memory:rel_new']

    assert old_state.root_id == 'om_state:s1_observational_memory:rel_old'
    assert old_state.superseded_by == 'om_state:s1_observational_memory:rel_new'
    assert old_state.lifecycle_status == 'superseded'
    assert old_state.history_meta['lineage_basis'] == 'endpoint_supersession'
    assert old_state.history_meta['derivation_level'] == 'native'
    assert new_state.root_id == 'om_state:s1_observational_memory:rel_old'
    assert new_state.parent_id == 'om_state:s1_observational_memory:rel_old'
    assert new_state.is_current is True
    assert limits['history_relation_lineages_projected'] == 1
