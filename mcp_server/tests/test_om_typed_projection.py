"""Tests for the simplified OMTypedProjectionService (Phase 2).

After the Phase 2 refactor, OMTypedProjectionService:
- Only projects unpromoted OM nodes as provisional episodes
- Never produces StateFact objects (state is ledger-canonical only)
- Does not perform topology/lineage reconstruction
"""

import pytest

from mcp_server.src.models.typed_memory import Episode
from mcp_server.src.services.om_typed_projection import OMTypedProjectionService


class _FakeSearchService:
    def __init__(self, *, node_rows=None):
        self.node_rows = node_rows or []

    def includes_observational_memory(self, group_ids):
        return True

    def _om_groups_in_scope(self, group_ids):
        return list(group_ids or ['s1_observational_memory'])

    async def search_observational_nodes(self, **_kwargs):
        return list(self.node_rows)


class _FakeGraphitiService:
    class config:
        class database:
            provider = 'neo4j'

    async def get_client(self):
        return self


# ---------------------------------------------------------------------------
# Core contract: provisional episodes for unpromoted OM nodes
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_projects_unpromoted_nodes_as_provisional_episodes():
    """Each OM node becomes a standalone provisional episode."""
    search = _FakeSearchService(
        node_rows=[
            {
                'uuid': 'node-1',
                'name': 'Heap allocation cap raised',
                'summary': 'Heap allocation cap raised to 4 GiB',
                'group_id': 's1_observational_memory',
                'created_at': '2026-03-01T10:00:00Z',
                'attributes': {
                    'status': 'open',
                    'semantic_domain': 'infrastructure',
                },
            },
            {
                'uuid': 'node-2',
                'name': 'GC frequency tuned',
                'summary': 'GC frequency tuned to 100ms intervals',
                'group_id': 's1_observational_memory',
                'created_at': '2026-03-02T10:00:00Z',
                'attributes': {
                    'status': 'open',
                    'semantic_domain': 'infrastructure',
                },
            },
        ]
    )
    service = OMTypedProjectionService(
        search_service=search,
        graphiti_service=_FakeGraphitiService(),
    )

    objects, overrides, limits = await service.project(
        query='heap',
        effective_group_ids=['s1_observational_memory'],
        object_types=set(),
        max_results=10,
    )

    assert len(objects) == 2
    for obj in objects:
        assert isinstance(obj, Episode)
        assert 'provisional' in obj.annotations
        assert 'unpromoted' in obj.annotations
        assert 'om_native' in obj.annotations
        assert obj.version == 1
        assert obj.is_current is True
        assert obj.parent_id is None
        assert obj.source_lane == 's1_observational_memory'
        assert obj.source_key.startswith('om:s1_observational_memory:node:')
        assert obj.history_meta['lineage_basis'] == 'provisional'
        assert obj.history_meta['derivation_level'] == 'provisional'
        assert len(obj.evidence_refs) == 1
        assert obj.evidence_refs[0].source_system == 'om'

    assert objects[0].object_id == 'om_episode:s1_observational_memory:node-1'
    assert objects[1].object_id == 'om_episode:s1_observational_memory:node-2'
    assert limits['enabled'] is True
    assert limits['reason'] == 'provisional_projection'
    assert limits['episodes_projected'] == 2
    assert limits['state_projected'] == 0


@pytest.mark.anyio
async def test_no_state_facts_are_ever_projected():
    """Projection never produces StateFact objects — state is ledger-only."""
    search = _FakeSearchService(
        node_rows=[
            {
                'uuid': 'node-1',
                'summary': 'Some content',
                'group_id': 's1_observational_memory',
                'created_at': '2026-03-01T10:00:00Z',
                'attributes': {},
            },
        ]
    )
    service = OMTypedProjectionService(
        search_service=search,
        graphiti_service=_FakeGraphitiService(),
    )

    objects, _, limits = await service.project(
        query='content',
        effective_group_ids=['s1_observational_memory'],
        object_types=set(),
        max_results=10,
    )

    for obj in objects:
        assert obj.object_type == 'episode', (
            f'Projection must never produce {obj.object_type} — state is ledger-canonical only'
        )
    assert limits['state_projected'] == 0


@pytest.mark.anyio
async def test_state_fact_request_skips_projection():
    """If caller only wants state_fact, projection returns nothing."""
    search = _FakeSearchService(
        node_rows=[
            {
                'uuid': 'node-1',
                'summary': 'Some content',
                'group_id': 's1_observational_memory',
                'created_at': '2026-03-01T10:00:00Z',
                'attributes': {},
            },
        ]
    )
    service = OMTypedProjectionService(
        search_service=search,
        graphiti_service=_FakeGraphitiService(),
    )

    objects, _, limits = await service.project(
        query='content',
        effective_group_ids=['s1_observational_memory'],
        object_types={'state_fact'},
        max_results=10,
    )

    assert objects == []
    assert limits['enabled'] is False
    assert limits['reason'] == 'episodes_not_requested'


# ---------------------------------------------------------------------------
# Edge cases and guard rails
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_no_graphiti_service_returns_disabled():
    service = OMTypedProjectionService(
        search_service=_FakeSearchService(),
        graphiti_service=None,
    )
    objects, _, limits = await service.project(
        query='anything',
        effective_group_ids=['s1_observational_memory'],
        object_types=set(),
        max_results=10,
    )
    assert objects == []
    assert limits['enabled'] is False
    assert limits['reason'] == 'graphiti_service_unavailable'


@pytest.mark.anyio
async def test_non_om_scope_returns_disabled():
    class _NonOMSearch(_FakeSearchService):
        def includes_observational_memory(self, group_ids):
            return False

    service = OMTypedProjectionService(
        search_service=_NonOMSearch(),
        graphiti_service=_FakeGraphitiService(),
    )
    objects, _, limits = await service.project(
        query='anything',
        effective_group_ids=['some_other_lane'],
        object_types=set(),
        max_results=10,
    )
    assert objects == []
    assert limits['enabled'] is False
    assert limits['reason'] == 'om_not_in_scope'


@pytest.mark.anyio
async def test_empty_search_results_returns_enabled_with_zero_episodes():
    service = OMTypedProjectionService(
        search_service=_FakeSearchService(node_rows=[]),
        graphiti_service=_FakeGraphitiService(),
    )
    objects, _, limits = await service.project(
        query='nonexistent',
        effective_group_ids=['s1_observational_memory'],
        object_types=set(),
        max_results=10,
    )
    assert objects == []
    assert limits['enabled'] is True
    assert limits['episodes_projected'] == 0


@pytest.mark.anyio
async def test_skips_rows_with_missing_identifiers():
    search = _FakeSearchService(
        node_rows=[
            {'uuid': '', 'group_id': 's1_observational_memory', 'attributes': {}},
            {'uuid': 'valid', 'group_id': '', 'attributes': {}},
            {'uuid': 'good', 'group_id': 's1_observational_memory',
             'summary': 'Valid node', 'created_at': '2026-03-01T10:00:00Z',
             'attributes': {}},
        ]
    )
    service = OMTypedProjectionService(
        search_service=search,
        graphiti_service=_FakeGraphitiService(),
    )

    objects, _, limits = await service.project(
        query='anything',
        effective_group_ids=['s1_observational_memory'],
        object_types=set(),
        max_results=10,
    )

    assert len(objects) == 1
    assert objects[0].object_id == 'om_episode:s1_observational_memory:good'


@pytest.mark.anyio
async def test_search_text_overrides_populated():
    search = _FakeSearchService(
        node_rows=[
            {
                'uuid': 'node-1',
                'summary': 'Important observation about memory allocation',
                'group_id': 's1_observational_memory',
                'created_at': '2026-03-01T10:00:00Z',
                'attributes': {'semantic_domain': 'infrastructure'},
            },
        ]
    )
    service = OMTypedProjectionService(
        search_service=search,
        graphiti_service=_FakeGraphitiService(),
    )

    objects, overrides, _ = await service.project(
        query='memory',
        effective_group_ids=['s1_observational_memory'],
        object_types=set(),
        max_results=10,
    )

    assert len(overrides) == 1
    search_text = overrides[objects[0].object_id]
    assert 'Important observation about memory allocation' in search_text
    assert 'provisional' in search_text
    assert 's1_observational_memory' in search_text


@pytest.mark.anyio
async def test_experimental_om_group_surfaces_through_projection():
    """Experimental OM-native groups produce provisional episodes."""
    search = _FakeSearchService(
        node_rows=[
            {
                'uuid': 'exp-node-1',
                'summary': 'Experimental observation',
                'group_id': 'ontbk15batch_20260310_om_f',
                'created_at': '2026-03-10T10:00:00Z',
                'attributes': {},
            },
        ]
    )
    service = OMTypedProjectionService(
        search_service=search,
        graphiti_service=_FakeGraphitiService(),
    )

    objects, _, limits = await service.project(
        query='experimental',
        effective_group_ids=['ontbk15batch_20260310_om_f'],
        object_types=set(),
        max_results=10,
    )

    assert len(objects) == 1
    assert objects[0].source_lane == 'ontbk15batch_20260310_om_f'
    assert objects[0].source_key == 'om:ontbk15batch_20260310_om_f:node:exp-node-1'
    assert limits['episodes_projected'] == 1


@pytest.mark.anyio
async def test_source_key_format_enables_coverage_tracking():
    """source_key uses om:{group}:node:{id} format for coverage suppression compat."""
    search = _FakeSearchService(
        node_rows=[
            {
                'uuid': 'tracked-node',
                'summary': 'Trackable',
                'group_id': 's1_observational_memory',
                'created_at': '2026-03-01T10:00:00Z',
                'attributes': {},
            },
        ]
    )
    service = OMTypedProjectionService(
        search_service=search,
        graphiti_service=_FakeGraphitiService(),
    )

    objects, _, _ = await service.project(
        query='track',
        effective_group_ids=['s1_observational_memory'],
        object_types=set(),
        max_results=10,
    )

    assert objects[0].source_key == 'om:s1_observational_memory:node:tracked-node'
    # Verify the source_key can be parsed by _om_node_key_from_source_key
    parts = objects[0].source_key.split(':')
    assert parts[0] == 'om'
    assert parts[1] == 's1_observational_memory'
    assert parts[2] == 'node'
    assert parts[3] == 'tracked-node'
