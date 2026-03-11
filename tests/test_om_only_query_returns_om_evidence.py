import asyncio

from tests.helpers_mcp_import import load_search_service

SearchService = load_search_service().SearchService


class _FakeNeo4jService:
    async def search_om_nodes(self, *_args, **_kwargs):
        return [
            {
                'uuid': 'om-node-1',
                'content': 'Observed recurring morning routine',
                'created_at': '2026-03-05T00:00:00Z',
                'group_id': 's1_observational_memory',
                'status': 'active',
            }
        ]

    async def search_om_facts(self, *_args, **_kwargs):
        return [
            {
                'uuid': 'om-fact-1',
                'relation_type': 'observed_pattern',
                'source_node_id': 'om-node-1',
                'target_node_id': 'om-node-2',
                'source_content': 'Morning workout block before 10:30',
                'target_content': 'Daily routine adherence',
                'created_at': '2026-03-05T00:00:00Z',
                'group_id': 's1_observational_memory',
            }
        ]


class _FakeClient:
    driver = object()


class _FakeGraphitiService:
    class config:  # noqa: D401 - minimal shim for provider lookup
        class database:
            provider = 'neo4j'

    async def get_client(self):
        return _FakeClient()


def test_om_only_scope_returns_om_evidence():
    async def _run():
        service = SearchService(neo4j_service=_FakeNeo4jService())
        graphiti = _FakeGraphitiService()

        facts = await service.search_observational_facts(
            graphiti_service=graphiti,
            query='What recurring observations exist?',
            group_ids=['s1_observational_memory'],
            max_facts=5,
            center_node_uuid=None,
        )

        assert len(facts) == 1
        fact = facts[0]
        assert fact['group_id'] == 's1_observational_memory'
        assert fact['attributes']['source'] == 'om_primitive'

    asyncio.run(_run())


class _SameUuidAcrossGroupsNeo4jService:
    async def search_om_nodes(self, *_args, **kwargs):
        group_id = kwargs['group_id']
        return [
            {
                'uuid': 'shared-node',
                'content': f'Observed recurring morning routine in {group_id}',
                'created_at': '2026-03-05T00:00:00Z',
                'group_id': group_id,
                'status': 'active',
            }
        ]

    async def search_om_facts(self, *_args, **kwargs):
        group_id = kwargs['group_id']
        return [
            {
                'uuid': 'shared-fact',
                'relation_type': 'observed_pattern',
                'source_node_id': 'shared-node',
                'target_node_id': 'om-node-2',
                'source_content': f'Morning workout block before 10:30 in {group_id}',
                'target_content': 'Daily routine adherence',
                'created_at': '2026-03-05T00:00:00Z',
                'group_id': group_id,
            }
        ]


def test_multi_group_scope_keeps_distinct_om_nodes_with_same_uuid():
    async def _run():
        service = SearchService(neo4j_service=_SameUuidAcrossGroupsNeo4jService())
        graphiti = _FakeGraphitiService()

        nodes = await service.search_observational_nodes(
            graphiti_service=graphiti,
            query='routine',
            group_ids=['s1_observational_memory', 'ontbk15batch_20260310_om_f'],
            max_nodes=5,
            entity_types=['OMNode'],
        )

        assert len(nodes) == 2
        assert [(node['group_id'], node['uuid']) for node in nodes] == [
            ('ontbk15batch_20260310_om_f', 'shared-node'),
            ('s1_observational_memory', 'shared-node'),
        ]

    asyncio.run(_run())


def test_multi_group_scope_keeps_distinct_om_facts_with_same_uuid():
    async def _run():
        service = SearchService(neo4j_service=_SameUuidAcrossGroupsNeo4jService())
        graphiti = _FakeGraphitiService()

        facts = await service.search_observational_facts(
            graphiti_service=graphiti,
            query='routine',
            group_ids=['s1_observational_memory', 'ontbk15batch_20260310_om_f'],
            max_facts=5,
            center_node_uuid=None,
        )

        assert len(facts) == 2
        assert [(fact['group_id'], fact['uuid']) for fact in facts] == [
            ('ontbk15batch_20260310_om_f', 'shared-fact'),
            ('s1_observational_memory', 'shared-fact'),
        ]

    asyncio.run(_run())
