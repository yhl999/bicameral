#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import asyncio
import sqlite3
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_server.src.models.typed_memory import Episode, EvidenceRef, Procedure, StateFact
from mcp_server.src.services.change_ledger import ChangeLedger
from mcp_server.src.services.evidence_callback import (
    EvidenceCallbackRegistry,
    PassThroughEvidenceCallback,
    QMDEvidenceCallback,
)
from mcp_server.src.services.om_typed_projection import OMTypedProjectionService
from mcp_server.src.services.typed_retrieval import TypedRetrievalService


def _ledger() -> ChangeLedger:
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    return ChangeLedger(conn)


def _message_ref(message_id: str) -> EvidenceRef:
    return EvidenceRef(
        kind='message',
        source_system='telegram',
        locator={
            'system': 'telegram',
            'conversation_id': 'chat-1',
            'message_id': message_id,
        },
        snippet=f'message-{message_id}',
    )


def _qmd_ref(chunk_id: str) -> EvidenceRef:
    return EvidenceRef(
        kind='qmd_chunk',
        source_system='qmd',
        locator={
            'collection': 'memory',
            'document_id': 'doc-1',
            'chunk_id': chunk_id,
        },
        snippet=f'qmd-snippet-{chunk_id}',
    )


def _seed_fixture(ledger: ChangeLedger) -> None:
    original = StateFact.model_validate(
        {
            'object_id': 'fact_coffee_v1',
            'root_id': 'fact_coffee_v1',
            'fact_type': 'preference',
            'subject': 'user:principal',
            'predicate': 'pref.coffee',
            'value': {'drink': 'espresso'},
            'scope': 'private',
            'policy_scope': 'private',
            'visibility_scope': 'private',
            'source_lane': 's1_sessions_main',
            'evidence_refs': [_message_ref('m1'), _qmd_ref('c1')],
            'created_at': '2026-03-08T22:00:00Z',
            'valid_at': '2026-03-08T22:00:00Z',
        }
    )
    updated = StateFact.model_validate(
        {
            'object_id': 'fact_coffee_v2',
            'root_id': 'fact_coffee_v1',
            'parent_id': 'fact_coffee_v1',
            'version': 2,
            'fact_type': 'preference',
            'subject': 'user:principal',
            'predicate': 'pref.coffee',
            'value': {'drink': 'filter'},
            'scope': 'private',
            'policy_scope': 'private',
            'visibility_scope': 'private',
            'source_lane': 's1_sessions_main',
            'evidence_refs': [_message_ref('m2'), _qmd_ref('c2')],
            'created_at': '2026-03-09T00:00:00Z',
            'valid_at': '2026-03-09T00:00:00Z',
        }
    )
    episode = Episode.model_validate(
        {
            'object_id': 'episode_launch_v1',
            'root_id': 'episode_launch_v1',
            'title': 'Launch retrospective',
            'summary': 'Discussed launch procedure and rollout issues',
            'annotations': ['launch', 'retro'],
            'policy_scope': 'private',
            'visibility_scope': 'private',
            'source_lane': 's1_sessions_main',
            'evidence_refs': [_message_ref('e1')],
            'created_at': '2026-03-09T01:00:00Z',
        }
    )
    procedure = Procedure.model_validate(
        {
            'object_id': 'procedure_launch_v1',
            'root_id': 'procedure_launch_v1',
            'name': 'Launch service safely',
            'trigger': 'production launch',
            'preconditions': ['repo clean', 'tests passing'],
            'steps': ['build artifacts', 'deploy service', 'verify health'],
            'expected_outcome': 'healthy rollout',
            'policy_scope': 'internal',
            'visibility_scope': 'internal',
            'source_lane': 'ops_lane',
            'evidence_refs': [_message_ref('p1')],
            'created_at': '2026-03-09T02:00:00Z',
        }
    )

    ledger.append_event('assert', actor_id='extractor', payload=original, recorded_at='2026-03-08T22:00:00Z')
    ledger.append_event('promote', actor_id='policy:v3', object_id='fact_coffee_v1', root_id='fact_coffee_v1')
    ledger.append_event(
        'supersede',
        actor_id='policy:v3',
        payload=updated,
        target_object_id='fact_coffee_v1',
        recorded_at='2026-03-09T00:00:00Z',
    )
    ledger.append_event('promote', actor_id='policy:v3', object_id='fact_coffee_v2', root_id='fact_coffee_v1')
    ledger.append_event('assert', actor_id='extractor', payload=episode, recorded_at='2026-03-09T01:00:00Z')
    ledger.append_event('promote', actor_id='policy:v3', object_id='episode_launch_v1', root_id='episode_launch_v1')
    ledger.append_event('assert', actor_id='extractor', payload=procedure, recorded_at='2026-03-09T02:00:00Z')
    ledger.append_event('promote', actor_id='policy:v3', object_id='procedure_launch_v1', root_id='procedure_launch_v1')


def _service() -> TypedRetrievalService:
    ledger = _ledger()
    _seed_fixture(ledger)
    evidence_registry = EvidenceCallbackRegistry(
        callbacks=[
            QMDEvidenceCallback(command='__missing_qmd__'),
            PassThroughEvidenceCallback(),
        ]
    )
    return TypedRetrievalService(ledger=ledger, evidence_registry=evidence_registry)


async def _run() -> None:
    service = _service()

    current = await service.search(
        query='what is the current coffee preference now',
        object_types=['state'],
        history_mode='auto',
        max_results=5,
        max_evidence=10,
    )
    assert current['query_mode'] == 'current'
    assert [item['object_id'] for item in current['state']] == ['fact_coffee_v2']
    assert current['state'][0]['is_current'] is True

    history = await service.search(
        query='what changed about coffee preference',
        object_types=['state'],
        history_mode='auto',
        max_results=5,
        max_evidence=10,
    )
    assert history['query_mode'] == 'history'
    assert [item['object_id'] for item in history['state']] == ['fact_coffee_v2', 'fact_coffee_v1']
    assert history['state'][0]['parent_id'] == 'fact_coffee_v1'
    assert history['state'][1]['superseded_by'] == 'fact_coffee_v2'

    mixed = await service.search(
        query='launch',
        metadata_filters={'source_lane': {'in': ['s1_sessions_main', 'ops_lane']}},
        history_mode='all',
        max_results=10,
        max_evidence=10,
    )
    assert [item['object_id'] for item in mixed['episodes']] == ['episode_launch_v1']
    assert [item['object_id'] for item in mixed['procedures']] == ['procedure_launch_v1']

    qmd_items = [item for item in history['evidence'] if item['kind'] == 'qmd_chunk']
    assert len(qmd_items) == 2
    assert all(item['resolver'] == 'qmd' for item in qmd_items)
    assert all(item['status'] == 'resolved' for item in qmd_items)

    print('typed retrieval smoke: PASS')
    print(f"  current-state objects: {[item['object_id'] for item in current['state']]}")
    print(f"  history chain: {[item['object_id'] for item in history['state']]}")
    print(f"  cross-type buckets: episodes={mixed['counts']['episodes']} procedures={mixed['counts']['procedures']}")
    print(f"  evidence items: {history['counts']['evidence']}")

    # ── OM typed projection smoke ───────────────────────────────────────────
    await _run_om_projection_smoke()


class _FakeOMSearchService:
    """Minimal stub that returns canned OM node/fact rows."""

    def includes_observational_memory(self, group_ids: list[str]) -> bool:
        return any('observational_memory' in gid or '_om_' in gid for gid in group_ids)

    def _om_groups_in_scope(self, group_ids: list[str]) -> list[str]:
        return [gid for gid in group_ids if 'observational_memory' in gid or '_om_' in gid]

    async def search_observational_nodes(self, *, graphiti_service, query, group_ids, max_nodes, entity_types):
        group_id = group_ids[0] if group_ids else 's1_observational_memory'
        return [
            {
                'uuid': 'plan_v2',
                'name': 'Heap cap mitigation plan',
                'summary': 'Ship the heap cap guardrail',
                'created_at': '2026-03-05T00:00:00Z',
                'group_id': group_id,
                'attributes': {
                    'source': 'om_primitive',
                    'status': 'active',
                    'semantic_domain': 'observational_memory',
                },
            }
        ]

    async def search_observational_facts(self, *, graphiti_service, query, group_ids, max_facts, center_node_uuid):
        group_id = group_ids[0] if group_ids else 's1_observational_memory'
        neighborhood_rows = {
            'plan_v1': [
                {
                    'uuid': 'rel_plan_v1_issue',
                    'name': 'ADDRESSES',
                    'fact': 'ADDRESSES: investigate heap cap -> heap cap issue',
                    'group_id': group_id,
                    'source_node_uuid': 'plan_v1',
                    'target_node_uuid': 'issue_1',
                    'created_at': '2026-03-01T00:00:00Z',
                    'attributes': {
                        'source_content': 'Investigate heap cap spike',
                        'target_content': 'Heap cap issue',
                    },
                }
            ],
            'plan_v2': [
                {
                    'uuid': 'rel_plan_v2_issue',
                    'name': 'ADDRESSES',
                    'fact': 'ADDRESSES: ship heap cap guardrail -> heap cap issue',
                    'group_id': group_id,
                    'source_node_uuid': 'plan_v2',
                    'target_node_uuid': 'issue_1',
                    'created_at': '2026-03-05T00:00:00Z',
                    'attributes': {
                        'source_content': 'Ship heap cap guardrail',
                        'target_content': 'Heap cap issue',
                    },
                }
            ],
            'issue_1': [
                {
                    'uuid': 'rel_fix_resolves_issue',
                    'name': 'RESOLVES',
                    'fact': 'RESOLVES: ship heap cap guardrail -> heap cap issue',
                    'group_id': group_id,
                    'source_node_uuid': 'fix_1',
                    'target_node_uuid': 'issue_1',
                    'created_at': '2026-03-06T00:00:00Z',
                    'attributes': {
                        'source_content': 'Ship heap cap guardrail',
                        'target_content': 'Heap cap issue',
                    },
                }
            ],
            'fix_1': [
                {
                    'uuid': 'rel_fix_resolves_issue',
                    'name': 'RESOLVES',
                    'fact': 'RESOLVES: ship heap cap guardrail -> heap cap issue',
                    'group_id': group_id,
                    'source_node_uuid': 'fix_1',
                    'target_node_uuid': 'issue_1',
                    'created_at': '2026-03-06T00:00:00Z',
                    'attributes': {
                        'source_content': 'Ship heap cap guardrail',
                        'target_content': 'Heap cap issue',
                    },
                }
            ],
        }
        if center_node_uuid is not None:
            return neighborhood_rows.get(center_node_uuid, [])
        return [
            {
                'uuid': 'rel_plan_v2_issue',
                'name': 'ADDRESSES',
                'fact': 'ADDRESSES: ship heap cap guardrail -> heap cap issue',
                'group_id': group_id,
                'source_node_uuid': 'plan_v2',
                'target_node_uuid': 'issue_1',
                'created_at': '2026-03-05T00:00:00Z',
                'attributes': {
                    'source_content': 'Ship heap cap guardrail',
                    'target_content': 'Heap cap issue',
                },
            },
            {
                'uuid': 'rel_fix_resolves_issue',
                'name': 'RESOLVES',
                'fact': 'RESOLVES: ship heap cap guardrail -> heap cap issue',
                'group_id': group_id,
                'source_node_uuid': 'fix_1',
                'target_node_uuid': 'issue_1',
                'created_at': '2026-03-06T00:00:00Z',
                'attributes': {
                    'source_content': 'Ship heap cap guardrail',
                    'target_content': 'Heap cap issue',
                },
            },
        ]


class _FakeOMDriver:
    def __init__(self):
        self.calls = []
        self._records = {
            'plan_v2': [
                {
                    'node_id': 'plan_v1',
                    'uuid': 'plan_v1',
                    'content': 'Investigate heap cap spike',
                    'created_at': '2026-03-01T00:00:00Z',
                    'status': 'open',
                    'semantic_domain': 'observational_memory',
                    'supersedes': [],
                },
                {
                    'node_id': 'plan_v2',
                    'uuid': 'plan_v2',
                    'content': 'Ship heap cap guardrail',
                    'created_at': '2026-03-05T00:00:00Z',
                    'status': 'active',
                    'semantic_domain': 'observational_memory',
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
                    'content': 'Heap cap issue',
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
                    'content': 'Ship heap cap guardrail',
                    'created_at': '2026-03-06T00:00:00Z',
                    'status': 'active',
                    'semantic_domain': 'observational_memory',
                    'supersedes': [],
                }
            ],
        }

    async def execute_query(self, _query, **params):
        self.calls.append(params)
        group_id = params.get('group_id') or 's1_observational_memory'
        seed_node_id = params.get('seed_node_id')
        rows = []
        for row in self._records.get(seed_node_id, []):
            copied = deepcopy(row)
            copied['group_id'] = group_id
            rows.append(copied)
        return rows, None, None


class _FakeOMGraphitiService:
    class config:
        class database:
            provider = 'neo4j'

    def __init__(self, driver):
        self.driver = driver

    async def get_client(self):
        return SimpleNamespace(driver=self.driver)


async def _run_om_projection_smoke() -> None:
    """Smoke test: OM projected objects surface through typed buckets."""
    fake_search = _FakeOMSearchService()
    fake_graphiti = _FakeOMGraphitiService(_FakeOMDriver())

    om_projection = OMTypedProjectionService(
        search_service=fake_search,
        graphiti_service=fake_graphiti,
    )

    # Service with OM projection but empty ledger
    ledger = ChangeLedger(sqlite3.connect(':memory:'))
    evidence_registry = EvidenceCallbackRegistry(callbacks=[PassThroughEvidenceCallback()])
    service = TypedRetrievalService(
        ledger=ledger,
        evidence_registry=evidence_registry,
        om_projection_service=om_projection,
    )

    # Canonical OM lane through typed buckets
    result = await service.search(
        query='heap cap',
        effective_group_ids=['s1_observational_memory'],
        history_mode='all',
        max_results=10,
        max_evidence=10,
    )
    assert result['result_format'] == 'typed'
    assert result['counts']['episodes'] >= 1, f"expected OM episodes, got {result['counts']}"
    assert result['counts']['state'] >= 1, f"expected OM state facts, got {result['counts']}"

    om_episodes = result['episodes']
    assert any('om_episode:' in ep['object_id'] for ep in om_episodes), 'expected om_episode: prefix'
    assert all(ep['source_lane'] == 's1_observational_memory' for ep in om_episodes)

    om_state = result['state']
    assert any('om_state:' in sf['object_id'] for sf in om_state), 'expected om_state: prefix'

    # Evidence refs point back to OM provenance
    evidence = result['evidence']
    om_evidence = [e for e in evidence if e.get('source_system') == 'om']
    assert len(om_evidence) >= 1, f'expected OM evidence refs, got {len(om_evidence)}'

    # OM projection limits reported
    om_limits = result['limits_applied']['materialization']['om_projection']
    assert om_limits['enabled'] is True
    assert om_limits['episodes_projected'] >= 1
    assert om_limits['state_projected'] >= 1

    # Experimental OM group through typed buckets
    result_exp = await service.search(
        query='heap cap',
        effective_group_ids=['ontbk15batch_20260310_om_f'],
        history_mode='all',
        max_results=10,
        max_evidence=10,
    )
    assert result_exp['counts']['episodes'] >= 1 or result_exp['counts']['state'] >= 1, \
        f"expected OM content for experimental group, got {result_exp['counts']}"

    print('om typed projection smoke: PASS')
    print(f"  canonical episodes: {result['counts']['episodes']}")
    print(f"  canonical state: {result['counts']['state']}")
    print(f"  canonical evidence: {result['counts']['evidence']}")
    print(f"  experimental group surfaced: {result_exp['counts']['episodes'] + result_exp['counts']['state']} objects")


if __name__ == '__main__':
    asyncio.run(_run())
