# ruff: noqa: E402, I001
from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_server.src.models.typed_memory import Episode, EvidenceRef, Procedure, StateFact
from mcp_server.src.services.change_ledger import ChangeLedger
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
    return TypedRetrievalService(ledger=ledger)


def test_current_state_queries_return_only_current_objects():
    service = _service()
    result = asyncio.run(
        service.search(
            query='what is the current coffee preference now',
            object_types=['state'],
            history_mode='auto',
            max_results=5,
            max_evidence=10,
        )
    )

    assert result['query_mode'] == 'current'
    assert [item['object_id'] for item in result['state']] == ['fact_coffee_v2']
    assert result['state'][0]['is_current'] is True
    assert result['state'][0]['value'] == {'drink': 'filter'}
    assert len(result['evidence']) == 2


def test_history_queries_return_supersession_chain():
    service = _service()
    result = asyncio.run(
        service.search(
            query='what changed about coffee preference',
            object_types=['state'],
            history_mode='auto',
            max_results=5,
            max_evidence=10,
        )
    )

    assert result['query_mode'] == 'history'
    assert [item['object_id'] for item in result['state']] == ['fact_coffee_v1', 'fact_coffee_v2']
    first, second = result['state']
    assert first['is_current'] is False
    assert first['superseded_by'] == 'fact_coffee_v2'
    assert second['parent_id'] == 'fact_coffee_v1'
    assert second['version'] == 2


def test_cross_type_retrieval_with_metadata_filters():
    service = _service()
    result = asyncio.run(
        service.search(
            query='launch',
            metadata_filters={'source_lane': {'in': ['s1_sessions_main', 'ops_lane']}},
            history_mode='all',
            max_results=10,
            max_evidence=10,
        )
    )

    assert result['episodes'][0]['object_id'] == 'episode_launch_v1'
    assert result['procedures'][0]['object_id'] == 'procedure_launch_v1'
    assert result['counts']['episodes'] == 1
    assert result['counts']['procedures'] == 1


def test_evidence_bucket_dedupes_and_uses_qmd_adapter():
    service = _service()
    result = asyncio.run(
        service.search(
            query='coffee',
            object_types=['state'],
            history_mode='history',
            max_results=5,
            max_evidence=10,
        )
    )

    qmd_items = [item for item in result['evidence'] if item['kind'] == 'qmd_chunk']
    assert len(qmd_items) == 2
    assert all(item['resolver'] == 'qmd' for item in qmd_items)
    assert all(item['status'] == 'resolved' for item in qmd_items)
