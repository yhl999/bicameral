from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_server.src.models.typed_memory import EvidenceRef, EntityRegistryEntry, Procedure, StateFact
from mcp_server.src.services.change_ledger import ChangeLedger


def _ledger() -> ChangeLedger:
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    return ChangeLedger(conn)


def _message_ref(message_id: str = 'm1') -> EvidenceRef:
    return EvidenceRef(
        kind='message',
        source_system='telegram',
        locator={
            'system': 'telegram',
            'conversation_id': 'chat-1',
            'message_id': message_id,
        },
    )


def test_evidence_ref_uses_locked_canonical_uri_shapes():
    msg = EvidenceRef(
        kind='message',
        source_system='telegram',
        locator={
            'system': 'telegram',
            'conversation_id': 'chat-1',
            'message_id': '42',
        },
    )
    file_ref = EvidenceRef(
        kind='file',
        source_system='workspace',
        locator={
            'repo': 'bicameral',
            'path': 'docs/architecture.md',
            'start_line': 10,
            'end_line': 20,
        },
    )
    sql_ref = EvidenceRef(
        kind='sql_row',
        source_system='taste_db',
        locator={
            'system': 'taste_db',
            'database': 'wine',
            'table': 'bottles',
            'pk_json': {'id': 7},
        },
    )

    assert msg.canonical_uri == 'msg://telegram/chat-1/42'
    assert file_ref.canonical_uri == 'file://bicameral/docs/architecture.md#L10-L20'
    assert sql_ref.canonical_uri == 'sql://taste_db/wine/bottles#pk=%7B%22id%22%3A7%7D'


def test_change_ledger_projects_state_fact_lifecycle():
    ledger = _ledger()
    original = StateFact.model_validate(
        {
            'object_id': 'fact_v1',
            'root_id': 'fact_v1',
            'fact_type': 'preference',
            'subject': 'user:principal',
            'predicate': 'pref.coffee',
            'value': {'drink': 'espresso'},
            'scope': 'private',
            'policy_scope': 'private',
            'visibility_scope': 'private',
            'evidence_refs': [_message_ref('m1')],
            'created_at': '2026-03-08T22:00:00Z',
            'valid_at': '2026-03-08T22:00:00Z',
        }
    )
    updated = StateFact.model_validate(
        {
            'object_id': 'fact_v2',
            'root_id': 'fact_v1',
            'parent_id': 'fact_v1',
            'version': 2,
            'fact_type': 'preference',
            'subject': 'user:principal',
            'predicate': 'pref.coffee',
            'value': {'drink': 'filter'},
            'scope': 'private',
            'policy_scope': 'private',
            'visibility_scope': 'private',
            'evidence_refs': [_message_ref('m2')],
            'created_at': '2026-03-09T00:00:00Z',
            'valid_at': '2026-03-09T00:00:00Z',
        }
    )

    ledger.append_event('assert', actor_id='extractor', payload=original, recorded_at='2026-03-08T22:00:00Z')
    ledger.append_event('promote', actor_id='policy:v3', object_id='fact_v1', root_id='fact_v1')
    ledger.append_event(
        'supersede',
        actor_id='policy:v3',
        payload=updated,
        target_object_id='fact_v1',
        recorded_at='2026-03-09T00:00:00Z',
    )
    ledger.append_event('invalidate', actor_id='policy:v3', object_id='fact_v2', root_id='fact_v1', recorded_at='2026-03-09T01:00:00Z')

    lineage = ledger.materialize_lineage('fact_v1')
    assert len(lineage) == 2

    first = next(obj for obj in lineage if obj.object_id == 'fact_v1')
    second = next(obj for obj in lineage if obj.object_id == 'fact_v2')

    assert first.is_current is False
    assert first.superseded_by == 'fact_v2'
    assert second.version == 2
    assert second.invalid_at == '2026-03-09T01:00:00Z'
    assert second.is_current is False
    assert ledger.current_object('fact_v1') is None


def test_procedure_success_and_failure_update_counters():
    ledger = _ledger()
    procedure = Procedure.model_validate(
        {
            'object_id': 'proc_v1',
            'root_id': 'proc_v1',
            'name': 'Launch code-loop-runner',
            'trigger': 'non-trivial coding task',
            'preconditions': ['repo clean'],
            'steps': ['prepare repo', 'launch runner', 'monitor logs'],
            'expected_outcome': 'runner completes without watchdog death',
            'policy_scope': 'internal',
            'visibility_scope': 'internal',
            'evidence_refs': [_message_ref('p1')],
            'created_at': '2026-03-08T22:00:00Z',
        }
    )

    ledger.append_event('assert', actor_id='extractor', payload=procedure, recorded_at='2026-03-08T22:00:00Z')
    ledger.append_event('promote', actor_id='ui:yuan', object_id='proc_v1', root_id='proc_v1')
    ledger.append_event('procedure_success', actor_id='runner', object_id='proc_v1', root_id='proc_v1')
    ledger.append_event('procedure_success', actor_id='runner', object_id='proc_v1', root_id='proc_v1')
    ledger.append_event('procedure_failure', actor_id='runner', object_id='proc_v1', root_id='proc_v1')

    materialized = ledger.materialize_object('proc_v1')
    assert isinstance(materialized, Procedure)
    assert materialized.promotion_status == 'promoted'
    assert materialized.success_count == 2
    assert materialized.fail_count == 1


def test_entity_registry_resolves_aliases_and_external_ids():
    ledger = _ledger()
    relationship = StateFact.model_validate(
        {
            'object_id': 'entity_fact_v1',
            'root_id': 'entity_fact_v1',
            'fact_type': 'relationship',
            'subject': 'entity:assistant',
            'predicate': 'relationship.entity_aliases',
            'value': {
                'entity_id': 'ent_archibald',
                'entity_type': 'assistant',
                'current_name': 'Archibald',
                'aliases': ['Archie'],
                'previous_names': ['Assistant'],
                'external_ids': [{'system': 'telegram', 'value': '1439681712'}],
            },
            'scope': 'private',
            'policy_scope': 'private',
            'visibility_scope': 'private',
            'evidence_refs': [_message_ref('e1')],
            'created_at': '2026-03-08T22:00:00Z',
        }
    )

    ledger.append_event('assert', actor_id='extractor', payload=relationship, recorded_at='2026-03-08T22:00:00Z')
    ledger.append_event('promote', actor_id='ui:yuan', object_id='entity_fact_v1', root_id='entity_fact_v1')

    registry = ledger.entity_registry()
    assert registry.resolve_name('Archie').entity_id == 'ent_archibald'
    assert registry.resolve_name('Assistant').entity_id == 'ent_archibald'
    assert registry.resolve_external_id('telegram', '1439681712').entity_id == 'ent_archibald'

    manual_registry = EntityRegistryEntry.model_validate(
        {
            'entity_id': 'ent_manual',
            'entity_type': 'person',
            'current_name': 'Yuan',
            'aliases': ['Yuan Han'],
        }
    )
    assert manual_registry.matches_name('yuan han') is True
