from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_server.src.models.typed_memory import Episode, StateFact
from mcp_server.src.services.change_ledger import ChangeLedger
from truth import candidates


def _candidate_db() -> sqlite3.Connection:
    conn = candidates.connect(':memory:')
    conn.row_factory = sqlite3.Row
    return conn


def _ledger() -> ChangeLedger:
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    return ChangeLedger(conn)


def _candidate_kwargs(**overrides):
    payload = {
        'subject': 'user:principal',
        'predicate': 'pref.editor',
        'scope': 'private',
        'assertion_type': 'preference',
        'value': {'value': 'vim'},
        'evidence_refs': [{'source_key': 'sessions:s1', 'evidence_id': 'msg-1', 'scope': 's1_sessions_main'}],
        'speaker_id': 'owner',
        'confidence': 0.93,
        'origin': 'extracted',
    }
    payload.update(overrides)
    return payload


def test_auto_promoted_candidate_materializes_promoted_state_fact():
    conn = _candidate_db()
    ledger = _ledger()

    result = candidates.upsert_candidate(conn, ledger=ledger, **_candidate_kwargs())
    assert result.status == 'approved'

    row = conn.execute('SELECT ledger_event_id FROM candidates WHERE candidate_id = ?', (result.candidate_id,)).fetchone()
    assert row['ledger_event_id']

    facts = ledger.current_state_facts()
    assert len(facts) == 1
    fact = facts[0]
    assert isinstance(fact, StateFact)
    assert fact.fact_type == 'preference'
    assert fact.promotion_status == 'promoted'
    assert fact.source_lane == 's1_sessions_main'
    assert fact.source_key == 'sessions:s1'
    assert fact.evidence_refs[0].canonical_uri == 'eventlog://sessions/sessions:s1/msg-1'


def test_auto_supersede_candidate_reuses_lineage_and_replaces_current_fact():
    conn = _candidate_db()
    ledger = _ledger()

    first = candidates.upsert_candidate(conn, ledger=ledger, **_candidate_kwargs(predicate='pref.os', value={'value': 'macos'}))
    first_fact = ledger.current_state_facts()[0]

    second = candidates.upsert_candidate(
        conn,
        ledger=ledger,
        **_candidate_kwargs(
            predicate='pref.os',
            value={'value': 'linux'},
            evidence_refs=[
                {'source_key': 'sessions:s1', 'evidence_id': 'msg-2', 'scope': 's1_sessions_main'},
                {'source_key': 'memory:s1', 'evidence_id': 'note-1', 'scope': 's1_memory_day1'},
            ],
            conflict_with_fact_id=first_fact.object_id,
            seeded_supersede_ok=True,
            explicit_update=True,
        ),
    )

    assert second.status == 'approved'

    lineage = ledger.materialize_lineage(first_fact.root_id)
    assert len(lineage) == 2
    old_fact = next(obj for obj in lineage if obj.object_id == first_fact.object_id)
    new_fact = next(obj for obj in lineage if obj.object_id != first_fact.object_id)

    assert isinstance(new_fact, StateFact)
    assert old_fact.is_current is False
    assert old_fact.superseded_by == new_fact.object_id
    assert new_fact.root_id == first_fact.root_id
    assert new_fact.parent_id == first_fact.object_id
    assert new_fact.version == 2
    assert new_fact.value == {'value': 'linux'}


def test_episode_candidate_maps_into_typed_episode_object():
    conn = _candidate_db()
    ledger = _ledger()

    result = candidates.upsert_candidate(
        conn,
        ledger=ledger,
        **_candidate_kwargs(
            predicate='episode.session_summary',
            assertion_type='episode',
            value={'summary': 'User planned the typed-memory rescope.'},
        ),
    )

    assert result.status == 'requires_approval'
    updated, ledger_event_id = candidates.promote_candidate(
        conn,
        result.candidate_id,
        actor_id='ui:yuan',
        reason='manual episode approval',
        ledger=ledger,
    )

    assert updated == 1
    assert ledger_event_id
    root = ledger.conn.execute('SELECT root_id FROM change_events ORDER BY recorded_at, rowid LIMIT 1').fetchone()['root_id']
    current = ledger.current_object(root)
    assert isinstance(current, Episode)
    assert current.summary == '{"summary": "User planned the typed-memory rescope."}'
