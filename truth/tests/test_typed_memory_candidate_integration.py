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

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# Issue #1 — Manual approval must preserve one-current-object rule
# ─────────────────────────────────────────────────────────────────────────────


def test_manual_approval_of_conflicting_candidate_one_current_rule():
    """Manually approving a conflicting candidate must supersede the prior fact.

    Previously, promote_candidate_fact only superseded when seeded_supersede_ok
    was True.  For manually-approved candidates the policy trace has
    seeded_supersede_ok=False (since policy said 'requires_approval'), so the
    old code would 'assert' a new fact instead of superseding — creating two
    current facts in the same conflict set.

    Fix: always supersede when conflict_with_fact_id is provided and the prior
    object exists in the ledger, regardless of seeded_supersede_ok.
    """
    conn = _candidate_db()
    ledger = _ledger()

    # Step 1: auto-promote the first fact (low-risk, owner speaker).
    first = candidates.upsert_candidate(
        conn,
        ledger=ledger,
        **_candidate_kwargs(predicate='pref.theme', value={'value': 'light'}),
    )
    assert first.status == 'approved'
    first_facts = ledger.current_state_facts()
    assert len(first_facts) == 1
    first_fact = first_facts[0]

    # Step 2: introduce a conflicting candidate that requires manual approval
    # (high confidence but conflict → policy says requires_approval, NOT auto-supersede,
    # because seeded_supersede_ok is False in the candidate).
    second = candidates.upsert_candidate(
        conn,
        ledger=ledger,
        **_candidate_kwargs(
            predicate='pref.theme',
            value={'value': 'dark'},
            conflict_with_fact_id=first_fact.object_id,
            seeded_supersede_ok=False,  # explicit: policy will NOT auto-supersede
            confidence=0.70,            # below auto-promote threshold
        ),
    )
    assert second.status in {'requires_approval', 'pending'}, (
        f'Expected requires_approval, got {second.status!r}'
    )

    # At this point the old fact is still current — no second current should exist.
    current_before = ledger.current_state_facts()
    assert len(current_before) == 1
    assert current_before[0].object_id == first_fact.object_id

    # Step 3: human manually approves.
    updated, ledger_event_id = candidates.promote_candidate(
        conn,
        second.candidate_id,
        actor_id='ui:yuan',
        reason='manual approval — dark theme confirmed',
        ledger=ledger,
    )
    assert updated == 1
    assert ledger_event_id, 'ledger_event_id must be set after manual promotion'

    # Step 4: one-current-object rule — exactly one current fact, the new one.
    current_after = ledger.current_state_facts()
    assert len(current_after) == 1, (
        f'Expected exactly 1 current fact after manual approval, got {len(current_after)}: '
        + str([(f.object_id, f.value) for f in current_after])
    )
    new_fact = current_after[0]
    assert new_fact.value == {'value': 'dark'}, (
        f'Current fact should be the newly approved one, got: {new_fact.value!r}'
    )

    # Step 5: lineage integrity — old fact superseded, new fact inherits root.
    lineage = ledger.materialize_lineage(first_fact.root_id)
    assert len(lineage) == 2
    old = next(obj for obj in lineage if obj.object_id == first_fact.object_id)
    new = next(obj for obj in lineage if obj.object_id != first_fact.object_id)
    assert old.is_current is False
    assert old.superseded_by == new.object_id
    assert new.root_id == first_fact.root_id
    assert new.version == 2


# ─────────────────────────────────────────────────────────────────────────────
# Issue #7 — Deny path must write invalidate event to the ledger
# ─────────────────────────────────────────────────────────────────────────────


def test_deny_promoted_candidate_writes_invalidate_to_ledger():
    """Denying a previously-promoted candidate must write an invalidate event.

    The old code skipped ledger writes entirely for ChangeLedger-based deny
    paths, breaking the append-only audit contract.
    """
    conn = _candidate_db()
    ledger = _ledger()

    # Promote a candidate first.
    result = candidates.upsert_candidate(
        conn,
        ledger=ledger,
        **_candidate_kwargs(predicate='pref.deny_test', value={'value': 'draft'}),
    )
    assert result.status == 'approved'

    current_before = ledger.current_state_facts()
    assert len(current_before) == 1
    promoted_fact = current_before[0]

    # Record event count before denial.
    events_before = ledger.conn.execute('SELECT count(*) FROM change_events').fetchone()[0]
    assert events_before == 2  # assert + promote

    # Now deny the promoted candidate.
    updated, _ = candidates.deny_candidate(
        conn,
        result.candidate_id,
        actor_id='ui:yuan',
        reason='changed mind',
        ledger=ledger,
    )
    assert updated == 1

    # Verify: an invalidate event was written.
    events_after = ledger.conn.execute('SELECT count(*) FROM change_events').fetchone()[0]
    assert events_after == 3, (
        f'Expected 3 events (assert + promote + invalidate), got {events_after}'
    )

    invalidate_row = ledger.conn.execute(
        "SELECT * FROM change_events WHERE event_type = 'invalidate'"
    ).fetchone()
    assert invalidate_row is not None, 'invalidate event not found in ledger'
    assert invalidate_row['object_id'] == promoted_fact.object_id
    assert invalidate_row['actor_id'] == 'ui:yuan'

    # Verify: the fact is no longer current.
    current_after = ledger.current_state_facts()
    assert len(current_after) == 0, (
        f'Fact should no longer be current after denial, found: {current_after!r}'
    )


def test_deny_unpromoted_candidate_no_ledger_event_needed():
    """Denying a candidate that was never promoted must not write to the ledger.

    Only the candidates DB row is updated.  Nothing entered the ledger, so
    nothing needs to be invalidated there.
    """
    conn = _candidate_db()
    ledger = _ledger()

    # Insert a requires-approval candidate (low confidence → not auto-promoted).
    result = candidates.upsert_candidate(
        conn,
        ledger=ledger,
        **_candidate_kwargs(
            predicate='pref.low_conf_deny',
            value={'value': 'draft'},
            confidence=0.50,    # below auto-promote threshold
        ),
    )
    assert result.status in {'requires_approval', 'pending'}

    events_before = ledger.conn.execute('SELECT count(*) FROM change_events').fetchone()[0]
    assert events_before == 0, 'No events should be in ledger for un-promoted candidate'

    updated, ledger_event_id = candidates.deny_candidate(
        conn,
        result.candidate_id,
        actor_id='ui:yuan',
        reason='not needed',
        ledger=ledger,
    )
    assert updated == 1

    # No ledger events — candidate was never promoted.
    events_after = ledger.conn.execute('SELECT count(*) FROM change_events').fetchone()[0]
    assert events_after == 0, (
        'No ledger events expected when denying an un-promoted candidate'
    )

    # Candidates DB correctly reflects denial.
    row = conn.execute(
        'SELECT status, decision FROM candidates WHERE candidate_id = ?',
        (result.candidate_id,),
    ).fetchone()
    assert row['status'] == 'denied'
    assert row['decision'] == 'denied'


# ─────────────────────────────────────────────────────────────────────────────
# Issue #4 — SQLite pragmas on the ledger connection
# ─────────────────────────────────────────────────────────────────────────────


def test_change_ledger_connect_enables_wal_and_foreign_keys(tmp_path):
    """The change_ledger.connect() helper must enable WAL and foreign_keys."""
    from mcp_server.src.services.change_ledger import connect

    db_path = tmp_path / 'test_pragmas.db'
    conn = connect(db_path)
    try:
        journal_mode = conn.execute('PRAGMA journal_mode').fetchone()[0]
        fk_on = conn.execute('PRAGMA foreign_keys').fetchone()[0]
        assert journal_mode == 'wal', f'Expected WAL mode, got: {journal_mode!r}'
        assert fk_on == 1, 'Expected foreign_keys=ON'
    finally:
        conn.close()
