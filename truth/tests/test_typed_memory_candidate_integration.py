# ruff: noqa: E402, I001
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

    candidates.upsert_candidate(conn, ledger=ledger, **_candidate_kwargs(predicate='pref.os', value={'value': 'macos'}))
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


# ─────────────────────────────────────────────────────────────────────────────
# GPT P1 Blocker 2 — deny → re-approve must not create ledger split-brain
# ─────────────────────────────────────────────────────────────────────────────


def test_deny_then_reapprove_is_blocked_prevents_split_brain():
    """Re-approving a denied candidate must be blocked to prevent ledger split-brain.

    Timeline that exposed the bug:
      1. Candidate promoted → ledger: assert+promote events; candidates.db: approved
      2. Candidate denied   → ledger: invalidate event;    candidates.db: denied
      3. promote_candidate called again → (before fix) candidates.db flipped to
         'approved' but ledger still had invalidate event making fact non-current.
         Result: split-brain — candidates.db says approved, ledger says non-current.

    Fix: 'denied' is a terminal candidates lifecycle state. promote_candidate
    must return (0, None) immediately when the candidate is denied, preventing
    any candidates.db update and leaving ledger state untouched.
    """
    conn = _candidate_db()
    ledger = _ledger()

    # Step 1: auto-promote (low-risk owner speaker → auto_promoted)
    result = candidates.upsert_candidate(
        conn, ledger=ledger, **_candidate_kwargs(predicate='pref.split_brain_test')
    )
    assert result.status == 'approved', f'Expected approved, got: {result.status!r}'

    # Verify: one current fact in ledger
    assert len(ledger.current_state_facts()) == 1

    # Step 2: deny the promoted candidate (writes invalidate to ledger)
    deny_updated, _ = candidates.deny_candidate(
        conn, result.candidate_id,
        actor_id='ui:yuan', reason='changed mind', ledger=ledger,
    )
    assert deny_updated == 1

    # Verify state after denial
    row_denied = conn.execute(
        'SELECT status FROM candidates WHERE candidate_id = ?', (result.candidate_id,)
    ).fetchone()
    assert row_denied['status'] == 'denied'
    assert len(ledger.current_state_facts()) == 0, (
        'Fact should be non-current in ledger after denial'
    )

    # Count ledger events: should be exactly 3 (assert, promote, invalidate)
    events_after_deny = ledger.conn.execute(
        'SELECT count(*) FROM change_events'
    ).fetchone()[0]
    assert events_after_deny == 3

    # Step 3: attempt re-approval — must be blocked
    re_updated, re_event_id = candidates.promote_candidate(
        conn, result.candidate_id,
        actor_id='ui:yuan', reason='oops, reverting denial', ledger=ledger,
    )

    # The guard must return (0, None) — no rows updated, no event id
    assert re_updated == 0, (
        f'Re-approval of denied candidate must be blocked; '
        f'promote_candidate returned rowcount={re_updated}'
    )
    assert re_event_id is None, (
        f'No ledger event id should be returned for blocked re-approval, got: {re_event_id!r}'
    )

    # Step 4: candidates.db must still be 'denied' — not 'approved'
    row_after = conn.execute(
        'SELECT status FROM candidates WHERE candidate_id = ?', (result.candidate_id,)
    ).fetchone()
    assert row_after['status'] == 'denied', (
        f'candidates.db status must remain denied after blocked re-approval, '
        f'got: {row_after["status"]!r}'
    )

    # Step 5: ledger must not have grown — still 3 events, still no current facts
    events_after_reapprove = ledger.conn.execute(
        'SELECT count(*) FROM change_events'
    ).fetchone()[0]
    assert events_after_reapprove == 3, (
        f'Ledger must not gain new events after blocked re-approval, '
        f'got {events_after_reapprove} (expected 3)'
    )
    assert len(ledger.current_state_facts()) == 0, (
        'Ledger must still show no current facts after blocked re-approval (no split-brain)'
    )


def test_deny_never_promoted_then_reapprove_is_also_blocked():
    """An unpromoted (never reached ledger) denied candidate must also be blocked
    from re-approval.  'denied' is always terminal regardless of whether the
    candidate ever touched the ledger.
    """
    conn = _candidate_db()
    ledger = _ledger()

    # Insert a low-confidence candidate that requires manual approval
    result = candidates.upsert_candidate(
        conn, ledger=ledger,
        **_candidate_kwargs(
            predicate='pref.unpromoted_deny_test',
            confidence=0.50,   # below auto-promote threshold
        ),
    )
    assert result.status in {'requires_approval', 'pending'}, (
        f'Expected requires_approval/pending, got: {result.status!r}'
    )

    # Deny it (never touched the ledger)
    candidates.deny_candidate(
        conn, result.candidate_id,
        actor_id='ui:yuan', reason='not needed', ledger=ledger,
    )
    assert conn.execute(
        'SELECT status FROM candidates WHERE candidate_id = ?', (result.candidate_id,)
    ).fetchone()['status'] == 'denied'

    # Attempt re-approval — must be blocked even though ledger was never involved
    re_updated, re_event_id = candidates.promote_candidate(
        conn, result.candidate_id,
        actor_id='ui:yuan', reason='trying to reinstate', ledger=ledger,
    )
    assert re_updated == 0, (
        f'Re-approval of denied (never-promoted) candidate must be blocked, '
        f'promote_candidate returned rowcount={re_updated}'
    )
    assert re_event_id is None

    # Ledger must remain empty
    events = ledger.conn.execute('SELECT count(*) FROM change_events').fetchone()[0]
    assert events == 0, f'Ledger must stay empty, got {events} events'


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


# ─────────────────────────────────────────────────────────────────────────────
# GPT P1 Blocker — Cross-store retry consistency / idempotency
#
# Regression tests for the partial-failure scenario where the ledger write
# succeeds but the subsequent candidates.db UPDATE/commit fails.
#
# Without the reconciliation fix:
#   - promote retry → duplicate promote (+ assert/supersede) events in ledger
#   - deny retry    → duplicate invalidate events in ledger
#
# With the fix, both paths are idempotent: a retry recovers the already-written
# ledger event_id (promote) or skips the duplicate write (invalidate), then
# completes only the missing candidates.db UPDATE.
# ─────────────────────────────────────────────────────────────────────────────


def _simulate_promote_ledger_write_only(
    conn: sqlite3.Connection,
    candidate_id: str,
    ledger: ChangeLedger,
) -> str:
    """Simulate the partial failure: ledger write succeeds, candidates.db update does not.

    Calls promote_candidate_fact() directly as promote_candidate() would, but
    does NOT touch candidates.db afterwards.  Returns the promote event_id that
    landed in the ledger.
    """
    from truth.candidates import (
        POLICY_VERSION_DEFAULT,
        _candidate_fact_payload,
        _normalize_reason,
        _now_iso,
    )

    row = conn.execute(
        'SELECT * FROM candidates WHERE candidate_id = ?', (candidate_id,)
    ).fetchone()
    assert row is not None, f'candidate {candidate_id!r} not found in DB'

    fact_payload = _candidate_fact_payload(row)
    decided_at = _now_iso()
    safe_reason = _normalize_reason('test_partial_failure', fallback='auto_promote')

    result = ledger.promote_candidate_fact(
        actor_id='policy:v3',
        reason=safe_reason,
        policy_version=row['policy_version'] or POLICY_VERSION_DEFAULT,
        candidate_id=candidate_id,
        fact=fact_payload,
        conflict_with_fact_id=row['conflict_with_fact_id'],
        seeded_supersede_ok=False,
        recorded_at=decided_at,
    )
    # Deliberately do NOT update candidates.db — this is the partial-failure state.
    return result.event_id


def test_promote_retry_after_partial_failure_no_duplicate_ledger_events():
    """Retry of promote_candidate after partial failure must not duplicate ledger events.

    Simulated failure sequence:
      1. promote_candidate_fact() writes assert+promote to ledger DB → committed
      2. candidates.db UPDATE/commit never happens (crash / connection drop)
      3. Caller retries promote_candidate() with the candidate still in 'pending' state

    Expected outcome:
      - Ledger event count stays at 2 (assert + promote, no duplicates)
      - candidates.db is updated to 'approved' with the recovered ledger_event_id
      - The promoted StateFact is the only current fact in the ledger
    """
    conn = _candidate_db()
    ledger = _ledger()

    # Insert a candidate that requires manual approval (low confidence → not auto-promoted).
    result = candidates.upsert_candidate(
        conn,
        ledger=ledger,
        **_candidate_kwargs(
            predicate='pref.retry_promote_test',
            value={'value': 'neovim'},
            confidence=0.50,  # below auto-promote threshold → requires_approval
        ),
    )
    assert result.status in {'requires_approval', 'pending'}, (
        f'Expected requires_approval/pending, got: {result.status!r}'
    )

    # Confirm: no ledger events yet (candidate was never promoted).
    events_before = ledger.conn.execute('SELECT count(*) FROM change_events').fetchone()[0]
    assert events_before == 0

    # Simulate partial failure: ledger write succeeds, candidates.db update fails.
    partial_event_id = _simulate_promote_ledger_write_only(
        conn, result.candidate_id, ledger
    )

    # Ledger now has 2 events (assert + promote); candidates.db still has original state.
    events_after_partial = ledger.conn.execute(
        'SELECT count(*) FROM change_events'
    ).fetchone()[0]
    assert events_after_partial == 2, (
        f'Expected 2 ledger events after partial write, got {events_after_partial}'
    )
    row_partial = conn.execute(
        'SELECT status, ledger_event_id FROM candidates WHERE candidate_id = ?',
        (result.candidate_id,),
    ).fetchone()
    assert row_partial['ledger_event_id'] is None, (
        'candidates.db ledger_event_id must be NULL (simulating failed DB update)'
    )
    assert row_partial['status'] != 'approved', (
        'candidates.db status must not be approved yet (simulating failed DB update)'
    )

    # ── Retry promote_candidate ───────────────────────────────────────────────
    # This is what the real caller would do after the partial failure.
    updated, recovered_event_id = candidates.promote_candidate(
        conn,
        result.candidate_id,
        actor_id='policy:v3',
        reason='retry after partial failure',
        ledger=ledger,
    )

    # Idempotency: no new ledger events were written.
    events_after_retry = ledger.conn.execute(
        'SELECT count(*) FROM change_events'
    ).fetchone()[0]
    assert events_after_retry == 2, (
        f'Retry must not duplicate ledger events: expected 2, got {events_after_retry}'
    )

    # The recovered event_id must match the one that was written during the
    # partial failure — not a new one.
    assert recovered_event_id == partial_event_id, (
        f'Retry must recover the existing promote event_id '
        f'(got {recovered_event_id!r}, expected {partial_event_id!r})'
    )

    # candidates.db must now reflect the successful promotion.
    assert updated == 1, 'candidates.db UPDATE must have applied on retry'
    row_after = conn.execute(
        'SELECT status, ledger_event_id FROM candidates WHERE candidate_id = ?',
        (result.candidate_id,),
    ).fetchone()
    assert row_after['status'] == 'approved', (
        f'candidates.db status must be approved after retry, got: {row_after["status"]!r}'
    )
    assert row_after['ledger_event_id'] == partial_event_id, (
        'candidates.db ledger_event_id must be set to the recovered promote event_id'
    )

    # Final state: exactly one current fact in the ledger.
    current_facts = ledger.current_state_facts()
    assert len(current_facts) == 1, (
        f'Expected exactly 1 current fact after retry, got {len(current_facts)}'
    )
    assert current_facts[0].value == {'value': 'neovim'}, (
        f'Current fact value mismatch: {current_facts[0].value!r}'
    )


def test_deny_retry_after_partial_failure_no_duplicate_invalidate_events():
    """Retry of deny_candidate after partial failure must not duplicate invalidate events.

    Simulated failure sequence:
      1. Candidate is fully promoted (assert+promote in ledger, 'approved' in candidates.db)
      2. deny_candidate() is called:
         a. ledger.append_event('invalidate', ...) succeeds → committed in ledger
         b. candidates.db UPDATE/commit never happens (crash / connection drop)
      3. Caller retries deny_candidate() — candidate is still 'approved' in candidates.db

    Expected outcome:
      - Ledger event count stays at 3 (assert + promote + invalidate, no duplicates)
      - candidates.db is updated to 'denied' on retry
      - The fact is still non-current in the ledger (invalidate already applied)
    """
    conn = _candidate_db()
    ledger = _ledger()

    # Step 1: Fully promote a candidate.
    result = candidates.upsert_candidate(
        conn,
        ledger=ledger,
        **_candidate_kwargs(
            predicate='pref.retry_deny_test',
            value={'value': 'emacs'},
        ),
    )
    assert result.status == 'approved', f'Expected approved, got: {result.status!r}'

    promoted_facts = ledger.current_state_facts()
    assert len(promoted_facts) == 1
    promoted_fact = promoted_facts[0]

    events_after_promote = ledger.conn.execute(
        'SELECT count(*) FROM change_events'
    ).fetchone()[0]
    assert events_after_promote == 2  # assert + promote

    # Step 2: Simulate partial deny failure.
    # Write the invalidate event directly to the ledger (as deny_candidate would),
    # but do NOT update candidates.db.
    inv_object_id = promoted_fact.object_id
    ledger.append_event(
        'invalidate',
        actor_id='ui:yuan',
        reason='test_partial_deny_failure',
        object_id=inv_object_id,
        root_id=ledger.root_id_for_object(inv_object_id),
    )

    # Ledger now has 3 events; candidates.db still says 'approved'.
    events_after_partial = ledger.conn.execute(
        'SELECT count(*) FROM change_events'
    ).fetchone()[0]
    assert events_after_partial == 3, (
        f'Expected 3 ledger events after partial invalidate write, got {events_after_partial}'
    )
    row_partial = conn.execute(
        'SELECT status FROM candidates WHERE candidate_id = ?',
        (result.candidate_id,),
    ).fetchone()
    assert row_partial['status'] == 'approved', (
        'candidates.db status must still be approved (simulating failed DB update)'
    )

    # Verify fact is already non-current in the ledger (invalidate landed).
    current_after_partial = ledger.current_state_facts()
    assert len(current_after_partial) == 0, (
        'Fact should already be non-current after partial ledger write'
    )

    # ── Retry deny_candidate ──────────────────────────────────────────────────
    updated, _ = candidates.deny_candidate(
        conn,
        result.candidate_id,
        actor_id='ui:yuan',
        reason='retry after partial deny failure',
        ledger=ledger,
    )

    # Idempotency: no new ledger events were written.
    events_after_retry = ledger.conn.execute(
        'SELECT count(*) FROM change_events'
    ).fetchone()[0]
    assert events_after_retry == 3, (
        f'Retry must not duplicate invalidate events: expected 3, got {events_after_retry}'
    )

    # candidates.db must now reflect the denial.
    assert updated == 1, 'candidates.db UPDATE must have applied on retry'
    row_after = conn.execute(
        'SELECT status FROM candidates WHERE candidate_id = ?',
        (result.candidate_id,),
    ).fetchone()
    assert row_after['status'] == 'denied', (
        f'candidates.db status must be denied after retry, got: {row_after["status"]!r}'
    )

    # Ledger state: fact is still non-current (no double-invalidate, no resurrection).
    current_after_retry = ledger.current_state_facts()
    assert len(current_after_retry) == 0, (
        f'Fact must remain non-current after retry (no ledger resurrection), '
        f'found: {current_after_retry!r}'
    )
