from __future__ import annotations

import sqlite3
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

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


# ─────────────────────────────────────────────────────────────────────────────
# Issue #3 — event ID must be collision-safe under concurrent writes
# ─────────────────────────────────────────────────────────────────────────────


def test_event_ids_are_collision_safe():
    """Rapid generation of event IDs must produce no duplicates.

    The old implementation used sha256(timestamp) which could collide across
    concurrent writes in the same second.  The new implementation uses
    secrets.token_hex(12) (96 bits of randomness).
    """
    from mcp_server.src.services.change_ledger import _new_event_id

    n = 10_000
    ids = [_new_event_id() for _ in range(n)]
    assert len(set(ids)) == n, 'event ID collision detected'
    for eid in ids:
        assert eid.startswith('evt_'), f'unexpected event_id format: {eid!r}'
        # 24 hex chars after the prefix = 96 bits of randomness
        assert len(eid) == len('evt_') + 24, f'unexpected event_id length: {eid!r}'


# ─────────────────────────────────────────────────────────────────────────────
# Issue #2 — promotion must be atomic (both events or neither)
# ─────────────────────────────────────────────────────────────────────────────


def test_promote_candidate_fact_is_atomic():
    """If the promote event insert fails, the create event must also be rolled back.

    promote_candidate_fact builds both rows and calls _do_insert twice, then
    does a single conn.commit().  If the second _do_insert raises before the
    commit, the first insert has not been committed and SQLite rolls back the
    open transaction when the exception propagates.

    We simulate failure on the second _do_insert to verify zero events are
    committed after the exception.
    """
    ledger = _ledger()
    fact_payload = {
        'subject': 'user:principal',
        'predicate': 'pref.atomicity_test',
        'scope': 'private',
        'assertion_type': 'preference',
        'value': 'vim',
        'evidence_refs': [{'source_key': 's:1', 'evidence_id': 'e-atomic', 'scope': 's1_sessions_main'}],
    }

    # patch.object on a class method: mock is looked up on the class and
    # called without Python's descriptor binding, so side_effect receives
    # just the positional args (no implicit self).  We capture `ledger` via
    # closure and call the original unbound method explicitly.
    original_do_insert = ChangeLedger._do_insert
    call_count = [0]

    def flaky_second_insert(row):  # no self — mock is NOT a descriptor
        call_count[0] += 1
        if call_count[0] == 2:
            raise RuntimeError('simulated disk failure on second insert')
        return original_do_insert(ledger, row)

    with patch.object(ChangeLedger, '_do_insert', side_effect=flaky_second_insert):
        with pytest.raises(RuntimeError, match='simulated disk failure'):
            ledger.promote_candidate_fact(
                actor_id='test:actor',
                reason='atomicity test',
                policy_version='v3',
                candidate_id='cand-atomic-test',
                fact=fact_payload,
            )

    count = ledger.conn.execute('SELECT count(*) FROM change_events').fetchone()[0]
    assert count == 0, (
        f'Atomicity broken: {count} event(s) committed before promotion was '
        'complete.  Both the create and promote events must commit as one unit.'
    )


def test_promote_candidate_fact_success_emits_exactly_two_events():
    """On success, exactly a create event and a promote event are committed."""
    ledger = _ledger()
    fact_payload = {
        'subject': 'user:principal',
        'predicate': 'pref.two_event_test',
        'scope': 'private',
        'assertion_type': 'preference',
        'value': 'emacs',
        'evidence_refs': [{'source_key': 's:1', 'evidence_id': 'e-two', 'scope': 's1_sessions_main'}],
    }

    result = ledger.promote_candidate_fact(
        actor_id='test:actor',
        reason='two-event test',
        policy_version='v3',
        candidate_id='cand-two-event',
        fact=fact_payload,
    )

    rows = ledger.conn.execute(
        'SELECT event_type FROM change_events ORDER BY rowid'
    ).fetchall()
    event_types = [r['event_type'] for r in rows]
    assert event_types == ['assert', 'promote'], f'unexpected event sequence: {event_types}'
    assert len(result.event_ids) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Issue #5 — EvidenceRef: legacy refs without stable IDs must not collide
# ─────────────────────────────────────────────────────────────────────────────


def test_legacy_ref_without_stable_id_generates_unique_canonical_uris():
    """Two structurally distinct legacy refs with no evidence_id must not share a URI.

    The old code fell back to 'unknown' as the event_id, producing identical
    canonical_uris for different refs.  The fix uses a content hash as fallback.
    """
    ref_a = {'source_key': 'sessions:s1', 'scope': 's1_sessions_main'}
    ref_b = {'source_key': 'memory:s1', 'scope': 's1_memory_day1'}
    # Both have no evidence_id / chunk_key / start_id / end_id

    ev_a = EvidenceRef.from_legacy_ref(ref_a)
    ev_b = EvidenceRef.from_legacy_ref(ref_b)

    assert ev_a.canonical_uri != ev_b.canonical_uri, (
        'Two distinct legacy refs without stable IDs produced the same '
        f'canonical_uri: {ev_a.canonical_uri!r}'
    )
    # URIs must be deterministic (same call → same URI)
    ev_a2 = EvidenceRef.from_legacy_ref(ref_a)
    assert ev_a.canonical_uri == ev_a2.canonical_uri, 'Legacy ref URI is not deterministic'


def test_legacy_ref_with_evidence_id_is_unchanged():
    """When a stable evidence_id is present, from_legacy_ref must not change it."""
    ref = {
        'source_key': 'sessions:s1',
        'evidence_id': 'stable-id-42',
        'scope': 's1_sessions_main',
    }
    ev = EvidenceRef.from_legacy_ref(ref)
    assert 'stable-id-42' in ev.canonical_uri


# ─────────────────────────────────────────────────────────────────────────────
# GPT P1 Blocker 1 — one-current-object rule enforced without conflict_with_fact_id
# ─────────────────────────────────────────────────────────────────────────────


def test_missing_conflict_id_does_not_yield_two_current_facts():
    """Promoting two facts with the same conflict set (subject+predicate+scope)
    must result in exactly one current fact even when conflict_with_fact_id is
    NOT passed on the second promotion call.

    Prior bug: the second promote_candidate_fact would issue an 'assert' event
    (instead of 'supersede') creating two is_current=True facts in the same
    conflict set.  The central enforcement scan in promote_candidate_fact must
    detect the existing current occupant and override creation_type to 'supersede'.
    """
    ledger = _ledger()

    fact_a_payload = {
        'subject': 'user:principal',
        'predicate': 'pref.missing_conflict_id_test',
        'scope': 'private',
        'assertion_type': 'preference',
        'value': 'vim',
        'evidence_refs': [{'source_key': 's:1', 'evidence_id': 'ev-a', 'scope': 's1_sessions_main'}],
    }
    fact_b_payload = {
        'subject': 'user:principal',
        'predicate': 'pref.missing_conflict_id_test',  # same conflict set
        'scope': 'private',
        'assertion_type': 'preference',
        'value': 'emacs',
        'evidence_refs': [{'source_key': 's:1', 'evidence_id': 'ev-b', 'scope': 's1_sessions_main'}],
    }

    result_a = ledger.promote_candidate_fact(
        actor_id='test:actor',
        reason='promote A',
        policy_version='v3',
        candidate_id='cand-no-conflict-id-a',
        fact=fact_a_payload,
        # conflict_with_fact_id intentionally absent
    )

    current_after_a = ledger.current_state_facts()
    assert len(current_after_a) == 1
    assert isinstance(current_after_a[0], StateFact)
    assert current_after_a[0].value == 'vim'

    result_b = ledger.promote_candidate_fact(
        actor_id='test:actor',
        reason='promote B — same conflict set, no conflict_with_fact_id',
        policy_version='v3',
        candidate_id='cand-no-conflict-id-b',
        fact=fact_b_payload,
        # conflict_with_fact_id intentionally absent — central enforcement must handle
    )

    current_after_b = ledger.current_state_facts()
    assert len(current_after_b) == 1, (
        f'Expected exactly 1 current fact after second promotion (no conflict_id), '
        f'got {len(current_after_b)}: '
        + str([(f.object_id, f.value) for f in current_after_b])
    )
    assert current_after_b[0].value == 'emacs', (
        f'Expected fact B (emacs) to be current, got: {current_after_b[0].value!r}'
    )

    # Lineage: A must have been superseded and share root with B
    lineage = ledger.materialize_lineage(result_a.root_id)
    assert len(lineage) == 2, f'Expected 2 objects in lineage, got {len(lineage)}'
    fact_a_obj = next(obj for obj in lineage if obj.object_id == result_a.object_id)
    fact_b_obj = next(obj for obj in lineage if obj.object_id == result_b.object_id)

    assert fact_a_obj.is_current is False, 'Fact A must not be current after supersession'
    assert fact_a_obj.superseded_by == fact_b_obj.object_id
    assert fact_b_obj.root_id == result_a.root_id, 'Fact B must inherit fact A root'
    assert fact_b_obj.version == 2


def test_stale_conflict_id_does_not_yield_two_current_facts():
    """A stale conflict_with_fact_id (pointing to a non-current predecessor) must
    not create two current facts.  The central enforcement scan must detect that
    the stale ID is no longer the current occupant of the conflict set and
    redirect the supersession to the actual current fact.

    Scenario: v1 superseded by v2; caller tries to promote v3 with
    conflict_with_fact_id=v1 (stale).  Without the fix, v3 would supersede v1
    (already non-current) while v2 remains current — two current facts in the
    same conflict set.
    """
    ledger = _ledger()

    def _fact(cand_id: str, val: str, ev_id: str) -> dict:
        return {
            'subject': 'user:principal',
            'predicate': 'pref.stale_conflict_test',
            'scope': 'private',
            'assertion_type': 'preference',
            'value': val,
            'evidence_refs': [{'source_key': 's:1', 'evidence_id': ev_id, 'scope': 's1_sessions_main'}],
        }

    # Promote v1 (becomes current)
    r1 = ledger.promote_candidate_fact(
        actor_id='test:actor', reason='v1', policy_version='v3',
        candidate_id='cand-stale-v1', fact=_fact('cand-stale-v1', 'one', 'ev-stale-1'),
    )
    # Promote v2 superseding v1 (v2 becomes current)
    r2 = ledger.promote_candidate_fact(
        actor_id='test:actor', reason='v2', policy_version='v3',
        candidate_id='cand-stale-v2', fact=_fact('cand-stale-v2', 'two', 'ev-stale-2'),
        conflict_with_fact_id=r1.object_id,
    )
    assert len(ledger.current_state_facts()) == 1
    assert ledger.current_state_facts()[0].value == 'two'

    # Promote v3 with STALE conflict_with_fact_id pointing at v1 (not v2)
    r3 = ledger.promote_candidate_fact(
        actor_id='test:actor', reason='v3 with stale conflict id', policy_version='v3',
        candidate_id='cand-stale-v3', fact=_fact('cand-stale-v3', 'three', 'ev-stale-3'),
        conflict_with_fact_id=r1.object_id,   # stale: v1 is not current
    )

    current_after_v3 = ledger.current_state_facts()
    assert len(current_after_v3) == 1, (
        f'Expected exactly 1 current fact after v3 promotion with stale conflict_id, '
        f'got {len(current_after_v3)}: '
        + str([(f.object_id, f.value) for f in current_after_v3])
    )
    assert current_after_v3[0].value == 'three', (
        f'Expected v3 to be current, got: {current_after_v3[0].value!r}'
    )

    # v2 must now be superseded
    v2_obj = ledger.materialize_object(r2.object_id)
    assert v2_obj is not None
    assert v2_obj.is_current is False, 'v2 must be superseded after v3 promotion'


def test_canonical_uri_path_segment_with_slash_is_encoded():
    """A locator component containing a literal slash must be percent-encoded.

    Unencoded slashes in path segments create ambiguous URIs where
    'a/b' and 'a' + sub-path 'b' are indistinguishable.
    """
    # message_id that contains a slash (edge case, e.g. group/thread IDs)
    ref = EvidenceRef(
        kind='message',
        source_system='slack',
        locator={
            'system': 'slack',
            'conversation_id': 'C1234',
            'message_id': 'p1234567890/123456',  # Slack thread ID format
        },
    )
    # The slash in message_id must be encoded as %2F
    assert '%2F' in ref.canonical_uri, (
        f'Slash in message_id was not encoded: {ref.canonical_uri!r}'
    )
    # The rest of the URI structure must still be correct
    assert ref.canonical_uri.startswith('msg://slack/C1234/')


# ─────────────────────────────────────────────────────────────────────────────
# P1 Blocker — one-current-object rule enforced under multiple SQLite connections
# ─────────────────────────────────────────────────────────────────────────────


def test_concurrent_connections_cannot_produce_two_current_facts(tmp_path):
    """Two concurrent SQLite connections racing to promote facts with the same
    conflict set (subject + predicate + scope) must never yield two
    is_current=True facts.

    Regression test for the check-then-act race in promote_candidate_fact():
    without BEGIN IMMEDIATE the conflict-set scan and the write are not
    serialized, so a second writer can slip in between them and both promotions
    land as independent 'assert' events — two current facts in one conflict set,
    a direct violation of the Phase 0 architecture lock.

    The fix: BEGIN IMMEDIATE in promote_candidate_fact() acquires the write
    reservation before reading current state, forcing all concurrent promotions
    to serialize at that gate.  With busy_timeout=5000 both promotions
    eventually succeed (serialized); the second one detects the first as the
    current occupant and issues a 'supersede' instead of 'assert'.  Final
    state: exactly one current fact.
    """
    from mcp_server.src.services.change_ledger import ChangeLedger

    db_path = tmp_path / 'test_concurrent_current.db'

    # Prime the schema by connecting once from the main thread before the race.
    ChangeLedger(db_path)

    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def _fact_payload(val: str, ev_id: str) -> dict:
        return {
            'subject': 'user:principal',
            'predicate': 'pref.concurrent_current_test',
            'scope': 'private',
            'assertion_type': 'preference',
            'value': val,
            'evidence_refs': [
                {'source_key': 's:1', 'evidence_id': ev_id, 'scope': 's1_sessions_main'}
            ],
        }

    def _promote(val: str, cand_id: str, ev_id: str) -> None:
        # Each thread creates its own ChangeLedger (and SQLite connection)
        # so there are no cross-thread connection ownership issues.
        ledger = ChangeLedger(db_path)
        try:
            barrier.wait()  # synchronize so both threads start together
            ledger.promote_candidate_fact(
                actor_id='test:actor',
                reason='concurrent race test',
                policy_version='v3',
                candidate_id=cand_id,
                fact=_fact_payload(val, ev_id),
            )
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=_promote, args=('vim', 'cand-race-1', 'ev-race-1'))
    t2 = threading.Thread(target=_promote, args=('emacs', 'cand-race-2', 'ev-race-2'))
    t1.start()
    t2.start()
    t1.join(timeout=15)
    t2.join(timeout=15)

    # Errors must only be SQLITE_BUSY (acceptable if busy_timeout is exceeded).
    # Any other exception type indicates a logic bug, not a resource contention.
    for exc in errors:
        assert isinstance(exc, sqlite3.OperationalError), (
            f'Unexpected error type in concurrent promotion: '
            f'{type(exc).__name__}: {exc}'
        )

    # Read current state through a fresh connection so we see the committed
    # state of both writers, not the in-memory state of either ledger.
    ledger3 = ChangeLedger(db_path)
    all_current = ledger3.current_state_facts()
    current_for_pred = [
        f for f in all_current
        if f.predicate == 'pref.concurrent_current_test'
    ]

    assert len(current_for_pred) == 1, (
        f'Expected exactly 1 current fact in conflict set after concurrent '
        f'promotions (BEGIN IMMEDIATE serialization fix), '
        f'got {len(current_for_pred)}: '
        + str([(f.object_id, f.value) for f in current_for_pred])
    )
