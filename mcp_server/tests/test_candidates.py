"""Exec 4 candidate lifecycle tests."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from mcp_server.src.models.typed_memory import EvidenceRef, StateFact
from mcp_server.src.routers import candidates as candidates_router
from mcp_server.src.services.change_ledger import ChangeLedger
from mcp_server.src.services.schema_validation import _validate_typed_object


@pytest.fixture
def ledger_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / 'change_ledger.db'
    monkeypatch.setenv('BICAMERAL_CHANGE_LEDGER_PATH', str(path))
    return path


def _make_candidate(
    *,
    db: candidates_router.CandidatesDB,
    candidate_id: str,
    candidate_type: str = 'preference',
    subject: str = 'Yuan',
    predicate: str = 'prefers',
    value: str = 'espresso',
    conflicting_fact_uuid: str | None = None,
    status: str = 'quarantine',
    confidence: float = 0.9,
):
    db.insert_candidate(
        uuid=candidate_id,
        type=candidate_type,
        subject=subject,
        predicate=predicate,
        value=value,
        conflicting_fact_uuid=conflicting_fact_uuid,
        confidence=confidence,
        status=status,
        metadata={
            'source': 'test',
            'evidence_refs': [
                {
                    'evidence_id': f'eid-{candidate_id}',
                    'source_key': f'source-{candidate_id}',
                    'scope': 'test',
                }
            ],
        },
    )


def _seed_state_fact(*, subject: str, predicate: str, value: str, object_id: str = 'existing') -> StateFact:
    evidence = EvidenceRef.model_validate(
        {
            'kind': 'event_log',
            'source_system': 'tests',
            'locator': {
                'system': 'pytest',
                'stream': 'candidates',
                'event_id': f'evt-{object_id}',
            },
            'observed_at': '2026-03-10T09:00:00Z',
            'retrieved_at': '2026-03-10T09:00:00Z',
        }
    )
    return StateFact.model_validate(
        {
            'object_id': object_id,
            'root_id': object_id,
            'version': 1,
            'fact_type': 'preference',
            'subject': subject,
            'predicate': predicate,
            'value': value,
            'scope': 'private',
            'policy_scope': 'private',
            'visibility_scope': 'private',
            'evidence_refs': [evidence],
            'created_at': '2026-03-10T09:00:00Z',
            'valid_at': '2026-03-10T09:00:00Z',
            'promotion_status': 'promoted',
        }
    )


def test_list_candidates_default_scope(ledger_path: Path):
    db = candidates_router.CandidatesDB(ledger_path)
    _make_candidate(db=db, candidate_id='cand-a', candidate_type='preference', confidence=0.4)
    _make_candidate(db=db, candidate_id='cand-b', candidate_type='state', confidence=0.95)
    _make_candidate(db=db, candidate_id='cand-c', candidate_type='state', status='promoted', confidence=0.99)

    assert asyncio.run(candidates_router.list_candidates()) == asyncio.run(
        candidates_router.list_candidates(status='quarantine')
    )
    state_candidates = asyncio.run(candidates_router.list_candidates(status='quarantine', type_filter='state'))
    assert len(state_candidates) == 1
    assert state_candidates[0]['uuid'] == 'cand-b'


def test_candidates_schema_migrates_legacy_rows_to_exec4_contract(ledger_path: Path):
    conn = sqlite3.connect(str(ledger_path))
    conn.execute(
        '''
        CREATE TABLE candidates (
            uuid TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            value TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        '''
    )
    conn.execute(
        '''
        INSERT INTO candidates(uuid, type, subject, predicate, value, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (
            'cand-legacy',
            'preference',
            'Yuan',
            'prefers',
            '"tea"',
            'pending',
            '2026-03-10T09:00:00Z',
            '2026-03-10T09:00:00Z',
        ),
    )
    conn.commit()
    conn.close()

    db = candidates_router.CandidatesDB(ledger_path)
    migrated = db.get_candidate('cand-legacy')
    assert migrated is not None
    assert migrated['status'] == 'quarantine'
    assert migrated['confidence'] == 0.0
    assert migrated['resolution'] is None
    assert migrated['metadata'] is None
    assert migrated['conflicting_fact_uuid'] is None

    columns = {
        row['name']
        for row in db.conn.execute("PRAGMA table_info('candidates')").fetchall()
    }
    assert {'conflicting_fact_uuid', 'resolution', 'confidence', 'reviewed_at', 'reviewed_by', 'promoted_at', 'promoted_by', 'reason', 'metadata_json'} <= columns

    quarantine = asyncio.run(candidates_router.list_candidates())
    assert [row['uuid'] for row in quarantine] == ['cand-legacy']


def test_candidate_rows_match_public_schema(ledger_path: Path):
    db = candidates_router.CandidatesDB(ledger_path)
    _make_candidate(db=db, candidate_id='cand-schema', candidate_type='procedure', confidence=0.66)

    candidates = asyncio.run(candidates_router.list_candidates(status='quarantine'))
    assert len(candidates) == 1

    ok, err = _validate_typed_object(candidates[0], 'Candidate')
    assert ok is True, err


def test_promote_candidate_supersede_marks_candidate_and_promotes_fact(ledger_path: Path):
    db = candidates_router.CandidatesDB(ledger_path)
    ledger = ChangeLedger(ledger_path)

    existing = _seed_state_fact(
        subject='Yuan',
        predicate='prefers',
        value='cold brew',
        object_id='existing-fact',
    )
    ledger.append_event(
        'assert',
        actor_id='seed',
        reason='seed',
        payload=existing,
        object_id=existing.object_id,
        object_type=existing.object_type,
        root_id=existing.root_id,
    )

    _make_candidate(
        db=db,
        candidate_id='cand-promote',
        value='pour over',
        conflicting_fact_uuid=existing.object_id,
    )

    result = asyncio.run(candidates_router.promote_candidate('cand-promote', resolution='supersede'))
    assert result.get('candidate', {}).get('status') == 'promoted'
    assert result['promotion']['event_id']

    candidate = db.get_candidate('cand-promote')
    assert candidate is not None
    assert candidate['status'] == 'promoted'
    assert candidate['resolution'] == 'supersede'
    assert candidate['promoted_by'] == 'system'
    assert candidate['promoted_at']
    assert candidate['reason'] == 'candidate_promoted_supersede'

    active = [fact for fact in ledger.current_state_facts() if fact.predicate == 'prefers']
    active_ids = {fact.object_id for fact in active}
    assert existing.object_id not in active_ids
    assert result['promotion']['object_id'] in active_ids


def test_promote_candidate_parallel_keeps_conflicting_fact_current(ledger_path: Path):
    db = candidates_router.CandidatesDB(ledger_path)
    ledger = ChangeLedger(ledger_path)

    existing = _seed_state_fact(
        subject='Yuan',
        predicate='prefers',
        value='cold brew',
        object_id='existing-fact-p',
    )
    ledger.append_event('assert', actor_id='seed', reason='seed', payload=existing, object_id=existing.object_id)

    _make_candidate(
        db=db,
        candidate_id='cand-parallel',
        value='tea',
        conflicting_fact_uuid=existing.object_id,
    )

    result = asyncio.run(candidates_router.promote_candidate('cand-parallel', resolution='parallel'))
    assert result.get('candidate', {}).get('status') == 'promoted'

    promoted_id = result.get('promotion', {}).get('object_id')
    assert promoted_id
    active = [fact for fact in ledger.current_state_facts() if fact.predicate == 'prefers']
    ids = {fact.object_id for fact in active}
    assert existing.object_id in ids
    assert promoted_id in ids


def test_promote_candidate_supersede_requires_materialized_target(ledger_path: Path):
    db = candidates_router.CandidatesDB(ledger_path)
    ledger = ChangeLedger(ledger_path)

    _make_candidate(
        db=db,
        candidate_id='cand-missing-target',
        value='oolong',
        conflicting_fact_uuid='missing-fact',
    )

    result = asyncio.run(candidates_router.promote_candidate('cand-missing-target', resolution='supersede'))
    assert result == {
        'error': "promote_candidate failed: explicit supersede requested but no valid supersede target was materialized for 'missing-fact'"
    }

    candidate = db.get_candidate('cand-missing-target')
    assert candidate is not None
    assert candidate['status'] == 'quarantine'
    assert candidate['resolution'] is None
    assert ledger.current_state_facts() == []
    assert ledger.conn.execute("SELECT count(*) FROM change_events").fetchone()[0] == 0


def test_promote_candidate_rejects_heterogeneous_supersede_target(ledger_path: Path):
    db = candidates_router.CandidatesDB(ledger_path)
    ledger = ChangeLedger(ledger_path)

    existing = _seed_state_fact(subject='Yuan', predicate='prefers', value='cold brew', object_id='existing-fact-h')
    ledger.append_event('assert', actor_id='seed', reason='seed', payload=existing, object_id=existing.object_id)

    _make_candidate(
        db=db,
        candidate_id='cand-procedure',
        candidate_type='procedure',
        subject='brew coffee',
        predicate='procedure',
        value={'steps': ['grind beans', 'brew']},
        conflicting_fact_uuid=existing.object_id,
    )

    result = asyncio.run(candidates_router.promote_candidate('cand-procedure', resolution='supersede'))
    assert 'error' in result
    assert 'incompatible supersede target' in result['error']

    candidate = db.get_candidate('cand-procedure')
    assert candidate is not None
    assert candidate['status'] == 'quarantine'

    active = [fact for fact in ledger.current_state_facts() if fact.predicate == 'prefers']
    assert {fact.object_id for fact in active} == {existing.object_id}
    assert all(obj.object_type != 'procedure' for obj in ledger.materialize_lineage(existing.root_id))


def test_promote_candidate_rejects_state_fact_supersede_across_conflict_sets(ledger_path: Path):
    db = candidates_router.CandidatesDB(ledger_path)
    ledger = ChangeLedger(ledger_path)

    existing = _seed_state_fact(subject='Yuan', predicate='prefers', value='cold brew', object_id='existing-fact-cs')
    ledger.append_event('assert', actor_id='seed', reason='seed', payload=existing, object_id=existing.object_id)

    _make_candidate(
        db=db,
        candidate_id='cand-mismatch',
        candidate_type='state',
        subject='Archibald',
        predicate='lives_in',
        value='New York',
        conflicting_fact_uuid=existing.object_id,
    )

    result = asyncio.run(candidates_router.promote_candidate('cand-mismatch', resolution='supersede'))
    assert 'error' in result
    assert 'conflict set mismatch' in result['error']

    candidate = db.get_candidate('cand-mismatch')
    assert candidate is not None
    assert candidate['status'] == 'quarantine'
    assert {fact.object_id for fact in ledger.current_state_facts()} == {existing.object_id}


def test_promote_candidate_reconciles_existing_ledger_promotion(ledger_path: Path):
    db = candidates_router.CandidatesDB(ledger_path)

    _make_candidate(db=db, candidate_id='cand-reconcile', value='flat white')
    first = asyncio.run(candidates_router.promote_candidate('cand-reconcile', resolution='parallel'))
    assert first.get('candidate', {}).get('status') == 'promoted'

    db.conn.execute(
        """
        UPDATE candidates
           SET status = 'quarantine',
               resolution = NULL,
               reviewed_at = NULL,
               reviewed_by = NULL,
               promoted_at = NULL,
               promoted_by = NULL,
               reason = NULL
         WHERE uuid = ?
        """,
        ('cand-reconcile',),
    )
    db.conn.commit()

    reconciled = asyncio.run(candidates_router.promote_candidate('cand-reconcile', resolution='parallel'))
    assert reconciled.get('candidate', {}).get('status') == 'promoted'
    assert reconciled['promotion']['object_id'] == first['promotion']['object_id']
    assert reconciled['promotion']['event_id'] == first['promotion']['event_id']

    candidate = db.get_candidate('cand-reconcile')
    assert candidate is not None
    assert candidate['status'] == 'promoted'
    assert candidate['resolution'] == 'parallel'


def test_cancel_and_reject_record_audit_fields(ledger_path: Path):
    db = candidates_router.CandidatesDB(ledger_path)
    _make_candidate(db=db, candidate_id='cand-cancel')
    _make_candidate(db=db, candidate_id='cand-reject')

    cancel_result = asyncio.run(candidates_router.promote_candidate('cand-cancel', resolution='cancel'))
    assert cancel_result.get('candidate', {}).get('status') == 'rejected'

    reject_result = asyncio.run(candidates_router.reject_candidate('cand-reject'))
    assert reject_result.get('candidate', {}).get('status') == 'rejected'

    cancel_candidate = db.get_candidate('cand-cancel')
    reject_candidate = db.get_candidate('cand-reject')
    assert cancel_candidate is not None
    assert reject_candidate is not None
    assert cancel_candidate['reviewed_by'] == 'system'
    assert reject_candidate['reviewed_by'] == 'system'
    assert cancel_candidate['resolution'] == 'cancel'
    assert cancel_candidate['reason'] == 'candidate_cancelled'
    assert reject_candidate['reason'] == 'candidate_rejected'
    assert cancel_candidate['promoted_by'] is None
    assert cancel_candidate['reviewed_at']


def test_list_candidates_invalid_filters_return_empty(ledger_path: Path):
    db = candidates_router.CandidatesDB(ledger_path)
    _make_candidate(db=db, candidate_id='cand-1', status='quarantine')

    assert asyncio.run(candidates_router.list_candidates(status='does-not-exist')) == []
    assert asyncio.run(candidates_router.list_candidates(age_days=0)) == []
    assert asyncio.run(candidates_router.list_candidates(min_confidence=2.0)) == []


def test_promote_and_reject_invalid_state_returns_error(ledger_path: Path):
    db = candidates_router.CandidatesDB(ledger_path)
    candidate_id = 'cand-invalid'
    _make_candidate(db=db, candidate_id=candidate_id, status='rejected')

    assert asyncio.run(
        candidates_router.promote_candidate(candidate_id='missing', resolution='parallel')
    ).get('error') == 'Candidate not found: missing'
    assert asyncio.run(
        candidates_router.promote_candidate(candidate_id=candidate_id, resolution='parallel')
    ).get('error') == f'Candidate {candidate_id} is already rejected'

    assert asyncio.run(candidates_router.reject_candidate(candidate_id='missing')).get('error') == (
        'Candidate not found: missing'
    )
    assert asyncio.run(candidates_router.reject_candidate(candidate_id=candidate_id)).get('error') == (
        f'Candidate {candidate_id} is already rejected'
    )

    assert 'error' in asyncio.run(candidates_router.promote_candidate(candidate_id='new-id', resolution='bogus'))


def test_candidate_lifecycle_end_to_end_flow(ledger_path: Path):
    db = candidates_router.CandidatesDB(ledger_path)
    ledger = ChangeLedger(ledger_path)

    existing = _seed_state_fact(
        subject='Yuan',
        predicate='prefers',
        value='cold brew',
        object_id='existing-e2e',
    )
    ledger.append_event('assert', actor_id='seed', reason='seed', payload=existing, object_id=existing.object_id)

    _make_candidate(
        db=db,
        candidate_id='cand-e2e-promote',
        candidate_type='Preference',
        value='matcha',
        conflicting_fact_uuid=existing.object_id,
    )
    _make_candidate(
        db=db,
        candidate_id='cand-e2e-reject',
        candidate_type='Preference',
        value='latte',
    )

    quarantine = asyncio.run(candidates_router.list_candidates())
    by_id = {row['uuid']: row for row in quarantine}
    assert by_id['cand-e2e-promote']['status'] == 'quarantine'
    assert by_id['cand-e2e-promote']['type'] == 'Preference'

    promote_result = asyncio.run(
        candidates_router.promote_candidate('cand-e2e-promote', resolution='supersede')
    )
    assert promote_result['candidate']['status'] == 'promoted'
    assert promote_result['candidate']['resolution'] == 'supersede'

    promoted = asyncio.run(candidates_router.list_candidates(status='promoted'))
    promoted_ids = {row['uuid'] for row in promoted}
    assert 'cand-e2e-promote' in promoted_ids

    reject_result = asyncio.run(candidates_router.reject_candidate('cand-e2e-reject'))
    assert reject_result['candidate']['status'] == 'rejected'

    rejected = asyncio.run(candidates_router.list_candidates(status='rejected'))
    rejected_ids = {row['uuid'] for row in rejected}
    assert 'cand-e2e-reject' in rejected_ids

    active = [fact for fact in ledger.current_state_facts() if fact.predicate == 'prefers']
    active_ids = {fact.object_id for fact in active}
    assert existing.object_id not in active_ids
    assert promote_result['promotion']['object_id'] in active_ids


def test_promote_candidate_serializes_concurrent_calls(ledger_path: Path):
    db = candidates_router.CandidatesDB(ledger_path)
    ledger = ChangeLedger(ledger_path)

    existing = _seed_state_fact(
        subject='Yuan',
        predicate='prefers',
        value='cold brew',
        object_id='existing-race',
    )
    ledger.append_event('assert', actor_id='seed', reason='seed', payload=existing, object_id=existing.object_id)

    _make_candidate(
        db=db,
        candidate_id='cand-race',
        value='pour over',
        conflicting_fact_uuid=existing.object_id,
    )

    barrier = threading.Barrier(3)

    def _promote_once() -> dict[str, object]:
        barrier.wait(timeout=5)
        return asyncio.run(candidates_router.promote_candidate('cand-race', resolution='supersede'))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_promote_once), pool.submit(_promote_once)]
        barrier.wait(timeout=5)
        results = [future.result(timeout=10) for future in futures]

    successes = [result for result in results if 'promotion' in result]
    errors = [result for result in results if 'error' in result]
    assert len(successes) == 1
    assert len(errors) == 1
    assert errors[0]['error'] == 'Candidate cand-race is already promoted'

    promote_event_count = int(
        ledger.conn.execute(
            "SELECT count(*) FROM change_events WHERE candidate_id = ? AND event_type = 'promote'",
            ('cand-race',),
        ).fetchone()[0]
    )
    assert promote_event_count == 1

    candidate = db.get_candidate('cand-race')
    assert candidate is not None
    assert candidate['status'] == 'promoted'
