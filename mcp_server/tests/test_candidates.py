"""Exec 4 candidate lifecycle tests."""

from __future__ import annotations

from pathlib import Path

import asyncio
import pytest

from mcp_server.src.models.typed_memory import EvidenceRef, StateFact
from mcp_server.src.routers import candidates as candidates_router
from mcp_server.src.services.change_ledger import ChangeLedger


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

    assert asyncio.run(candidates_router.list_candidates()) == asyncio.run(candidates_router.list_candidates(status='quarantine'))
    state_candidates = asyncio.run(candidates_router.list_candidates(status='quarantine', type_filter='state'))
    assert len(state_candidates) == 1
    assert state_candidates[0]['uuid'] == 'cand-b'


def test_promote_candidate_supersede_marks_candidate_and_promotes_fact(ledger_path: Path):
    db = candidates_router.CandidatesDB(ledger_path)
    ledger = ChangeLedger(ledger_path)

    existing = _seed_state_fact(subject='Yuan', predicate='prefers', value='cold brew', object_id='existing-fact')
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

    # Superseded fact should no longer be current after promotion.
    active = [f for f in ledger.current_state_facts() if f.predicate == 'prefers']
    active_ids = {f.object_id for f in active}
    assert existing.object_id not in active_ids
    assert result['promotion']['object_id'] in active_ids


def test_promote_candidate_parallel_keeps_conflicting_fact_current(ledger_path: Path):
    db = candidates_router.CandidatesDB(ledger_path)
    ledger = ChangeLedger(ledger_path)

    existing = _seed_state_fact(subject='Yuan', predicate='prefers', value='cold brew', object_id='existing-fact-p')
    ledger.append_event('assert', actor_id='seed', reason='seed', payload=existing, object_id=existing.object_id)

    _make_candidate(
        db=db,
        candidate_id='cand-parallel',
        value='tea',
        conflicting_fact_uuid=existing.object_id,
    )

    result = asyncio.run(candidates_router.promote_candidate('cand-parallel', resolution='parallel'))
    assert result.get('candidate', {}).get('status') == 'promoted'

    # Both facts should remain in current state when resolved as parallel.
    promoted_id = result.get('promotion', {}).get('object_id')
    assert promoted_id
    active = [f for f in ledger.current_state_facts() if f.predicate == 'prefers']
    ids = {f.object_id for f in active}
    assert existing.object_id in ids
    assert promoted_id in ids


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

    assert asyncio.run(candidates_router.promote_candidate(candidate_id='missing', resolution='parallel')).get('error') == 'Candidate not found: missing'
    assert asyncio.run(candidates_router.promote_candidate(candidate_id=candidate_id, resolution='parallel')).get('error') == (
        f'Candidate {candidate_id} is already rejected'
    )

    assert asyncio.run(candidates_router.reject_candidate(candidate_id='missing')).get('error') == 'Candidate not found: missing'
    assert asyncio.run(candidates_router.reject_candidate(candidate_id=candidate_id)).get('error') == (
        f'Candidate {candidate_id} is already rejected'
    )

    assert 'error' in asyncio.run(candidates_router.promote_candidate(candidate_id='new-id', resolution='bogus'))
