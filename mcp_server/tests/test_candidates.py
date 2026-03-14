"""Candidate lifecycle tests — adapted for the CandidateStore / integrated-surface API.

These tests exercise the same behavioral contracts as the original CandidatesDB
tests but against the new CandidateStore + candidates router (integrated branch).
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from mcp_server.src.models.typed_memory import EvidenceRef, StateFact
from mcp_server.src.routers import candidates as candidates_router
from mcp_server.src.services.candidate_store import CandidateStore
from mcp_server.src.services.change_ledger import ChangeLedger
from mcp_server.src.services.schema_validation import _validate_typed_object


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store_and_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Shared in-temp-dir store + ledger + auth bypass for all tests."""
    ledger_path = tmp_path / 'change_ledger.db'
    candidates_path = tmp_path / 'candidates.db'

    monkeypatch.setenv('BICAMERAL_CHANGE_LEDGER_PATH', str(ledger_path))
    monkeypatch.setenv('BICAMERAL_CANDIDATES_DB_PATH', str(candidates_path))

    # Reset module-level cached singletons so env-var paths take effect.
    monkeypatch.setattr(candidates_router, '_change_ledger', None)
    monkeypatch.setattr(candidates_router, '_candidate_store', None)

    # Bypass reviewer auth — tests use direct store/ledger manipulation.
    monkeypatch.setattr(candidates_router, '_require_reviewer', lambda ctx: 'system')

    store = CandidateStore(db_path=candidates_path)
    ledger = ChangeLedger(ledger_path)
    yield store, ledger
    ledger.close()


def _make_candidate(
    *,
    store: CandidateStore,
    candidate_id: str,
    fact_type: str = 'preference',
    subject: str = 'Yuan',
    predicate: str = 'prefers',
    value: str = 'espresso',
    conflict_with_fact_id: str | None = None,
    status: str = 'pending',
    confidence: float = 0.9,
):
    store.create_candidate(
        payload={
            'subject': subject,
            'predicate': predicate,
            'value': value,
            'fact_type': fact_type,
        },
        status=status,
        conflict_with_fact_id=conflict_with_fact_id,
        candidate_id=candidate_id,
        metadata={'confidence': confidence, 'source': 'test'},
        raw_hint={
            'evidence_refs': [
                {
                    'evidence_id': f'eid-{candidate_id}',
                    'source_key': f'source-{candidate_id}',
                    'scope': 'test',
                }
            ],
        },
    )


def _seed_state_fact(
    *,
    subject: str,
    predicate: str,
    value: str,
    object_id: str = 'existing',
    source_lane: str | None = None,
) -> StateFact:
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
            'source_lane': source_lane,
            'evidence_refs': [evidence],
            'created_at': '2026-03-10T09:00:00Z',
            'valid_at': '2026-03-10T09:00:00Z',
            'promotion_status': 'promoted',
        }
    )


# ---------------------------------------------------------------------------
# list_candidates
# ---------------------------------------------------------------------------

def test_list_candidates_default_scope(store_and_ledger):
    store, ledger = store_and_ledger
    _make_candidate(store=store, candidate_id='cand-a', fact_type='preference', confidence=0.4)
    _make_candidate(store=store, candidate_id='cand-b', fact_type='state', confidence=0.95)
    _make_candidate(store=store, candidate_id='cand-c', fact_type='state', status='promoted', confidence=0.99)

    result_default = asyncio.run(candidates_router.list_candidates())
    result_quarantine = asyncio.run(candidates_router.list_candidates(status='quarantine'))
    # Default should match explicit quarantine filter
    assert result_default['candidates'] == result_quarantine['candidates']


def test_candidates_schema_migrates_legacy_rows_to_exec4_contract(store_and_ledger):
    """CandidateStore correctly exposes the Exec-4 public schema shape."""
    store, ledger = store_and_ledger
    _make_candidate(store=store, candidate_id='cand-legacy', fact_type='preference', confidence=0.0)

    result = asyncio.run(candidates_router.list_candidates())
    assert result['status'] == 'ok'
    candidates = result['candidates']
    assert len(candidates) == 1
    cand = candidates[0]
    # Status is 'quarantine' in public contract (not internal 'pending')
    assert cand['status'] == 'quarantine'
    assert cand['uuid'] == 'cand-legacy'


def test_candidate_rows_match_public_schema(store_and_ledger):
    store, ledger = store_and_ledger
    _make_candidate(store=store, candidate_id='cand-schema', fact_type='procedure', confidence=0.66)

    result = asyncio.run(candidates_router.list_candidates(status='quarantine'))
    candidates = result['candidates']
    assert len(candidates) == 1

    # Validate against the public Candidate schema (Candidate.json).
    # Strip internal storage fields and backward-compat aliases that are not
    # part of the public contract.
    _PUBLIC_CANDIDATE_FIELDS = frozenset({
        'uuid', 'type', 'subject', 'predicate', 'value', 'conflicting_fact_uuid',
        'status', 'created_at', 'updated_at', 'reviewed_at', 'reviewed_by',
        'promoted_at', 'promoted_by', 'reason', 'resolution', 'confidence', 'metadata',
    })
    candidate_public = {k: v for k, v in candidates[0].items() if k in _PUBLIC_CANDIDATE_FIELDS}
    ok, err = _validate_typed_object(candidate_public, 'Candidate')
    assert ok is True, err


# ---------------------------------------------------------------------------
# promote_candidate — supersede
# ---------------------------------------------------------------------------

def test_promote_candidate_supersede_marks_candidate_and_promotes_fact(store_and_ledger):
    store, ledger = store_and_ledger

    existing = _seed_state_fact(
        subject='Yuan', predicate='prefers', value='cold brew', object_id='existing-fact',
    )
    ledger.append_event(
        'assert', actor_id='seed', reason='seed', payload=existing,
        object_id=existing.object_id, object_type=existing.object_type, root_id=existing.root_id,
    )

    _make_candidate(
        store=store, candidate_id='cand-promote', value='pour over',
        conflict_with_fact_id=existing.object_id,
    )

    result = asyncio.run(candidates_router.promote_candidate('cand-promote', resolution='supersede'))
    assert result.get('status') == 'ok', result
    assert result.get('candidate', {}).get('status') == 'promoted'
    assert result['promotion']['event_id']

    candidate = store.get_candidate('cand-promote')
    assert candidate is not None
    assert candidate['status'] == 'promoted'
    assert candidate['resolution'] == 'supersede'

    active = [fact for fact in ledger.current_state_facts() if fact.predicate == 'prefers']
    active_ids = {fact.object_id for fact in active}
    assert existing.object_id not in active_ids
    assert result['promotion']['object_id'] in active_ids


def test_promote_candidate_parallel_resolution_rejected_on_integrated_surface(store_and_ledger):
    """The integrated surface rejects the 'parallel' resolution as unsupported."""
    store, ledger = store_and_ledger

    existing = _seed_state_fact(
        subject='Yuan', predicate='prefers', value='cold brew', object_id='existing-fact-p',
    )
    ledger.append_event('assert', actor_id='seed', reason='seed', payload=existing,
                        object_id=existing.object_id)

    _make_candidate(
        store=store, candidate_id='cand-parallel', value='tea',
        conflict_with_fact_id=existing.object_id,
    )

    # parallel is explicitly rejected on the integrated surface
    result = asyncio.run(candidates_router.promote_candidate('cand-parallel', resolution='parallel'))
    assert result.get('status') == 'error', result
    assert result.get('error_type') == 'validation_error'
    # The original fact is still current — no side effects from the rejected resolution
    active = [f for f in ledger.current_state_facts() if f.predicate == 'prefers']
    assert {f.object_id for f in active} == {existing.object_id}


def test_promote_candidate_supersede_requires_materialized_target(store_and_ledger):
    store, ledger = store_and_ledger

    _make_candidate(
        store=store, candidate_id='cand-missing-target', value='oolong',
        conflict_with_fact_id='missing-fact',
    )

    result = asyncio.run(candidates_router.promote_candidate('cand-missing-target', resolution='supersede'))
    assert 'error' in result or result.get('status') == 'error', result
    # Candidate should remain in pending state
    candidate = store.get_candidate('cand-missing-target')
    assert candidate is not None
    assert candidate['status'] == 'pending'
    assert candidate.get('resolution') is None
    assert ledger.current_state_facts() == []
    assert ledger.conn.execute("SELECT count(*) FROM change_events").fetchone()[0] == 0


def test_promote_candidate_rejects_heterogeneous_supersede_target(store_and_ledger):
    store, ledger = store_and_ledger

    existing = _seed_state_fact(
        subject='Yuan', predicate='prefers', value='cold brew', object_id='existing-fact-h',
    )
    ledger.append_event('assert', actor_id='seed', reason='seed', payload=existing,
                        object_id=existing.object_id)

    _make_candidate(
        store=store, candidate_id='cand-procedure', fact_type='procedure',
        subject='brew coffee', predicate='procedure',
        value={'steps': ['grind beans', 'brew']},
        conflict_with_fact_id=existing.object_id,
    )

    result = asyncio.run(candidates_router.promote_candidate('cand-procedure', resolution='supersede'))
    assert 'error' in result or result.get('status') == 'error', result
    err_msg = result.get('message', '') + result.get('error', '')
    assert 'incompatible' in err_msg or 'supersede' in err_msg

    candidate = store.get_candidate('cand-procedure')
    assert candidate is not None
    assert candidate['status'] == 'pending'

    active = [fact for fact in ledger.current_state_facts() if fact.predicate == 'prefers']
    assert {fact.object_id for fact in active} == {existing.object_id}


def test_promote_candidate_rejects_state_fact_supersede_across_conflict_sets(store_and_ledger):
    store, ledger = store_and_ledger

    existing = _seed_state_fact(
        subject='Yuan', predicate='prefers', value='cold brew', object_id='existing-fact-cs',
    )
    ledger.append_event('assert', actor_id='seed', reason='seed', payload=existing,
                        object_id=existing.object_id)

    _make_candidate(
        store=store, candidate_id='cand-mismatch', fact_type='state',
        subject='Archibald', predicate='lives_in', value='New York',
        conflict_with_fact_id=existing.object_id,
    )

    result = asyncio.run(candidates_router.promote_candidate('cand-mismatch', resolution='supersede'))
    assert 'error' in result or result.get('status') == 'error', result
    err_msg = result.get('message', '') + result.get('error', '')
    assert 'conflict set mismatch' in err_msg or 'supersede' in err_msg

    candidate = store.get_candidate('cand-mismatch')
    assert candidate is not None
    assert candidate['status'] == 'pending'
    assert {fact.object_id for fact in ledger.current_state_facts()} == {existing.object_id}


def test_promote_candidate_reconciles_existing_ledger_promotion(store_and_ledger):
    """Idempotent promotion: if ledger already has a promote event, re-calling returns same result."""
    store, ledger = store_and_ledger

    # Seed existing fact so supersede has a valid target
    existing = _seed_state_fact(
        subject='Yuan', predicate='prefers', value='flat white', object_id='existing-reconcile',
    )
    ledger.append_event('assert', actor_id='seed', reason='seed', payload=existing,
                        object_id=existing.object_id)

    _make_candidate(
        store=store, candidate_id='cand-reconcile', value='oat latte',
        conflict_with_fact_id=existing.object_id,
    )
    first = asyncio.run(candidates_router.promote_candidate('cand-reconcile', resolution='supersede'))
    assert first.get('status') == 'ok', first
    assert first.get('candidate', {}).get('status') == 'promoted'

    # Simulate partial failure: reset store status back to pending
    store.update_candidate_status('cand-reconcile', 'pending', resolution=None)

    # Second call hits the reconcile path (ledger already has the promote event)
    reconciled = asyncio.run(candidates_router.promote_candidate('cand-reconcile', resolution='supersede'))
    assert reconciled.get('status') == 'ok', reconciled
    assert reconciled.get('candidate', {}).get('status') == 'promoted'
    assert reconciled['promotion']['object_id'] == first['promotion']['object_id']
    assert reconciled['promotion']['event_id'] == first['promotion']['event_id']

    candidate = store.get_candidate('cand-reconcile')
    assert candidate is not None
    assert candidate['status'] == 'promoted'


# ---------------------------------------------------------------------------
# cancel + reject
# ---------------------------------------------------------------------------

def test_cancel_and_reject_record_audit_fields(store_and_ledger):
    store, ledger = store_and_ledger
    _make_candidate(store=store, candidate_id='cand-cancel')
    _make_candidate(store=store, candidate_id='cand-reject')

    cancel_result = asyncio.run(candidates_router.promote_candidate('cand-cancel', resolution='cancel'))
    assert cancel_result.get('status') == 'ok', cancel_result
    assert cancel_result.get('candidate', {}).get('status') == 'rejected'

    reject_result = asyncio.run(candidates_router.reject_candidate('cand-reject'))
    assert reject_result.get('status') == 'ok', reject_result
    assert reject_result.get('candidate', {}).get('status') == 'rejected'

    cancel_candidate = store.get_candidate('cand-cancel')
    reject_candidate = store.get_candidate('cand-reject')
    assert cancel_candidate is not None
    assert reject_candidate is not None
    assert cancel_candidate['status'] == 'rejected'
    assert reject_candidate['status'] == 'rejected'
    assert cancel_candidate.get('resolution') == 'cancel'
    # reject stores resolution as 'rejected' (no reason) or 'rejected: <reason>'
    assert reject_candidate.get('resolution', '').startswith('rejected')


# ---------------------------------------------------------------------------
# Invalid filter / state guards
# ---------------------------------------------------------------------------

def test_list_candidates_invalid_filters_return_empty(store_and_ledger):
    store, ledger = store_and_ledger
    _make_candidate(store=store, candidate_id='cand-1', status='pending')

    result = asyncio.run(candidates_router.list_candidates(status='does-not-exist'))
    assert result.get('error_type') == 'validation_error'

    result_age0 = asyncio.run(candidates_router.list_candidates(age_days=0))
    assert result_age0.get('error_type') == 'validation_error'

    result_conf_high = asyncio.run(candidates_router.list_candidates(min_confidence=2.0))
    assert result_conf_high.get('error_type') == 'validation_error'


def test_promote_and_reject_invalid_state_returns_error(store_and_ledger):
    store, ledger = store_and_ledger
    candidate_id = 'cand-invalid'
    _make_candidate(store=store, candidate_id=candidate_id, status='rejected')

    result_missing = asyncio.run(
        candidates_router.promote_candidate(candidate_id='missing', resolution='supersede')
    )
    assert result_missing.get('error_type') == 'not_found' or 'not found' in result_missing.get('message', '').lower()

    result_already_rejected = asyncio.run(
        candidates_router.promote_candidate(candidate_id=candidate_id, resolution='supersede')
    )
    assert result_already_rejected.get('error_type') == 'invalid_state'

    result_reject_missing = asyncio.run(candidates_router.reject_candidate(candidate_id='missing'))
    assert result_reject_missing.get('error_type') == 'not_found' or 'not found' in result_reject_missing.get('message', '').lower()

    result_reject_already = asyncio.run(candidates_router.reject_candidate(candidate_id=candidate_id))
    assert result_reject_already.get('error_type') == 'invalid_state'

    result_bogus_resolution = asyncio.run(
        candidates_router.promote_candidate(candidate_id='new-id', resolution='bogus')
    )
    assert result_bogus_resolution.get('error_type') == 'validation_error' or 'error' in result_bogus_resolution


# ---------------------------------------------------------------------------
# End-to-end lifecycle
# ---------------------------------------------------------------------------

def test_candidate_lifecycle_end_to_end_flow(store_and_ledger):
    store, ledger = store_and_ledger

    existing = _seed_state_fact(
        subject='Yuan', predicate='prefers', value='cold brew', object_id='existing-e2e',
    )
    ledger.append_event('assert', actor_id='seed', reason='seed', payload=existing,
                        object_id=existing.object_id)

    _make_candidate(
        store=store, candidate_id='cand-e2e-promote', fact_type='preference',
        value='matcha', conflict_with_fact_id=existing.object_id,
    )
    _make_candidate(
        store=store, candidate_id='cand-e2e-reject', fact_type='preference', value='latte',
    )

    quarantine = asyncio.run(candidates_router.list_candidates())
    assert quarantine['status'] == 'ok'
    by_id = {row['uuid']: row for row in quarantine['candidates']}
    assert by_id['cand-e2e-promote']['status'] == 'quarantine'
    assert by_id['cand-e2e-promote']['type'].lower() == 'preference'

    promote_result = asyncio.run(
        candidates_router.promote_candidate('cand-e2e-promote', resolution='supersede')
    )
    assert promote_result['status'] == 'ok'
    assert promote_result['candidate']['status'] == 'promoted'
    assert promote_result['candidate']['resolution'] == 'supersede'

    promoted = asyncio.run(candidates_router.list_candidates(status='promoted'))
    promoted_ids = {row['uuid'] for row in promoted['candidates']}
    assert 'cand-e2e-promote' in promoted_ids

    reject_result = asyncio.run(candidates_router.reject_candidate('cand-e2e-reject'))
    assert reject_result['status'] == 'ok'
    assert reject_result['candidate']['status'] == 'rejected'

    rejected = asyncio.run(candidates_router.list_candidates(status='rejected'))
    rejected_ids = {row['uuid'] for row in rejected['candidates']}
    assert 'cand-e2e-reject' in rejected_ids

    active = [fact for fact in ledger.current_state_facts() if fact.predicate == 'prefers']
    active_ids = {fact.object_id for fact in active}
    assert existing.object_id not in active_ids
    assert promote_result['promotion']['object_id'] in active_ids


# ---------------------------------------------------------------------------
# Concurrent promotion serialization
# ---------------------------------------------------------------------------

def test_promote_candidate_serializes_concurrent_calls(store_and_ledger):
    store, ledger = store_and_ledger

    existing = _seed_state_fact(
        subject='Yuan', predicate='prefers', value='cold brew', object_id='existing-race',
    )
    ledger.append_event('assert', actor_id='seed', reason='seed', payload=existing,
                        object_id=existing.object_id)

    _make_candidate(
        store=store, candidate_id='cand-race', value='pour over',
        conflict_with_fact_id=existing.object_id,
    )

    barrier = threading.Barrier(3)

    def _promote_once() -> dict[str, object]:
        barrier.wait(timeout=5)
        return asyncio.run(candidates_router.promote_candidate('cand-race', resolution='supersede'))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_promote_once), pool.submit(_promote_once)]
        barrier.wait(timeout=5)
        results = [future.result(timeout=10) for future in futures]

    successes = [result for result in results if result.get('status') == 'ok']
    errors = [result for result in results if result.get('status') == 'error']
    assert len(successes) == 1, f"Expected 1 success, got {results}"
    assert len(errors) == 1, f"Expected 1 error, got {results}"
    assert errors[0].get('error_type') == 'invalid_state'

    promote_event_count = int(
        ledger.conn.execute(
            "SELECT count(*) FROM change_events WHERE candidate_id = ? AND event_type = 'promote'",
            ('cand-race',),
        ).fetchone()[0]
    )
    assert promote_event_count == 1

    candidate = store.get_candidate('cand-race')
    assert candidate is not None
    assert candidate['status'] == 'promoted'
