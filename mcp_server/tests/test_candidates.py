from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mcp_server.src.models.typed_memory import EvidenceRef, StateFact
from mcp_server.src.routers import candidates as candidates_router
from mcp_server.src.services.change_ledger import ChangeLedger


@pytest.fixture
def tmp_ledger_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / 'change_ledger.db'
    monkeypatch.setenv('BICAMERAL_CHANGE_LEDGER_PATH', str(db_path))
    return db_path


def _insert_candidate(
    ledger: ChangeLedger,
    *,
    candidate_id: str,
    candidate_type: str = 'preference',
    subject: str = 'Yuan',
    predicate: str = 'prefers',
    value: str = 'espresso',
    conflicting_fact_uuid: str | None = None,
    status: str = 'quarantine',
    confidence: float = 0.9,
) -> None:
    conn = ledger.conn
    candidates_router.CandidatesDB(conn).ensure_schema()
    conn.execute(
        """
        INSERT INTO candidates(
            uuid, type, subject, predicate, value, conflicting_fact_uuid,
            status, resolution, confidence, created_at, updated_at, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)
        """,
        (
            candidate_id,
            candidate_type,
            subject,
            predicate,
            json.dumps(value),
            conflicting_fact_uuid,
            status,
            confidence,
            '2026-03-10T10:00:00Z',
            '2026-03-10T10:00:00Z',
            json.dumps({'source': 'test'}),
        ),
    )
    conn.commit()


def _seed_existing_fact(ledger: ChangeLedger, *, object_id: str = 'fact_existing') -> StateFact:
    evidence = EvidenceRef.model_validate(
        {
            'kind': 'event_log',
            'source_system': 'tests',
            'locator': {'system': 'pytest', 'stream': 'candidates', 'event_id': f'evt-{object_id}'},
            'observed_at': '2026-03-10T09:00:00Z',
            'retrieved_at': '2026-03-10T09:00:00Z',
        }
    )
    fact = StateFact.model_validate(
        {
            'object_id': object_id,
            'root_id': object_id,
            'version': 1,
            'fact_type': 'preference',
            'subject': 'Yuan',
            'predicate': 'prefers',
            'value': 'espresso',
            'scope': 'private',
            'policy_scope': 'private',
            'visibility_scope': 'private',
            'evidence_refs': [evidence],
            'created_at': '2026-03-10T09:00:00Z',
            'valid_at': '2026-03-10T09:00:00Z',
            'promotion_status': 'promoted',
        }
    )
    ledger.append_event(
        'assert',
        actor_id='test',
        reason='seed',
        payload=fact,
        object_id=fact.object_id,
        object_type=fact.object_type,
        root_id=fact.root_id,
    )
    return fact


def test_list_candidates_with_filters(tmp_ledger_path: Path):
    ledger = ChangeLedger(tmp_ledger_path)
    _insert_candidate(ledger, candidate_id='cand_a', confidence=0.2)
    _insert_candidate(ledger, candidate_id='cand_b', candidate_type='world_state', confidence=0.95)

    result = asyncio.run(
        candidates_router.list_candidates(status='quarantine', type_filter='world_state', min_confidence=0.9)
    )
    assert 'error' not in result
    assert len(result['candidates']) == 1
    assert result['candidates'][0]['uuid'] == 'cand_b'


def test_promote_candidate_supersede(tmp_ledger_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('BICAMERAL_CHANGE_LEDGER_PATH', str(tmp_ledger_path))
    ledger = ChangeLedger(tmp_ledger_path)
    existing = _seed_existing_fact(ledger)
    _insert_candidate(
        ledger,
        candidate_id='cand_promote',
        value='pour over',
        conflicting_fact_uuid=existing.object_id,
    )

    result = asyncio.run(candidates_router.promote_candidate('cand_promote', 'supersede'))
    assert 'error' not in result
    assert result['status'] == 'promoted'
    assert result['resolution'] == 'supersede'

    candidate = candidates_router.CandidatesDB(tmp_ledger_path).get_candidate('cand_promote')
    assert candidate is not None
    assert candidate['status'] == 'promoted'


def test_promote_candidate_cancel_marks_rejected(tmp_ledger_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('BICAMERAL_CHANGE_LEDGER_PATH', str(tmp_ledger_path))
    ledger = ChangeLedger(tmp_ledger_path)
    _insert_candidate(ledger, candidate_id='cand_cancel')

    result = asyncio.run(candidates_router.promote_candidate('cand_cancel', 'cancel'))
    assert 'error' not in result
    assert result['status'] == 'rejected'

    candidate = candidates_router.CandidatesDB(tmp_ledger_path).get_candidate('cand_cancel')
    assert candidate is not None
    assert candidate['status'] == 'rejected'


def test_reject_candidate(tmp_ledger_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('BICAMERAL_CHANGE_LEDGER_PATH', str(tmp_ledger_path))
    ledger = ChangeLedger(tmp_ledger_path)
    _insert_candidate(ledger, candidate_id='cand_reject')

    result = asyncio.run(candidates_router.reject_candidate('cand_reject'))
    assert 'error' not in result
    assert result['status'] == 'rejected'
