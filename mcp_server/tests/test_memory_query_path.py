from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mcp_server.src.models.typed_memory import EvidenceRef, StateFact
from mcp_server.src.routers import memory as memory_router
from mcp_server.src.services.change_ledger import ChangeLedger


@pytest.fixture
def tmp_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ChangeLedger:
    db_path = tmp_path / 'change_ledger.db'
    monkeypatch.setenv('BICAMERAL_CHANGE_LEDGER_PATH', str(db_path))
    memory_router._LEDGER = None
    return ChangeLedger(db_path)


def _fact(
    *,
    object_id: str,
    subject: str,
    predicate: str,
    value: str,
    created_at: str,
    root_id: str | None = None,
    parent_id: str | None = None,
    version: int = 1,
) -> StateFact:
    evidence = EvidenceRef.model_validate(
        {
            'kind': 'event_log',
            'source_system': 'tests',
            'locator': {
                'system': 'pytest',
                'stream': 'memory-query',
                'event_id': f'evt-{object_id}',
            },
            'observed_at': created_at,
            'retrieved_at': created_at,
        }
    )
    return StateFact.model_validate(
        {
            'object_id': object_id,
            'root_id': root_id or object_id,
            'parent_id': parent_id,
            'version': version,
            'fact_type': 'preference',
            'subject': subject,
            'predicate': predicate,
            'value': value,
            'scope': 'private',
            'policy_scope': 'private',
            'visibility_scope': 'private',
            'evidence_refs': [evidence],
            'created_at': created_at,
            'valid_at': created_at,
            'source_lane': 'private',
            'source_key': 'tests',
            'promotion_status': 'promoted',
        }
    )


def _seed_superseded_preference(ledger: ChangeLedger) -> tuple[StateFact, StateFact]:
    old_fact = _fact(
        object_id='fact_old',
        subject='Yuan',
        predicate='prefers',
        value='espresso',
        created_at='2026-03-10T10:00:00Z',
    )
    ledger.append_event(
        'assert',
        actor_id='test',
        reason='seed-old',
        payload=old_fact,
        object_id=old_fact.object_id,
        object_type=old_fact.object_type,
        root_id=old_fact.root_id,
    )

    new_fact = _fact(
        object_id='fact_new',
        root_id=old_fact.root_id,
        parent_id=old_fact.object_id,
        version=2,
        subject='Yuan',
        predicate='prefers',
        value='pour over',
        created_at='2026-03-10T12:00:00Z',
    )
    ledger.append_event(
        'supersede',
        actor_id='test',
        reason='seed-new',
        payload=new_fact,
        object_id=new_fact.object_id,
        object_type=new_fact.object_type,
        root_id=new_fact.root_id,
        parent_id=new_fact.parent_id,
        target_object_id=old_fact.object_id,
    )

    return old_fact, new_fact


def test_get_current_state_returns_latest_current_fact(tmp_ledger: ChangeLedger):
    _old, new = _seed_superseded_preference(tmp_ledger)

    result = asyncio.run(memory_router.get_current_state(subject='Yuan', predicate='prefers'))
    assert 'error' not in result
    assert len(result['facts']) == 1
    assert result['facts'][0]['object_id'] == new.object_id
    assert result['facts'][0]['value'] == 'pour over'


def test_get_current_state_as_of_returns_historical_snapshot(tmp_ledger: ChangeLedger):
    old, _new = _seed_superseded_preference(tmp_ledger)

    result = asyncio.run(
        memory_router.get_current_state(
            subject='Yuan',
            predicate='prefers',
            as_of='2026-03-10T11:00:00Z',
        )
    )
    assert 'error' not in result
    assert len(result['facts']) == 1
    assert result['facts'][0]['object_id'] == old.object_id
    assert result['facts'][0]['value'] == 'espresso'


def test_get_history_returns_versions_newest_first(tmp_ledger: ChangeLedger):
    old, new = _seed_superseded_preference(tmp_ledger)

    result = asyncio.run(memory_router.get_history(subject='Yuan', predicate='prefers'))
    assert 'error' not in result
    assert [item['object_id'] for item in result['history']] == [new.object_id, old.object_id]


def test_get_history_limit_and_validation(tmp_ledger: ChangeLedger):
    _seed_superseded_preference(tmp_ledger)

    limited = asyncio.run(memory_router.get_history(subject='Yuan', predicate='prefers', limit=1))
    assert 'error' not in limited
    assert len(limited['history']) == 1

    bad_subject = asyncio.run(memory_router.get_current_state(subject=''))
    assert 'error' in bad_subject

    bad_limit = asyncio.run(memory_router.get_history(subject='Yuan', limit=0))
    assert 'error' in bad_limit
