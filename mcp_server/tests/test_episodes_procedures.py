from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mcp_server.src.models.typed_memory import Episode, EvidenceRef
from mcp_server.src.routers import episodes_procedures as router
from mcp_server.src.services.change_ledger import ChangeLedger
from mcp_server.src.services.procedure_service import ProcedureService


@pytest.fixture
def tmp_ledger_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / 'change_ledger.db'
    monkeypatch.setenv('BICAMERAL_CHANGE_LEDGER_PATH', str(db_path))
    router._LEDGER = None
    router._PROCEDURES = None
    return db_path


def _episode(object_id: str, title: str, summary: str, started_at: str, ended_at: str) -> Episode:
    evidence = EvidenceRef.model_validate(
        {
            'kind': 'event_log',
            'source_system': 'tests',
            'locator': {'system': 'pytest', 'stream': 'episodes', 'event_id': f'evt-{object_id}'},
            'observed_at': started_at,
            'retrieved_at': ended_at,
        }
    )
    return Episode.model_validate(
        {
            'object_id': object_id,
            'root_id': object_id,
            'title': title,
            'summary': summary,
            'started_at': started_at,
            'ended_at': ended_at,
            'policy_scope': 'private',
            'visibility_scope': 'private',
            'evidence_refs': [evidence],
            'created_at': started_at,
        }
    )


def _seed_data(ledger_path: Path) -> tuple[str, str]:
    ledger = ChangeLedger(ledger_path)
    ep = _episode(
        'ep_1',
        title='Self Audit: memory checks',
        summary='self_audit run detected stale context map',
        started_at='2026-03-10T10:00:00Z',
        ended_at='2026-03-10T10:05:00Z',
    )
    ledger.append_event(
        'assert',
        actor_id='test',
        reason='seed-episode',
        payload=ep,
        object_id=ep.object_id,
        object_type=ep.object_type,
        root_id=ep.root_id,
    )

    procedure_service = ProcedureService(ledger)
    promoted = procedure_service.create_procedure(
        actor_id='test',
        name='Handle urgent outage',
        trigger='urgent outage',
        steps=['ack alert', 'diagnose', 'rollback'],
        expected_outcome='service restored',
        evidence_refs=[{'source_key': 'seed', 'evidence_id': 'proc-promoted'}],
        promote=True,
    )
    proposed = procedure_service.create_procedure(
        actor_id='test',
        name='Handle low-priority bug',
        trigger='minor bug',
        steps=['triage', 'backlog'],
        expected_outcome='scheduled fix',
        evidence_refs=[{'source_key': 'seed', 'evidence_id': 'proc-proposed'}],
        promote=False,
    )
    return promoted.object_id, proposed.object_id


def test_search_episodes_and_time_range(tmp_ledger_path: Path):
    _seed_data(tmp_ledger_path)

    found = asyncio.run(router.search_episodes('self_audit'))
    assert 'error' not in found
    assert len(found['episodes']) == 1

    none = asyncio.run(
        router.search_episodes(
            'self_audit',
            time_range={'start': '2026-03-10T11:00:00Z', 'end': '2026-03-10T12:00:00Z'},
        )
    )
    assert 'error' not in none
    assert none['episodes'] == []


def test_get_episode_not_found(tmp_ledger_path: Path):
    _seed_data(tmp_ledger_path)
    result = asyncio.run(router.get_episode('missing'))
    assert 'error' in result


def test_search_procedures_promoted_only_by_default(tmp_ledger_path: Path):
    promoted_id, proposed_id = _seed_data(tmp_ledger_path)

    promoted_only = asyncio.run(router.search_procedures('handle'))
    assert 'error' not in promoted_only
    ids = {item['object_id'] for item in promoted_only['procedures']}
    assert promoted_id in ids
    assert proposed_id not in ids

    include_all = asyncio.run(router.search_procedures('handle', include_all=True))
    ids_all = {item['object_id'] for item in include_all['procedures']}
    assert promoted_id in ids_all
    assert proposed_id in ids_all


def test_get_procedure_by_id(tmp_ledger_path: Path):
    promoted_id, _ = _seed_data(tmp_ledger_path)
    result = asyncio.run(router.get_procedure(promoted_id))
    assert 'error' not in result
    assert result['object_id'] == promoted_id
