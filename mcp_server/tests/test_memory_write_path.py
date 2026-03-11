from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mcp_server.src.routers import memory as memory_router
from mcp_server.src.services.change_ledger import ChangeLedger


@pytest.fixture
def tmp_ledger_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / 'change_ledger.db'
    monkeypatch.setenv('BICAMERAL_CHANGE_LEDGER_PATH', str(db_path))
    monkeypatch.delenv('BICAMERAL_DISABLE_NEO4J_MATERIALIZATION', raising=False)
    memory_router._LEDGER = None
    return db_path


def test_remember_fact_writes_state_fact(tmp_ledger_path: Path):
    result = asyncio.run(
        memory_router.remember_fact(
            'Yuan prefers espresso',
            hint={'source_lane': 'private'},
        )
    )

    assert 'error' not in result
    assert result['message'] == 'Fact remembered'
    typed_fact = result['typed_fact']
    assert typed_fact['subject'] == 'Yuan'
    assert typed_fact['predicate'] == 'prefers'
    assert typed_fact['value'] == 'espresso'

    ledger = ChangeLedger(tmp_ledger_path)
    current = ledger.current_state_facts()
    assert len(current) == 1
    assert current[0].subject == 'Yuan'


def test_remember_fact_conflict_returns_dialog_without_write(tmp_ledger_path: Path):
    first = asyncio.run(memory_router.remember_fact('Yuan prefers espresso'))
    assert 'error' not in first

    second = asyncio.run(memory_router.remember_fact('Yuan prefers pour over'))
    assert second['status'] == 'conflict'
    assert 'conflict_dialog' in second
    assert second['conflict_dialog']['options'] == ['supersede', 'parallel', 'cancel']

    ledger = ChangeLedger(tmp_ledger_path)
    current = ledger.current_state_facts()
    assert len(current) == 1
    assert current[0].value == 'espresso'


def test_remember_fact_supersede_writes_new_version(tmp_ledger_path: Path):
    first = asyncio.run(memory_router.remember_fact('Yuan prefers espresso'))
    assert 'error' not in first
    first_fact = first['typed_fact']

    second = asyncio.run(
        memory_router.remember_fact(
            'Yuan prefers pour over',
            hint={'conflict_resolution': 'supersede'},
        )
    )
    assert 'error' not in second
    second_fact = second['typed_fact']

    assert second['conflict_resolution'] == 'supersede'
    assert second_fact['root_id'] == first_fact['root_id']
    assert second_fact['version'] == first_fact['version'] + 1

    ledger = ChangeLedger(tmp_ledger_path)
    current = ledger.current_state_facts()
    assert len(current) == 1
    assert current[0].value == 'pour over'
    assert current[0].version == 2


def test_remember_fact_neo4j_failure_is_fail_open(
    tmp_ledger_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    async def _boom(self, *_args, **_kwargs):
        raise RuntimeError('neo4j unavailable')

    monkeypatch.setattr(
        memory_router.Neo4jMaterializationService,
        'materialize_state_fact',
        _boom,
    )

    result = asyncio.run(memory_router.remember_fact('Yuan prefers natural wine'))
    assert 'error' not in result
    assert result['neo4j']['materialized'] is False
    assert 'neo4j unavailable' in (result['neo4j']['error'] or '')
