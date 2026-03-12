from __future__ import annotations

import asyncio
import json

import pytest
from mcp_server.src.services.candidate_store import CandidateStore
from mcp_server.src.services.change_ledger import ChangeLedger

try:
    from mcp_server.src.routers import memory
except ImportError:
    from routers import memory  # type: ignore


@pytest.fixture
def isolated_memory(tmp_path, monkeypatch):
    """Isolate memory router state per test with dedicated ledger/candidate stores."""
    change_db = tmp_path / 'change_ledger.db'
    candidate_db = tmp_path / 'candidates.db'

    ledger = ChangeLedger(change_db)
    candidates = CandidateStore(candidate_db)

    async def fake_materialize_typed_fact(
        *,
        fact,
        source='caller_asserted_unverified',
        superseded_fact_id=None,
        graphiti_client=None,
    ):
        # Deterministic, side-effect-free for unit tests.
        fake_materialize_typed_fact.calls.append(
            (fact.object_id, source, superseded_fact_id is not None, graphiti_client is not None)
        )
        return True, None

    fake_materialize_typed_fact.calls = []  # type: ignore[attr-defined]

    class _FakeMaterializer:
        def materialize_typed_fact(
            self,
            *,
            fact,
            source='caller_asserted_unverified',
            superseded_fact_id=None,
            graphiti_client=None,
        ):
            return fake_materialize_typed_fact(
                fact=fact,
                source=source,
                superseded_fact_id=superseded_fact_id,
                graphiti_client=graphiti_client,
            )

    monkeypatch.setattr(memory, '_change_ledger', ledger)
    monkeypatch.setattr(memory, '_candidate_store', candidates)
    monkeypatch.setattr(memory, '_materializer', _FakeMaterializer())

    return {
        'ledger': ledger,
        'candidates': candidates,
        'materialize_calls': fake_materialize_typed_fact.calls,
    }


def _subject_state_facts(ledger: ChangeLedger, subject: str):
    return [
        fact
        for fact in ledger.current_state_facts()
        if fact.object_type == 'state_fact' and fact.subject == subject
    ]


def _run(coro):
    return asyncio.run(coro)


def _owner_trust(*, allow_conflict_supersede: bool = False) -> dict[str, object]:
    return {
        'trust': {
            'verified': True,
            'is_owner': True,
            'actor_id': 'owner:archibald',
            'source': 'owner_asserted',
            'allow_conflict_supersede': allow_conflict_supersede,
        }
    }


def test_remember_fact_writes_typed_fact(isolated_memory):
    result = _run(
        memory.remember_fact(
            'I prefer dark mode for my working context',
            {'type': 'Preference', 'subject': 'UI preferences'},
        )
    )

    assert result['status'] == 'ok'
    assert result['fact']['subject'] == 'UI preferences'
    assert result['fact']['fact_type'] == 'preference'
    assert result['fact']['predicate'] == 'preference'
    assert result['fact']['is_current']
    assert isolated_memory['materialize_calls'], 'expected materialization attempt'

    current = _subject_state_facts(isolated_memory['ledger'], 'UI preferences')
    assert len(current) == 1


def test_remember_fact_duplicate_returns_duplicate(isolated_memory):
    first = _run(
        memory.remember_fact(
            'I prefer dark mode',
            {'type': 'Preference', 'subject': 'UI preferences'},
        )
    )
    second = _run(
        memory.remember_fact(
            'I prefer dark mode',
            {'type': 'Preference', 'subject': 'UI preferences'},
        )
    )

    assert first['status'] == 'ok'
    assert second['status'] == 'duplicate'

    current = _subject_state_facts(isolated_memory['ledger'], 'UI preferences')
    assert len(current) == 1


def test_remember_fact_invalid_type_is_validation_error(isolated_memory):
    result = _run(
        memory.remember_fact(
            'I prefer dark mode',
            {'type': 'NotAType', 'subject': 'UI preferences'},
        )
    )

    assert result['status'] == 'error'
    assert result['error_type'] == 'validation_error'


def test_remember_fact_missing_subject_is_validation_error(isolated_memory):
    result = _run(
        memory.remember_fact(
            'I prefer dark mode',
            {'type': 'Preference'},
        )
    )

    assert result['status'] == 'error'
    assert result['error_type'] == 'validation_error'


def test_remember_fact_conflict_returns_dialog_and_candidate(isolated_memory):
    first = _run(
        memory.remember_fact(
            'I prefer dark mode',
            {'type': 'Preference', 'subject': 'UI preferences'},
        )
    )
    assert first['status'] == 'ok'

    conflict = _run(
        memory.remember_fact(
            'I prefer light mode',
            {'type': 'Preference', 'subject': 'UI preferences'},
        )
    )

    assert conflict['status'] == 'conflict'
    assert conflict['type'] == 'ConflictDialog'
    assert len(conflict['options']) == 3
    assert conflict['candidate_uuid']
    assert len(isolated_memory['candidates'].list_candidates(status='pending')) == 1


def test_remember_fact_supersede_without_privilege_is_quarantined(isolated_memory):
    first = _run(
        memory.remember_fact(
            'I prefer dark mode',
            {'type': 'Preference', 'subject': 'UI preferences'},
        )
    )
    assert first['status'] == 'ok'

    second = _run(
        memory.remember_fact(
            'I prefer light mode',
            {'type': 'Preference', 'subject': 'UI preferences', 'supersede': True},
        )
    )

    assert second['status'] == 'conflict'
    assert second['supersede_requested'] is True
    assert second['supersede_allowed'] is False


def test_remember_fact_supersedes_existing_when_privileged(isolated_memory):
    first = _run(
        memory.remember_fact(
            'I prefer dark mode',
            {'type': 'Preference', 'subject': 'UI preferences'},
        )
    )
    assert first['status'] == 'ok'

    second_hint = {
        'type': 'Preference',
        'subject': 'UI preferences',
        'supersede': True,
        **_owner_trust(allow_conflict_supersede=True),
    }
    second = _run(memory.remember_fact('I prefer light mode', second_hint))

    assert second['status'] == 'ok'
    current = _subject_state_facts(isolated_memory['ledger'], 'UI preferences')
    assert len(current) == 1
    assert current[0].value == 'light mode'


def test_remember_fact_rejects_unknown_hint_keys(isolated_memory):
    result = _run(
        memory.remember_fact(
            'I prefer dark mode',
            {'type': 'Preference', 'subject': 'UI preferences', 'junk': 'oops'},
        )
    )

    assert result['status'] == 'error'
    assert result['error_type'] == 'validation_error'
    assert 'unknown field' in result['message']


def test_remember_fact_rejects_oversized_hint_metadata(isolated_memory):
    result = _run(
        memory.remember_fact(
            'I prefer dark mode',
            {
                'type': 'Preference',
                'subject': 'UI preferences',
                'metadata': {'blob': 'x' * 9000},
            },
        )
    )

    assert result['status'] == 'error'
    assert result['error_type'] == 'validation_error'
    assert 'max size' in result['message']


def test_remember_fact_does_not_default_to_owner_stamp_without_trust(isolated_memory):
    result = _run(
        memory.remember_fact(
            'I prefer dark mode',
            {'type': 'Preference', 'subject': 'UI preferences'},
        )
    )

    assert result['status'] == 'ok'
    root_id = result['fact']['root_id']
    rows = isolated_memory['ledger'].events_for_root(root_id)
    assert rows
    assert rows[0].actor_id == 'caller:unverified'

    metadata = json.loads(rows[0].metadata_json or '{}')
    assert metadata['source'] == 'caller_asserted_unverified'


def test_remember_fact_owner_stamp_requires_verified_trust(isolated_memory):
    result = _run(
        memory.remember_fact(
            'I prefer dark mode',
            {
                'type': 'Preference',
                'subject': 'UI preferences',
                **_owner_trust(),
            },
        )
    )

    assert result['status'] == 'ok'
    root_id = result['fact']['root_id']
    rows = isolated_memory['ledger'].events_for_root(root_id)
    assert rows[0].actor_id == 'owner:archibald'

    metadata = json.loads(rows[0].metadata_json or '{}')
    assert metadata['source'] == 'owner_asserted'


def test_remember_fact_multi_hop_supersession_preserves_root_lineage(isolated_memory):
    trusted = _owner_trust(allow_conflict_supersede=True)

    first = _run(
        memory.remember_fact(
            'I prefer dark mode',
            {'type': 'Preference', 'subject': 'UI preferences'},
        )
    )
    assert first['status'] == 'ok'

    second = _run(
        memory.remember_fact(
            'I prefer light mode',
            {
                'type': 'Preference',
                'subject': 'UI preferences',
                'supersede': True,
                **trusted,
            },
        )
    )
    assert second['status'] == 'ok'

    third = _run(
        memory.remember_fact(
            'I prefer solarized',
            {
                'type': 'Preference',
                'subject': 'UI preferences',
                'supersede': True,
                **trusted,
            },
        )
    )
    assert third['status'] == 'ok'

    rows = isolated_memory['ledger'].conn.execute(
        """
        SELECT payload_json
          FROM change_events
         WHERE event_type IN ('assert', 'supersede')
         ORDER BY rowid
        """
    ).fetchall()
    payloads = [json.loads(row['payload_json']) for row in rows]

    assert len(payloads) == 3
    assert len({payload['root_id'] for payload in payloads}) == 1
    assert payloads[1]['parent_id'] == payloads[0]['object_id']
    assert payloads[2]['parent_id'] == payloads[1]['object_id']


def test_remember_fact_neo4j_failure_is_observable_and_non_blocking(
    isolated_memory,
    monkeypatch,
):
    class _FailingMaterializer:
        async def materialize_typed_fact(
            self,
            *,
            fact,
            source='caller_asserted_unverified',
            superseded_fact_id=None,
            graphiti_client=None,
        ):
            return False, 'neo4j_unreachable'

    monkeypatch.setattr(memory, '_materializer', _FailingMaterializer())

    result = _run(
        memory.remember_fact(
            'I prefer dark mode',
            {'type': 'Preference', 'subject': 'UI preferences'},
        )
    )

    assert result['status'] == 'ok'
    assert result['materialized'] is False
    assert result['materialization_error'] == 'neo4j_unreachable'
    assert result['neo4j_materialization']['status'] == 'failed'
    assert result['warnings'][0]['code'] == 'neo4j_materialization_failed'

    current = _subject_state_facts(isolated_memory['ledger'], 'UI preferences')
    assert len(current) == 1
