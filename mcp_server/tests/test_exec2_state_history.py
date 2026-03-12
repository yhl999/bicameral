from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_HELPER_PATH = REPO_ROOT / 'tests' / 'helpers_mcp_import.py'
_HELPER_SPEC = spec_from_file_location('repo_tests_helpers_mcp_import', _HELPER_PATH)
assert _HELPER_SPEC is not None and _HELPER_SPEC.loader is not None
_HELPER_MODULE = module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPER_MODULE)
load_graphiti_mcp_server = _HELPER_MODULE.load_graphiti_mcp_server

server = load_graphiti_mcp_server()

from mcp_server.src.models.typed_memory import EvidenceRef, StateFact  # noqa: E402
from mcp_server.src.services.change_ledger import ChangeLedger  # noqa: E402


def _config(default_group_id: str | None = 's1_sessions_main'):
    return SimpleNamespace(
        database=SimpleNamespace(provider='neo4j'),
        graphiti=SimpleNamespace(
            group_id=default_group_id,
            lane_aliases={
                'sessions_main': ['s1_sessions_main'],
                'curated': ['s1_curated_refs'],
            },
        ),
    )


def _evidence_ref(tag: str, lane: str) -> EvidenceRef:
    return EvidenceRef.from_legacy_ref(
        {
            'source_key': 'unit-test',
            'evidence_id': tag,
            'scope': lane,
        }
    )


def _state_fact(
    *,
    object_id: str,
    root_id: str,
    subject: str,
    predicate: str,
    value: str,
    created_at: str,
    scope: str = 'private',
    source_lane: str = 's1_sessions_main',
    policy_scope: str = 'private',
    visibility_scope: str = 'private',
) -> StateFact:
    return StateFact.model_validate(
        {
            'object_id': object_id,
            'root_id': root_id,
            'object_type': 'state_fact',
            'fact_type': 'preference',
            'subject': subject,
            'predicate': predicate,
            'value': value,
            'scope': scope,
            'source_lane': source_lane,
            'policy_scope': policy_scope,
            'visibility_scope': visibility_scope,
            'created_at': created_at,
            'evidence_refs': [_evidence_ref(object_id, source_lane)],
        }
    )


@pytest.fixture(autouse=True)
def _server_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(server, 'config', _config())


@pytest.fixture
def ledger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    temp_ledger = ChangeLedger(tmp_path / 'change_ledger.db')
    monkeypatch.setattr(server, 'change_ledger', temp_ledger)
    return temp_ledger


@pytest.mark.anyio
async def test_get_current_state_returns_scoped_fact_envelope(ledger: ChangeLedger):
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='theme-main',
            root_id='r-theme-main',
            subject='UI',
            predicate='theme',
            value='dark',
            created_at='2026-01-01T10:00:00Z',
        ),
        root_id='r-theme-main',
        recorded_at='2026-01-01T10:00:00Z',
    )
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='font-main',
            root_id='r-font-main',
            subject='UI',
            predicate='font',
            value='serif',
            created_at='2026-01-01T10:05:00Z',
        ),
        root_id='r-font-main',
        recorded_at='2026-01-01T10:05:00Z',
    )
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='theme-curated',
            root_id='r-theme-curated',
            subject='UI',
            predicate='theme',
            value='solarized',
            created_at='2026-01-01T10:10:00Z',
            source_lane='s1_curated_refs',
        ),
        root_id='r-theme-curated',
        recorded_at='2026-01-01T10:10:00Z',
    )
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='theme-public',
            root_id='r-theme-public',
            subject='UI',
            predicate='theme',
            value='public',
            created_at='2026-01-01T10:15:00Z',
            visibility_scope='public',
        ),
        root_id='r-theme-public',
        recorded_at='2026-01-01T10:15:00Z',
    )

    result = await server.get_current_state('UI', lane_alias=['sessions_main'])

    assert 'error' not in result
    assert result['metadata'] == {
        'subject': 'UI',
        'predicate': None,
        'scope': None,
        'group_ids': ['s1_sessions_main'],
        'lane_alias': ['sessions_main'],
        'limit': 100,
        'result_count': 2,
        'truncated': False,
    }
    assert [fact['predicate'] for fact in result['facts']] == ['font', 'theme']
    assert {fact['value'] for fact in result['facts']} == {'serif', 'dark'}
    assert all(fact['source_lane'] == 's1_sessions_main' for fact in result['facts'])


@pytest.mark.anyio
async def test_get_current_state_empty_match_returns_empty_envelope(ledger: ChangeLedger):
    result = await server.get_current_state('missing', group_ids=['s1_sessions_main'])

    assert result == {
        'message': "No current state facts found for subject='missing'",
        'facts': [],
        'metadata': {
            'subject': 'missing',
            'predicate': None,
            'scope': None,
            'group_ids': ['s1_sessions_main'],
            'lane_alias': None,
            'limit': 100,
            'result_count': 0,
            'truncated': False,
        },
    }


@pytest.mark.anyio
async def test_get_current_state_can_filter_same_subject_predicate_by_scope(ledger: ChangeLedger):
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='theme-private',
            root_id='r-theme-private',
            subject='UI',
            predicate='theme',
            value='dark',
            created_at='2026-01-01T10:00:00Z',
            scope='private',
        ),
        root_id='r-theme-private',
        recorded_at='2026-01-01T10:00:00Z',
    )
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='theme-team',
            root_id='r-theme-team',
            subject='UI',
            predicate='theme',
            value='light',
            created_at='2026-01-01T10:05:00Z',
            scope='team',
        ),
        root_id='r-theme-team',
        recorded_at='2026-01-01T10:05:00Z',
    )

    unfiltered = await server.get_current_state('UI', predicate='theme', group_ids=['s1_sessions_main'])
    assert 'error' not in unfiltered
    assert [fact['scope'] for fact in unfiltered['facts']] == ['team', 'private']

    filtered = await server.get_current_state(
        'UI',
        predicate='theme',
        scope='private',
        group_ids=['s1_sessions_main'],
    )
    assert 'error' not in filtered
    assert [fact['value'] for fact in filtered['facts']] == ['dark']
    assert filtered['metadata'] == {
        'subject': 'UI',
        'predicate': 'theme',
        'scope': 'private',
        'group_ids': ['s1_sessions_main'],
        'lane_alias': None,
        'limit': 100,
        'result_count': 1,
        'truncated': False,
    }


@pytest.mark.anyio
async def test_get_history_uses_lifecycle_timestamp_for_superseded_parent(ledger: ChangeLedger):
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='chain-1',
            root_id='r-chain',
            subject='Workspace',
            predicate='theme',
            value='light',
            created_at='2026-01-01T10:00:00Z',
        ),
        root_id='r-chain',
        recorded_at='2026-01-01T10:00:00Z',
    )
    ledger.append_event(
        'refine',
        payload=_state_fact(
            object_id='chain-2',
            root_id='r-chain',
            subject='Workspace',
            predicate='theme',
            value='dark',
            created_at='2026-01-01T11:00:00Z',
        ),
        target_object_id='chain-1',
        root_id='r-chain',
        recorded_at='2026-01-01T11:00:00Z',
    )
    ledger.append_event(
        'refine',
        payload=_state_fact(
            object_id='chain-3',
            root_id='r-chain',
            subject='Workspace',
            predicate='theme',
            value='system',
            created_at='2026-01-01T12:00:00Z',
        ),
        target_object_id='chain-2',
        root_id='r-chain',
        recorded_at='2026-01-01T12:00:00Z',
    )

    result = await server.get_history('Workspace', predicate='theme', group_ids=['s1_sessions_main'])

    assert 'error' not in result
    assert [event['value'] for event in result['history']] == ['light', 'dark', 'system']
    assert [event['timestamp'] for event in result['history']] == [
        '2026-01-01T11:00:00Z',
        '2026-01-01T12:00:00Z',
        '2026-01-01T12:00:00Z',
    ]
    assert [event['status'] for event in result['history']] == ['superseded', 'superseded', 'active']


@pytest.mark.anyio
async def test_get_history_returns_scoped_history_envelope_with_recent_limit(ledger: ChangeLedger):
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='chain-1',
            root_id='r-chain',
            subject='Workspace',
            predicate='theme',
            value='light',
            created_at='2026-01-01T10:00:00Z',
        ),
        root_id='r-chain',
        recorded_at='2026-01-01T10:00:00Z',
    )
    ledger.append_event(
        'refine',
        payload=_state_fact(
            object_id='chain-2',
            root_id='r-chain',
            subject='Workspace',
            predicate='theme',
            value='dark',
            created_at='2026-01-01T11:00:00Z',
        ),
        target_object_id='chain-1',
        root_id='r-chain',
        recorded_at='2026-01-01T11:00:00Z',
    )
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='chain-unrelated',
            root_id='r-chain-unrelated',
            subject='Workspace',
            predicate='theme',
            value='blue',
            created_at='2026-01-01T11:30:00Z',
        ),
        root_id='r-chain-unrelated',
        recorded_at='2026-01-01T11:30:00Z',
    )
    ledger.append_event(
        'refine',
        payload=_state_fact(
            object_id='chain-3',
            root_id='r-chain',
            subject='Workspace',
            predicate='theme',
            value='system',
            created_at='2026-01-01T12:00:00Z',
        ),
        target_object_id='chain-2',
        root_id='r-chain',
        recorded_at='2026-01-01T12:00:00Z',
    )
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='chain-curated',
            root_id='r-chain-curated',
            subject='Workspace',
            predicate='theme',
            value='neon',
            created_at='2026-01-01T12:30:00Z',
            source_lane='s1_curated_refs',
        ),
        root_id='r-chain-curated',
        recorded_at='2026-01-01T12:30:00Z',
    )
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='chain-public',
            root_id='r-chain-public',
            subject='Workspace',
            predicate='theme',
            value='public',
            created_at='2026-01-01T12:45:00Z',
            policy_scope='public',
        ),
        root_id='r-chain-public',
        recorded_at='2026-01-01T12:45:00Z',
    )

    result = await server.get_history('Workspace', predicate='theme', group_ids=['s1_sessions_main'], limit=2)

    assert 'error' not in result
    assert [event['value'] for event in result['history']] == ['dark', 'system']
    assert [event['timestamp'] for event in result['history']] == ['2026-01-01T12:00:00Z', '2026-01-01T12:00:00Z']
    assert [event['status'] for event in result['history']] == ['superseded', 'active']
    assert all(event['source_lane'] == 's1_sessions_main' for event in result['history'])
    assert result['metadata'] == {
        'subject': 'Workspace',
        'predicate': 'theme',
        'scope': None,
        'group_ids': ['s1_sessions_main'],
        'lane_alias': None,
        'limit': 2,
        'result_count': 2,
        'truncated': True,
    }


@pytest.mark.anyio
async def test_get_history_can_filter_same_subject_predicate_by_scope(ledger: ChangeLedger):
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='theme-private-1',
            root_id='r-theme-private',
            subject='Workspace',
            predicate='theme',
            value='dark',
            created_at='2026-01-01T10:00:00Z',
            scope='private',
        ),
        root_id='r-theme-private',
        recorded_at='2026-01-01T10:00:00Z',
    )
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='theme-team-1',
            root_id='r-theme-team',
            subject='Workspace',
            predicate='theme',
            value='light',
            created_at='2026-01-01T10:05:00Z',
            scope='team',
        ),
        root_id='r-theme-team',
        recorded_at='2026-01-01T10:05:00Z',
    )

    unfiltered = await server.get_history('Workspace', predicate='theme', group_ids=['s1_sessions_main'])
    assert 'error' not in unfiltered
    assert [event['scope'] for event in unfiltered['history']] == ['private', 'team']

    filtered = await server.get_history(
        'Workspace',
        predicate='theme',
        scope='team',
        group_ids=['s1_sessions_main'],
    )
    assert 'error' not in filtered
    assert [event['value'] for event in filtered['history']] == ['light']
    assert filtered['metadata'] == {
        'subject': 'Workspace',
        'predicate': 'theme',
        'scope': 'team',
        'group_ids': ['s1_sessions_main'],
        'lane_alias': None,
        'limit': 200,
        'result_count': 1,
        'truncated': False,
    }


@pytest.mark.anyio
async def test_get_history_preserves_invalidated_status_on_wire(ledger: ChangeLedger):
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='theme-1',
            root_id='r-theme',
            subject='Workspace',
            predicate='theme',
            value='dark',
            created_at='2026-01-01T10:00:00Z',
        ),
        root_id='r-theme',
        recorded_at='2026-01-01T10:00:00Z',
    )
    ledger.append_event(
        'invalidate',
        object_id='theme-1',
        target_object_id='theme-1',
        root_id='r-theme',
        recorded_at='2026-01-01T12:00:00Z',
    )

    result = await server.get_history('Workspace', predicate='theme', group_ids=['s1_sessions_main'])

    assert 'error' not in result
    assert result['history'] == [
        {
            'uuid': 'theme-1',
            'type': 'preference',
            'subject': 'Workspace',
            'predicate': 'theme',
            'scope': 'private',
            'value': 'dark',
            'timestamp': '2026-01-01T12:00:00Z',
            'source': 'owner_asserted',
            'source_lane': 's1_sessions_main',
            'status': 'invalidated',
            'supersedes': None,
            'superseded_by': None,
        }
    ]


@pytest.mark.anyio
async def test_get_history_orders_lifecycle_only_events_by_latest_ledger_time(ledger: ChangeLedger):
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='theme-1',
            root_id='r-theme',
            subject='Workspace',
            predicate='theme',
            value='dark',
            created_at='2026-01-01T10:00:00Z',
        ),
        root_id='r-theme',
        recorded_at='2026-01-01T10:00:00Z',
    )
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='layout-1',
            root_id='r-layout',
            subject='Workspace',
            predicate='layout',
            value='grid',
            created_at='2026-01-01T11:00:00Z',
        ),
        root_id='r-layout',
        recorded_at='2026-01-01T11:00:00Z',
    )
    ledger.append_event(
        'invalidate',
        object_id='layout-1',
        target_object_id='layout-1',
        root_id='r-layout',
        recorded_at='2026-01-01T13:00:00Z',
    )
    ledger.append_event(
        'promote',
        object_id='theme-1',
        root_id='r-theme',
        recorded_at='2026-01-01T14:00:00Z',
    )

    result = await server.get_history('Workspace', group_ids=['s1_sessions_main'])

    assert 'error' not in result
    assert [(event['predicate'], event['timestamp']) for event in result['history']] == [
        ('layout', '2026-01-01T13:00:00Z'),
        ('theme', '2026-01-01T14:00:00Z'),
    ]
    assert [event['status'] for event in result['history']] == ['invalidated', 'active']


@pytest.mark.anyio
async def test_state_and_history_reject_empty_subject_consistently(ledger: ChangeLedger):
    current = await server.get_current_state('   ')
    history = await server.get_history('\n\t')

    assert current == {'error': 'invalid_input', 'message': 'subject must be a non-empty string'}
    assert history == {'error': 'invalid_input', 'message': 'subject must be a non-empty string'}


@pytest.mark.anyio
async def test_state_and_history_reject_empty_predicate_consistently(ledger: ChangeLedger):
    current = await server.get_current_state('UI', predicate='   ')
    history = await server.get_history('UI', predicate='   ')

    assert current == {'error': 'invalid_input', 'message': 'predicate must be a non-empty string'}
    assert history == {'error': 'invalid_input', 'message': 'predicate must be a non-empty string'}


@pytest.mark.anyio
async def test_state_and_history_fail_closed_without_group_scope(
    ledger: ChangeLedger,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(server, 'config', _config(default_group_id=None))

    current = await server.get_current_state('UI')
    history = await server.get_history('UI')

    expected = {
        'error': 'group_scope_required',
        'message': (
            'state/history retrieval requires explicit group_ids/lane_alias '
            'or a configured default group'
        ),
    }
    assert current == expected
    assert history == expected


@pytest.mark.anyio
async def test_get_current_state_returns_ledger_error_on_failure(monkeypatch: pytest.MonkeyPatch):
    def _boom():
        raise RuntimeError('boom')

    monkeypatch.setattr(server, '_change_ledger', _boom)
    result = await server.get_current_state('UI', group_ids=['s1_sessions_main'])

    assert result == {'error': 'ledger_error', 'message': 'boom'}


@pytest.mark.anyio
async def test_get_history_lifecycle_only_root_not_dropped_at_cap(
    ledger: ChangeLedger,
    monkeypatch: pytest.MonkeyPatch,
):
    """Regression: roots with recent lifecycle-only events (invalidate/promote)
    must not be dropped when the root cap is hit.

    The old ``_state_history_root_ids`` query ranked roots by
    ``max(recorded_at) WHERE payload_json IS NOT NULL``.  ``invalidate`` and
    ``promote`` events carry no ``payload_json``, so a root whose latest payload
    event is old but whose latest *lifecycle* event is recent would rank by its
    stale payload timestamp and fall off the cap.

    The fix switches the query to ``typed_roots`` whose ``latest_recorded_at``
    column is updated on every append — including lifecycle-only events.
    """
    # Reduce the cap to 2 so only 3 roots are enough to trigger the bug.
    monkeypatch.setattr(server, '_MAX_STATE_HISTORY_ROOTS', 2)

    # Root A (r-font): old assert at T=10, recent promote at T=14.
    # With the old bug this root ranks by T=10 (payload-only) and is dropped.
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='cap-font-1',
            root_id='r-cap-font',
            subject='CapSubject',
            predicate='font',
            value='serif',
            created_at='2026-01-01T10:00:00Z',
        ),
        root_id='r-cap-font',
        recorded_at='2026-01-01T10:00:00Z',
    )
    ledger.append_event(
        'promote',
        object_id='cap-font-1',
        root_id='r-cap-font',
        recorded_at='2026-01-01T14:00:00Z',
    )

    # Root B (r-theme): assert at T=11, no lifecycle → ranks by T=11.
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='cap-theme-1',
            root_id='r-cap-theme',
            subject='CapSubject',
            predicate='theme',
            value='dark',
            created_at='2026-01-01T11:00:00Z',
        ),
        root_id='r-cap-theme',
        recorded_at='2026-01-01T11:00:00Z',
    )

    # Root C (r-layout): assert at T=12, no lifecycle → ranks by T=12.
    ledger.append_event(
        'assert',
        payload=_state_fact(
            object_id='cap-layout-1',
            root_id='r-cap-layout',
            subject='CapSubject',
            predicate='layout',
            value='grid',
            created_at='2026-01-01T12:00:00Z',
        ),
        root_id='r-cap-layout',
        recorded_at='2026-01-01T12:00:00Z',
    )

    # With cap=2 the fix must return the two roots with the HIGHEST
    # latest_recorded_at: r-cap-font (T=14 via promote) and r-cap-layout (T=12).
    # The old (buggy) query would return r-cap-layout (T=12) and r-cap-theme
    # (T=11), silently dropping r-cap-font despite it being the most recently
    # touched root.
    result = await server.get_history('CapSubject', group_ids=['s1_sessions_main'])

    assert 'error' not in result
    predicates = {event['predicate'] for event in result['history']}
    assert 'font' in predicates, (
        'root with recent lifecycle-only event (promote at T=14) must not be '
        'dropped when root cap is hit'
    )
    assert 'layout' in predicates, 'second-most-recent root must be included'
    assert 'theme' not in predicates, (
        'oldest root (by latest_recorded_at) must be the one dropped at cap'
    )
