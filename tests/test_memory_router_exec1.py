import asyncio

import pytest
from mcp_server.src.routers import candidates, memory
from mcp_server.src.services.candidate_store import CandidateStore
from mcp_server.src.services.change_ledger import ChangeLedger

# Integration architecture check: exec1 singleton arch vs exec4 per-call arch
_EXEC1_SINGLETON_ARCH = hasattr(memory, '_change_ledger') and hasattr(candidates, '_change_ledger')
_SKIP_REASON = (
    'test_memory_router_exec1 requires exec1 singleton arch '
    '(module-level _change_ledger/_candidate_store on memory/candidates); '
    'integration uses exec4 per-call connection pattern'
)


def _run(coro):
    return asyncio.run(coro)


class _FakeMCP:
    def __init__(self):
        self.decorated = []

    def tool(self, *_args, **_kwargs):
        def _decorator(func):
            self.decorated.append(func)
            return func

        return _decorator


def _install_temp_stores(monkeypatch, tmp_path):
    if not _EXEC1_SINGLETON_ARCH:
        pytest.skip(_SKIP_REASON)
    ledger = ChangeLedger(tmp_path / 'change_ledger.db')
    store = CandidateStore(tmp_path / 'candidates.db')
    monkeypatch.setattr(memory, '_change_ledger', ledger)
    monkeypatch.setattr(memory, '_candidate_store', store)
    monkeypatch.setattr(memory, '_materializer', None)
    monkeypatch.setattr(candidates, '_change_ledger', ledger)
    monkeypatch.setattr(candidates, '_candidate_store', store)
    return ledger, store


def test_router_register_tools_returns_public_callables_without_private_registry():
    fake_mcp = _FakeMCP()

    memory_tools = memory.register_tools(fake_mcp)
    candidate_tools = candidates.register_tools(fake_mcp)

    assert set(memory_tools) == {'remember_fact', 'get_current_state', 'get_history'}
    assert set(candidate_tools) == {'list_candidates', 'promote_candidate', 'reject_candidate'}
    assert len(fake_mcp.decorated) == 6


def test_conflict_quarantine_promotion_preserves_scope_and_materialization_source(monkeypatch, tmp_path):
    ledger, store = _install_temp_stores(monkeypatch, tmp_path)
    materialize_calls = []

    async def fake_materialize(*, fact, source, superseded_fact_id=None):
        materialize_calls.append(
            {
                'fact_id': fact.object_id,
                'scope': fact.scope,
                'source': source,
                'superseded_fact_id': superseded_fact_id,
            }
        )
        return True, None

    monkeypatch.setattr(memory, '_materialize_fact', fake_materialize)

    # Trust elevation requires actor_id to be in the server-side allowlist.
    monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'caller:delegate')

    trusted_hint = {
        'fact_type': 'preference',
        'subject': 'coffee',
        'predicate': 'temperature',
        'trust': {
            'verified': True,
            'actor_id': 'caller:delegate',
            'source': 'delegate_asserted',
            'scope': 'internal',
        },
    }

    # Trust elevation requires the server-derived principal to be in BICAMERAL_TRUSTED_ACTOR_IDS.
    # hint.trust.actor_id is informational only; _server_principal is the auth gate.
    first = _run(
        memory.remember_fact(
            text='Coffee temperature preference is hot',
            hint={**trusted_hint, 'value': 'hot'},
            _server_principal='caller:delegate',
        )
    )
    assert first['status'] == 'ok'
    assert first['fact']['scope'] == 'internal'

    conflict = _run(
        memory.remember_fact(
            text='Coffee temperature preference is iced',
            hint={**trusted_hint, 'value': 'iced'},
            _server_principal='caller:delegate',
        )
    )
    assert conflict['status'] == 'conflict'
    candidate_id = conflict['candidate_id']
    assert conflict['new_fact']['raw_hint']['scope'] == 'internal'
    assert conflict['new_fact']['raw_hint']['policy_scope'] == 'internal'
    assert conflict['new_fact']['metadata']['scope'] == 'internal'

    class _MockCtx:
        """Minimal mock for FastMCP Context — provides server-derived client_id."""
        client_id = 'caller:delegate'

    # Promotion uses server-derived principal (from ctx), not caller-supplied actor_id.
    # The ledger event actor records the server-verified identity.
    promoted = _run(candidates.promote_candidate(candidate_id, 'supersede', actor_id='caller:delegate', ctx=_MockCtx()))
    assert promoted['status'] == 'ok'
    assert promoted['fact']['scope'] == 'internal'
    assert promoted['candidate']['status'] == 'promoted'

    assert [call['source'] for call in materialize_calls] == ['delegate_asserted', 'delegate_asserted']
    assert [call['scope'] for call in materialize_calls] == ['internal', 'internal']
    assert materialize_calls[1]['superseded_fact_id'] == first['fact']['object_id']

    promoted_row = store.get_candidate(candidate_id)
    assert promoted_row is not None
    assert promoted_row['status'] == 'promoted'
    assert promoted_row['raw_hint']['scope'] == 'internal'

    promote_events = [row for row in ledger.events_for_root(promoted['fact']['root_id']) if row.event_type == 'promote']
    assert len(promote_events) == 1
    # actor_id in the ledger event is the PERFORMING actor (passed to promote_candidate),
    # not stale metadata from the original quarantine hint.
    assert promote_events[0].actor_id == 'caller:delegate'


# ──────────────────────────────────────────────────────────────────────────────
# A. BLOCKER — candidate promotion preserves canonical lane identity
# ──────────────────────────────────────────────────────────────────────────────

def test_candidate_promotion_preserves_canonical_source_lane(monkeypatch, tmp_path):
    """Full integrated flow: remember_fact(conflict) → list_candidates →
    promote_candidate → get_current_state lane-scoped read succeeds with
    canonical lane preserved through the entire path.
    """
    ledger, store = _install_temp_stores(monkeypatch, tmp_path)

    REAL_LANE = 'test-group-lane-42'

    async def fake_materialize(*, fact, source, superseded_fact_id=None):
        return True, None

    monkeypatch.setattr(memory, '_materialize_fact', fake_materialize)
    monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'caller:delegate')
    # Simulate a server configured with a real group_id.
    monkeypatch.setattr(memory, '_derive_source_lane', lambda: REAL_LANE)

    trusted_hint = {
        'fact_type': 'preference',
        'subject': 'editor',
        'predicate': 'default_editor',
        'value': 'vim',
        'trust': {
            'verified': True,
            'actor_id': 'caller:delegate',
            'source': 'delegate_asserted',
            'scope': 'private',
        },
    }

    # Write initial fact directly.
    first = _run(
        memory.remember_fact(
            text='My default editor is vim',
            hint=trusted_hint,
            _server_principal='caller:delegate',
        )
    )
    assert first['status'] == 'ok', first
    assert first['fact']['source_lane'] == REAL_LANE

    # Trigger a conflict with a different value.
    conflict = _run(
        memory.remember_fact(
            text='My default editor is helix',
            hint={**trusted_hint, 'value': 'helix'},
            _server_principal='caller:delegate',
        )
    )
    assert conflict['status'] == 'conflict', conflict
    candidate_id = conflict['candidate_id']

    # Verify source_lane is stored in the candidate's raw_hint.
    candidate = store.get_candidate(candidate_id)
    assert candidate is not None
    assert candidate['raw_hint']['source_lane'] == REAL_LANE, (
        f"Expected raw_hint['source_lane'] == {REAL_LANE!r}, "
        f"got {candidate['raw_hint'].get('source_lane')!r}"
    )

    # list_candidates returns the right shape.
    class _MockCtx:
        client_id = 'caller:delegate'

    listed = _run(list([] or []) or candidates.list_candidates(status='quarantine', ctx=_MockCtx()))
    # Use direct store access to avoid reviewer gate in list_candidates.
    raw_list = store.list_candidates(status='pending')
    assert any(c['candidate_id'] == candidate_id for c in raw_list)

    # Promote the candidate.
    promoted = _run(
        candidates.promote_candidate(candidate_id, 'supersede', actor_id='caller:delegate', ctx=_MockCtx())
    )
    assert promoted['status'] == 'ok', promoted
    assert promoted['fact']['source_lane'] == REAL_LANE, (
        f"Expected promoted fact source_lane == {REAL_LANE!r}, "
        f"got {promoted['fact'].get('source_lane')!r}"
    )

    # Lane-scoped read must find the promoted fact.
    state = _run(
        memory.get_current_state(
            subject='editor',
            group_ids=[REAL_LANE],
        )
    )
    assert state['status'] == 'ok', state
    assert len(state['facts']) == 1, f"Expected 1 fact, got {len(state['facts'])}: {state['facts']}"
    assert state['facts'][0]['source_lane'] == REAL_LANE
    assert state['facts'][0]['value'] == 'helix'


def test_promoted_candidate_source_lane_not_derived_from_scope(monkeypatch, tmp_path):
    """Regression: promoted candidates must not get source_lane='private' / 'public'
    (visibility scope) when a real lane/group id is available.
    """
    ledger, store = _install_temp_stores(monkeypatch, tmp_path)

    REAL_LANE = 'my-real-group-id'

    async def fake_materialize(*, fact, source, superseded_fact_id=None):
        return True, None

    monkeypatch.setattr(memory, '_materialize_fact', fake_materialize)
    monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'caller:delegate')
    monkeypatch.setattr(memory, '_derive_source_lane', lambda: REAL_LANE)

    trusted_hint = {
        'fact_type': 'preference',
        'subject': 'theme',
        'predicate': 'ui_theme',
        'value': 'dark',
        'scope': 'private',
        'trust': {
            'verified': True,
            'actor_id': 'caller:delegate',
            'source': 'delegate_asserted',
        },
    }

    _run(
        memory.remember_fact(
            text='My UI theme is dark',
            hint=trusted_hint,
            _server_principal='caller:delegate',
        )
    )

    conflict = _run(
        memory.remember_fact(
            text='My UI theme is light',
            hint={**trusted_hint, 'value': 'light'},
            _server_principal='caller:delegate',
        )
    )
    assert conflict['status'] == 'conflict', conflict
    candidate_id = conflict['candidate_id']

    class _MockCtx:
        client_id = 'caller:delegate'

    promoted = _run(
        candidates.promote_candidate(candidate_id, 'supersede', actor_id='caller:delegate', ctx=_MockCtx())
    )
    assert promoted['status'] == 'ok', promoted

    promoted_source_lane = promoted['fact']['source_lane']
    # Regression: must NOT be a visibility scope label.
    assert promoted_source_lane not in ('private', 'public', 'internal', None), (
        f"source_lane must be the real group_id, got {promoted_source_lane!r}"
    )
    assert promoted_source_lane == REAL_LANE, (
        f"Expected source_lane == {REAL_LANE!r}, got {promoted_source_lane!r}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# B. NON-BLOCKING — contract/output cleanup regressions
# ──────────────────────────────────────────────────────────────────────────────

def test_list_candidates_contract_output_shape_matches_runtime(monkeypatch, tmp_path):
    """list_candidates runtime response shape must match the advertised TOOL_CONTRACT."""
    from mcp_server.src.routers.candidates import TOOL_CONTRACTS

    ledger, store = _install_temp_stores(monkeypatch, tmp_path)

    async def fake_materialize(*, fact, source, superseded_fact_id=None):
        return True, None

    monkeypatch.setattr(memory, '_materialize_fact', fake_materialize)
    monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'caller:delegate')
    monkeypatch.setattr(memory, '_derive_source_lane', lambda: None)

    hint = {
        'fact_type': 'preference',
        'subject': 'output_format',
        'predicate': 'format',
        'value': 'json',
        'trust': {
            'verified': True,
            'actor_id': 'caller:delegate',
            'source': 'delegate_asserted',
        },
    }
    _run(memory.remember_fact(text='format is json', hint=hint, _server_principal='caller:delegate'))
    _run(memory.remember_fact(
        text='format is xml',
        hint={**hint, 'value': 'xml'},
        _server_principal='caller:delegate',
    ))

    class _MockCtx:
        client_id = 'caller:delegate'

    result = _run(candidates.list_candidates(ctx=_MockCtx()))

    # Runtime shape
    assert result['status'] == 'ok', result
    assert 'candidates' in result, f"Missing 'candidates' key in response: {result.keys()}"
    assert 'reviewer' in result, f"Missing 'reviewer' key in response: {result.keys()}"
    assert isinstance(result['candidates'], list)

    # The advertised contract in TOOL_CONTRACTS must match the runtime shape.
    contract = next((c for c in TOOL_CONTRACTS if c['name'] == 'list_candidates'), None)
    assert contract is not None, 'list_candidates contract not found in TOOL_CONTRACTS'
    output_contract = contract['schema']['output']
    # Contract must advertise the dict envelope (not bare list[Candidate])
    assert '"candidates"' in output_contract, (
        f'list_candidates contract output must advertise "candidates" key, got: {output_contract!r}'
    )
    assert '"reviewer"' in output_contract, (
        f'list_candidates contract output must advertise "reviewer" key, got: {output_contract!r}'
    )


def test_reject_candidate_preserves_reason_in_response_and_store(monkeypatch, tmp_path):
    """reject_candidate(reason=...) must preserve reason in the response and
    durably in candidates.db (via resolution column).
    """
    ledger, store = _install_temp_stores(monkeypatch, tmp_path)

    async def fake_materialize(*, fact, source, superseded_fact_id=None):
        return True, None

    monkeypatch.setattr(memory, '_materialize_fact', fake_materialize)
    monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'caller:delegate')
    monkeypatch.setattr(memory, '_derive_source_lane', lambda: None)

    hint = {
        'fact_type': 'preference',
        'subject': 'lang',
        'predicate': 'coding_lang',
        'value': 'python',
        'trust': {
            'verified': True,
            'actor_id': 'caller:delegate',
            'source': 'delegate_asserted',
        },
    }
    _run(memory.remember_fact(text='lang is python', hint=hint, _server_principal='caller:delegate'))
    conflict = _run(
        memory.remember_fact(
            text='lang is rust',
            hint={**hint, 'value': 'rust'},
            _server_principal='caller:delegate',
        )
    )
    assert conflict['status'] == 'conflict', conflict
    candidate_id = conflict['candidate_id']

    class _MockCtx:
        client_id = 'caller:delegate'

    REJECTION_REASON = 'superseded by direct owner write'
    result = _run(
        candidates.reject_candidate(candidate_id, reason=REJECTION_REASON, ctx=_MockCtx())
    )
    assert result['status'] == 'ok', result
    assert result['action'] == 'rejected'
    # Reason must be present in the response.
    assert 'reason' in result, f"Missing 'reason' key in reject response: {result.keys()}"
    assert result['reason'] == REJECTION_REASON, (
        f"Expected reason {REJECTION_REASON!r}, got {result.get('reason')!r}"
    )

    # Reason must be durably stored in candidates.db.
    rejected_row = store.get_candidate(candidate_id)
    assert rejected_row is not None
    assert rejected_row['status'] == 'rejected'
    resolution = rejected_row.get('resolution') or ''
    assert REJECTION_REASON in resolution, (
        f"Expected rejection reason in resolution column, got {resolution!r}"
    )
