import asyncio

from mcp_server.src.routers import candidates, memory
from mcp_server.src.services.candidate_store import CandidateStore
from mcp_server.src.services.change_ledger import ChangeLedger


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
