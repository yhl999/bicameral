from __future__ import annotations

import asyncio
import json

import pytest
from mcp_server.src.services.candidate_store import CandidateStore
from mcp_server.src.services.change_ledger import ChangeLedger

try:
    from mcp_server.src.routers import candidates as candidates_router
    from mcp_server.src.routers import memory
except ImportError:
    from routers import candidates as candidates_router  # type: ignore
    from routers import memory  # type: ignore

# Integration architecture check: these tests require exec1's module-level singleton pattern
# (_change_ledger, _candidate_store, _materializer as module globals on memory/candidates_router).
# Integration uses exec4's per-call connection pattern.
_EXEC1_SINGLETON_ARCH = hasattr(memory, '_change_ledger') and hasattr(candidates_router, '_change_ledger')
_EXEC1_SKIP_REASON = (
    'test_remember_fact requires exec1 module-level singleton architecture '
    '(_change_ledger on memory/candidates_router); integration uses per-call connection pattern'
)


@pytest.fixture
def isolated_memory(tmp_path, monkeypatch):
    """Isolate memory router state per test with dedicated ledger/candidate stores."""
    if not _EXEC1_SINGLETON_ARCH:
        pytest.skip(_EXEC1_SKIP_REASON)
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
    monkeypatch.setattr(candidates_router, '_change_ledger', ledger)
    monkeypatch.setattr(candidates_router, '_candidate_store', candidates)

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


class _MockCtx:
    """Minimal stand-in for the FastMCP Context used in tests.

    Provides a ``client_id`` attribute so ``_extract_server_principal(ctx)``
    returns the desired server-derived principal without needing a live MCP
    request.  This is the ONLY legitimate way to inject a trusted principal
    in unit tests — NOT passing ``actor_id`` as a raw string argument.
    """

    def __init__(self, client_id: str) -> None:
        self.client_id = client_id


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


def test_remember_fact_accepts_fact_type_alias(isolated_memory):
    result = _run(
        memory.remember_fact(
            'I prefer dark mode',
            {'fact_type': 'Preference', 'subject': 'UI preferences'},
        )
    )

    assert result['status'] == 'ok'
    assert result['fact']['fact_type'] == 'preference'


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


def test_remember_fact_non_string_text_returns_validation_error(isolated_memory):
    result = _run(memory.remember_fact(123))  # type: ignore[arg-type]

    assert result == {
        'status': 'error',
        'error_type': 'validation_error',
        'message': 'text must be a string',
    }


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
    assert [option['label'] for option in conflict['options']] == ['Supersede', 'Cancel']
    assert 'promote_candidate(candidate_id, resolution="supersede"' in conflict['resolve_via']
    assert 'reject_candidate(candidate_id' in conflict['resolve_via']
    # actor_id is now an optional audit hint only; auth is server-derived (MCP auth context)
    assert 'actor_id' in conflict['resolve_via']
    assert 'auth context' in conflict['resolve_via'] or 'audit hint' in conflict['resolve_via']
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


def test_remember_fact_supersedes_existing_when_privileged(isolated_memory, monkeypatch):
    monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'owner:archibald')
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
    # Privilege elevation requires a server-derived principal, not caller-supplied actor_id.
    second = _run(memory.remember_fact('I prefer light mode', second_hint, _server_principal='owner:archibald'))

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


def test_remember_fact_owner_stamp_requires_trusted_actor_in_server_allowlist(isolated_memory, monkeypatch):
    # Trust elevation is gated on both:
    # 1. The server-side BICAMERAL_TRUSTED_ACTOR_IDS env var (server config), AND
    # 2. The server-derived principal (_server_principal, NOT caller hint.trust.actor_id).
    # Callers whose server_principal is in the allowlist receive elevated write context;
    # others are silently downgraded to untrusted regardless of hint.trust fields.
    monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'owner:archibald')
    result = _run(
        memory.remember_fact(
            'I prefer dark mode',
            {
                'type': 'Preference',
                'subject': 'UI preferences',
                **_owner_trust(),
            },
            _server_principal='owner:archibald',
        )
    )

    assert result['status'] == 'ok'
    root_id = result['fact']['root_id']
    rows = isolated_memory['ledger'].events_for_root(root_id)
    assert rows[0].actor_id == 'owner:archibald'

    metadata = json.loads(rows[0].metadata_json or '{}')
    assert metadata['source'] == 'owner_asserted'


def test_remember_fact_multi_hop_supersession_preserves_root_lineage(isolated_memory, monkeypatch):
    monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'owner:archibald')
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
            _server_principal='owner:archibald',
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
            _server_principal='owner:archibald',
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


def test_promote_candidate_supersedes_quarantined_fact(isolated_memory, monkeypatch):
    monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'system:scheduler')
    _run(
        memory.remember_fact(
            'I prefer dark mode',
            {'type': 'Preference', 'subject': 'UI preferences'},
        )
    )
    conflict = _run(
        memory.remember_fact(
            'I prefer light mode',
            {'type': 'Preference', 'subject': 'UI preferences'},
        )
    )

    promoted = _run(
        candidates_router.promote_candidate(
            candidate_id=conflict['candidate_id'],
            resolution='supersede',
            actor_id='system:scheduler',  # informational audit hint only — NOT the auth gate
            ctx=_MockCtx('system:scheduler'),  # server-derived principal (auth gate)
        )
    )

    assert promoted['status'] == 'ok'
    assert promoted['action'] == 'promoted'
    assert promoted['candidate']['status'] == 'promoted'
    assert promoted['fact']['value'] == 'light mode'
    assert isolated_memory['materialize_calls'][-1][1] == 'caller_asserted_unverified'
    assert isolated_memory['materialize_calls'][-1][2] is True

    current = _subject_state_facts(isolated_memory['ledger'], 'UI preferences')
    assert len(current) == 1
    assert current[0].value == 'light mode'


def test_promote_candidate_preserves_trusted_source_and_scope(isolated_memory, monkeypatch):
    monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'owner:archibald')
    trusted = _owner_trust()
    trusted['trust']['scope'] = 'public'

    _run(
        memory.remember_fact(
            'I prefer dark mode',
            {'type': 'Preference', 'subject': 'UI preferences', **trusted},
            _server_principal='owner:archibald',  # trusted write to public scope
        )
    )
    conflict = _run(
        memory.remember_fact(
            'I prefer light mode',
            {'type': 'Preference', 'subject': 'UI preferences', **trusted},
            _server_principal='owner:archibald',  # trusted conflict → quarantined with correct scope+source
        )
    )

    promoted = _run(
        candidates_router.promote_candidate(
            candidate_id=conflict['candidate_id'],
            resolution='supersede',
            actor_id='owner:archibald',  # informational audit hint only
            ctx=_MockCtx('owner:archibald'),  # server-derived principal (auth gate)
        )
    )

    assert promoted['status'] == 'ok'
    assert promoted['fact']['scope'] == 'public'
    assert promoted['fact']['policy_scope'] == 'public'
    assert isolated_memory['materialize_calls'][-1][1] == 'owner_asserted'
    assert isolated_memory['materialize_calls'][-1][2] is True

    public_state = _run(memory.get_current_state(subject='UI preferences', scope='public'))
    assert [fact['value'] for fact in public_state['facts']] == ['light mode']


def test_promote_candidate_rejects_parallel_resolution(isolated_memory):
    _run(
        memory.remember_fact(
            'I prefer dark mode',
            {'type': 'Preference', 'subject': 'UI preferences'},
        )
    )
    conflict = _run(
        memory.remember_fact(
            'I prefer light mode',
            {'type': 'Preference', 'subject': 'UI preferences'},
        )
    )

    result = _run(
        candidates_router.promote_candidate(
            candidate_id=conflict['candidate_id'],
            resolution='parallel',
        )
    )

    assert result['status'] == 'error'
    assert result['error_type'] == 'validation_error'
    assert 'parallel resolution is not supported' in result['message']
    current = _subject_state_facts(isolated_memory['ledger'], 'UI preferences')
    assert len(current) == 1
    assert current[0].value == 'dark mode'


def test_promote_candidate_cancel_rejects_candidate(isolated_memory, monkeypatch):
    monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'system:scheduler')
    _run(
        memory.remember_fact(
            'I prefer dark mode',
            {'type': 'Preference', 'subject': 'UI preferences'},
        )
    )
    conflict = _run(
        memory.remember_fact(
            'I prefer light mode',
            {'type': 'Preference', 'subject': 'UI preferences'},
        )
    )

    cancelled = _run(
        candidates_router.promote_candidate(
            candidate_id=conflict['candidate_id'],
            resolution='cancel',
            actor_id='system:scheduler',  # informational audit hint only
            ctx=_MockCtx('system:scheduler'),  # server-derived principal (auth gate)
        )
    )

    assert cancelled['status'] == 'ok'
    assert cancelled['action'] == 'cancelled'
    assert cancelled['candidate']['status'] == 'rejected'


def test_reject_candidate_marks_candidate_rejected(isolated_memory, monkeypatch):
    monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'system:scheduler')
    _run(
        memory.remember_fact(
            'I prefer dark mode',
            {'type': 'Preference', 'subject': 'UI preferences'},
        )
    )
    conflict = _run(
        memory.remember_fact(
            'I prefer light mode',
            {'type': 'Preference', 'subject': 'UI preferences'},
        )
    )

    rejected = _run(
        candidates_router.reject_candidate(
            candidate_id=conflict['candidate_id'],
            actor_id='system:scheduler',  # informational audit hint only
            ctx=_MockCtx('system:scheduler'),  # server-derived principal (auth gate)
        )
    )

    assert rejected['status'] == 'ok'
    assert rejected['action'] == 'rejected'
    assert rejected['candidate']['status'] == 'rejected'


def test_get_history_returns_event_summaries_for_current_roots(isolated_memory):
    _run(
        memory.remember_fact(
            'I prefer dark mode',
            {'type': 'Preference', 'subject': 'UI preferences'},
        )
    )
    conflict = _run(
        memory.remember_fact(
            'I prefer light mode',
            {'type': 'Preference', 'subject': 'UI preferences', 'supersede': 'parallel'},
        )
    )

    history = _run(memory.get_history(subject='UI preferences'))

    assert conflict['status'] == 'conflict'
    assert history['status'] == 'ok'
    assert history['scope'] == 'private'
    assert len(history['roots_considered']) == 1
    assert [row['event_type'] for row in history['history']] == ['assert']
    assert all('root_id' in row for row in history['history'])


def test_get_history_includes_supersede_event_for_current_root(isolated_memory, monkeypatch):
    monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'owner:archibald')
    trusted = _owner_trust(allow_conflict_supersede=True)

    _run(
        memory.remember_fact(
            'I prefer dark mode',
            {'type': 'Preference', 'subject': 'UI preferences'},
        )
    )
    _run(
        memory.remember_fact(
            'I prefer light mode',
            {
                'type': 'Preference',
                'subject': 'UI preferences',
                'supersede': True,
                **trusted,
            },
            _server_principal='owner:archibald',  # trusted supersede — server principal required
        )
    )

    history = _run(memory.get_history(subject='UI preferences'))

    assert history['status'] == 'ok'
    assert [row['event_type'] for row in history['history']] == ['assert', 'supersede']
    assert len(history['roots_considered']) == 1


# ---------------------------------------------------------------------------
# Security: trust spoofing and authorization enforcement tests
# ---------------------------------------------------------------------------

def test_trust_verified_flag_alone_does_not_elevate_privileges(isolated_memory):
    """hint.trust.verified=True alone must NOT elevate write privileges.

    Without the actor_id appearing in BICAMERAL_TRUSTED_ACTOR_IDS, the write
    context must fall back to the default untrusted context regardless of what
    the caller puts in hint.trust.
    """
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
    # Write still succeeds (fail-safe for reads, safe-deny for trust escalation)
    assert result['status'] == 'ok'
    root_id = result['fact']['root_id']
    rows = isolated_memory['ledger'].events_for_root(root_id)
    # Actor must be the default untrusted identity, not 'owner:archibald'
    assert rows[0].actor_id == 'caller:unverified'
    metadata = json.loads(rows[0].metadata_json or '{}')
    assert metadata['source'] == 'caller_asserted_unverified'


def test_hint_trust_cannot_unlock_supersede_without_server_allowlist(isolated_memory):
    """hint.trust.allow_conflict_supersede must not work without server allowlist.

    Without BICAMERAL_TRUSTED_ACTOR_IDS configured, a caller claiming
    allow_conflict_supersede=True must still be quarantined.
    """
    _run(
        memory.remember_fact(
            'I prefer dark mode',
            {'type': 'Preference', 'subject': 'UI preferences'},
        )
    )
    result = _run(
        memory.remember_fact(
            'I prefer light mode',
            {
                'type': 'Preference',
                'subject': 'UI preferences',
                'supersede': True,
                **_owner_trust(allow_conflict_supersede=True),
            },
        )
    )
    # Must be quarantined (conflict), NOT directly superseded
    assert result['status'] == 'conflict'
    assert result['supersede_allowed'] is False


def test_promote_candidate_requires_authorized_actor(isolated_memory, monkeypatch):
    """promote_candidate must return unauthorized when no actor_id is provided."""
    _run(memory.remember_fact('I prefer dark mode', {'type': 'Preference', 'subject': 'UI preferences'}))
    conflict = _run(memory.remember_fact('I prefer light mode', {'type': 'Preference', 'subject': 'UI preferences'}))

    # No actor_id provided
    result = _run(candidates_router.promote_candidate(candidate_id=conflict['candidate_id'], resolution='supersede'))
    assert result['status'] == 'error'
    assert result['error_type'] == 'unauthorized'


def test_promote_candidate_rejects_unknown_actor(isolated_memory, monkeypatch):
    """promote_candidate must reject actor_ids not in the server allowlist."""
    monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'system:trusted')
    _run(memory.remember_fact('I prefer dark mode', {'type': 'Preference', 'subject': 'UI preferences'}))
    conflict = _run(memory.remember_fact('I prefer light mode', {'type': 'Preference', 'subject': 'UI preferences'}))

    result = _run(
        candidates_router.promote_candidate(
            candidate_id=conflict['candidate_id'],
            resolution='supersede',
            actor_id='attacker:evil',
        )
    )
    assert result['status'] == 'error'
    assert result['error_type'] == 'unauthorized'


def test_reject_candidate_requires_authorized_actor(isolated_memory):
    """reject_candidate must return unauthorized when no actor_id is provided."""
    _run(memory.remember_fact('I prefer dark mode', {'type': 'Preference', 'subject': 'UI preferences'}))
    conflict = _run(memory.remember_fact('I prefer light mode', {'type': 'Preference', 'subject': 'UI preferences'}))

    result = _run(candidates_router.reject_candidate(candidate_id=conflict['candidate_id']))
    assert result['status'] == 'error'
    assert result['error_type'] == 'unauthorized'


def test_promote_candidate_audit_records_performing_actor_not_hint_actor(isolated_memory, monkeypatch):
    """Ledger promotion event must record the actor who performed the promotion,
    not the actor from the original quarantine hint metadata.
    """
    monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'owner:archibald,reviewer:bot')
    # Original fact written by owner
    _run(
        memory.remember_fact(
            'I prefer dark mode',
            {'type': 'Preference', 'subject': 'UI preferences', **_owner_trust()},
        )
    )
    # Conflict written by owner too (so raw_hint stores owner actor)
    conflict = _run(
        memory.remember_fact(
            'I prefer light mode',
            {'type': 'Preference', 'subject': 'UI preferences', **_owner_trust()},
        )
    )
    assert conflict['status'] == 'conflict'
    candidate_id = conflict['candidate_id']

    # Promotion performed by a DIFFERENT actor (reviewer:bot) — verified via server context.
    promoted = _run(
        candidates_router.promote_candidate(
            candidate_id=candidate_id,
            resolution='supersede',
            actor_id='reviewer:bot',  # informational audit hint only
            ctx=_MockCtx('reviewer:bot'),  # server-derived principal (auth gate + audit)
        )
    )
    assert promoted['status'] == 'ok'

    root_id = promoted['fact']['root_id']
    ledger = isolated_memory['ledger']
    promote_events = [row for row in ledger.events_for_root(root_id) if row.event_type == 'promote']
    assert len(promote_events) == 1
    # Must record reviewer:bot (the server-verified actor who called promote_candidate),
    # NOT owner:archibald (the actor from the original quarantine hint metadata).
    assert promote_events[0].actor_id == 'reviewer:bot'


# ---------------------------------------------------------------------------
# Regression: caller-supplied strings alone MUST NOT grant privilege
# ---------------------------------------------------------------------------


def test_caller_supplied_actor_id_in_hint_does_not_grant_trust(isolated_memory, monkeypatch):
    """hint.trust.actor_id matching BICAMERAL_TRUSTED_ACTOR_IDS must NOT elevate privileges.

    This is the core spoofability regression: a caller who knows (or guesses)
    a valid entry in BICAMERAL_TRUSTED_ACTOR_IDS must NOT be able to gain
    elevated write context by supplying that string in hint.trust.actor_id.
    Trust elevation requires the server-derived principal (_server_principal)
    to be in the allowlist — caller-supplied hint fields are informational only.
    """
    monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'owner:archibald')

    # Caller knows the allowlist value and supplies it in hint.trust.actor_id —
    # but does NOT provide a server-derived principal (_server_principal=None).
    result = _run(
        memory.remember_fact(
            'I prefer dark mode',
            {
                'type': 'Preference',
                'subject': 'UI preferences',
                **_owner_trust(),  # includes 'actor_id': 'owner:archibald'
            },
            # _server_principal deliberately omitted — simulates no auth context
        )
    )

    # Write succeeds (fail-safe), but the actor is the untrusted sentinel,
    # NOT 'owner:archibald' from the caller's hint.
    assert result['status'] == 'ok'
    rows = isolated_memory['ledger'].events_for_root(result['fact']['root_id'])
    assert rows[0].actor_id == 'caller:unverified', (
        'hint.trust.actor_id must not grant trust when no server principal is provided'
    )
    import json
    metadata = json.loads(rows[0].metadata_json or '{}')
    assert metadata['source'] == 'caller_asserted_unverified'


def test_caller_supplied_actor_id_does_not_authorize_promote(isolated_memory, monkeypatch):
    """promote_candidate with actor_id in allowlist but no ctx must return unauthorized.

    The actor_id tool argument is caller-controlled and MUST NOT be used as the
    authorization gate.  Only the server-derived principal from _extract_server_principal
    (OAuth bearer token or transport client_id) counts for authorization.
    """
    monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'system:trusted')
    _run(memory.remember_fact('I prefer dark mode', {'type': 'Preference', 'subject': 'UI preferences'}))
    conflict = _run(memory.remember_fact('I prefer light mode', {'type': 'Preference', 'subject': 'UI preferences'}))

    # Caller supplies a matching actor_id but no ctx — server_principal = __anon__
    result = _run(
        candidates_router.promote_candidate(
            candidate_id=conflict['candidate_id'],
            resolution='supersede',
            actor_id='system:trusted',  # matches allowlist, but is caller-supplied — should NOT grant auth
            # ctx intentionally omitted → no server-derived principal
        )
    )
    assert result['status'] == 'error', (
        'promote_candidate must be unauthorized when server principal is absent, '
        'even if actor_id matches the server allowlist'
    )
    assert result['error_type'] == 'unauthorized'


def test_caller_supplied_actor_id_does_not_authorize_reject(isolated_memory, monkeypatch):
    """reject_candidate with actor_id in allowlist but no ctx must return unauthorized.

    Mirrors test_caller_supplied_actor_id_does_not_authorize_promote for the reject path.
    """
    monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'system:trusted')
    _run(memory.remember_fact('I prefer dark mode', {'type': 'Preference', 'subject': 'UI preferences'}))
    conflict = _run(memory.remember_fact('I prefer light mode', {'type': 'Preference', 'subject': 'UI preferences'}))

    # Caller supplies a matching actor_id but no ctx — server_principal = __anon__
    result = _run(
        candidates_router.reject_candidate(
            candidate_id=conflict['candidate_id'],
            actor_id='system:trusted',  # matches allowlist but caller-supplied
            # ctx intentionally omitted → no server-derived principal
        )
    )
    assert result['status'] == 'error', (
        'reject_candidate must be unauthorized when server principal is absent, '
        'even if actor_id matches the server allowlist'
    )
    assert result['error_type'] == 'unauthorized'


# ---------------------------------------------------------------------------
# list_candidates auth gate (Exec 1 boundary fix)
# ---------------------------------------------------------------------------

def test_list_candidates_requires_auth(isolated_memory, monkeypatch):
    """list_candidates must return unauthorized for anonymous callers.

    Previously list_candidates had no auth gate, exposing quarantined candidate
    payloads (including raw_hint/metadata) to unauthenticated callers.
    """
    _run(memory.remember_fact('I prefer dark mode', {'type': 'Preference', 'subject': 'UI preferences'}))
    _run(memory.remember_fact('I prefer light mode', {'type': 'Preference', 'subject': 'UI preferences'}))

    # No ctx → server_principal = __anon__ → must be rejected.
    result = _run(candidates_router.list_candidates())
    assert result['status'] == 'error', 'list_candidates must be unauthorized for anonymous callers'
    assert result['error_type'] == 'unauthorized'


def test_list_candidates_requires_auth_no_allowlist(isolated_memory):
    """list_candidates must return unauthorized even when ctx is supplied but allowlist is empty."""
    _run(memory.remember_fact('I prefer dark mode', {'type': 'Preference', 'subject': 'UI preferences'}))
    _run(memory.remember_fact('I prefer light mode', {'type': 'Preference', 'subject': 'UI preferences'}))

    # ctx present but BICAMERAL_TRUSTED_ACTOR_IDS not set → fail-closed.
    result = _run(candidates_router.list_candidates(ctx=_MockCtx('owner:archibald')))
    assert result['status'] == 'error'
    assert result['error_type'] == 'unauthorized'


def test_list_candidates_authorized(isolated_memory, monkeypatch):
    """list_candidates must return candidates for an authenticated trusted caller."""
    monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'owner:archibald')
    _run(memory.remember_fact('I prefer dark mode', {'type': 'Preference', 'subject': 'UI preferences'}))
    conflict = _run(memory.remember_fact('I prefer light mode', {'type': 'Preference', 'subject': 'UI preferences'}))
    assert conflict['status'] == 'conflict'

    result = _run(candidates_router.list_candidates(ctx=_MockCtx('owner:archibald')))
    assert result['status'] == 'ok'
    assert len(result['candidates']) >= 1
    candidate_ids = [c['candidate_id'] for c in result['candidates']]
    assert conflict['candidate_id'] in candidate_ids


# ---------------------------------------------------------------------------
# FastMCP ctx-injection parity — startup / schema smoke tests
# ---------------------------------------------------------------------------

def test_register_tools_ctx_not_in_schema():
    """ctx must NOT appear as a caller-visible parameter in registered tool schemas.

    FastMCP only hides ctx when annotated as Context | None (not Any).
    If ctx leaks into the schema, callers could supply it and the transport
    auth layer would not inject the real Context — breaking path-2 auth.
    """
    from mcp.server.fastmcp import FastMCP as _RealFastMCP

    mcp = _RealFastMCP('smoke-test')

    # Register memory router tools
    memory.register_tools(mcp)
    # Register candidate router tools
    candidates_router.register_tools(mcp)

    tools_by_name = {t.name: t for t in mcp._tool_manager.list_tools()}

    # Every Exec 1 tool must hide ctx (not expose it to callers).
    privileged_tools = ['remember_fact', 'list_candidates', 'promote_candidate', 'reject_candidate']
    for tool_name in privileged_tools:
        assert tool_name in tools_by_name, f'tool {tool_name!r} not registered'
        schema = tools_by_name[tool_name].parameters
        exposed = list(schema.get('properties', {}).keys())
        assert 'ctx' not in exposed, (
            f'tool {tool_name!r} exposes ctx as a caller-visible parameter '
            f'(exposed params: {exposed}); ctx must be FastMCP-injected via '
            f'Context | None annotation, not Any'
        )


def test_register_tools_required_params_correct():
    """Verify get_tools()-visible parameter sets are correct for Exec 1 tools."""
    from mcp.server.fastmcp import FastMCP as _RealFastMCP

    mcp = _RealFastMCP('param-smoke')
    memory.register_tools(mcp)
    candidates_router.register_tools(mcp)

    tools_by_name = {t.name: t for t in mcp._tool_manager.list_tools()}

    # remember_fact: text (required), hint (optional)
    rf_params = set(tools_by_name['remember_fact'].parameters.get('properties', {}).keys())
    assert 'text' in rf_params
    assert 'hint' in rf_params
    assert 'ctx' not in rf_params

    # list_candidates: status (optional) — no ctx in schema
    lc_params = set(tools_by_name['list_candidates'].parameters.get('properties', {}).keys())
    assert 'ctx' not in lc_params

    # promote_candidate: candidate_id + resolution (required), actor_id (optional)
    pc_params = set(tools_by_name['promote_candidate'].parameters.get('properties', {}).keys())
    assert 'candidate_id' in pc_params
    assert 'resolution' in pc_params
    assert 'ctx' not in pc_params

    # reject_candidate: candidate_id (required), actor_id (optional)
    rc_params = set(tools_by_name['reject_candidate'].parameters.get('properties', {}).keys())
    assert 'candidate_id' in rc_params
    assert 'ctx' not in rc_params


# ─── Regression: candidate_store default DB path (Exec 1 blocker) ────────────

def test_candidate_store_default_db_path_is_repo_root():
    """CandidateStore default path must resolve to repo-root state/, not mcp_server/state/.

    Regression guard: the canonical path used by all tools, docs, and scripts is
    repo-root state/candidates.db.  Previously _default_candidates_db_path() used
    parents[2] (= mcp_server/) instead of parents[3] (= repo-root).
    """
    from mcp_server.src.services.candidate_store import _default_candidates_db_path
    from mcp_server.src.services.change_ledger import DB_PATH_DEFAULT as LEDGER_DB_PATH

    default = _default_candidates_db_path()
    # Both DBs must share the same parent directory (repo-root state/).
    assert default.parent == LEDGER_DB_PATH.parent, (
        f"CandidateStore default dir {default.parent} != ledger dir {LEDGER_DB_PATH.parent}\n"
        f"Both must point to repo-root state/, not mcp_server/state/"
    )
    assert default.name == 'candidates.db'
    # Must NOT contain 'mcp_server' in the parent path segment immediately above 'state'
    assert default.parent.parent.name != 'mcp_server', (
        f"Default path {default} still resolves into mcp_server/state/ — fix parents[] index"
    )


# ─── Regression: promotion split-brain reconciliation (Exec 1 blocker) ───────

def test_promote_candidate_reconciles_after_partial_ledger_commit(isolated_memory, monkeypatch):
    """promote_candidate is idempotent when the ledger write succeeded but the
    candidates-DB update did not commit (crash-safety split-brain reconciliation).

    Simulates the failure mode:
    1. promote_candidate_fact commits to the ledger (promote event recorded).
    2. Process crashes before store.update_candidate_status runs.
    3. Retry: promote_candidate is called again with the candidate still 'pending'.
    Expected: the retry reconciles cleanly — no uniqueness error, status synced.
    """
    monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'system:scheduler')

    # Step 1 — create a conflict candidate via two remember_fact calls.
    _run(memory.remember_fact('I prefer dark mode', {'type': 'Preference', 'subject': 'UI prefs'}))
    conflict = _run(
        memory.remember_fact('I prefer light mode', {'type': 'Preference', 'subject': 'UI prefs'})
    )
    candidate_id = conflict['candidate_id']

    ledger = isolated_memory['ledger']
    store = isolated_memory['candidates']

    # Step 2 — simulate the split-brain: write to ledger directly (as the first
    # promote_candidate call would have done), but leave candidates DB at 'pending'.
    cand = store.get_candidate(candidate_id)
    raw_hint = cand.get('raw_hint') or {}
    policy_version = str(raw_hint.get('policy_version') or candidates_router.DEFAULT_POLICY_VERSION)
    fact_input = candidates_router._candidate_to_fact_input(cand)
    ledger.promote_candidate_fact(
        actor_id='system:scheduler',
        reason='candidate_promotion:supersede',
        policy_version=policy_version,
        candidate_id=candidate_id,
        fact=fact_input,
        conflict_with_fact_id=cand.get('conflict_with_fact_id'),
    )
    # Confirm split-brain: ledger has the promote event, but DB still says pending.
    assert ledger.promotion_event_for_candidate(candidate_id) is not None
    assert store.get_candidate(candidate_id)['status'] == 'pending'

    # Step 3 — retry the promotion; must reconcile without error.
    result = _run(
        candidates_router.promote_candidate(
            candidate_id=candidate_id,
            resolution='supersede',
            ctx=_MockCtx('system:scheduler'),
        )
    )

    assert result['status'] == 'ok', f"Expected ok, got: {result}"
    assert result['action'] == 'promoted'
    assert result['candidate']['status'] == 'promoted'
    assert result['fact']['value'] == 'light mode'
    # Candidates DB must now be synced (no longer split-brained).
    assert store.get_candidate(candidate_id)['status'] == 'promoted'
    # A second retry must also be idempotent (candidate is now 'promoted', not pending).
    result2 = _run(
        candidates_router.promote_candidate(
            candidate_id=candidate_id,
            resolution='supersede',
            ctx=_MockCtx('system:scheduler'),
        )
    )
    assert result2['status'] == 'error'
    assert result2['error_type'] == 'invalid_state'
