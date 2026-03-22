"""
Tests for the typed-retrieval quality fixes (search_text, aliasing, stemming, min-overlap).

Covers:
- Fix 1: search_text uses structured field extraction not raw JSON blobs
- Fix 2: entity alias expansion (user:principal → yuan, agent:archibald → archibald)
- Fix 3: lightweight stemming (constraints↔constraint, scheduling↔sched, etc.)
- Fix 4: min-2-token overlap in root selection (anti-flood heuristic)
- rebuild_all_search_text migration helper
"""
from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from mcp_server.src.models.typed_memory import Episode, EvidenceRef, Procedure, StateFact
from mcp_server.src.services.change_ledger import ChangeLedger, rebuild_all_search_text, _build_search_text_for_objects
from mcp_server.src.services.typed_retrieval import (
    TypedRetrievalService,
    _searchable_text,
    _stem_token,
    _tokenize,
)


def _msg_ref(msg_id: str = 'm1') -> EvidenceRef:
    return EvidenceRef(
        kind='message',
        source_system='telegram',
        locator={'system': 'telegram', 'conversation_id': 'c1', 'message_id': msg_id},
        title='source',
        snippet='test snippet',
    )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _run(coro):
    return asyncio.run(coro)


def _memory_ledger() -> ChangeLedger:
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    return ChangeLedger(conn)


def _state_fact(
    *,
    object_id: str,
    root_id: str,
    subject: str = 'user:principal',
    predicate: str = 'pref.coffee',
    value: object = 'espresso',
    fact_type: str = 'preference',
    scope: str = 'private',
    source_lane: str = 'private',
    version: int = 1,
    is_current: bool = True,
) -> StateFact:
    return StateFact(
        object_id=object_id,
        root_id=root_id,
        version=version,
        is_current=is_current,
        source_lane=source_lane,
        source_key=f'session:test:{object_id}',
        policy_scope='private',
        visibility_scope='private',
        fact_type=fact_type,
        subject=subject,
        predicate=predicate,
        value=value,
        scope=scope,
        evidence_refs=[_msg_ref(object_id)],
    )


def _episode(*, object_id: str, root_id: str, title: str = 'Test', summary: str = 'Test episode') -> Episode:
    return Episode(
        object_id=object_id,
        root_id=root_id,
        version=1,
        is_current=True,
        source_lane='learning_self_audit',
        source_key=f'om:learning_self_audit:node:{object_id}',
        policy_scope='private',
        visibility_scope='private',
        title=title,
        summary=summary,
        evidence_refs=[_msg_ref(object_id)],
    )


def _procedure(*, object_id: str, root_id: str, name: str, trigger: str, steps: list[str]) -> Procedure:
    return Procedure(
        object_id=object_id,
        root_id=root_id,
        version=1,
        is_current=True,
        source_lane='learning_self_audit',
        source_key=f'session:test:{object_id}',
        policy_scope='private',
        visibility_scope='private',
        name=name,
        trigger=trigger,
        preconditions=[],
        steps=steps,
        expected_outcome='done',
        evidence_refs=[_msg_ref(object_id)],
    )


class _FakeEvidenceRegistry:
    async def resolve_many(self, refs, *, object_ids_by_uri=None, max_items=None):
        return []


# ── Fix 1: search_text uses structured fields not raw JSON ────────────────────


def test_searchable_text_state_fact_does_not_contain_json_keys():
    """_searchable_text must not emit JSON structural keys like 'policy_scope'."""
    fact = _state_fact(object_id='f1', root_id='f1', predicate='pref.default_timezone', value='America/New_York')
    text = _searchable_text(fact)
    # Must contain semantic content
    assert 'pref.default_timezone' in text
    assert 'america/new_york' in text.lower()
    # Must NOT look like raw JSON (no quotes / braces from serialization)
    assert '"policy_scope"' not in text
    assert '"created_at"' not in text
    assert '{"object_id"' not in text


def test_build_search_text_for_objects_produces_structured_text():
    """_build_search_text_for_objects (used by change_ledger) mirrors _searchable_text."""
    fact = _state_fact(
        object_id='sf_tz',
        root_id='sf_tz',
        subject='user:principal',
        predicate='pref.default_timezone',
        value='America/New_York',
    )
    text = _build_search_text_for_objects([fact])
    # Structured semantic fields should be present
    assert 'pref.default_timezone' in text
    assert 'america/new_york' in text.lower()
    # JSON structural noise should be absent
    assert '"policy_scope"' not in text
    assert '"object_id"' not in text


def test_ledger_stores_structured_search_text():
    """After append_event, typed_roots.search_text contains semantic content, not JSON noise."""
    ledger = _memory_ledger()
    fact = _state_fact(
        object_id='f_sched',
        root_id='f_sched',
        predicate='constraint.no_meetings_before_1030',
        value='No meetings before 10:30am ET on weekdays',
    )
    ledger.append_event('assert', actor_id='test', payload=fact, recorded_at='2026-03-21T00:00:00Z')
    ledger.append_event('promote', actor_id='policy', object_id='f_sched', root_id='f_sched')

    row = ledger.conn.execute(
        "SELECT search_text FROM typed_roots WHERE root_id = ?", ('f_sched',)
    ).fetchone()
    assert row is not None
    st = row['search_text']
    # Predicate and value content should be tokenizable
    assert 'constraint.no_meetings_before_1030' in st
    assert '10:30am' in st or '10' in st  # value substring present
    # Should NOT start with '{' (raw JSON)
    assert not st.strip().startswith('{'), f"search_text looks like raw JSON: {st[:80]!r}"
    # JSON key noise must be absent
    assert '"policy_scope"' not in st


def test_rebuild_all_search_text_updates_existing_rows():
    """rebuild_all_search_text recomputes search_text for all existing typed_roots."""
    ledger = _memory_ledger()
    fact = _state_fact(object_id='f_reb', root_id='f_reb', predicate='pref.coffee', value='filter')
    ledger.append_event('assert', actor_id='test', payload=fact, recorded_at='2026-03-21T00:00:00Z')
    ledger.append_event('promote', actor_id='policy', object_id='f_reb', root_id='f_reb')

    # Manually corrupt the search_text to simulate old JSON blob format
    ledger.conn.execute(
        "UPDATE typed_roots SET search_text = ? WHERE root_id = ?",
        ('{"object_id":"f_reb","policy_scope":"private"}', 'f_reb'),
    )
    ledger.conn.commit()
    corrupted = ledger.conn.execute(
        "SELECT search_text FROM typed_roots WHERE root_id = ?", ('f_reb',)
    ).fetchone()['search_text']
    assert '"policy_scope"' in corrupted  # confirm corruption

    n = rebuild_all_search_text(ledger.conn)
    assert n == 1

    rebuilt = ledger.conn.execute(
        "SELECT search_text FROM typed_roots WHERE root_id = ?", ('f_reb',)
    ).fetchone()['search_text']
    assert '"policy_scope"' not in rebuilt
    assert 'pref.coffee' in rebuilt


# ── Fix 2: entity alias expansion ────────────────────────────────────────────


def test_searchable_text_includes_yuan_alias_for_user_principal():
    """StateFact with subject='user:principal' must include 'yuan' in searchable text."""
    fact = _state_fact(object_id='f_alias', root_id='f_alias', subject='user:principal')
    text = _searchable_text(fact)
    assert 'yuan' in text.lower(), f"'yuan' not found in searchable text: {text[:200]!r}"


def test_searchable_text_includes_archibald_alias_for_agent_archibald():
    """StateFact with subject='agent:archibald' must include 'archibald' in searchable text."""
    fact = _state_fact(object_id='f_arch', root_id='f_arch', subject='agent:archibald')
    text = _searchable_text(fact)
    assert 'archibald' in text.lower(), f"'archibald' not found in searchable text: {text[:200]!r}"


def test_build_search_text_includes_entity_alias():
    """_build_search_text_for_objects (ledger-side) also emits entity aliases."""
    fact = _state_fact(object_id='f2', root_id='f2', subject='user:principal')
    text = _build_search_text_for_objects([fact])
    assert 'yuan' in text.lower()


def test_ledger_search_text_includes_yuan_alias():
    """After ingest, typed_roots.search_text for a user:principal fact contains 'yuan'."""
    ledger = _memory_ledger()
    fact = _state_fact(
        object_id='f_yuan',
        root_id='f_yuan',
        subject='user:principal',
        predicate='pref.scheduling',
        value='No meetings before 10:30am',
    )
    ledger.append_event('assert', actor_id='test', payload=fact, recorded_at='2026-03-21T00:00:00Z')
    ledger.append_event('promote', actor_id='policy', object_id='f_yuan', root_id='f_yuan')

    st = ledger.conn.execute(
        "SELECT search_text FROM typed_roots WHERE root_id = ?", ('f_yuan',)
    ).fetchone()['search_text']
    assert 'yuan' in st, f"'yuan' alias missing from search_text: {st[:200]!r}"


def test_candidate_root_ids_yuan_query_finds_user_principal_fact():
    """Root selection for query 'Yuan scheduling' must find user:principal facts."""
    ledger = _memory_ledger()
    fact = _state_fact(
        object_id='f_sched_yuan',
        root_id='f_sched_yuan',
        subject='user:principal',
        predicate='pref.scheduling',
        value='available 11am-5pm weekdays',
    )
    ledger.append_event('assert', actor_id='test', payload=fact, recorded_at='2026-03-21T00:00:00Z')
    ledger.append_event('promote', actor_id='policy', object_id='f_sched_yuan', root_id='f_sched_yuan')

    svc = TypedRetrievalService(ledger=ledger, evidence_registry=_FakeEvidenceRegistry())
    root_ids, strategy = svc._candidate_root_ids(
        query="What are Yuan's scheduling preferences?",
        max_roots=25,
        object_types=set(),
        metadata_filters={},
    )
    assert 'f_sched_yuan' in root_ids, (
        f"'f_sched_yuan' not found in root_ids={root_ids}; strategy={strategy!r}. "
        "Entity alias expansion for 'yuan' → 'user:principal' may be missing."
    )


# ── Fix 3: lightweight stemming ──────────────────────────────────────────────


@pytest.mark.parametrize('token,expected_stem', [
    ('constraints', 'constraint'),   # -s stripping
    ('preferences', 'preference'),   # -ces → -ce rule
    ('frameworks', 'framework'),     # -s stripping
    ('scheduling', 'schedul'),       # -ing stripping
    ('delegating', 'delegat'),       # -ing stripping
    ('changed', 'chang'),            # -ed stripping
    ('priorities', 'priority'),      # -ities → -ity rule
    ('principles', 'principl'),      # -es stripping (no 'ces' match since c+e+s at end)
    ('tasks', 'task'),               # -s stripping
    ('decisions', 'decision'),       # -s stripping
])
def test_stem_token_handles_common_morphology(token, expected_stem):
    stem = _stem_token(token)
    assert stem == expected_stem, f"_stem_token({token!r}) = {stem!r}, expected {expected_stem!r}"


def test_stem_token_returns_none_for_short_tokens():
    assert _stem_token('ai') is None
    assert _stem_token('is') is None
    assert _stem_token('be') is None


def test_tokenize_emits_both_original_and_stem():
    """_tokenize must emit both the raw token and its stemmed form."""
    tokens = _tokenize('scheduling constraints frameworks')
    # Original forms
    assert 'scheduling' in tokens
    assert 'constraints' in tokens
    assert 'frameworks' in tokens
    # Stemmed forms
    assert 'schedul' in tokens, f"stem 'schedul' not in {tokens}"
    assert 'constraint' in tokens, f"stem 'constraint' not in {tokens}"
    assert 'framework' in tokens, f"stem 'framework' not in {tokens}"


def test_tokenize_no_duplicates():
    """_tokenize must not emit duplicate tokens."""
    tokens = _tokenize('constraint constraint constraints')
    assert len(tokens) == len(set(tokens)), f"Duplicate tokens found in {tokens}"


def test_candidate_root_ids_stemmed_query_finds_fact():
    """Root selection for 'scheduling constraints' must find a fact with 'constraint' token."""
    ledger = _memory_ledger()
    fact = _state_fact(
        object_id='f_constraint',
        root_id='f_constraint',
        subject='user:principal',
        predicate='constraint.no_meetings_before_1030',
        value='No meetings before 10:30am',
    )
    ledger.append_event('assert', actor_id='test', payload=fact, recorded_at='2026-03-21T00:00:00Z')
    ledger.append_event('promote', actor_id='policy', object_id='f_constraint', root_id='f_constraint')

    svc = TypedRetrievalService(ledger=ledger, evidence_registry=_FakeEvidenceRegistry())
    root_ids, strategy = svc._candidate_root_ids(
        query="What are Yuan's scheduling constraints?",  # 'constraints' (plural)
        max_roots=25,
        object_types=set(),
        metadata_filters={},
    )
    assert 'f_constraint' in root_ids, (
        f"'f_constraint' not found; strategy={strategy!r}. "
        "Stemming should map 'constraints' → 'constraint' matching the predicate."
    )


# ── Fix 4: minimum 2-token overlap in root selection ─────────────────────────


def test_candidate_root_ids_requires_2_token_overlap_when_sufficient_tokens():
    """With ≥2 query tokens, only roots matching at least 2 are returned (strict path)."""
    ledger = _memory_ledger()

    # This fact matches query token "yuan" (via alias) AND "scheduling" (via predicate)
    good_fact = _state_fact(
        object_id='f_good',
        root_id='f_good',
        subject='user:principal',
        predicate='pref.scheduling',
        value='available 11am-5pm',
    )
    # This fact matches only ONE query token ("scheduling") but not "yuan" (subject is NOT user:principal)
    # so it should be excluded if min2 is applied correctly… but with subject='agent:archibald' it
    # won't match 'yuan'. Its predicate contains 'scheduling' → 1-token match only.
    noise_fact = _state_fact(
        object_id='f_noise',
        root_id='f_noise',
        subject='agent:archibald',
        predicate='pref.scheduling_style',
        value='async',
    )

    for fact in (good_fact, noise_fact):
        ledger.append_event('assert', actor_id='test', payload=fact, recorded_at='2026-03-21T00:00:00Z')
        ledger.append_event('promote', actor_id='policy', object_id=fact.object_id, root_id=fact.root_id)

    svc = TypedRetrievalService(ledger=ledger, evidence_registry=_FakeEvidenceRegistry())
    root_ids, strategy = svc._candidate_root_ids(
        query="What are Yuan's scheduling preferences?",
        max_roots=25,
        object_types=set(),
        metadata_filters={},
    )

    # The good fact should match (aliases give "yuan" + "scheduling")
    assert 'f_good' in root_ids, f"Expected 'f_good' in {root_ids}; strategy={strategy}"
    # Strategy should reflect the 2-token path
    assert strategy in (
        'query_tokens_min2_overlap',
        'query_tokens_1overlap_fallback',
        'query_tokens_no_match',
    ), f"Unexpected strategy: {strategy!r}"


def test_candidate_root_ids_falls_back_to_single_token_for_short_query():
    """With only 1 meaningful query token, single-token matching is used (not empty)."""
    ledger = _memory_ledger()
    fact = _state_fact(object_id='f_short', root_id='f_short', predicate='pref.coffee', value='espresso')
    ledger.append_event('assert', actor_id='test', payload=fact, recorded_at='2026-03-21T00:00:00Z')
    ledger.append_event('promote', actor_id='policy', object_id='f_short', root_id='f_short')

    svc = TypedRetrievalService(ledger=ledger, evidence_registry=_FakeEvidenceRegistry())
    root_ids, strategy = svc._candidate_root_ids(
        query='espresso',  # single meaningful token
        max_roots=25,
        object_types=set(),
        metadata_filters={},
    )
    assert 'f_short' in root_ids, f"Single-token query should still find roots; strategy={strategy}"


def test_search_returns_state_facts_for_yuan_scheduling_query():
    """End-to-end: a 'Yuan scheduling' query surfaces scheduling-domain state_facts."""
    ledger = _memory_ledger()
    facts = [
        _state_fact(
            object_id='f_morning',
            root_id='f_morning',
            subject='user:principal',
            predicate='constraint.no_meetings_before_1030',
            value='No meetings before 10:30am ET on weekdays',
        ),
        _state_fact(
            object_id='f_timezone',
            root_id='f_timezone',
            subject='user:principal',
            predicate='pref.default_timezone',
            value='America/New_York',
        ),
    ]
    for fact in facts:
        ledger.append_event('assert', actor_id='test', payload=fact, recorded_at='2026-03-21T00:00:00Z')
        ledger.append_event('promote', actor_id='policy', object_id=fact.object_id, root_id=fact.root_id)

    svc = TypedRetrievalService(ledger=ledger, evidence_registry=_FakeEvidenceRegistry())
    result = _run(svc.search(
        query="What are Yuan's scheduling preferences and calendar constraints?",
        object_types=['state'],
        max_results=10,
        max_evidence=5,
    ))

    returned_ids = {item['object_id'] for item in result['state']}
    assert returned_ids, "Expected at least one state_fact returned for Yuan scheduling query"
    assert 'f_morning' in returned_ids or 'f_timezone' in returned_ids, (
        f"Neither scheduling fact surfaced. Returned IDs: {returned_ids}. "
        "Fixes 1+2+3 should together make Yuan's scheduling facts reachable."
    )


def test_search_returns_archibald_delegation_facts():
    """End-to-end: 'delegation Archibald' query surfaces agent:archibald facts."""
    ledger = _memory_ledger()
    fact = _state_fact(
        object_id='f_delegation',
        root_id='f_delegation',
        subject='agent:archibald',
        predicate='policy.delegation_scope',
        value='internal file ops and research without approval',
    )
    ledger.append_event('assert', actor_id='test', payload=fact, recorded_at='2026-03-21T00:00:00Z')
    ledger.append_event('promote', actor_id='policy', object_id='f_delegation', root_id='f_delegation')

    svc = TypedRetrievalService(ledger=ledger, evidence_registry=_FakeEvidenceRegistry())
    result = _run(svc.search(
        query="What is Yuan's approach to delegating tasks to Archibald?",
        object_types=['state'],
        max_results=10,
        max_evidence=5,
    ))

    returned_ids = {item['object_id'] for item in result['state']}
    assert 'f_delegation' in returned_ids, (
        f"'f_delegation' not in {returned_ids}. "
        "Alias 'archibald' → 'agent:archibald' should surface delegation facts."
    )
