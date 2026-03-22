import asyncio
import json
import sqlite3
from types import SimpleNamespace

from mcp_server.src.models.typed_memory import Episode, EvidenceRef, StateFact
from mcp_server.src.services.change_ledger import ChangeLedger
from mcp_server.src.services.om_typed_projection import OMTypedProjectionService
from mcp_server.src.services.typed_retrieval import ScoredObject, TypedRetrievalService


class _FakeEvidenceRegistry:
    def __init__(self, payload=None):
        self.payload = payload or []
        self.calls = []

    async def resolve_many(self, refs, *, object_ids_by_uri=None, max_items=None):
        self.calls.append(
            {
                'refs': refs,
                'object_ids_by_uri': object_ids_by_uri,
                'max_items': max_items,
            }
        )
        return list(self.payload)


class _FakeLedger:
    conn = None


def _run(coro):
    return asyncio.run(coro)


def _state_fact(
    *,
    object_id: str,
    root_id: str,
    version: int,
    value: str,
    is_current: bool = True,
    source_lane: str = 's1_sessions_main',
    source_key: str = 'session:1',
    invalid_at: str | None = None,
    lifecycle_status: str = 'asserted',
    promotion_status: str = 'proposed',
):
    return StateFact(
        object_id=object_id,
        root_id=root_id,
        version=version,
        is_current=is_current,
        source_lane=source_lane,
        source_key=source_key,
        policy_scope='private',
        visibility_scope='private',
        evidence_refs=[
            EvidenceRef(
                kind='message',
                source_system='telegram',
                locator={'system': 'telegram', 'conversation_id': 'c1', 'message_id': object_id},
                title='source',
                snippet=value,
            )
        ],
        fact_type='preference',
        subject='yuan',
        predicate='likes',
        value=value,
        scope='private',
        invalid_at=invalid_at,
        lifecycle_status=lifecycle_status,
        promotion_status=promotion_status,
    )


def _memory_ledger() -> ChangeLedger:
    return ChangeLedger(sqlite3.connect(':memory:'))


def _episode(*, object_id: str, root_id: str, version: int, summary: str, is_current: bool = True, parent_id: str | None = None, superseded_by: str | None = None, invalid_at: str | None = None, source_key: str | None = None):
    return Episode(
        object_id=object_id,
        root_id=root_id,
        parent_id=parent_id,
        version=version,
        is_current=is_current,
        source_lane='s1_observational_memory',
        source_key=source_key or f'om:s1_observational_memory:node:{object_id}',
        policy_scope='private',
        visibility_scope='private',
        title=summary[:40],
        summary=summary,
        invalid_at=invalid_at,
        superseded_by=superseded_by,
        evidence_refs=[
            EvidenceRef(
                kind='event_log',
                source_system='om',
                locator={'system': 'om', 'stream': 's1_observational_memory:node', 'event_id': object_id},
                title='source',
                snippet=summary,
            )
        ],
    )


def _seed_assert(ledger: ChangeLedger, obj: StateFact, *, recorded_at: str = '2026-03-09T05:00:00Z') -> None:
    ledger.append_event(
        'assert',
        actor_id='tester',
        reason='seed',
        recorded_at=recorded_at,
        payload=obj.model_dump(mode='json'),
    )


def test_history_mode_keeps_best_root_score_for_multi_version_lineage():
    service = TypedRetrievalService(ledger=_FakeLedger())
    current = _state_fact(object_id='obj_current', root_id='root_a', version=2, value='espresso', is_current=True)
    previous = _state_fact(object_id='obj_previous', root_id='root_a', version=1, value='coffee', is_current=False)
    other = _state_fact(object_id='obj_other', root_id='root_b', version=1, value='tea', is_current=True)

    ranked = [
        ScoredObject(obj=current, score=9.0),
        ScoredObject(obj=other, score=7.0),
        ScoredObject(obj=previous, score=1.5),
    ]
    filtered = [current, previous, other]

    selected = service._select_objects(ranked, filtered, 'history', max_results=1)

    assert [item.obj.object_id for item in selected] == ['obj_current', 'obj_previous']
    assert {item.obj.object_id: item.score for item in selected} == {
        'obj_current': 9.0,
        'obj_previous': 9.0,
    }


def test_search_caps_typed_limits_and_reports_applied_limits():
    registry = _FakeEvidenceRegistry(payload=[{'canonical_uri': 'msg://telegram/c1/e1'}])
    service = TypedRetrievalService(ledger=_FakeLedger(), evidence_registry=registry)
    obj = _state_fact(object_id='obj_1', root_id='root_1', version=1, value='espresso')

    service._materialize_candidate_objects = lambda **kwargs: (
        [obj],
        {
            'candidate_roots': 1,
            'materialized_roots': 1,
            'snapshot_only_roots_over_event_cap': 0,
            'skipped_roots_over_event_cap': 0,
            'root_selection_strategy': 'test',
            'max_candidate_roots': 250,
            'max_lineage_events': 256,
        },
        {},
    )

    response = _run(
        service.search(
            query='espresso',
            object_types=['state'],
            metadata_filters={'source_lane': {'eq': 's1_sessions_main'}},
            max_results=999,
            max_evidence=999,
        )
    )

    assert response['retrieval_mode'] == 'typed'
    assert response['result_format'] == 'typed'  # deprecated compat alias still present
    assert response['counts'] == {'state': 1, 'episodes': 0, 'procedures': 0, 'evidence': 1}
    assert response['limits_applied']['max_results'] == {'requested': 999, 'effective': 200}
    assert response['limits_applied']['max_evidence'] == {'requested': 999, 'effective': 200}
    assert response['filters_applied'] == {
        'object_types': ['state_fact'],
        'metadata_filters': {'source_lane': {'eq': 's1_sessions_main'}},
    }
    assert registry.calls[0]['max_items'] == 200


def test_candidate_root_prefilter_uses_root_index_and_scope_filters():
    executed = {}

    class _Conn:
        def execute(self, sql, params):
            executed['sql'] = sql
            executed['params'] = list(params)
            return SimpleNamespace(fetchall=lambda: [])

    service = TypedRetrievalService(ledger=SimpleNamespace(conn=_Conn()))

    root_ids, strategy = service._candidate_root_ids(
        query='coffee preference',
        max_roots=25,
        object_types={'state_fact'},
        metadata_filters={'source_lane': {'in': ['lane_a', 'lane_b']}},
    )

    assert root_ids == []
    assert strategy == 'query_tokens_no_match'
    assert 'FROM typed_roots' in executed['sql']
    assert 'change_events' not in executed['sql']
    assert 'object_type IN (?)' in executed['sql']
    assert 'source_lane IN (?, ?)' in executed['sql']
    assert executed['params'][:3] == ['state_fact', 'lane_a', 'lane_b']


def test_candidate_root_prefilter_finds_provisional_om_episode_by_plain_group_id():
    """Regression: provisional ledger episodes written by om_compressor must be discoverable
    via _candidate_root_ids when the scoped source_lane filter uses the plain group id.

    Previously, om_compressor wrote source_lane='om:<group_id>' (e.g. 'om:s1_observational_memory')
    while scoped typed retrieval filtered by the plain group id ('s1_observational_memory').
    The filter mismatch caused provisional episodes to be silently dropped from scoped queries.

    After the fix, om_compressor writes source_lane=<group_id> (no prefix), so the filter matches.
    """
    group_id = 's1_observational_memory'
    ep = Episode(
        object_id='om_provisional_episode:s1_observational_memory:node_fix_test',
        root_id='om_provisional_root:s1_observational_memory:node_fix_test',
        version=1,
        is_current=True,
        source_lane=group_id,  # canonical: plain group id, NOT "om:s1_observational_memory"
        source_key=f'om:{group_id}:node:node_fix_test',
        policy_scope='observational',
        visibility_scope='owner',
        title='provisional episode for retrieval regression test',
        summary='Yuan prefers Ethiopian coffee observed during session replay',
        annotations=['om_native', 'provisional', 'unpromoted'],
        history_meta={
            'lineage_kind': 'om_provisional',
            'lineage_basis': 'write_time_ledger',
            'derivation_level': 'provisional',
            'om_node_id': 'node_fix_test',
            'om_group_id': group_id,
            'chunk_id': 'chunk_regression',
            'promotion_status': 'unpromoted',
        },
        evidence_refs=[
            EvidenceRef(
                kind='event_log',
                source_system='om',
                locator={'system': 'om', 'stream': f'{group_id}:node', 'event_id': 'node_fix_test'},
                title='OM node',
                snippet='Yuan prefers Ethiopian coffee observed during session replay',
            )
        ],
        created_at='2026-03-11T00:00:00Z',
        valid_at='2026-03-11T00:00:00Z',
    )

    ledger = _memory_ledger()
    ledger.append_event(
        'assert',
        actor_id='om_compressor',
        reason='provisional_write',
        recorded_at='2026-03-11T00:00:00Z',
        root_id=ep.root_id,
        payload=ep.model_dump(mode='json'),
    )

    service = TypedRetrievalService(ledger=ledger, evidence_registry=_FakeEvidenceRegistry())

    # Scoped filter using the plain group id — must find the provisional episode.
    root_ids, strategy = service._candidate_root_ids(
        query='Ethiopian coffee',
        max_roots=25,
        object_types=set(),
        metadata_filters={'source_lane': {'in': [group_id]}},
    )

    assert ep.root_id in root_ids, (
        f"Provisional episode root '{ep.root_id}' not found in candidate roots {root_ids}. "
        "om_compressor must write source_lane=<group_id> (not 'om:<group_id>') so scoped "
        "typed retrieval can discover provisional ledger episodes."
    )
    # Strategy reflects 2-token minimum overlap logic (min2 when >=2 tokens match,
    # fallback to single-token if strict filter is empty).
    assert strategy in ('query_tokens_min2_overlap', 'query_tokens_1overlap_fallback'), (
        f"Unexpected strategy: {strategy!r}"
    )


def test_episode_only_query_returns_shadow_episode_not_provisional_for_promoted_om_node():
    """Regression: episode-only typed queries must return the ledger-backed shadow
    episode for a promoted OM node, not the stale provisional ledger episode.

    Bug: _candidate_root_ids() emitted ``object_type IN ('episode')`` when the caller
    passed object_types=['episode'].  That SQL filter excluded the promoted state_fact
    root (object_type='state_fact') from the candidate pool.
    _derive_ledger_backed_om_history() then received no StateFact objects and returned
    covered_om_episode_nodes=set(), so the provisional episode was never suppressed and
    the shadow episode was never created.

    Fix: when 'episode' is in object_types, also include 'state_fact' in the SQL type
    filter so the promoted state root survives candidate selection.  state_fact objects
    are removed by _matches_object_type() in the filtered_objects pass and never reach
    the caller.
    """
    group_id = 's1_observational_memory'
    node_id = 'travel-style'

    # Promoted OM state fact — stored in typed_roots with object_type='state_fact'.
    # Before the fix this root was filtered out when object_types=['episode'].
    promoted_fact = _state_fact(
        object_id='om_state_promoted',
        root_id='om_state_root',
        version=1,
        value='carry-on only',
        source_lane=group_id,
        source_key=f'om:{group_id}:node:{node_id}',
        lifecycle_status='promoted',
        promotion_status='promoted',
    )
    # Provisional episode for the same OM node — stored with object_type='episode'.
    # This is the stale placeholder that must be suppressed after promotion.
    provisional_ep = _om_provisional_episode(
        object_id=f'om_provisional_episode:{group_id}:{node_id}',
        summary='carry-on only packing preference provisional',
        group_id=group_id,
    )
    # _om_provisional_episode derives source_key as
    # 'om:<group_id>:node:<last segment of object_id>', which resolves to
    # 'om:s1_observational_memory:node:travel-style' — same as promoted_fact.source_key.

    ledger = _memory_ledger()
    _seed_assert(ledger, promoted_fact)
    _seed_assert(ledger, provisional_ep)

    service = TypedRetrievalService(ledger=ledger, evidence_registry=_FakeEvidenceRegistry())

    response = _run(
        service.search(
            query='carry-on',
            object_types=['episode'],
            max_results=10,
            max_evidence=10,
        )
    )

    episode_ids = [ep['object_id'] for ep in response['episodes']]

    assert 'om_episode_shadow:om_state_promoted' in episode_ids, (
        f'Shadow episode not returned for promoted OM node. Got: {episode_ids}. '
        'Prefilter bug: state_fact root was excluded from candidate selection '
        "when object_types=['episode'], so _derive_ledger_backed_om_history() "
        'received no state facts and could not derive the shadow episode.'
    )
    assert f'om_provisional_episode:{group_id}:{node_id}' not in episode_ids, (
        f'Stale provisional episode not suppressed after promotion. Got: {episode_ids}'
    )
    # state_fact objects must not leak into the episodes bucket
    assert response['counts']['state'] == 0, (
        f"state_fact objects leaked into state bucket: {response['state']}"
    )


def test_search_does_not_return_recent_unrelated_objects_for_nonmatching_query():
    ledger = _memory_ledger()
    _seed_assert(ledger, _state_fact(object_id='obj_1', root_id='root_1', version=1, value='espresso'))
    service = TypedRetrievalService(ledger=ledger, evidence_registry=_FakeEvidenceRegistry())

    response = _run(service.search(query='dragonfruit', object_types=['state']))

    assert response['counts']['state'] == 0
    assert response['message'] == 'No relevant typed memory found'
    assert response['limits_applied']['materialization']['root_selection_strategy'] == 'query_tokens_no_match'


def test_search_weak_query_fails_closed_instead_of_falling_back_to_recent_roots():
    ledger = _memory_ledger()
    _seed_assert(ledger, _state_fact(object_id='obj_1', root_id='root_1', version=1, value='espresso'))
    service = TypedRetrievalService(ledger=ledger, evidence_registry=_FakeEvidenceRegistry())

    response = _run(service.search(query='zz', object_types=['state']))

    assert response['counts']['state'] == 0
    assert response['limits_applied']['materialization']['root_selection_strategy'] == 'query_too_weak'


def test_search_unicode_tokenless_query_can_hit_exact_match_root_prefilter():
    ledger = _memory_ledger()
    _seed_assert(ledger, _state_fact(object_id='obj_1', root_id='root_1', version=1, value='咖啡'))
    service = TypedRetrievalService(ledger=ledger, evidence_registry=_FakeEvidenceRegistry())

    response = _run(service.search(query='咖啡', object_types=['state']))

    assert response['counts']['state'] == 1
    assert response['state'][0]['object_id'] == 'obj_1'
    assert response['limits_applied']['materialization']['root_selection_strategy'] == 'query_text_exact'



def test_search_single_character_cjk_query_can_hit_exact_match_root_prefilter():
    ledger = _memory_ledger()
    _seed_assert(ledger, _state_fact(object_id='obj_1', root_id='root_1', version=1, value='咖啡'))
    service = TypedRetrievalService(ledger=ledger, evidence_registry=_FakeEvidenceRegistry())

    response = _run(service.search(query='咖', object_types=['state']))

    assert response['counts']['state'] == 1
    assert response['state'][0]['object_id'] == 'obj_1'
    assert response['limits_applied']['materialization']['root_selection_strategy'] == 'query_text_exact'



def test_search_single_character_ascii_query_still_fails_closed():
    ledger = _memory_ledger()
    _seed_assert(ledger, _state_fact(object_id='obj_1', root_id='root_1', version=1, value='coffee'))
    service = TypedRetrievalService(ledger=ledger, evidence_registry=_FakeEvidenceRegistry())

    response = _run(service.search(query='c', object_types=['state']))

    assert response['counts']['state'] == 0
    assert response['limits_applied']['materialization']['root_selection_strategy'] == 'query_too_weak'



def test_lineage_over_cap_uses_root_snapshot_instead_of_disappearing():
    ledger = _memory_ledger()
    obj = _state_fact(object_id='obj_1', root_id='root_1', version=42, value='espresso')
    payload_json = json.dumps(obj.model_dump(mode='json'), ensure_ascii=False, sort_keys=True)
    lineage_search_text = '\n'.join([payload_json.lower(), 'historical preference coffee'])
    ledger.conn.execute(
        """
        INSERT INTO typed_roots(
            root_id, latest_recorded_at, object_type, source_lane,
            current_object_id, current_version, current_payload_json,
            search_text, lineage_event_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            'root_1',
            '2026-03-09T05:00:00Z',
            'state_fact',
            's1_sessions_main',
            'obj_1',
            42,
            payload_json,
            lineage_search_text,
            999,
        ),
    )
    ledger.conn.commit()

    service = TypedRetrievalService(ledger=ledger, evidence_registry=_FakeEvidenceRegistry())
    service._candidate_root_ids = lambda **kwargs: (['root_1'], 'test_snapshot')
    service._events_for_root_limited = lambda **kwargs: (_ for _ in ()).throw(AssertionError('should not read lineage'))

    response = _run(service.search(query='coffee', object_types=['state']))

    assert response['counts']['state'] == 1
    assert response['state'][0]['object_id'] == 'obj_1'
    assert response['limits_applied']['materialization']['snapshot_only_roots_over_event_cap'] == 1
    assert response['limits_applied']['materialization']['skipped_roots_over_event_cap'] == 0


# ── OM typed projection merge tests ──────────────────────────────────────────


class _FakeOMProjectionService:
    """Minimal OM projection stub that returns canned typed objects."""

    def __init__(self, objects=None, search_overrides=None, limits=None):
        self._objects = objects or []
        self._search_overrides = search_overrides or {}
        self._limits = limits or {'enabled': True, 'reason': 'projected', 'groups_considered': [], 'episodes_projected': 0, 'state_projected': 0, 'max_results': 10}
        self.calls = []

    async def project(self, *, query, effective_group_ids, object_types, max_results, query_mode):
        self.calls.append({
            'query': query,
            'effective_group_ids': effective_group_ids,
            'object_types': object_types,
            'max_results': max_results,
            'query_mode': query_mode,
        })
        return self._objects, self._search_overrides, self._limits


def _om_provisional_episode(*, object_id: str, summary: str = 'om provisional observation', group_id: str = 's1_observational_memory'):
    """Create a provisional OM episode matching the Phase 2 projection contract."""
    return Episode(
        object_id=object_id,
        root_id=object_id,
        version=1,
        is_current=True,
        source_lane=group_id,
        source_key=f'om:{group_id}:node:{object_id.split(":")[-1]}',
        policy_scope='private',
        visibility_scope='private',
        title=summary[:40],
        summary=summary,
        annotations=['om_native', 'provisional', 'unpromoted'],
        history_meta={'lineage_kind': 'om_node', 'lineage_basis': 'provisional', 'derivation_level': 'provisional'},
        evidence_refs=[
            EvidenceRef(
                kind='event_log',
                source_system='om',
                locator={'system': 'om', 'stream': f'{group_id}:node', 'event_id': object_id.split(':')[-1]},
                title='OM node',
                snippet=summary,
            )
        ],
    )


def test_om_projection_merges_objects_into_typed_search():
    """OM projected provisional episodes surface through typed buckets."""
    om_obj = _om_provisional_episode(
        object_id='om_episode:s1_observational_memory:node-1',
        summary='resolves latency spike',
    )
    projection = _FakeOMProjectionService(
        objects=[om_obj],
        search_overrides={'om_episode:s1_observational_memory:node-1': 'resolves latency spike'},
        limits={
            'enabled': True,
            'reason': 'provisional_projection',
            'groups_considered': ['s1_observational_memory'],
            'episodes_projected': 1,
            'state_projected': 0,
            'max_results': 10,
        },
    )
    service = TypedRetrievalService(
        ledger=_FakeLedger(),
        evidence_registry=_FakeEvidenceRegistry(),
        om_projection_service=projection,
    )
    service._materialize_candidate_objects = lambda **kwargs: ([], {'candidate_roots': 0, 'materialized_roots': 0, 'snapshot_only_roots_over_event_cap': 0, 'skipped_roots_over_event_cap': 0, 'root_selection_strategy': 'empty', 'max_candidate_roots': 250, 'max_lineage_events': 256}, {})

    response = _run(
        service.search(
            query='latency spike',
            effective_group_ids=['s1_observational_memory'],
            max_results=10,
            max_evidence=10,
        )
    )

    assert response['counts']['episodes'] == 1
    assert response['episodes'][0]['object_id'] == 'om_episode:s1_observational_memory:node-1'
    assert response['episodes'][0]['source_lane'] == 's1_observational_memory'
    assert 'provisional' in response['episodes'][0]['annotations']
    assert response['limits_applied']['materialization']['om_projection']['enabled'] is True
    assert projection.calls[0]['effective_group_ids'] == ['s1_observational_memory']
    assert projection.calls[0]['object_types'] == {'episode'}


def test_om_projection_deduplicates_against_ledger_objects():
    """OM projected episodes with same ID as existing objects are not duplicated."""
    ledger_episode = _episode(
        object_id='om_episode:s1_observational_memory:shared',
        root_id='om_episode:s1_observational_memory:shared',
        version=1,
        summary='resolves latency spike',
    )
    om_obj = _om_provisional_episode(
        object_id='om_episode:s1_observational_memory:shared',
        summary='resolves latency spike',
    )

    projection = _FakeOMProjectionService(
        objects=[om_obj],
        search_overrides={'om_episode:s1_observational_memory:shared': 'resolves latency spike'},
    )
    service = TypedRetrievalService(
        ledger=_FakeLedger(),
        evidence_registry=_FakeEvidenceRegistry(),
        om_projection_service=projection,
    )
    service._materialize_candidate_objects = lambda **kwargs: ([ledger_episode], {'candidate_roots': 1, 'materialized_roots': 1, 'snapshot_only_roots_over_event_cap': 0, 'skipped_roots_over_event_cap': 0, 'root_selection_strategy': 'test', 'max_candidate_roots': 250, 'max_lineage_events': 256}, {'om_episode:s1_observational_memory:shared': 'resolves latency spike'})

    response = _run(
        service.search(
            query='resolves',
            effective_group_ids=['s1_observational_memory'],
            max_results=10,
            max_evidence=10,
        )
    )

    assert response['counts']['episodes'] == 1


def test_om_projection_skips_when_only_state_facts_requested():
    """Projection never produces state facts — skip entirely when only state is requested."""
    ledger_obj = _state_fact(
        object_id='ledger_state',
        root_id='ledger_root',
        version=1,
        value='carry-on only',
        source_lane='s1_observational_memory',
        source_key='om:s1_observational_memory:node:travel-style',
    )
    projection = _FakeOMProjectionService(
        objects=[_om_provisional_episode(
            object_id='om_episode:s1_observational_memory:node-shadow',
            summary='carry-on only',
        )],
        search_overrides={'om_episode:s1_observational_memory:node-shadow': 'carry-on only'},
    )
    service = TypedRetrievalService(
        ledger=_FakeLedger(),
        evidence_registry=_FakeEvidenceRegistry(),
        om_projection_service=projection,
    )
    service._materialize_candidate_objects = lambda **kwargs: (
        [ledger_obj],
        {
            'candidate_roots': 1,
            'materialized_roots': 1,
            'snapshot_only_roots_over_event_cap': 0,
            'skipped_roots_over_event_cap': 0,
            'root_selection_strategy': 'test',
            'max_candidate_roots': 250,
            'max_lineage_events': 256,
        },
        {'ledger_state': 'carry-on only'},
    )

    response = _run(
        service.search(
            query='carry-on',
            object_types=['state'],
            effective_group_ids=['s1_observational_memory'],
            max_results=10,
            max_evidence=10,
        )
    )

    # Projection is skipped entirely when only state_fact is requested
    assert projection.calls == []
    assert response['counts']['state'] == 1
    assert response['state'][0]['object_id'] == 'ledger_state'
    assert response['limits_applied']['materialization']['om_projection']['reason'] == 'episodes_not_requested'



def test_om_projection_always_projects_only_episodes():
    """Projection always produces only episodes (state is ledger-canonical)."""
    ledger_obj = _state_fact(
        object_id='ledger_state',
        root_id='ledger_root',
        version=1,
        value='carry-on only',
        source_lane='s1_observational_memory',
        source_key='om:s1_observational_memory:node:travel-style',
    )
    om_episode = _om_provisional_episode(
        object_id='om_episode:s1_observational_memory:travel-style-v1',
        summary='travel style before the ledger promotion',
    )
    projection = _FakeOMProjectionService(
        objects=[om_episode],
        search_overrides={
            'om_episode:s1_observational_memory:travel-style-v1': 'travel style before the ledger promotion',
        },
        limits={
            'enabled': True,
            'reason': 'provisional_projection',
            'groups_considered': ['s1_observational_memory'],
            'episodes_projected': 1,
            'state_projected': 0,
            'max_results': 10,
        },
    )
    service = TypedRetrievalService(
        ledger=_FakeLedger(),
        evidence_registry=_FakeEvidenceRegistry(),
        om_projection_service=projection,
    )
    service._materialize_candidate_objects = lambda **kwargs: (
        [ledger_obj],
        {
            'candidate_roots': 1,
            'materialized_roots': 1,
            'snapshot_only_roots_over_event_cap': 0,
            'skipped_roots_over_event_cap': 0,
            'root_selection_strategy': 'test',
            'max_candidate_roots': 250,
            'max_lineage_events': 256,
        },
        {'ledger_state': 'carry-on only'},
    )

    response = _run(
        service.search(
            query='carry-on travel style',
            effective_group_ids=['s1_observational_memory'],
            max_results=10,
            max_evidence=10,
        )
    )

    assert len(projection.calls) == 1
    assert projection.calls[0]['object_types'] == {'episode'}
    assert response['counts']['state'] == 1
    assert response['counts']['episodes'] == 1
    assert [item['object_id'] for item in response['state']] == ['ledger_state']
    assert [item['object_id'] for item in response['episodes']] == [
        'om_episode:s1_observational_memory:travel-style-v1',
    ]


def test_ledger_backed_om_history_derives_episode_lineage_for_promoted_om_state():
    previous = _state_fact(
        object_id='ledger_state_v1',
        root_id='ledger_root',
        version=1,
        value='checked bag only',
        is_current=False,
        source_lane='s1_observational_memory',
        source_key='om:s1_observational_memory:node:travel-style-v1',
        invalid_at='2026-03-09T00:00:00Z',
        lifecycle_status='superseded',
        promotion_status='promoted',
    )
    current = _state_fact(
        object_id='ledger_state_v2',
        root_id='ledger_root',
        version=2,
        value='carry-on only',
        source_lane='s1_observational_memory',
        source_key='om:s1_observational_memory:node:travel-style-v2',
        lifecycle_status='promoted',
        promotion_status='promoted',
    )
    projection = _FakeOMProjectionService(
        objects=[
            _episode(
                object_id='om_episode:s1_observational_memory:travel-style-v1',
                root_id='om_episode:s1_observational_memory:travel-style-v1',
                version=1,
                summary='checked bag only',
                is_current=False,
                invalid_at='2026-03-09T00:00:00Z',
                superseded_by='om_episode:s1_observational_memory:travel-style-v2',
                source_key='om:s1_observational_memory:node:travel-style-v1',
            ),
            _episode(
                object_id='om_episode:s1_observational_memory:travel-style-v2',
                root_id='om_episode:s1_observational_memory:travel-style-v1',
                version=2,
                summary='carry-on only',
                parent_id='om_episode:s1_observational_memory:travel-style-v1',
                source_key='om:s1_observational_memory:node:travel-style-v2',
            ),
        ],
        search_overrides={
            'om_episode:s1_observational_memory:travel-style-v1': 'checked bag only before change',
            'om_episode:s1_observational_memory:travel-style-v2': 'carry-on only now',
        },
        limits={
            'enabled': True,
            'reason': 'projected_history',
            'groups_considered': ['s1_observational_memory'],
            'episodes_projected': 2,
            'state_projected': 0,
            'max_results': 10,
            'history_mode': True,
            'history_candidates': 1,
            'history_lineages_projected': 1,
            'history_state_projection_supported': False,
            'unsupported_object_types': [],
            'skipped_candidates': [],
        },
    )
    service = TypedRetrievalService(
        ledger=_FakeLedger(),
        evidence_registry=_FakeEvidenceRegistry(),
        om_projection_service=projection,
    )
    service._materialize_candidate_objects = lambda **kwargs: (
        [previous, current],
        {
            'candidate_roots': 1,
            'materialized_roots': 1,
            'snapshot_only_roots_over_event_cap': 0,
            'skipped_roots_over_event_cap': 0,
            'root_selection_strategy': 'test',
            'max_candidate_roots': 250,
            'max_lineage_events': 256,
        },
        {
            'ledger_state_v1': 'checked bag only before change',
            'ledger_state_v2': 'carry-on only now',
        },
    )

    response = _run(
        service.search(
            query='what changed with travel style carry-on',
            effective_group_ids=['s1_observational_memory'],
            object_types=['episode'],
            history_mode='history',
            max_results=10,
            max_evidence=10,
        )
    )

    assert len(projection.calls) == 1
    assert projection.calls[0]['object_types'] == {'episode'}
    assert [item['object_id'] for item in response['episodes']] == [
        'om_episode_shadow:ledger_state_v2',
        'om_episode_shadow:ledger_state_v1',
    ]
    assert response['episodes'][0]['root_id'] == 'ledger_root'
    assert response['episodes'][0]['parent_id'] == 'om_episode_shadow:ledger_state_v1'
    assert response['episodes'][1]['superseded_by'] == 'om_episode_shadow:ledger_state_v2'
    assert response['limits_applied']['materialization']['ledger_backed_om_history'] == {
        'enabled': True,
        'reason': 'ledger_promoted_om_state_history',
        'episodes_derived': 2,
        'roots_covered': 1,
        'source_nodes_covered': 2,
    }
    assert response['limits_applied']['materialization']['om_projection']['suppressed_projected_episodes'] == 2
    assert response['limits_applied']['materialization']['om_projection']['suppressed_projected_episode_nodes'] == 2


def test_ledger_backed_om_history_keeps_uncovered_projected_episodes():
    ledger_obj = _state_fact(
        object_id='ledger_state',
        root_id='ledger_root',
        version=1,
        value='carry-on only',
        source_lane='s1_observational_memory',
        source_key='om:s1_observational_memory:node:travel-style',
        lifecycle_status='promoted',
        promotion_status='promoted',
    )
    covered_episode = _episode(
        object_id='om_episode:s1_observational_memory:travel-style',
        root_id='om_episode:s1_observational_memory:travel-style',
        version=1,
        summary='travel style before promotion',
        source_key='om:s1_observational_memory:node:travel-style',
    )
    uncovered_episode = _episode(
        object_id='om_episode:s1_observational_memory:packing-habit',
        root_id='om_episode:s1_observational_memory:packing-habit',
        version=1,
        summary='packs a charger in every bag',
        source_key='om:s1_observational_memory:node:packing-habit',
    )
    projection = _FakeOMProjectionService(
        objects=[covered_episode, uncovered_episode],
        search_overrides={
            'om_episode:s1_observational_memory:travel-style': 'carry-on only travel style',
            'om_episode:s1_observational_memory:packing-habit': 'carry-on only and always packs charger',
        },
        limits={
            'enabled': True,
            'reason': 'projected',
            'groups_considered': ['s1_observational_memory'],
            'episodes_projected': 2,
            'state_projected': 0,
            'max_results': 10,
        },
    )
    service = TypedRetrievalService(
        ledger=_FakeLedger(),
        evidence_registry=_FakeEvidenceRegistry(),
        om_projection_service=projection,
    )
    service._materialize_candidate_objects = lambda **kwargs: (
        [ledger_obj],
        {
            'candidate_roots': 1,
            'materialized_roots': 1,
            'snapshot_only_roots_over_event_cap': 0,
            'skipped_roots_over_event_cap': 0,
            'root_selection_strategy': 'test',
            'max_candidate_roots': 250,
            'max_lineage_events': 256,
        },
        {'ledger_state': 'carry-on only'},
    )

    response = _run(
        service.search(
            query='carry-on',
            effective_group_ids=['s1_observational_memory'],
            object_types=['episode'],
            max_results=10,
            max_evidence=10,
        )
    )

    assert [item['object_id'] for item in response['episodes']] == [
        'om_episode_shadow:ledger_state',
        'om_episode:s1_observational_memory:packing-habit',
    ]
    assert response['limits_applied']['materialization']['om_projection']['suppressed_projected_episodes'] == 1


def test_om_projection_auto_history_query_merges_lineage_objects():
    """Auto-history queries should merge explicit OM lineage into typed buckets."""
    previous = _episode(
        object_id='om_episode:s1_observational_memory:node-v1',
        root_id='om_episode:s1_observational_memory:node-v1',
        version=1,
        summary='latency issue before the fix',
        is_current=False,
        superseded_by='om_episode:s1_observational_memory:node-v2',
        invalid_at='2026-03-09T00:00:00Z',
    )
    current = _episode(
        object_id='om_episode:s1_observational_memory:node-v2',
        root_id='om_episode:s1_observational_memory:node-v1',
        parent_id='om_episode:s1_observational_memory:node-v1',
        version=2,
        summary='latency issue after the fix',
        is_current=True,
    )
    projection = _FakeOMProjectionService(
        objects=[previous, current],
        search_overrides={'om_episode:s1_observational_memory:node-v2': 'what changed for latency spike'},
        limits={
            'enabled': True,
            'reason': 'projected_history',
            'groups_considered': ['s1_observational_memory'],
            'episodes_projected': 2,
            'state_projected': 0,
            'max_results': 10,
            'history_mode': True,
            'history_candidates': 1,
            'history_lineages_projected': 1,
            'history_state_projection_supported': False,
            'unsupported_object_types': [],
            'skipped_candidates': [],
        },
    )
    service = TypedRetrievalService(
        ledger=_FakeLedger(),
        evidence_registry=_FakeEvidenceRegistry(),
        om_projection_service=projection,
    )
    service._materialize_candidate_objects = lambda **kwargs: ([], {'candidate_roots': 0, 'materialized_roots': 0, 'snapshot_only_roots_over_event_cap': 0, 'skipped_roots_over_event_cap': 0, 'root_selection_strategy': 'empty', 'max_candidate_roots': 250, 'max_lineage_events': 256}, {})

    response = _run(
        service.search(
            query='what changed with latency spike',
            effective_group_ids=['s1_observational_memory'],
            max_results=10,
            max_evidence=10,
        )
    )

    assert response['query_mode'] == 'history'
    assert response['counts']['episodes'] == 2
    assert [item['object_id'] for item in response['episodes']] == [
        'om_episode:s1_observational_memory:node-v2',
        'om_episode:s1_observational_memory:node-v1',
    ]
    assert response['limits_applied']['materialization']['om_projection']['history_lineages_projected'] == 1
    assert projection.calls[0]['query_mode'] == 'history'


def test_om_projection_history_fail_closed_preserves_limits_metadata():
    """History-mode OM projection should surface fail-closed metadata instead of fabricating lineage."""
    projection = _FakeOMProjectionService(
        objects=[],
        limits={
            'enabled': True,
            'reason': 'projected_history',
            'groups_considered': ['s1_observational_memory'],
            'episodes_projected': 0,
            'state_projected': 0,
            'max_results': 10,
            'history_mode': True,
            'history_candidates': 1,
            'history_lineages_projected': 0,
            'history_state_projection_supported': False,
            'unsupported_object_types': [],
            'skipped_candidates': [
                {
                    'group_id': 's1_observational_memory',
                    'node_id': 'node-branch',
                    'reason': 'ambiguous_supersession_graph',
                }
            ],
        },
    )
    service = TypedRetrievalService(
        ledger=_FakeLedger(),
        evidence_registry=_FakeEvidenceRegistry(),
        om_projection_service=projection,
    )
    service._materialize_candidate_objects = lambda **kwargs: ([], {'candidate_roots': 0, 'materialized_roots': 0, 'snapshot_only_roots_over_event_cap': 0, 'skipped_roots_over_event_cap': 0, 'root_selection_strategy': 'empty', 'max_candidate_roots': 250, 'max_lineage_events': 256}, {})

    response = _run(
        service.search(
            query='what changed with latency spike',
            effective_group_ids=['s1_observational_memory'],
            max_results=10,
            max_evidence=10,
        )
    )

    assert response['query_mode'] == 'history'
    assert response['counts']['episodes'] == 0
    assert response['limits_applied']['materialization']['om_projection']['skipped_candidates'] == [
        {
            'group_id': 's1_observational_memory',
            'node_id': 'node-branch',
            'reason': 'ambiguous_supersession_graph',
        }
    ]
    assert projection.calls[0]['query_mode'] == 'history'


def test_om_projection_evidence_includes_om_provenance():
    """OM projected provisional episodes carry evidence refs with om source_system."""
    om_obj = _om_provisional_episode(
        object_id='om_episode:s1_observational_memory:node-prov',
        summary='resolves latency spike',
    )
    projection = _FakeOMProjectionService(
        objects=[om_obj],
        search_overrides={'om_episode:s1_observational_memory:node-prov': 'resolves latency spike'},
    )
    service = TypedRetrievalService(
        ledger=_FakeLedger(),
        evidence_registry=_FakeEvidenceRegistry(
            payload=[{
                'canonical_uri': 'eventlog://om/s1_observational_memory:node/node-prov',
                'kind': 'event_log',
                'source_system': 'om',
                'resolver': 'passthrough',
                'status': 'resolved',
            }]
        ),
        om_projection_service=projection,
    )
    service._materialize_candidate_objects = lambda **kwargs: ([], {'candidate_roots': 0, 'materialized_roots': 0, 'snapshot_only_roots_over_event_cap': 0, 'skipped_roots_over_event_cap': 0, 'root_selection_strategy': 'empty', 'max_candidate_roots': 250, 'max_lineage_events': 256}, {})

    response = _run(
        service.search(
            query='resolves latency',
            effective_group_ids=['s1_observational_memory'],
            max_results=10,
            max_evidence=10,
        )
    )

    assert response['counts']['evidence'] >= 1
    assert any(e.get('source_system') == 'om' for e in response['evidence'])


def test_om_projection_disabled_when_no_service():
    """Without OM projection service, om_projection limits show disabled."""
    service = TypedRetrievalService(
        ledger=_FakeLedger(),
        evidence_registry=_FakeEvidenceRegistry(),
    )
    service._materialize_candidate_objects = lambda **kwargs: ([], {'candidate_roots': 0, 'materialized_roots': 0, 'snapshot_only_roots_over_event_cap': 0, 'skipped_roots_over_event_cap': 0, 'root_selection_strategy': 'empty', 'max_candidate_roots': 250, 'max_lineage_events': 256}, {})

    response = _run(
        service.search(
            query='test',
            max_results=10,
            max_evidence=10,
        )
    )

    assert response['limits_applied']['materialization']['om_projection']['enabled'] is False
    assert response['limits_applied']['materialization']['om_projection']['reason'] == 'no_projection_service'


def test_om_projection_experimental_group_surfaces_through_typed_buckets():
    """Experimental OM-native groups surface provisional episodes when explicitly scoped."""
    om_obj = _om_provisional_episode(
        object_id='om_episode:ontbk15batch_20260310_om_f:node-exp',
        summary='experimental bakeoff observation',
        group_id='ontbk15batch_20260310_om_f',
    )
    projection = _FakeOMProjectionService(
        objects=[om_obj],
        search_overrides={'om_episode:ontbk15batch_20260310_om_f:node-exp': 'experimental bakeoff observation'},
        limits={
            'enabled': True,
            'reason': 'provisional_projection',
            'groups_considered': ['ontbk15batch_20260310_om_f'],
            'episodes_projected': 1,
            'state_projected': 0,
            'max_results': 10,
        },
    )
    service = TypedRetrievalService(
        ledger=_FakeLedger(),
        evidence_registry=_FakeEvidenceRegistry(),
        om_projection_service=projection,
    )
    service._materialize_candidate_objects = lambda **kwargs: ([], {'candidate_roots': 0, 'materialized_roots': 0, 'snapshot_only_roots_over_event_cap': 0, 'skipped_roots_over_event_cap': 0, 'root_selection_strategy': 'empty', 'max_candidate_roots': 250, 'max_lineage_events': 256}, {})

    response = _run(
        service.search(
            query='experimental bakeoff',
            effective_group_ids=['ontbk15batch_20260310_om_f'],
            max_results=10,
            max_evidence=10,
        )
    )

    assert response['counts']['episodes'] == 1
    assert response['episodes'][0]['source_lane'] == 'ontbk15batch_20260310_om_f'
    assert 'provisional' in response['episodes'][0]['annotations']


class _HistoryAwareSearchService:
    def __init__(self, *, node_rows=None, fact_rows=None, neighborhood_rows=None):
        self.node_rows = node_rows or []
        self.fact_rows = fact_rows or []
        self.neighborhood_rows = neighborhood_rows or {}

    def includes_observational_memory(self, group_ids):
        return True

    def _om_groups_in_scope(self, group_ids):
        return list(group_ids or ['s1_observational_memory'])

    async def search_observational_nodes(self, **_kwargs):
        return list(self.node_rows)

    async def search_observational_facts(self, *, center_node_uuid=None, **_kwargs):
        if center_node_uuid is not None:
            return list(self.neighborhood_rows.get(center_node_uuid, []))
        return list(self.fact_rows)


class _HistoryAwareDriver:
    def __init__(self, records_by_seed=None):
        self.records_by_seed = records_by_seed or {}

    async def execute_query(self, _query, **params):
        tuple_key = (params.get('group_id'), params.get('seed_node_id'))
        rows = self.records_by_seed.get(tuple_key)
        if rows is None:
            rows = self.records_by_seed.get(params.get('seed_node_id'), [])
        return list(rows), None, None


class _HistoryAwareGraphitiService:
    class config:
        class database:
            provider = 'neo4j'

    def __init__(self, driver):
        self.driver = driver

    async def get_client(self):
        return SimpleNamespace(driver=self.driver)


def test_real_om_projection_surfaces_provisional_episodes_across_modes():
    """Real OMTypedProjectionService produces provisional episodes that work with all query modes."""
    search_service = _HistoryAwareSearchService(
        node_rows=[
            {
                'uuid': 'drink_v2',
                'name': 'Current drink',
                'summary': 'espresso after training',
                'group_id': 's1_observational_memory',
                'created_at': '2026-03-09T00:00:00Z',
                'attributes': {'status': 'active', 'semantic_domain': 'observational_memory'},
            }
        ]
    )
    projection = OMTypedProjectionService(
        search_service=search_service,
        graphiti_service=_HistoryAwareGraphitiService(_HistoryAwareDriver()),
    )
    service = TypedRetrievalService(
        ledger=_FakeLedger(),
        evidence_registry=_FakeEvidenceRegistry(),
        om_projection_service=projection,
    )
    service._materialize_candidate_objects = lambda **kwargs: ([], {'candidate_roots': 0, 'materialized_roots': 0, 'snapshot_only_roots_over_event_cap': 0, 'skipped_roots_over_event_cap': 0, 'root_selection_strategy': 'empty', 'max_candidate_roots': 250, 'max_lineage_events': 256}, {})

    all_response = _run(
        service.search(
            query='espresso',
            effective_group_ids=['s1_observational_memory'],
            object_types=['episode'],
            history_mode='all',
            max_results=10,
            max_evidence=10,
        )
    )
    current_response = _run(
        service.search(
            query='espresso',
            effective_group_ids=['s1_observational_memory'],
            object_types=['episode'],
            history_mode='current',
            max_results=10,
            max_evidence=10,
        )
    )

    # Provisional episodes are flat — each is standalone is_current=True
    assert all_response['query_mode'] == 'all'
    assert all_response['counts']['episodes'] == 1
    assert all_response['episodes'][0]['object_id'] == 'om_episode:s1_observational_memory:drink_v2'
    assert 'provisional' in all_response['episodes'][0]['annotations']

    assert current_response['query_mode'] == 'current'
    assert current_response['counts']['episodes'] == 1

    assert all_response['limits_applied']['materialization']['om_projection']['reason'] == 'provisional_projection'


def test_real_om_projection_preserves_same_node_uuid_across_multiple_groups():
    """Same node UUID in different groups produces distinct provisional episodes."""
    search_service = _HistoryAwareSearchService(
        node_rows=[
            {
                'uuid': 'shared_node',
                'name': 'Shared node group A',
                'summary': 'espresso after training in group A',
                'group_id': 'group_a',
                'created_at': '2026-03-09T00:00:00Z',
                'attributes': {'status': 'active', 'semantic_domain': 'observational_memory'},
            },
            {
                'uuid': 'shared_node',
                'name': 'Shared node group B',
                'summary': 'espresso after training in group B',
                'group_id': 'group_b',
                'created_at': '2026-03-09T00:00:00Z',
                'attributes': {'status': 'active', 'semantic_domain': 'observational_memory'},
            },
        ]
    )
    projection = OMTypedProjectionService(
        search_service=search_service,
        graphiti_service=_HistoryAwareGraphitiService(_HistoryAwareDriver()),
    )
    service = TypedRetrievalService(
        ledger=_FakeLedger(),
        evidence_registry=_FakeEvidenceRegistry(),
        om_projection_service=projection,
    )
    service._materialize_candidate_objects = lambda **kwargs: ([], {'candidate_roots': 0, 'materialized_roots': 0, 'snapshot_only_roots_over_event_cap': 0, 'skipped_roots_over_event_cap': 0, 'root_selection_strategy': 'empty', 'max_candidate_roots': 250, 'max_lineage_events': 256}, {})

    response = _run(
        service.search(
            query='espresso',
            effective_group_ids=['group_a', 'group_b'],
            object_types=['episode'],
            history_mode='all',
            max_results=10,
            max_evidence=10,
        )
    )

    assert response['counts']['episodes'] == 2
    assert {(item['source_lane'], item['object_id']) for item in response['episodes']} == {
        ('group_a', 'om_episode:group_a:shared_node'),
        ('group_b', 'om_episode:group_b:shared_node'),
    }


# ---------------------------------------------------------------------------
# Contract: retrieval_mode field in response payloads (P2 fix)
# ---------------------------------------------------------------------------

def test_search_response_includes_retrieval_mode_field_on_empty_results():
    """Zero-results response must include retrieval_mode='typed' (canonical) alongside result_format."""
    ledger = _memory_ledger()
    service = TypedRetrievalService(ledger=ledger, evidence_registry=_FakeEvidenceRegistry())

    response = _run(service.search(query='xyzzy_no_match', object_types=['state']))

    assert response['message'] == 'No relevant typed memory found'
    assert response['retrieval_mode'] == 'typed', (
        "retrieval_mode must be 'typed' in zero-results payload (canonical contract field)"
    )
    assert response['result_format'] == 'typed', (
        "result_format must still be present for backward-compat callers"
    )
