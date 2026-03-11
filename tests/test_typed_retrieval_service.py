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
    )


def _memory_ledger() -> ChangeLedger:
    return ChangeLedger(sqlite3.connect(':memory:'))


def _episode(*, object_id: str, root_id: str, version: int, summary: str, is_current: bool = True, parent_id: str | None = None, superseded_by: str | None = None, invalid_at: str | None = None):
    return Episode(
        object_id=object_id,
        root_id=root_id,
        parent_id=parent_id,
        version=version,
        is_current=is_current,
        source_lane='s1_observational_memory',
        source_key=f'om:s1_observational_memory:node:{object_id}',
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

    assert response['result_format'] == 'typed'
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


def _om_state_fact(*, object_id: str, group_id: str = 's1_observational_memory'):
    return StateFact(
        object_id=object_id,
        root_id=object_id,
        version=1,
        is_current=True,
        source_lane=group_id,
        source_key=f'om:{group_id}:relation:{object_id}',
        policy_scope='private',
        visibility_scope='private',
        evidence_refs=[
            EvidenceRef(
                kind='event_log',
                source_system='om',
                locator={'system': 'om', 'stream': f'{group_id}:relation', 'event_id': object_id},
                title='OM relation',
                snippet='om projected fact',
            )
        ],
        fact_type='relationship',
        subject='om_node:source',
        predicate='om_relation:resolves',
        value='om projected',
        scope='private',
    )


def test_om_projection_merges_objects_into_typed_search():
    """OM projected objects surface through typed buckets when projection service is wired."""
    om_obj = _om_state_fact(object_id='om_state:s1_observational_memory:rel-1')
    projection = _FakeOMProjectionService(
        objects=[om_obj],
        search_overrides={'om_state:s1_observational_memory:rel-1': 'resolves latency spike'},
        limits={
            'enabled': True,
            'reason': 'projected',
            'groups_considered': ['s1_observational_memory'],
            'episodes_projected': 0,
            'state_projected': 1,
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

    assert response['counts']['state'] == 1
    assert response['state'][0]['object_id'] == 'om_state:s1_observational_memory:rel-1'
    assert response['state'][0]['source_lane'] == 's1_observational_memory'
    assert response['limits_applied']['materialization']['om_projection']['enabled'] is True
    assert projection.calls[0]['effective_group_ids'] == ['s1_observational_memory']


def test_om_projection_deduplicates_against_ledger_objects():
    """OM projected objects with same ID as ledger objects are not duplicated."""
    ledger_obj = _state_fact(object_id='shared_id', root_id='root_shared', version=1, value='resolves')
    om_obj = _om_state_fact(object_id='shared_id')

    projection = _FakeOMProjectionService(
        objects=[om_obj],
        search_overrides={'shared_id': 'resolves latency spike'},
    )
    service = TypedRetrievalService(
        ledger=_FakeLedger(),
        evidence_registry=_FakeEvidenceRegistry(),
        om_projection_service=projection,
    )
    service._materialize_candidate_objects = lambda **kwargs: ([ledger_obj], {'candidate_roots': 1, 'materialized_roots': 1, 'snapshot_only_roots_over_event_cap': 0, 'skipped_roots_over_event_cap': 0, 'root_selection_strategy': 'test', 'max_candidate_roots': 250, 'max_lineage_events': 256}, {'shared_id': 'resolves latency spike'})

    response = _run(
        service.search(
            query='resolves',
            effective_group_ids=['s1_observational_memory'],
            max_results=10,
            max_evidence=10,
        )
    )

    assert response['counts']['state'] == 1


def test_om_projection_skips_state_projection_when_ledger_already_has_om_state():
    """Ledger-backed OM facts should be the canonical state path; projection becomes fallback-only."""
    ledger_obj = _state_fact(
        object_id='ledger_state',
        root_id='ledger_root',
        version=1,
        value='carry-on only',
        source_lane='s1_observational_memory',
        source_key='om:s1_observational_memory:node:travel-style',
    )
    projection = _FakeOMProjectionService(
        objects=[_om_state_fact(object_id='om_state:s1_observational_memory:rel-shadow')],
        search_overrides={'om_state:s1_observational_memory:rel-shadow': 'carry-on only'},
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

    assert projection.calls == []
    assert response['counts']['state'] == 1
    assert response['state'][0]['object_id'] == 'ledger_state'
    assert response['limits_applied']['materialization']['om_projection']['reason'] == 'ledger_canonical_om_state'
    assert response['limits_applied']['materialization']['om_projection']['suppressed_object_types'] == ['state_fact']



def test_om_projection_only_projects_episodes_when_ledger_already_has_om_state():
    """When OM state is ledger-backed, read-time projection should only fill the episode/history gap."""
    ledger_obj = _state_fact(
        object_id='ledger_state',
        root_id='ledger_root',
        version=1,
        value='carry-on only',
        source_lane='s1_observational_memory',
        source_key='om:s1_observational_memory:node:travel-style',
    )
    om_episode = _episode(
        object_id='om_episode:s1_observational_memory:travel-style-v1',
        root_id='om_episode:s1_observational_memory:travel-style-v1',
        version=1,
        summary='travel style before the ledger promotion',
    )
    class _FilteringProjection(_FakeOMProjectionService):
        async def project(self, *, query, effective_group_ids, object_types, max_results, query_mode):
            self.calls.append({
                'query': query,
                'effective_group_ids': effective_group_ids,
                'object_types': object_types,
                'max_results': max_results,
                'query_mode': query_mode,
            })
            projected = list(self._objects)
            if object_types:
                projected = [obj for obj in projected if obj.object_type in object_types]
            limits = dict(self._limits)
            limits['episodes_projected'] = sum(1 for obj in projected if obj.object_type == 'episode')
            limits['state_projected'] = sum(1 for obj in projected if obj.object_type == 'state_fact')
            return projected, self._search_overrides, limits

    projection = _FilteringProjection(
        objects=[_om_state_fact(object_id='om_state:s1_observational_memory:rel-shadow'), om_episode],
        search_overrides={
            'om_state:s1_observational_memory:rel-shadow': 'carry-on only',
            'om_episode:s1_observational_memory:travel-style-v1': 'travel style before the ledger promotion',
        },
        limits={
            'enabled': True,
            'reason': 'projected',
            'groups_considered': ['s1_observational_memory'],
            'episodes_projected': 1,
            'state_projected': 1,
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
    assert response['limits_applied']['materialization']['om_projection']['suppressed_object_types'] == ['state_fact']


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
    """OM projected objects carry evidence refs with om source_system."""
    om_obj = _om_state_fact(object_id='om_state:s1_observational_memory:rel-prov')
    projection = _FakeOMProjectionService(
        objects=[om_obj],
        search_overrides={'om_state:s1_observational_memory:rel-prov': 'resolves latency spike'},
    )
    service = TypedRetrievalService(
        ledger=_FakeLedger(),
        evidence_registry=_FakeEvidenceRegistry(
            payload=[{
                'canonical_uri': 'eventlog://om/s1_observational_memory:relation/rel-prov',
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
    """Experimental OM-native groups surface when explicitly scoped."""
    om_obj = _om_state_fact(
        object_id='om_state:ontbk15batch_20260310_om_f:rel-exp',
        group_id='ontbk15batch_20260310_om_f',
    )
    projection = _FakeOMProjectionService(
        objects=[om_obj],
        search_overrides={'om_state:ontbk15batch_20260310_om_f:rel-exp': 'experimental bakeoff resolves'},
        limits={
            'enabled': True,
            'reason': 'projected',
            'groups_considered': ['ontbk15batch_20260310_om_f'],
            'episodes_projected': 0,
            'state_projected': 1,
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

    assert response['counts']['state'] == 1
    assert response['state'][0]['source_lane'] == 'ontbk15batch_20260310_om_f'


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


def test_real_om_projection_distinguishes_history_current_and_all_modes():
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
    driver = _HistoryAwareDriver(
        records_by_seed={
            'drink_v2': [
                {
                    'node_id': 'drink_v1',
                    'uuid': 'drink_v1',
                    'group_id': 's1_observational_memory',
                    'content': 'coffee before training',
                    'created_at': '2026-03-01T00:00:00Z',
                    'status': 'open',
                    'semantic_domain': 'observational_memory',
                    'supersedes': [],
                },
                {
                    'node_id': 'drink_v2',
                    'uuid': 'drink_v2',
                    'group_id': 's1_observational_memory',
                    'content': 'espresso after training',
                    'created_at': '2026-03-09T00:00:00Z',
                    'status': 'active',
                    'semantic_domain': 'observational_memory',
                    'supersedes': [
                        {
                            'target_id': 'drink_v1',
                            'created_at': '2026-03-09T00:00:00Z',
                            'relation_uuid': 'rel_drink_v2_v1',
                        }
                    ],
                },
            ]
        }
    )
    projection = OMTypedProjectionService(
        search_service=search_service,
        graphiti_service=_HistoryAwareGraphitiService(driver),
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
    history_response = _run(
        service.search(
            query='espresso',
            effective_group_ids=['s1_observational_memory'],
            object_types=['episode'],
            history_mode='history',
            max_results=10,
            max_evidence=10,
        )
    )

    assert all_response['query_mode'] == 'all'
    assert [item['object_id'] for item in all_response['episodes']] == [
        'om_episode:s1_observational_memory:drink_v2',
        'om_episode:s1_observational_memory:drink_v1',
    ]
    assert all_response['episodes'][1]['is_current'] is False

    assert current_response['query_mode'] == 'current'
    assert [item['object_id'] for item in current_response['episodes']] == [
        'om_episode:s1_observational_memory:drink_v2',
    ]

    assert history_response['query_mode'] == 'history'
    assert [item['object_id'] for item in history_response['episodes']] == [
        'om_episode:s1_observational_memory:drink_v2',
        'om_episode:s1_observational_memory:drink_v1',
    ]
    assert history_response['episodes'][1]['is_current'] is False
    assert history_response['episodes'][1]['invalid_at'] == '2026-03-09T00:00:00Z'
    assert history_response['limits_applied']['materialization']['om_projection']['history_lineages_projected'] == 1


def test_real_om_projection_preserves_same_node_uuid_across_multiple_groups():
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
    driver = _HistoryAwareDriver(
        records_by_seed={
            ('group_a', 'shared_node'): [
                {
                    'node_id': 'shared_node',
                    'uuid': 'shared_node',
                    'group_id': 'group_a',
                    'content': 'espresso after training in group A',
                    'created_at': '2026-03-09T00:00:00Z',
                    'status': 'active',
                    'semantic_domain': 'observational_memory',
                    'supersedes': [],
                }
            ],
            ('group_b', 'shared_node'): [
                {
                    'node_id': 'shared_node',
                    'uuid': 'shared_node',
                    'group_id': 'group_b',
                    'content': 'espresso after training in group B',
                    'created_at': '2026-03-09T00:00:00Z',
                    'status': 'active',
                    'semantic_domain': 'observational_memory',
                    'supersedes': [],
                }
            ],
        }
    )
    projection = OMTypedProjectionService(
        search_service=search_service,
        graphiti_service=_HistoryAwareGraphitiService(driver),
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
