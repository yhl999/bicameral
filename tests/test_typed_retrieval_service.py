import asyncio
import json
import sqlite3
from types import SimpleNamespace

from mcp_server.src.models.typed_memory import EvidenceRef, StateFact
from mcp_server.src.services.change_ledger import ChangeLedger
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


def _state_fact(*, object_id: str, root_id: str, version: int, value: str, is_current: bool = True):
    return StateFact(
        object_id=object_id,
        root_id=root_id,
        version=version,
        is_current=is_current,
        source_lane='s1_sessions_main',
        source_key='session:1',
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
