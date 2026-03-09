import asyncio
from types import SimpleNamespace

from mcp_server.src.models.typed_memory import EvidenceRef, StateFact
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
            'skipped_roots_over_event_cap': 0,
            'root_selection_strategy': 'test',
            'max_candidate_roots': 250,
            'max_lineage_events': 256,
        },
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


def test_candidate_root_prefilter_uses_object_type_and_source_lane_scope():
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
    assert strategy == 'recent_roots'
    assert 'object_type IN (?)' in executed['sql']
    assert "json_extract(payload_json, '$.source_lane') IN (?, ?)" in executed['sql']
    assert executed['params'][:3] == ['state_fact', 'lane_a', 'lane_b']
