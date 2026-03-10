from __future__ import annotations

from unittest.mock import patch

import pytest

from scripts import om_compressor, om_convergence


class _FakeResult:
    def __init__(self, single_row=None):  # noqa: ANN001
        self._single = single_row or {'ok': True}

    def single(self):
        return self._single

    def consume(self):
        return None


class _RecordingTx:
    def __init__(self):
        self.calls: list[dict] = []

    def run(self, query: str, params=None):  # noqa: ANN001
        self.calls.append({'query': query, 'params': params or {}})
        return _FakeResult()


class _RecordingSession:
    def __init__(self):
        self.calls: list[dict] = []

    def run(self, query: str, params=None):  # noqa: ANN001
        self.calls.append({'query': query, 'params': params or {}})
        return _FakeResult()


def _cfg() -> om_compressor.ExtractorConfig:
    return om_compressor.ExtractorConfig(
        schema_version='v1',
        prompt_template='OM_PROMPT_TEMPLATE_V1',
        model_id='gpt-5.1-codex-mini',
        extractor_version='test-extractor',
    )


def _message(message_id: str, created_at: str) -> om_compressor.MessageRow:
    return om_compressor.MessageRow(
        message_id=message_id,
        source_session_id='sess-1',
        content=f'message {message_id}',
        created_at=created_at,
        content_embedding=[],
        om_extract_attempts=0,
    )


@pytest.mark.parametrize('relation_type', ['ADDRESSES', 'MOTIVATES'])
def test_process_chunk_writes_native_node_and_edge_lifecycle_fields(
    monkeypatch: pytest.MonkeyPatch,
    relation_type: str,
) -> None:
    tx = _RecordingTx()
    messages = [
        _message('m1', '2026-03-01T00:00:00Z'),
        _message('m2', '2026-03-03T12:00:00Z'),
    ]
    extracted = om_compressor.ExtractedChunk(
        nodes=[
            om_compressor.ExtractionNode(
                node_id='plan_v1',
                node_type='Action',
                semantic_domain='observational_memory',
                content='Ship the cache fix',
                urgency_score=4,
                source_session_id='sess-1',
                source_message_ids=['m1'],
            ),
            om_compressor.ExtractionNode(
                node_id='issue_1',
                node_type='Friction',
                semantic_domain='observational_memory',
                content='Latency issue',
                urgency_score=5,
                source_session_id='sess-1',
                source_message_ids=['m2'],
            ),
        ],
        edges=[
            om_compressor.ExtractionEdge(
                source_node_id='plan_v1',
                target_node_id='issue_1',
                relation_type=relation_type,
            )
        ],
    )

    monkeypatch.setattr(om_compressor, '_extract_items', lambda messages, cfg: extracted)
    monkeypatch.setattr(om_compressor, '_embed_text', lambda text, **_kwargs: [0.1, 0.2, 0.3])
    monkeypatch.setattr(om_compressor, 'now_iso', lambda: '2026-03-10T00:00:00Z')

    om_compressor._process_chunk_tx(
        tx,
        messages=messages,
        chunk_id='chunk-1',
        cfg=_cfg(),
        observed_node_ids=[],
        group_id='s1_observational_memory',
    )

    node_calls = [call for call in tx.calls if 'MERGE (n:OMNode {node_id:$node_id})' in call['query']]
    assert node_calls, 'expected OMNode upserts'
    plan_call = next(call for call in node_calls if call['params']['node_id'] == 'plan_v1')
    issue_call = next(call for call in node_calls if call['params']['node_id'] == 'issue_1')
    assert "n.valid_at = $valid_at" in plan_call['query']
    assert plan_call['params']['created_at'] == '2026-03-01T00:00:00Z'
    assert plan_call['params']['valid_at'] == '2026-03-01T00:00:00Z'
    assert issue_call['params']['created_at'] == '2026-03-03T12:00:00Z'

    edge_call = next(call for call in tx.calls if f'MERGE (s)-[r:{relation_type}]->(t)' in call['query'])
    assert edge_call['params']['relation_uuid'].startswith('omrel:')
    assert edge_call['params']['valid_at'] == '2026-03-03T12:00:00Z'
    assert 'r.relation_root_id = $relation_uuid' in edge_call['query']
    assert "r.lifecycle_status = 'asserted'" in edge_call['query']
    assert 'r.lineage_basis = \'singleton_native\'' in edge_call['query']

    relation_supersession_call = next(
        call for call in tx.calls if 'old_rel.superseded_by_relation_id = $relation_uuid' in call['query']
    )
    assert relation_supersession_call['params']['relation_uuid'] == edge_call['params']['relation_uuid']
    assert relation_supersession_call['params']['valid_at'] == '2026-03-03T12:00:00Z'


def test_process_chunk_writes_native_node_supersession_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    tx = _RecordingTx()
    messages = [
        _message('m1', '2026-03-01T00:00:00Z'),
        _message('m2', '2026-03-05T00:00:00Z'),
    ]
    extracted = om_compressor.ExtractedChunk(
        nodes=[
            om_compressor.ExtractionNode(
                node_id='plan_v1',
                node_type='Action',
                semantic_domain='observational_memory',
                content='Ship the cache fix',
                urgency_score=4,
                source_session_id='sess-1',
                source_message_ids=['m1'],
            ),
            om_compressor.ExtractionNode(
                node_id='plan_v2',
                node_type='Action',
                semantic_domain='observational_memory',
                content='Ship the cache fix with retry guard',
                urgency_score=4,
                source_session_id='sess-1',
                source_message_ids=['m2'],
            ),
        ],
        edges=[
            om_compressor.ExtractionEdge(
                source_node_id='plan_v2',
                target_node_id='plan_v1',
                relation_type='SUPERSEDES',
            )
        ],
    )

    monkeypatch.setattr(om_compressor, '_extract_items', lambda messages, cfg: extracted)
    monkeypatch.setattr(om_compressor, '_embed_text', lambda text, **_kwargs: [0.1, 0.2, 0.3])
    monkeypatch.setattr(om_compressor, 'now_iso', lambda: '2026-03-10T00:00:00Z')

    om_compressor._process_chunk_tx(
        tx,
        messages=messages,
        chunk_id='chunk-2',
        cfg=_cfg(),
        observed_node_ids=[],
        group_id='s1_observational_memory',
    )

    supersession_call = next(
        call for call in tx.calls if 'older.superseded_by_node_id = CASE' in call['query']
    )
    assert supersession_call['params'] == {
        'older_node_id': 'plan_v1',
        'newer_node_id': 'plan_v2',
        'valid_at': '2026-03-05T00:00:00Z',
    }
    assert 'newer.lineage_parent_id = $older_node_id' in supersession_call['query']
    assert "older.lifecycle_status = 'superseded'" in supersession_call['query']
    assert "older.transition_cause = 'superseded_by_newer_node'" in supersession_call['query']


def test_apply_transition_closed_invalidates_node_and_active_addresses() -> None:
    session = _RecordingSession()

    om_convergence._apply_transition(
        session,
        node_id='issue_1',
        node_type='Friction',
        target_status='closed',
        now_utc='2026-03-10T00:00:00Z',
    )

    joined = '\n'.join(call['query'] for call in session.calls)
    assert 'n.closed_at = $now_iso' in joined
    assert 'n.invalid_at = $now_iso' in joined
    assert "n.lifecycle_status = 'invalidated'" in joined
    assert 'MATCH (source:OMNode)-[r:ADDRESSES]->(target:OMNode {node_id:$node_id})' in joined
    assert 'r.invalid_at = $now_iso' in joined
    assert "r.transition_cause = $transition_cause" in joined
    assert 'MERGE (j)-[r:RESOLVES]->(n)' in joined
    assert 'r.valid_at = $now_iso' in joined


def test_apply_transition_reopened_restores_current_node_lifecycle_truth() -> None:
    session = _RecordingSession()

    om_convergence._apply_transition(
        session,
        node_id='issue_1',
        node_type='Friction',
        target_status='reopened',
        now_utc='2026-03-11T00:00:00Z',
    )

    joined = '\n'.join(call['query'] for call in session.calls)
    assert 'n.previous_status = n.status' in joined
    assert 'n.reopened_at = $now_iso' in joined
    assert 'n.invalid_at = NULL' in joined
    assert "n.lifecycle_status = 'asserted'" in joined
    assert "n.transition_cause = 'reopened_after_reappearance'" in joined
