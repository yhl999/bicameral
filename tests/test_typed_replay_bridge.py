# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingest.registry import IngestRegistry
from mcp_server.src.models.typed_memory import EvidenceRef
from mcp_server.src.services.typed_replay_bridge import (
    TypedReplayBridge,
    build_bridge_metadata,
    build_session_chunk_episode,
)
from scripts import mcp_ingest_sessions as ingest_sessions



def test_legacy_evidence_ref_normalizes_session_prefixes():
    ref = EvidenceRef.from_legacy_ref(
        {
            'source_key': 'sessionS:main:abc123',
            'evidence_id': 'sessionS:main:abc123:c0',
        }
    )

    assert ref.source_system == 'sessions'
    assert ref.canonical_uri == 'eventlog://sessions/session:main:abc123/session:main:abc123:c0'



def test_build_session_chunk_episode_preserves_canonical_provenance():
    episode = build_session_chunk_episode(
        object_id='episode-1',
        source_lane='s1_sessions_main',
        source_key='sessionS:main:abc123',
        source_episode_id='sessionS:main:abc123:c0',
        source_message_id='msg-1',
        scope='private',
        summary='User said they prefer coffee after lunch.',
        started_at='2026-03-08T10:00:00Z',
        ended_at='2026-03-08T10:05:00Z',
        chunk_key='sessionS:main:abc123:c0',
        evidence_id='evidence-1',
        start_id='msg-1',
        end_id='msg-3',
        annotations=['bridge:typed_replay'],
        title='session:main:abc123',
    )

    assert episode.object_id == 'episode-1'
    assert episode.source_lane == 's1_sessions_main'
    assert episode.source_key == 'session:main:abc123'
    assert episode.source_episode_id == 'session:main:abc123:c0'
    assert episode.source_message_id == 'msg-1'
    assert episode.policy_scope == 'private'
    assert episode.visibility_scope == 'private'
    assert episode.started_at == '2026-03-08T10:00:00Z'
    assert episode.ended_at == '2026-03-08T10:05:00Z'
    assert episode.evidence_refs[0].canonical_uri == (
        'eventlog://sessions/session:main:abc123/session:main:abc123:c0'
    )



def test_build_bridge_metadata_preserves_canonical_provenance():
    metadata = build_bridge_metadata(
        group_id='s1_sessions_main',
        source_key='sessionS:main:abc123',
        chunk_key='sessionS:main:abc123:c0',
        message_ids=['msg-1', 'msg-2'],
        evidence_id='evidence-1',
        content_hash='hash-1',
        scope='private',
        started_at='2026-03-08T10:00:00Z',
        ended_at='2026-03-08T10:05:00Z',
    )

    assert metadata['source_key'] == 'session:main:abc123'
    assert metadata['chunk_key'] == 'session:main:abc123:c0'


def test_typed_replay_bridge_assert_episode_once_is_idempotent(tmp_path: Path):
    ledger_path = tmp_path / 'change_ledger.db'
    bridge = TypedReplayBridge(ledger_path)
    episode = build_session_chunk_episode(
        object_id='episode-1',
        source_lane='s1_sessions_main',
        source_key='session:main:abc123',
        source_episode_id='session:main:abc123:c0',
        source_message_id='msg-1',
        scope='private',
        summary='Typed replay bridge smoke test.',
        started_at='2026-03-08T10:00:00Z',
        ended_at='2026-03-08T10:05:00Z',
        chunk_key='session:main:abc123:c0',
        evidence_id='evidence-1',
        start_id='msg-1',
        end_id='msg-2',
    )
    metadata = build_bridge_metadata(
        group_id='s1_sessions_main',
        source_key='session:main:abc123',
        chunk_key='session:main:abc123:c0',
        message_ids=['msg-1', 'msg-2'],
        evidence_id='evidence-1',
        content_hash='hash-1',
        scope='private',
        started_at='2026-03-08T10:00:00Z',
        ended_at='2026-03-08T10:05:00Z',
    )

    first = bridge.assert_episode_once(episode, metadata=metadata)
    second = bridge.assert_episode_once(episode, metadata=metadata)

    assert first.created is True
    assert second.created is False
    count = bridge.ledger.conn.execute('SELECT COUNT(*) FROM change_events').fetchone()[0]
    assert count == 1
    row = bridge.ledger.conn.execute('SELECT metadata_json FROM change_events').fetchone()
    assert json.loads(row['metadata_json'])['bridge'] == 'typed_replay'



def test_typed_replay_bridge_assert_episode_once_is_concurrency_safe(tmp_path: Path):
    ledger_path = tmp_path / 'change_ledger.db'
    episode = build_session_chunk_episode(
        object_id='episode-1',
        source_lane='s1_sessions_main',
        source_key='session:main:abc123',
        source_episode_id='session:main:abc123:c0',
        source_message_id='msg-1',
        scope='private',
        summary='Concurrent typed replay bridge smoke test.',
        started_at='2026-03-08T10:00:00Z',
        ended_at='2026-03-08T10:05:00Z',
        chunk_key='session:main:abc123:c0',
        evidence_id='evidence-1',
        start_id='msg-1',
        end_id='msg-2',
    )
    metadata = build_bridge_metadata(
        group_id='s1_sessions_main',
        source_key='session:main:abc123',
        chunk_key='session:main:abc123:c0',
        message_ids=['msg-1', 'msg-2'],
        evidence_id='evidence-1',
        content_hash='hash-1',
        scope='private',
        started_at='2026-03-08T10:00:00Z',
        ended_at='2026-03-08T10:05:00Z',
    )

    # Initialize schema once before concurrent writers race on the same DB.
    TypedReplayBridge(ledger_path)

    barrier = threading.Barrier(2)
    created_flags: list[bool] = []

    def _worker() -> None:
        worker_bridge = TypedReplayBridge(ledger_path)
        barrier.wait(timeout=5)
        result = worker_bridge.assert_episode_once(episode, metadata=metadata)
        created_flags.append(result.created)

    threads = [threading.Thread(target=_worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(created_flags) == [False, True]
    bridge = TypedReplayBridge(ledger_path)
    count = bridge.ledger.conn.execute('SELECT COUNT(*) FROM change_events').fetchone()[0]
    assert count == 1



def test_run_evidence_mode_typed_only_materializes_change_ledger(tmp_path: Path, monkeypatch):
    evidence_path = tmp_path / 'all_evidence.json'
    evidence_path.write_text(
        json.dumps(
            [
                {
                    'evidence_id': 'ev-1',
                    'source_key': 'session:main:abc123',
                    'chunk_key': 'session:main:abc123:c0',
                    'scope': 'private',
                    'timestamp_range': {
                        'start': '2026-03-08T10:00:00Z',
                        'end': '2026-03-08T10:05:00Z',
                    },
                    'start_id': 'msg-1',
                    'end_id': 'msg-2',
                    'message_ids': ['msg-1', 'msg-2'],
                    'source': {
                        'type': 'session',
                        'agent_id': 'main',
                        'session_id': 'abc123',
                    },
                    'content': 'User: I now prefer oolong tea.\n\nAssistant: noted.',
                }
            ]
        ),
        encoding='utf-8',
    )
    ledger_path = tmp_path / 'change_ledger.db'
    registry_path = tmp_path / 'ingest_registry.db'
    registry = IngestRegistry(registry_path)
    monkeypatch.setattr(ingest_sessions, 'get_registry', lambda: registry)

    args = argparse.Namespace(
        subchunk_size=10_000,
        evidence=str(evidence_path),
        group_id='s1_sessions_main',
        limit=10,
        offset=0,
        sleep=0.0,
        shards=1,
        shard_index=0,
        incremental=False,
        overlap=10,
        dry_run=False,
        force=False,
        typed_truth_mode='only',
        change_ledger_db=str(ledger_path),
        mcp_url='http://localhost:8000/mcp',
    )
    ap = ingest_sessions.build_parser()

    ingest_sessions._run_evidence_mode(args, ap)

    bridge = TypedReplayBridge(ledger_path)
    episodes = bridge.list_current_episodes()
    assert len(episodes) == 1
    assert episodes[0].source_lane == 's1_sessions_main'
    assert episodes[0].source_episode_id == 'session:main:abc123:c0'
    assert registry.get_stats()['chunk_count'] == 0
    stats = registry.get_extraction_stats(group_id='s1_sessions_main')
    assert stats['total'] == 0



def test_run_evidence_mode_typed_only_subchunks_drop_precise_message_bounds(tmp_path: Path, monkeypatch):
    evidence_path = tmp_path / 'all_evidence.json'
    evidence_path.write_text(
        json.dumps(
            [
                {
                    'evidence_id': 'ev-1',
                    'source_key': 'session:main:abc123',
                    'chunk_key': 'session:main:abc123:c0',
                    'scope': 'private',
                    'timestamp_range': {
                        'start': '2026-03-08T10:00:00Z',
                        'end': '2026-03-08T10:05:00Z',
                    },
                    'start_id': 'msg-1',
                    'end_id': 'msg-4',
                    'message_ids': ['msg-1', 'msg-2', 'msg-3', 'msg-4'],
                    'source': {
                        'type': 'session',
                        'agent_id': 'main',
                        'session_id': 'abc123',
                    },
                    'content': 'A' * 32,
                }
            ]
        ),
        encoding='utf-8',
    )
    monkeypatch.setattr(ingest_sessions, 'get_registry', lambda: IngestRegistry(tmp_path / 'unused.db'))

    args = argparse.Namespace(
        subchunk_size=10,
        evidence=str(evidence_path),
        group_id='s1_sessions_main',
        limit=10,
        offset=0,
        sleep=0.0,
        shards=1,
        shard_index=0,
        incremental=False,
        overlap=10,
        dry_run=False,
        force=False,
        typed_truth_mode='only',
        change_ledger_db=str(tmp_path / 'change_ledger.db'),
        mcp_url='http://localhost:8000/mcp',
    )
    ap = ingest_sessions.build_parser()

    ingest_sessions._run_evidence_mode(args, ap)

    bridge = TypedReplayBridge(tmp_path / 'change_ledger.db')
    episodes = bridge.list_current_episodes()
    assert len(episodes) > 1
    for episode in episodes:
        locator = episode.evidence_refs[0].locator
        assert 'start_id' not in locator
        assert 'end_id' not in locator
        assert episode.source_message_id is None
        assert 'provenance:subchunk' in episode.annotations
        assert any(a.startswith('source_chunk:session:main:abc123:c0') for a in episode.annotations)



def test_run_evidence_mode_sidecar_writes_typed_even_when_registry_chunk_exists(
    tmp_path: Path, monkeypatch
):
    evidence_path = tmp_path / 'all_evidence.json'
    evidence = {
        'evidence_id': 'ev-1',
        'source_key': 'sessionS:main:abc123',
        'chunk_key': 'sessionS:main:abc123:c0',
        'scope': 'private',
        'timestamp_range': {
            'start': '2026-03-08T10:00:00Z',
            'end': '2026-03-08T10:05:00Z',
        },
        'start_id': 'msg-1',
        'end_id': 'msg-2',
        'message_ids': ['msg-1', 'msg-2'],
        'source': {
            'type': 'session',
            'agent_id': 'main',
            'session_id': 'abc123',
        },
        'content': 'User: I now prefer oolong tea.\n\nAssistant: noted.',
    }
    evidence_path.write_text(json.dumps([evidence]), encoding='utf-8')

    registry = IngestRegistry(tmp_path / 'ingest_registry.db')
    monkeypatch.setattr(ingest_sessions, 'get_registry', lambda: registry)

    content_hash = IngestRegistry.compute_content_hash(evidence['content'])
    chunk_uuid = IngestRegistry.compute_chunk_uuid(
        source_key=evidence['source_key'],
        chunk_key=evidence['chunk_key'],
        content_hash=content_hash,
    )
    registry.record_chunk(
        chunk_uuid=chunk_uuid,
        source_key=evidence['source_key'],
        chunk_key=evidence['chunk_key'],
        content_hash=content_hash,
        evidence_id=evidence['evidence_id'],
    )

    calls: list[dict[str, object]] = []

    class DummyMCPClient:
        def __init__(self, _url: str):
            pass

        def call_tool(self, tool_name: str, payload: dict[str, object]) -> dict[str, object]:
            calls.append({'tool_name': tool_name, 'payload': payload})
            return {}

    monkeypatch.setattr(ingest_sessions, 'MCPClient', DummyMCPClient)

    ledger_path = tmp_path / 'change_ledger.db'
    args = argparse.Namespace(
        subchunk_size=10_000,
        evidence=str(evidence_path),
        group_id='s1_sessions_main',
        limit=10,
        offset=0,
        sleep=0.0,
        shards=1,
        shard_index=0,
        incremental=False,
        overlap=10,
        dry_run=False,
        force=False,
        typed_truth_mode='sidecar',
        change_ledger_db=str(ledger_path),
        mcp_url='http://localhost:8000/mcp',
    )
    ap = ingest_sessions.build_parser()

    ingest_sessions._run_evidence_mode(args, ap)

    assert calls == []
    bridge = TypedReplayBridge(ledger_path)
    episodes = bridge.list_current_episodes()
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.source_key == 'session:main:abc123'
    assert episode.source_episode_id == 'session:main:abc123:c0'

    row = bridge.ledger.conn.execute('SELECT metadata_json FROM change_events').fetchone()
    metadata = json.loads(row['metadata_json'])
    assert metadata['source_key'] == 'session:main:abc123'
    assert metadata['chunk_key'] == 'session:main:abc123:c0'


def test_run_evidence_mode_dry_run_does_not_create_change_ledger_db(tmp_path: Path, monkeypatch):
    evidence_path = tmp_path / 'all_evidence.json'
    evidence_path.write_text(
        json.dumps(
            [
                {
                    'evidence_id': 'ev-1',
                    'source_key': 'session:main:abc123',
                    'chunk_key': 'session:main:abc123:c0',
                    'scope': 'private',
                    'timestamp_range': {'start': '2026-03-08T10:00:00Z'},
                    'content': 'User: test',
                }
            ]
        ),
        encoding='utf-8',
    )
    monkeypatch.setattr(ingest_sessions, 'get_registry', lambda: IngestRegistry(tmp_path / 'unused.db'))

    ledger_path = tmp_path / 'change_ledger.db'
    args = argparse.Namespace(
        subchunk_size=10_000,
        evidence=str(evidence_path),
        group_id='s1_sessions_main',
        limit=10,
        offset=0,
        sleep=0.0,
        shards=1,
        shard_index=0,
        incremental=False,
        overlap=10,
        dry_run=True,
        force=False,
        typed_truth_mode='only',
        change_ledger_db=str(ledger_path),
        mcp_url='http://localhost:8000/mcp',
    )
    ap = ingest_sessions.build_parser()

    ingest_sessions._run_evidence_mode(args, ap)

    assert not ledger_path.exists()


def test_build_parser_typed_truth_mode_stays_off_by_default():
    ap = ingest_sessions.build_parser()
    assert ap._option_string_actions['--typed-truth-mode'].default == 'off'
