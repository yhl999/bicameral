# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import json
import sys
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
            'source_key': 'session:main:abc123',
            'evidence_id': 'chunk-1',
        }
    )

    assert ref.source_system == 'sessions'
    assert ref.canonical_uri == 'eventlog://sessions/session:main:abc123/chunk-1'



def test_build_session_chunk_episode_preserves_canonical_provenance():
    episode = build_session_chunk_episode(
        object_id='episode-1',
        source_lane='s1_sessions_main',
        source_key='session:main:abc123',
        source_episode_id='session:main:abc123:c0',
        source_message_id='msg-1',
        scope='private',
        summary='User said they prefer coffee after lunch.',
        started_at='2026-03-08T10:00:00Z',
        ended_at='2026-03-08T10:05:00Z',
        chunk_key='session:main:abc123:c0',
        source_event_id='episode-1',
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
    assert episode.evidence_refs[0].canonical_uri == 'eventlog://sessions/session:main:abc123/episode-1'



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
        source_event_id='episode-1',
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
    stats = registry.get_extraction_stats(group_id='s1_sessions_main')
    assert stats['succeeded'] == 1
