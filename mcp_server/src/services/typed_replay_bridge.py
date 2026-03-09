from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models.typed_memory import Episode, EvidenceRef
from .change_ledger import DB_PATH_DEFAULT, ChangeLedger

BRIDGE_ACTOR_ID = 'bridge:typed-replay'
BRIDGE_REASON = 'phase2_pre_ingestion_bridge'
BRIDGE_EXTRACTOR_VERSION = 'phase2-typed-replay-bridge/v1'
_TYPED_SCOPE_FALLBACK = 'private'
_SOURCE_SYSTEM_ALIASES = {
    'session': 'sessions',
    'sessions': 'sessions',
}


@dataclass(frozen=True)
class BridgeWriteResult:
    episode: Episode
    created: bool
    event_id: str | None = None


def canonical_source_system(source_key: str | None, *, default: str = 'legacy') -> str:
    prefix = str(source_key or '').split(':', 1)[0].strip().lower()
    if not prefix:
        return default
    return _SOURCE_SYSTEM_ALIASES.get(prefix, prefix)


def canonical_scope(scope: str | None) -> str:
    value = str(scope or '').strip().lower()
    if value in {'private', 'group', 'internal', 'public'}:
        return value
    return _TYPED_SCOPE_FALLBACK


def canonical_session_identity(value: str | None) -> str:
    raw = str(value or '').strip()
    if not raw:
        return raw
    prefix, sep, rest = raw.partition(':')
    if prefix.strip().lower() in {'session', 'sessions'}:
        return f'session:{rest}' if sep else 'session'
    return raw


def build_session_chunk_evidence_ref(
    *,
    source_key: str,
    source_lane: str,
    source_episode_id: str,
    chunk_key: str,
    evidence_id: str | None,
    start_id: str | None,
    end_id: str | None,
    observed_at: str | None,
    snippet: str | None,
) -> EvidenceRef:
    system = canonical_source_system(source_key, default='sessions')
    canonical_stream = canonical_session_identity(source_key)
    canonical_source_episode_id = canonical_session_identity(source_episode_id)
    canonical_chunk_key = canonical_session_identity(chunk_key)
    locator: dict[str, Any] = {
        'system': system,
        'stream': canonical_stream,
        'event_id': canonical_source_episode_id or canonical_chunk_key,
        'lane': source_lane,
        'source_episode_id': canonical_source_episode_id,
        'chunk_key': canonical_chunk_key,
    }
    if evidence_id:
        locator['evidence_id'] = evidence_id
    if start_id:
        locator['start_id'] = start_id
    if end_id:
        locator['end_id'] = end_id
    return EvidenceRef(
        kind='event_log',
        source_system=system,
        locator=locator,
        observed_at=observed_at,
        snippet=snippet,
    )


def build_session_chunk_episode(
    *,
    object_id: str,
    source_lane: str,
    source_key: str,
    source_episode_id: str,
    source_message_id: str | None,
    scope: str | None,
    summary: str,
    started_at: str | None,
    ended_at: str | None,
    chunk_key: str,
    evidence_id: str | None,
    start_id: str | None,
    end_id: str | None,
    annotations: list[str] | None = None,
    title: str | None = None,
) -> Episode:
    canonical_source_key = canonical_session_identity(source_key) or str(source_key or '')
    canonical_source_episode_id = canonical_session_identity(source_episode_id) or str(
        source_episode_id or ''
    )
    canonical_chunk_key = canonical_session_identity(chunk_key) or str(chunk_key or '')
    effective_scope = canonical_scope(scope)
    evidence_ref = build_session_chunk_evidence_ref(
        source_key=canonical_source_key,
        source_lane=source_lane,
        source_episode_id=canonical_source_episode_id,
        chunk_key=canonical_chunk_key,
        evidence_id=evidence_id,
        start_id=start_id,
        end_id=end_id,
        observed_at=started_at or ended_at,
        snippet=summary[:280] if summary else None,
    )
    created_at = started_at or ended_at
    payload: dict[str, Any] = {
        'object_id': object_id,
        'root_id': object_id,
        'object_type': 'episode',
        'source_lane': source_lane,
        'source_episode_id': canonical_source_episode_id,
        'source_message_id': source_message_id,
        'source_key': canonical_source_key,
        'policy_scope': effective_scope,
        'visibility_scope': effective_scope,
        'evidence_refs': [evidence_ref],
        'extractor_version': BRIDGE_EXTRACTOR_VERSION,
        'title': title,
        'summary': summary,
        'started_at': started_at,
        'ended_at': ended_at,
        'annotations': annotations or [],
    }
    if created_at:
        payload['created_at'] = created_at
        payload['valid_at'] = created_at
    return Episode.model_validate(payload)


class TypedReplayBridge:
    def __init__(self, db_path: str | Path = DB_PATH_DEFAULT):
        self.db_path = Path(db_path)
        self.ledger = ChangeLedger(self.db_path)

    def get_episode(self, object_id: str) -> Episode | None:
        existing = self.ledger.materialize_object(object_id)
        if existing is None:
            return None
        if not isinstance(existing, Episode):
            raise ValueError(
                f'Existing object_id {object_id!r} is not an Episode: {existing.object_type}'
            )
        return existing

    def assert_episode_once(
        self,
        episode: Episode,
        *,
        actor_id: str = BRIDGE_ACTOR_ID,
        reason: str = BRIDGE_REASON,
        metadata: dict[str, Any] | None = None,
    ) -> BridgeWriteResult:
        try:
            self.ledger.conn.execute('BEGIN IMMEDIATE')
            existing = self.get_episode(episode.object_id)
            if existing is not None:
                self.ledger.conn.commit()
                return BridgeWriteResult(episode=existing, created=False, event_id=None)

            event = self.ledger.append_event(
                'assert',
                actor_id=actor_id,
                reason=reason,
                recorded_at=episode.created_at,
                payload=episode,
                metadata=metadata,
                _autocommit=False,
            )
            self.ledger.conn.commit()
            return BridgeWriteResult(episode=episode, created=True, event_id=event.event_id)
        except sqlite3.IntegrityError:
            self.ledger.conn.rollback()
            existing = self.get_episode(episode.object_id)
            if existing is not None:
                return BridgeWriteResult(episode=existing, created=False, event_id=None)
            raise
        except Exception:
            self.ledger.conn.rollback()
            raise

    def list_current_episodes(self) -> list[Episode]:
        rows = self.ledger.conn.execute(
            "SELECT DISTINCT root_id FROM change_events WHERE object_type = 'episode' ORDER BY root_id"
        ).fetchall()
        episodes: list[Episode] = []
        for row in rows:
            current = self.ledger.current_object(str(row['root_id']))
            if isinstance(current, Episode):
                episodes.append(current)
        return episodes


def build_bridge_metadata(
    *,
    group_id: str,
    source_key: str,
    chunk_key: str,
    message_ids: list[str] | None,
    evidence_id: str | None,
    content_hash: str,
    scope: str | None,
    started_at: str | None,
    ended_at: str | None,
) -> dict[str, Any]:
    canonical_source_key = canonical_session_identity(source_key) or str(source_key or '')
    canonical_chunk_key = canonical_session_identity(chunk_key) or str(chunk_key or '')
    return {
        'bridge': 'typed_replay',
        'group_id': group_id,
        'source_key': canonical_source_key,
        'chunk_key': canonical_chunk_key,
        'message_ids': list(message_ids or []),
        'evidence_id': evidence_id,
        'content_hash': content_hash,
        'scope': canonical_scope(scope),
        'started_at': started_at,
        'ended_at': ended_at,
        'bridge_version': BRIDGE_EXTRACTOR_VERSION,
    }


def metadata_json(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
