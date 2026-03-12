"""SQLite-backed candidate quarantine store for owner-asserted conflicts."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


def _default_candidates_db_path() -> Path:
    base_dir = Path(__file__).resolve().parents[2] / 'state'
    return Path(os.environ.get('BICAMERAL_CANDIDATES_DB_PATH', base_dir / 'candidates.db'))


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    fact_type TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'promoted', 'rejected')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    conflict_with_fact_id TEXT,
    resolution TEXT,
    source TEXT,
    raw_hint_json TEXT,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidates(status);
"""


def _load_json(value: str | None) -> Any:
    if value in (None, ''):
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def _coerce_status(status: str | None) -> str:
    candidate_status = str(status or 'pending').strip().lower()
    if candidate_status not in {'pending', 'promoted', 'rejected'}:
        raise ValueError(f"Invalid candidate status: {status!r}")
    return candidate_status


@dataclass(slots=True)
class CandidateStore:
    db_path: Path | str = field(default_factory=_default_candidates_db_path)

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(CREATE_SQL)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            'subject': str(payload.get('subject') or '').strip(),
            'predicate': str(payload.get('predicate') or '').strip(),
            'value': payload.get('value'),
            'fact_type': str(payload.get('fact_type') or payload.get('type') or '').strip(),
        }

    def create_candidate(
        self,
        *,
        payload: dict[str, Any],
        status: str = 'pending',
        conflict_with_fact_id: str | None = None,
        resolution: str | None = None,
        source: str | None = None,
        raw_hint: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        candidate_id: str | None = None,
    ) -> dict[str, Any]:
        normalized = self._normalize_payload(payload)
        if not normalized['subject'] or not normalized['predicate']:
            raise ValueError('Candidate requires subject and predicate')
        if not normalized['fact_type']:
            raise ValueError('Candidate requires fact_type')

        candidate_id = candidate_id or f'cand-{uuid4().hex[:24]}'
        candidate_status = _coerce_status(status)
        now = _utc_iso()

        with self._connect() as conn:
            conn.execute(
                '''
                INSERT INTO candidates (
                    candidate_id,
                    fact_type,
                    subject,
                    predicate,
                    value_json,
                    status,
                    created_at,
                    updated_at,
                    conflict_with_fact_id,
                    resolution,
                    source,
                    raw_hint_json,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''' ,
                (
                    candidate_id,
                    normalized['fact_type'],
                    normalized['subject'],
                    normalized['predicate'],
                    json.dumps(normalized['value'], ensure_ascii=False, sort_keys=True),
                    candidate_status,
                    now,
                    now,
                    conflict_with_fact_id,
                    resolution,
                    source,
                    json.dumps(raw_hint, ensure_ascii=False, sort_keys=True) if raw_hint is not None else None,
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True) if metadata is not None else None,
                ),
            )
            conn.commit()

        candidate = {
            'candidate_id': candidate_id,
            'fact_type': normalized['fact_type'],
            'subject': normalized['subject'],
            'predicate': normalized['predicate'],
            'value': normalized['value'],
            'status': candidate_status,
            'created_at': now,
            'updated_at': now,
        }
        if conflict_with_fact_id:
            candidate['conflict_with_fact_id'] = conflict_with_fact_id
        if resolution is not None:
            candidate['resolution'] = resolution
        if source is not None:
            candidate['source'] = source
        if raw_hint is not None:
            candidate['raw_hint'] = raw_hint
        if metadata is not None:
            candidate['metadata'] = metadata

        logger.debug(
            'created candidate id=%s fact_type=%s subject=%s predicate=%s status=%s',
            candidate_id,
            normalized['fact_type'],
            normalized['subject'],
            normalized['predicate'],
            candidate_status,
        )

        return candidate

    def list_candidates(self, status: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM candidates"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(_coerce_status(status))
        query += " ORDER BY created_at DESC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_candidate(row) for row in rows]

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                'SELECT * FROM candidates WHERE candidate_id = ? LIMIT 1',
                (candidate_id,),
            ).fetchone()

        if row is None:
            return None
        return self._row_to_candidate(row)

    def update_candidate_status(
        self,
        candidate_id: str,
        status: str,
        *,
        resolution: str | None = None,
    ) -> dict[str, Any] | None:
        status_value = _coerce_status(status)
        now = _utc_iso()

        with self._connect() as conn:
            row = conn.execute(
                'SELECT * FROM candidates WHERE candidate_id = ? LIMIT 1',
                (candidate_id,),
            ).fetchone()
            if row is None:
                return None

            conn.execute(
                '''
                UPDATE candidates
                   SET status = ?,
                       resolution = ?,
                       updated_at = ?
                 WHERE candidate_id = ?
                ''',
                (status_value, resolution, now, candidate_id),
            )
            conn.commit()

            row = conn.execute(
                'SELECT * FROM candidates WHERE candidate_id = ? LIMIT 1',
                (candidate_id,),
            ).fetchone()

        return self._row_to_candidate(row)

    def _row_to_candidate(self, row: sqlite3.Row) -> dict[str, Any]:
        candidate = {
            'candidate_id': str(row['candidate_id']),
            'fact_type': str(row['fact_type']),
            'subject': str(row['subject']),
            'predicate': str(row['predicate']),
            'value': _load_json(row['value_json']),
            'status': str(row['status']),
            'created_at': str(row['created_at']) if row['created_at'] is not None else None,
            'updated_at': str(row['updated_at']) if row['updated_at'] is not None else None,
        }

        if row['conflict_with_fact_id'] is not None:
            candidate['conflict_with_fact_id'] = str(row['conflict_with_fact_id'])
        if row['resolution'] is not None:
            candidate['resolution'] = str(row['resolution'])
        if row['source'] is not None:
            candidate['source'] = str(row['source'])
        hint = _load_json(row['raw_hint_json'])
        if hint is not None:
            candidate['raw_hint'] = hint
        metadata = _load_json(row['metadata_json'])
        if metadata is not None:
            candidate['metadata'] = metadata
        return candidate
