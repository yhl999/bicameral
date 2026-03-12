"""Candidate lifecycle MCP tools.

These tools expose quarantine/promotion/rejection operations against a persisted
`candidates` table and promote accepted items into the main change ledger.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from ..services.change_ledger import DB_PATH_DEFAULT, ChangeLedger
except ImportError:  # pragma: no cover - top-level import fallback
    from services.change_ledger import DB_PATH_DEFAULT, ChangeLedger  # type: ignore[no-redef]


logger = logging.getLogger(__name__)

VALID_STATUSES = frozenset({'quarantine', 'promoted', 'rejected'})
VALID_RESOLUTIONS = frozenset({'supersede', 'parallel', 'cancel'})

CANDIDATES_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS candidates (
    uuid TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value TEXT NOT NULL,
    conflicting_fact_uuid TEXT,
    status TEXT NOT NULL DEFAULT 'quarantine',
    resolution TEXT,
    confidence REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    reviewed_at TEXT,
    reviewed_by TEXT,
    promoted_at TEXT,
    promoted_by TEXT,
    reason TEXT,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_candidates_status_created
    ON candidates(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_candidates_type_confidence
    ON candidates(type, confidence DESC);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _ledger_path() -> Path:
    override = os.getenv('BICAMERAL_CHANGE_LEDGER_PATH', '').strip()
    return Path(override) if override else Path(DB_PATH_DEFAULT)


def _actor_id() -> str:
    return os.getenv('BICAMERAL_MCP_ACTOR_ID', os.getenv('BICAMERAL_ACTOR_ID', 'system')).strip() or 'system'


def _ensure_columns(conn: sqlite3.Connection) -> None:
    existing = {
        row['name']
        for row in conn.execute("PRAGMA table_info('candidates')").fetchall()
    }

    missing_to_sql = {
        'reviewed_at': 'ALTER TABLE candidates ADD COLUMN reviewed_at TEXT',
        'reviewed_by': 'ALTER TABLE candidates ADD COLUMN reviewed_by TEXT',
        'promoted_at': 'ALTER TABLE candidates ADD COLUMN promoted_at TEXT',
        'promoted_by': 'ALTER TABLE candidates ADD COLUMN promoted_by TEXT',
        'reason': 'ALTER TABLE candidates ADD COLUMN reason TEXT',
    }

    for col, statement in missing_to_sql.items():
        if col not in existing:
            conn.execute(statement)

    conn.commit()


class CandidatesDB:
    """Persistence for candidate quarantine rows."""

    def __init__(self, conn_or_path: sqlite3.Connection | str | Path = DB_PATH_DEFAULT):
        if isinstance(conn_or_path, sqlite3.Connection):
            self.conn = conn_or_path
        else:
            db_path = Path(conn_or_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(db_path))
            self.conn.row_factory = sqlite3.Row
            self.conn.execute('PRAGMA journal_mode=WAL')
            self.conn.execute('PRAGMA foreign_keys=ON')
            self.conn.execute('PRAGMA busy_timeout=5000')
        self.conn.row_factory = sqlite3.Row
        self.ensure_schema()

    def ensure_schema(self) -> None:
        self.conn.executescript(CANDIDATES_SCHEMA_SQL)
        _ensure_columns(self.conn)
        self.conn.commit()

    def list_candidates(
        self,
        *,
        status: str,
        type_filter: str | None = None,
        min_confidence: float | None = None,
        max_age_days: int | None = None,
        age_days: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = ['status = ?']
        params: list[Any] = [status]

        if type_filter:
            clauses.append('type = ?')
            params.append(type_filter)
        if min_confidence is not None:
            clauses.append('confidence >= ?')
            params.append(float(min_confidence))

        # exec4 branch contract uses age_days; keep max_age_days for compatibility.
        effective_age = age_days if age_days is not None else max_age_days
        if effective_age is not None:
            cutoff = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(days=effective_age)
            clauses.append('created_at >= ?')
            params.append(cutoff.isoformat().replace('+00:00', 'Z'))

        where = ' AND '.join(clauses)
        rows = self.conn.execute(
            f"""
            SELECT *
              FROM candidates
             WHERE {where}
             ORDER BY created_at DESC
            """,
            params,
        ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        row = self.conn.execute('SELECT * FROM candidates WHERE uuid = ?', (candidate_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def update_status(
        self,
        candidate_id: str,
        *,
        status: str,
        resolution: str | None = None,
        actor: str | None = None,
        reason: str | None = None,
    ) -> bool:
        now = _now_iso()
        payload = {
            'status': status,
            'updated_at': now,
            'reviewed_by': actor,
            'reviewed_at': now,
            'reason': reason,
        }

        if status == 'promoted':
            payload.update({'promoted_by': actor, 'promoted_at': now})

        if resolution is not None:
            payload['resolution'] = resolution

        set_clause = ', '.join([f"{k} = ?" for k in payload.keys()])
        params = list(payload.values()) + [candidate_id]

        updated = self.conn.execute(
            f"""
            UPDATE candidates
               SET {set_clause}
             WHERE uuid = ?
            """,
            params,
        )
        self.conn.commit()
        return updated.rowcount > 0

    def insert_candidate(
        self,
        *,
        uuid: str,
        type: str,
        subject: str,
        predicate: str,
        value: Any,
        conflicting_fact_uuid: str | None = None,
        confidence: float = 0.0,
        status: str = 'quarantine',
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = _now_iso()
        payload = {
            'uuid': uuid,
            'type': type,
            'subject': subject,
            'predicate': predicate,
            'value': json.dumps(value),
            'conflicting_fact_uuid': conflicting_fact_uuid,
            'status': status,
            'confidence': confidence,
            'created_at': now,
            'updated_at': now,
            'metadata_json': json.dumps(metadata or {}, sort_keys=True),
        }
        self.conn.execute(
            '''
            INSERT OR REPLACE INTO candidates(
                uuid, type, subject, predicate, value, conflicting_fact_uuid,
                status, confidence, created_at, updated_at, metadata_json
            ) VALUES (
                :uuid, :type, :subject, :predicate, :value, :conflicting_fact_uuid,
                :status, :confidence, :created_at, :updated_at, :metadata_json
            )
            ''',
            payload,
        )
        self.conn.commit()



def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)

    metadata_json = item.pop('metadata_json', None)
    if metadata_json:
        try:
            item['metadata'] = json.loads(metadata_json)
        except (json.JSONDecodeError, TypeError):
            item['metadata'] = None
    else:
        item['metadata'] = None

    raw_value = item.get('value')
    if isinstance(raw_value, str):
        try:
            item['value'] = json.loads(raw_value)
        except json.JSONDecodeError:
            pass
    return item


async def list_candidates(
    status: str | None = None,
    type_filter: str | None = None,
    age_days: int | None = None,
    min_confidence: float | None = None,
    max_age_days: int | None = None,
) -> list[dict[str, Any]]:
    effective_status = (status or 'quarantine').strip().lower()
    if effective_status and effective_status not in VALID_STATUSES:
        return []

    if min_confidence is not None:
        min_confidence = float(min_confidence)
        if not (0.0 <= min_confidence <= 1.0):
            return []

    if age_days is not None and int(age_days) <= 0:
        return []
    if max_age_days is not None and int(max_age_days) <= 0:
        return []

    db = CandidatesDB(_ledger_path())
    return db.list_candidates(
        status=effective_status,
        type_filter=(type_filter or None),
        age_days=age_days,
        min_confidence=min_confidence,
        max_age_days=max_age_days,
    )



def _build_candidate_fact(candidate: dict[str, Any]) -> dict[str, Any]:
    metadata = candidate.get('metadata')
    if metadata is None:
        metadata = {}
    elif not isinstance(metadata, dict):
        metadata = {'metadata': metadata}

    value = candidate['value']
    return {
        'assertion_type': candidate['type'],
        'subject': candidate['subject'],
        'predicate': candidate['predicate'],
        'value': value,
        'scope': 'private',
        'evidence_refs': metadata.get('evidence_refs')
        or [
            {
                'evidence_id': candidate['uuid'],
                'source_key': f"candidate:{candidate['uuid']}",
                'scope': 'private',
            }
        ],
    }


async def promote_candidate(
    candidate_id: str,
    resolution: str,
    reason: str = 'promoted',
) -> dict[str, Any]:
    candidate_id = str(candidate_id or '').strip()
    resolution = str(resolution or '').strip().lower()
    if not candidate_id:
        return {'error': 'candidate_id is required'}
    if resolution not in VALID_RESOLUTIONS:
        return {'error': f'invalid resolution {resolution!r}; expected one of {sorted(VALID_RESOLUTIONS)}'}

    db = CandidatesDB(_ledger_path())
    candidate = db.get_candidate(candidate_id)
    if candidate is None:
        return {'error': f'Candidate not found: {candidate_id}'}
    if candidate['status'] != 'quarantine':
        return {'error': f'Candidate {candidate_id} is already {candidate["status"]}'}

    actor = _actor_id()

    if resolution == 'cancel':
        db.update_status(candidate_id, status='rejected', resolution='cancel', actor=actor, reason=reason)
        updated = db.get_candidate(candidate_id)
        return {
            'candidate': updated,
            'action': 'cancelled',
        }

    conflict_with = candidate.get('conflicting_fact_uuid')
    if resolution == 'parallel':
        conflict_with = None

    fact = _build_candidate_fact(candidate)

    ledger = ChangeLedger(_ledger_path())
    try:
        promotion = ledger.promote_candidate_fact(
            actor_id=actor,
            reason=f'candidate_promotion:{resolution}',
            policy_version='candidate_lifecycle_v1',
            candidate_id=candidate_id,
            fact=fact,
            conflict_with_fact_id=conflict_with,
            allow_parallel=(resolution == 'parallel'),
        )
    except Exception as exc:
        logger.exception('promote_candidate failed')
        return {'error': f'promote_candidate failed: {exc}'}

    db.update_status(
        candidate_id,
        status='promoted',
        resolution=resolution,
        actor=actor,
        reason=reason,
    )
    updated = db.get_candidate(candidate_id)

    return {
        'candidate': updated,
        'promotion': {
            'object_id': promotion.object_id,
            'root_id': promotion.root_id,
            'event_id': promotion.event_id,
            'event_ids': promotion.event_ids,
        },
    }


async def reject_candidate(
    candidate_id: str,
    reason: str = 'rejected',
) -> dict[str, Any]:
    candidate_id = str(candidate_id or '').strip()
    if not candidate_id:
        return {'error': 'candidate_id is required'}

    db = CandidatesDB(_ledger_path())
    candidate = db.get_candidate(candidate_id)
    if candidate is None:
        return {'error': f'Candidate not found: {candidate_id}'}
    if candidate['status'] != 'quarantine':
        return {'error': f'Candidate {candidate_id} is already {candidate["status"]}'}

    actor = _actor_id()
    db.update_status(candidate_id, status='rejected', actor=actor, reason=reason)
    return {'candidate': db.get_candidate(candidate_id), 'action': 'rejected'}


def register_tools(mcp: Any) -> None:
    mcp.tool()(list_candidates)
    mcp.tool()(promote_candidate)
    mcp.tool()(reject_candidate)



