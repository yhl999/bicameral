"""Candidate lifecycle MCP tools.

These tools expose quarantine/promotion/rejection operations against a persisted
`candidates` table and promote accepted items into the main change ledger.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import suppress
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
_POLICY_VERSION = 'candidate_lifecycle_v1'

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
"""


CANDIDATES_INDEXES_SQL = """
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
    return (
        os.getenv('BICAMERAL_MCP_ACTOR_ID', os.getenv('BICAMERAL_ACTOR_ID', 'system')).strip()
        or 'system'
    )


def _normalize_reason(reason: str | None) -> str | None:
    normalized = str(reason or '').strip()
    return normalized or None


def _connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA foreign_keys=ON')
    conn.execute('PRAGMA busy_timeout=5000')
    return conn


def _ensure_columns(conn: sqlite3.Connection) -> None:
    existing = {
        row['name']
        for row in conn.execute("PRAGMA table_info('candidates')").fetchall()
    }

    missing_to_sql = {
        'conflicting_fact_uuid': 'ALTER TABLE candidates ADD COLUMN conflicting_fact_uuid TEXT',
        'resolution': 'ALTER TABLE candidates ADD COLUMN resolution TEXT',
        'confidence': 'ALTER TABLE candidates ADD COLUMN confidence REAL NOT NULL DEFAULT 0.0',
        'reviewed_at': 'ALTER TABLE candidates ADD COLUMN reviewed_at TEXT',
        'reviewed_by': 'ALTER TABLE candidates ADD COLUMN reviewed_by TEXT',
        'promoted_at': 'ALTER TABLE candidates ADD COLUMN promoted_at TEXT',
        'promoted_by': 'ALTER TABLE candidates ADD COLUMN promoted_by TEXT',
        'reason': 'ALTER TABLE candidates ADD COLUMN reason TEXT',
        'metadata_json': 'ALTER TABLE candidates ADD COLUMN metadata_json TEXT',
    }

    for col, statement in missing_to_sql.items():
        if col not in existing:
            conn.execute(statement)

    conn.execute(
        "UPDATE candidates SET status = 'quarantine' WHERE status = 'pending'"
    )
    conn.commit()


def _resolution_from_promotion_reason(reason: str | None) -> str | None:
    if not reason:
        return None
    prefix = 'candidate_promotion:'
    if not reason.startswith(prefix):
        return None
    resolution = reason[len(prefix) :].strip().lower()
    if resolution in VALID_RESOLUTIONS:
        return resolution
    return None


def _candidate_status_error(candidate_id: str, candidate_status: str) -> dict[str, Any]:
    return {'error': f'Candidate {candidate_id} is already {candidate_status}'}


class CandidatesDB:
    """Persistence for candidate quarantine rows."""

    def __init__(self, conn_or_path: sqlite3.Connection | str | Path = DB_PATH_DEFAULT):
        if isinstance(conn_or_path, sqlite3.Connection):
            self.conn = conn_or_path
        else:
            db_path = Path(conn_or_path)
            self.conn = _connect_db(db_path)
        self.conn.row_factory = sqlite3.Row
        self.ensure_schema()

    def ensure_schema(self) -> None:
        self.conn.executescript(CANDIDATES_SCHEMA_SQL)
        _ensure_columns(self.conn)
        self.conn.executescript(CANDIDATES_INDEXES_SQL)
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
        expected_current_status: str | None = None,
        commit: bool = True,
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

        set_clause = ', '.join([f'{key} = ?' for key in payload])
        where_clause = 'uuid = ?'
        params: list[Any] = list(payload.values()) + [candidate_id]
        if expected_current_status is not None:
            where_clause += ' AND status = ?'
            params.append(expected_current_status)

        updated = self.conn.execute(
            f"""
            UPDATE candidates
               SET {set_clause}
             WHERE {where_clause}
            """,
            params,
        )
        if commit:
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
        with suppress(json.JSONDecodeError):
            item['value'] = json.loads(raw_value)
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
        try:
            min_confidence = float(min_confidence)
        except (TypeError, ValueError):
            return []
        if not (0.0 <= min_confidence <= 1.0):
            return []

    for value in (age_days, max_age_days):
        if value is None:
            continue
        try:
            if int(value) <= 0:
                return []
        except (TypeError, ValueError):
            return []

    conn = _connect_db(_ledger_path())
    try:
        db = CandidatesDB(conn)
        return db.list_candidates(
            status=effective_status,
            type_filter=(type_filter or None),
            age_days=age_days,
            min_confidence=min_confidence,
            max_age_days=max_age_days,
        )
    finally:
        conn.close()


def _build_candidate_fact(candidate: dict[str, Any]) -> dict[str, Any]:
    metadata = candidate.get('metadata')
    if metadata is None:
        metadata = {}
    elif not isinstance(metadata, dict):
        metadata = {'metadata': metadata}

    return {
        'assertion_type': candidate['type'],
        'subject': candidate['subject'],
        'predicate': candidate['predicate'],
        'value': candidate['value'],
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
    reason: str | None = None,
) -> dict[str, Any]:
    candidate_id = str(candidate_id or '').strip()
    resolution = str(resolution or '').strip().lower()
    reason = _normalize_reason(reason)

    if not candidate_id:
        return {'error': 'candidate_id is required'}
    if resolution not in VALID_RESOLUTIONS:
        return {'error': f'invalid resolution {resolution!r}; expected one of {sorted(VALID_RESOLUTIONS)}'}

    conn = _connect_db(_ledger_path())
    db = CandidatesDB(conn)
    ledger = ChangeLedger(conn)
    actor = _actor_id()

    try:
        conn.execute('BEGIN IMMEDIATE')

        candidate = db.get_candidate(candidate_id)
        if candidate is None:
            conn.rollback()
            return {'error': f'Candidate not found: {candidate_id}'}

        if candidate['status'] != 'quarantine':
            conn.rollback()
            return _candidate_status_error(candidate_id, candidate['status'])

        existing_promotion = ledger.promotion_event_for_candidate(candidate_id)
        if existing_promotion is not None:
            reconciled_resolution = _resolution_from_promotion_reason(existing_promotion.reason)
            db.update_status(
                candidate_id,
                status='promoted',
                resolution=reconciled_resolution,
                actor=actor,
                reason=reason or 'candidate_promoted_reconciled',
                expected_current_status='quarantine',
                commit=False,
            )
            conn.commit()
            return {
                'candidate': db.get_candidate(candidate_id),
                'promotion': {
                    'object_id': existing_promotion.object_id,
                    'root_id': existing_promotion.root_id,
                    'event_id': existing_promotion.event_id,
                    'event_ids': [existing_promotion.event_id],
                    'reconciled': True,
                },
            }

        if resolution == 'cancel':
            transitioned = db.update_status(
                candidate_id,
                status='rejected',
                resolution='cancel',
                actor=actor,
                reason=reason or 'candidate_cancelled',
                expected_current_status='quarantine',
                commit=False,
            )
            if not transitioned:
                conn.rollback()
                candidate = db.get_candidate(candidate_id)
                if candidate is None:
                    return {'error': f'Candidate not found: {candidate_id}'}
                return _candidate_status_error(candidate_id, candidate['status'])

            conn.commit()
            return {
                'candidate': db.get_candidate(candidate_id),
                'action': 'cancelled',
            }

        conflict_with = candidate.get('conflicting_fact_uuid')
        if resolution == 'parallel':
            conflict_with = None

        promotion = ledger.promote_candidate_fact(
            actor_id=actor,
            reason=f'candidate_promotion:{resolution}',
            policy_version=_POLICY_VERSION,
            candidate_id=candidate_id,
            fact=_build_candidate_fact(candidate),
            conflict_with_fact_id=conflict_with,
            allow_parallel=(resolution == 'parallel'),
            manage_transaction=False,
        )

        transitioned = db.update_status(
            candidate_id,
            status='promoted',
            resolution=resolution,
            actor=actor,
            reason=reason or f'candidate_promoted_{resolution}',
            expected_current_status='quarantine',
            commit=False,
        )
        if not transitioned:
            raise RuntimeError('candidate status transition failed after ledger promotion')

        conn.commit()
        return {
            'candidate': db.get_candidate(candidate_id),
            'promotion': {
                'object_id': promotion.object_id,
                'root_id': promotion.root_id,
                'event_id': promotion.event_id,
                'event_ids': promotion.event_ids,
            },
        }
    except Exception as exc:
        if conn.in_transaction:
            conn.rollback()
        logger.exception('promote_candidate failed')
        return {'error': f'promote_candidate failed: {exc}'}
    finally:
        conn.close()


async def reject_candidate(
    candidate_id: str,
    reason: str | None = None,
) -> dict[str, Any]:
    candidate_id = str(candidate_id or '').strip()
    reason = _normalize_reason(reason)
    if not candidate_id:
        return {'error': 'candidate_id is required'}

    conn = _connect_db(_ledger_path())
    db = CandidatesDB(conn)
    ledger = ChangeLedger(conn)
    actor = _actor_id()

    try:
        conn.execute('BEGIN IMMEDIATE')

        candidate = db.get_candidate(candidate_id)
        if candidate is None:
            conn.rollback()
            return {'error': f'Candidate not found: {candidate_id}'}

        if candidate['status'] != 'quarantine':
            conn.rollback()
            return _candidate_status_error(candidate_id, candidate['status'])

        existing_promotion = ledger.promotion_event_for_candidate(candidate_id)
        if existing_promotion is not None:
            db.update_status(
                candidate_id,
                status='promoted',
                resolution=_resolution_from_promotion_reason(existing_promotion.reason),
                actor=actor,
                reason='candidate_promoted_reconciled',
                expected_current_status='quarantine',
                commit=False,
            )
            conn.commit()
            return _candidate_status_error(candidate_id, 'promoted')

        transitioned = db.update_status(
            candidate_id,
            status='rejected',
            actor=actor,
            reason=reason or 'candidate_rejected',
            expected_current_status='quarantine',
            commit=False,
        )
        if not transitioned:
            conn.rollback()
            candidate = db.get_candidate(candidate_id)
            if candidate is None:
                return {'error': f'Candidate not found: {candidate_id}'}
            return _candidate_status_error(candidate_id, candidate['status'])

        conn.commit()
        return {'candidate': db.get_candidate(candidate_id), 'action': 'rejected'}
    except Exception as exc:
        if conn.in_transaction:
            conn.rollback()
        logger.exception('reject_candidate failed')
        return {'error': f'reject_candidate failed: {exc}'}
    finally:
        conn.close()


def register_tools(mcp: Any) -> dict[str, Any]:
    mcp.tool()(list_candidates)
    mcp.tool()(promote_candidate)
    mcp.tool()(reject_candidate)
    return {
        'list_candidates': list_candidates,
        'promote_candidate': promote_candidate,
        'reject_candidate': reject_candidate,
    }
