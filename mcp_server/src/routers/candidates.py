"""Candidate lifecycle MCP tools: list, promote, reject."""

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


class CandidatesDB:
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
        self.conn.commit()

    def list_candidates(
        self,
        *,
        status: str,
        type_filter: str | None = None,
        min_confidence: float | None = None,
        max_age_days: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ['status = ?']
        params: list[Any] = [status]

        if type_filter:
            clauses.append('type = ?')
            params.append(type_filter)
        if min_confidence is not None:
            clauses.append('confidence >= ?')
            params.append(float(min_confidence))
        if max_age_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
            clauses.append('created_at >= ?')
            params.append(cutoff.replace(microsecond=0).isoformat().replace('+00:00', 'Z'))

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

    def set_status(self, candidate_id: str, *, status: str, resolution: str | None = None) -> bool:
        now = _now_iso()
        updated = self.conn.execute(
            """
            UPDATE candidates
               SET status = ?,
                   resolution = COALESCE(?, resolution),
                   updated_at = ?
             WHERE uuid = ?
            """,
            (status, resolution, now, candidate_id),
        )
        self.conn.commit()
        return updated.rowcount > 0



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
            item['value'] = raw_value
    return item


async def list_candidates(
    status: str | None = None,
    type_filter: str | None = None,
    min_confidence: float | None = None,
    max_age_days: int | None = None,
) -> dict[str, Any]:
    effective_status = (status or 'quarantine').strip().lower()
    if effective_status not in VALID_STATUSES:
        return {'error': f'invalid status {effective_status!r}; expected one of {sorted(VALID_STATUSES)}'}

    if min_confidence is not None and not (0.0 <= float(min_confidence) <= 1.0):
        return {'error': 'min_confidence must be between 0.0 and 1.0'}
    if max_age_days is not None and int(max_age_days) <= 0:
        return {'error': 'max_age_days must be >= 1'}

    try:
        db = CandidatesDB(_ledger_path())
        rows = db.list_candidates(
            status=effective_status,
            type_filter=(type_filter or None),
            min_confidence=min_confidence,
            max_age_days=max_age_days,
        )
        return {
            'message': f'Found {len(rows)} candidate(s)',
            'status': effective_status,
            'candidates': rows,
        }
    except Exception as e:
        logger.exception('list_candidates failed')
        return {'error': f'list_candidates failed: {e}'}


async def promote_candidate(
    candidate_id: str,
    resolution: str,
) -> dict[str, Any]:
    candidate_id = str(candidate_id or '').strip()
    resolution = str(resolution or '').strip().lower()
    if not candidate_id:
        return {'error': 'candidate_id is required'}
    if resolution not in VALID_RESOLUTIONS:
        return {'error': f'invalid resolution {resolution!r}; expected one of {sorted(VALID_RESOLUTIONS)}'}

    try:
        db = CandidatesDB(_ledger_path())
        candidate = db.get_candidate(candidate_id)
        if candidate is None:
            return {'error': f'Candidate not found: {candidate_id}'}
        if candidate['status'] != 'quarantine':
            return {'error': f'Candidate {candidate_id} is already {candidate["status"]}'}

        if resolution == 'cancel':
            db.set_status(candidate_id, status='rejected', resolution='cancel')
            return {
                'message': f'Candidate {candidate_id} cancelled',
                'candidate_id': candidate_id,
                'status': 'rejected',
                'resolution': 'cancel',
            }

        fact = {
            'assertion_type': candidate['type'],
            'subject': candidate['subject'],
            'predicate': candidate['predicate'],
            'value': candidate['value'],
            'scope': 'private',
            'evidence_refs': [
                {
                    'evidence_id': candidate_id,
                    'source_key': f'candidate:{candidate_id}',
                    'scope': 'private',
                }
            ],
        }

        conflict_with = candidate.get('conflicting_fact_uuid')
        if resolution == 'parallel':
            conflict_with = None

        ledger = ChangeLedger(_ledger_path())
        promotion = ledger.promote_candidate_fact(
            actor_id='candidate_lifecycle',
            reason=f'candidate_promotion:{resolution}',
            policy_version='candidate_lifecycle_v1',
            candidate_id=candidate_id,
            fact=fact,
            conflict_with_fact_id=conflict_with,
        )

        db.set_status(candidate_id, status='promoted', resolution=resolution)
        return {
            'message': f'Candidate {candidate_id} promoted',
            'candidate_id': candidate_id,
            'status': 'promoted',
            'resolution': resolution,
            'object_id': promotion.object_id,
            'root_id': promotion.root_id,
            'event_ids': promotion.event_ids,
        }
    except Exception as e:
        logger.exception('promote_candidate failed')
        return {'error': f'promote_candidate failed: {e}'}


async def reject_candidate(candidate_id: str) -> dict[str, Any]:
    candidate_id = str(candidate_id or '').strip()
    if not candidate_id:
        return {'error': 'candidate_id is required'}

    try:
        db = CandidatesDB(_ledger_path())
        candidate = db.get_candidate(candidate_id)
        if candidate is None:
            return {'error': f'Candidate not found: {candidate_id}'}
        if candidate['status'] != 'quarantine':
            return {'error': f'Candidate {candidate_id} is already {candidate["status"]}'}

        db.set_status(candidate_id, status='rejected')
        return {
            'message': f'Candidate {candidate_id} rejected',
            'candidate_id': candidate_id,
            'status': 'rejected',
        }
    except Exception as e:
        logger.exception('reject_candidate failed')
        return {'error': f'reject_candidate failed: {e}'}


def register_tools(mcp: Any) -> None:
    mcp.tool()(list_candidates)
    mcp.tool()(promote_candidate)
    mcp.tool()(reject_candidate)
