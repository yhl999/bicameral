"""Memory router — remember_fact, get_current_state, get_history.

Owned by: Exec 1 (remember_fact) and Exec 2 (get_current_state, get_history).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from ..models.typed_memory import StateFact
    from ..services.change_ledger import DB_PATH_DEFAULT, ChangeLedger
except ImportError:  # pragma: no cover - top-level import fallback
    from models.typed_memory import StateFact  # type: ignore[no-redef]
    from services.change_ledger import DB_PATH_DEFAULT, ChangeLedger  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

_LEDGER: ChangeLedger | None = None


def _ledger_path() -> Path:
    override = os.getenv('BICAMERAL_CHANGE_LEDGER_PATH', '').strip()
    return Path(override) if override else Path(DB_PATH_DEFAULT)


def _get_ledger() -> ChangeLedger:
    global _LEDGER
    if _LEDGER is None:
        path = _ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        _LEDGER = ChangeLedger(path)
    return _LEDGER


def _match_fact(
    fact: StateFact,
    *,
    subject: str,
    predicate: str | None,
) -> bool:
    if fact.subject != subject:
        return False
    if predicate is not None and fact.predicate != predicate:
        return False
    return True


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace('Z', '+00:00'))
    except ValueError:
        return None


def _as_of_current_state(
    ledger: ChangeLedger,
    *,
    as_of: datetime,
    include_invalidated: bool,
) -> list[StateFact]:
    rows = ledger.conn.execute(
        """
        SELECT DISTINCT root_id
          FROM change_events
         WHERE object_type = 'state_fact'
           AND root_id IS NOT NULL
        """
    ).fetchall()

    selected: list[StateFact] = []
    for row in rows:
        lineage = ledger.materialize_lineage(str(row['root_id']))
        best: StateFact | None = None
        for obj in lineage:
            if not isinstance(obj, StateFact):
                continue
            valid_at = _parse_iso(obj.valid_at) or _parse_iso(obj.created_at)
            if valid_at is None or valid_at > as_of:
                continue
            if best is None:
                best = obj
                continue
            best_valid_at = _parse_iso(best.valid_at) or _parse_iso(best.created_at)
            if best_valid_at is None or (valid_at, obj.version) >= (best_valid_at, best.version):
                best = obj

        if best is None:
            continue

        if not include_invalidated and best.invalid_at is not None:
            invalid_at = _parse_iso(best.invalid_at)
            if invalid_at is not None and invalid_at <= as_of:
                continue

        selected.append(best)

    return selected


async def remember_fact(
    text: str,
    hint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a typed fact to the memory ledger.

    Stub in Exec 2 branch; implemented in Exec 1 branch.
    """
    logger.debug('remember_fact stub called with text=%r', text[:80] if text else '')
    return {
        'status': 'stub',
        'message': 'remember_fact not yet implemented (Phase 0 stub)',
        'text': text,
    }


async def get_current_state(
    subject: str,
    predicate: str | None = None,
    include_invalidated: bool = False,
    as_of: str | None = None,
) -> dict[str, Any]:
    """Query current state facts from the ledger.

    Returns currently-active facts by default; when ``as_of`` is provided,
    computes state as-of a historical timestamp.
    """
    subject = str(subject or '').strip()
    if not subject:
        return {'error': 'subject is required'}

    try:
        ledger = _get_ledger()

        if as_of:
            as_of_dt = _parse_iso(as_of)
            if as_of_dt is None:
                return {'error': 'as_of must be an ISO 8601 timestamp'}
            pool = _as_of_current_state(
                ledger,
                as_of=as_of_dt,
                include_invalidated=include_invalidated,
            )
        elif include_invalidated:
            # Include latest version for each root, regardless of currentness.
            rows = ledger.conn.execute(
                """
                SELECT DISTINCT root_id
                  FROM change_events
                 WHERE object_type = 'state_fact'
                   AND root_id IS NOT NULL
                """
            ).fetchall()
            pool = []
            for row in rows:
                lineage = ledger.materialize_lineage(str(row['root_id']))
                latest = None
                for obj in lineage:
                    if isinstance(obj, StateFact):
                        latest = obj
                if latest is not None:
                    pool.append(latest)
        else:
            pool = ledger.current_state_facts()

        matches = [fact for fact in pool if _match_fact(fact, subject=subject, predicate=predicate)]
        matches.sort(key=lambda item: (item.valid_at or item.created_at or ''), reverse=True)

        return {
            'message': f'Found {len(matches)} current fact(s)',
            'subject': subject,
            'predicate': predicate,
            'include_invalidated': include_invalidated,
            'as_of': as_of,
            'facts': [fact.model_dump(mode='json') for fact in matches],
        }
    except Exception as e:
        logger.exception('get_current_state failed')
        return {'error': f'get_current_state failed: {e}'}


async def get_history(
    subject: str,
    predicate: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Retrieve full version history for subject/predicate facts."""
    subject = str(subject or '').strip()
    if not subject:
        return {'error': 'subject is required'}

    try:
        requested_limit = int(limit)
    except (TypeError, ValueError):
        return {'error': 'limit must be an integer'}

    if requested_limit <= 0:
        return {'error': 'limit must be >= 1'}

    effective_limit = min(requested_limit, 500)

    try:
        ledger = _get_ledger()
        rows = ledger.conn.execute(
            """
            SELECT DISTINCT root_id
              FROM change_events
             WHERE object_type = 'state_fact'
               AND root_id IS NOT NULL
            """
        ).fetchall()

        history: list[StateFact] = []
        for row in rows:
            lineage = ledger.materialize_lineage(str(row['root_id']))
            for obj in lineage:
                if not isinstance(obj, StateFact):
                    continue
                if not _match_fact(obj, subject=subject, predicate=predicate):
                    continue
                history.append(obj)

        history.sort(
            key=lambda item: (
                item.valid_at or item.created_at or '',
                item.version,
            ),
            reverse=True,
        )
        history = history[:effective_limit]

        return {
            'message': f'Found {len(history)} historical fact(s)',
            'subject': subject,
            'predicate': predicate,
            'limit': effective_limit,
            'history': [item.model_dump(mode='json') for item in history],
        }
    except Exception as e:
        logger.exception('get_history failed')
        return {'error': f'get_history failed: {e}'}


def register_tools(mcp: Any) -> None:
    """Register all memory router tools with the MCP server instance."""
    mcp.tool()(remember_fact)
    mcp.tool()(get_current_state)
    mcp.tool()(get_history)
