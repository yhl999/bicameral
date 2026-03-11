"""Episode and procedure retrieval tools backed by typed ChangeLedger data."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from ..models.typed_memory import Episode, Procedure
    from ..services.change_ledger import DB_PATH_DEFAULT, ChangeLedger
    from ..services.procedure_service import ProcedureService
except ImportError:  # pragma: no cover - top-level import fallback
    from models.typed_memory import Episode, Procedure  # type: ignore[no-redef]
    from services.change_ledger import DB_PATH_DEFAULT, ChangeLedger  # type: ignore[no-redef]
    from services.procedure_service import ProcedureService  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

_LEDGER: ChangeLedger | None = None
_PROCEDURES: ProcedureService | None = None


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


def _get_procedure_service() -> ProcedureService:
    global _PROCEDURES
    if _PROCEDURES is None:
        _PROCEDURES = ProcedureService(_get_ledger())
    return _PROCEDURES


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace('Z', '+00:00'))
    except ValueError:
        return None


def _episode_in_range(episode: Episode, time_range: dict[str, Any] | None) -> bool:
    if not time_range:
        return True

    start = _parse_iso(str(time_range.get('start') or ''))
    end = _parse_iso(str(time_range.get('end') or ''))
    episode_start = _parse_iso(episode.started_at) or _parse_iso(episode.created_at)
    episode_end = _parse_iso(episode.ended_at) or episode_start

    if start is not None and (episode_start is None or episode_start < start):
        return False
    if end is not None and (episode_end is None or episode_end > end):
        return False
    return True


async def search_episodes(
    query: str,
    time_range: dict[str, Any] | None = None,
    include_history: bool = False,
) -> dict[str, Any]:
    """Search typed Episode objects from ChangeLedger."""
    q = (query or '').strip().lower()

    try:
        ledger = _get_ledger()
        rows = ledger.conn.execute(
            """
            SELECT DISTINCT root_id
              FROM change_events
             WHERE object_type = 'episode'
               AND root_id IS NOT NULL
            """
        ).fetchall()

        episodes: list[Episode] = []
        for row in rows:
            root_id = str(row['root_id'])
            if include_history:
                for obj in ledger.materialize_lineage(root_id):
                    if isinstance(obj, Episode):
                        episodes.append(obj)
            else:
                current = ledger.current_object(root_id)
                if isinstance(current, Episode):
                    episodes.append(current)

        filtered: list[Episode] = []
        for episode in episodes:
            blob = f"{episode.title or ''}\n{episode.summary or ''}".lower()
            if q and q not in blob:
                continue
            if not _episode_in_range(episode, time_range):
                continue
            filtered.append(episode)

        filtered.sort(key=lambda item: item.created_at or '', reverse=True)
        return {
            'message': f'Found {len(filtered)} episode(s)',
            'query': query,
            'episodes': [episode.model_dump(mode='json') for episode in filtered],
        }
    except Exception as e:
        logger.exception('search_episodes failed')
        return {'error': f'search_episodes failed: {e}'}


async def get_episode(episode_id: str) -> dict[str, Any]:
    try:
        episode = _get_ledger().materialize_object(str(episode_id or '').strip())
        if not isinstance(episode, Episode):
            return {'error': f'No episode with id {episode_id}'}
        return episode.model_dump(mode='json')
    except Exception as e:
        logger.exception('get_episode failed')
        return {'error': f'get_episode failed: {e}'}


async def search_procedures(
    query: str,
    include_all: bool = False,
) -> dict[str, Any]:
    q = (query or '').strip().lower()
    try:
        service = _get_procedure_service()
        procedures = service.list_current_procedures(include_proposed=include_all)

        filtered = []
        for procedure in procedures:
            blob = f"{procedure.name}\n{procedure.trigger}".lower()
            if q and q not in blob:
                continue
            filtered.append(procedure)

        filtered.sort(key=lambda item: item.success_count, reverse=True)
        return {
            'message': f'Found {len(filtered)} procedure(s)',
            'query': query,
            'procedures': [item.model_dump(mode='json') for item in filtered],
        }
    except Exception as e:
        logger.exception('search_procedures failed')
        return {'error': f'search_procedures failed: {e}'}


async def get_procedure(procedure_id: str) -> dict[str, Any]:
    pid = str(procedure_id or '').strip()
    try:
        service = _get_procedure_service()
        for procedure in service.list_current_procedures(include_proposed=True):
            if procedure.object_id == pid:
                return procedure.model_dump(mode='json')
        return {'error': f'No procedure with id {procedure_id}'}
    except Exception as e:
        logger.exception('get_procedure failed')
        return {'error': f'get_procedure failed: {e}'}


async def record_procedure_success(
    procedure_id: str,
    episode_id: str,
    evidence_refs: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        result = _get_procedure_service().record_feedback(
            procedure_id,
            outcome='success',
            actor_id='procedure_feedback',
            episode_id=episode_id,
            evidence_refs=evidence_refs,
        )
        return {
            'message': 'procedure success recorded',
            'procedure': result.procedure.model_dump(mode='json'),
            'auto_promoted': result.auto_promoted,
        }
    except Exception as e:
        logger.exception('record_procedure_success failed')
        return {'error': f'record_procedure_success failed: {e}'}


async def record_procedure_failure(
    procedure_id: str,
    episode_id: str,
    evidence_refs: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        result = _get_procedure_service().record_feedback(
            procedure_id,
            outcome='failure',
            actor_id='procedure_feedback',
            episode_id=episode_id,
            evidence_refs=evidence_refs,
        )
        return {
            'message': 'procedure failure recorded',
            'procedure': result.procedure.model_dump(mode='json'),
            'auto_promoted': result.auto_promoted,
        }
    except Exception as e:
        logger.exception('record_procedure_failure failed')
        return {'error': f'record_procedure_failure failed: {e}'}


def register_tools(mcp: Any) -> None:
    mcp.tool()(search_episodes)
    mcp.tool()(get_episode)
    mcp.tool()(search_procedures)
    mcp.tool()(get_procedure)
    mcp.tool()(record_procedure_success)
    mcp.tool()(record_procedure_failure)
