"""Memory router — remember_fact, get_current_state, get_history.

Owned by: Exec 1 (remember_fact) and Exec 2 (get_current_state, get_history).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from ..models.typed_memory import EvidenceRef, StateFact
    from ..services.change_ledger import DB_PATH_DEFAULT, ChangeLedger
    from ..services.neo4j_materialization import Neo4jMaterializationService
except ImportError:  # pragma: no cover - top-level import fallback
    from models.typed_memory import EvidenceRef, StateFact  # type: ignore[no-redef]
    from services.change_ledger import DB_PATH_DEFAULT, ChangeLedger  # type: ignore[no-redef]
    from services.neo4j_materialization import Neo4jMaterializationService  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

_LEDGER: ChangeLedger | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


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


def _build_evidence_ref(*, text: str, source_key: str = 'remember_fact:manual') -> EvidenceRef:
    event_seed = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    return EvidenceRef.model_validate(
        {
            'kind': 'event_log',
            'source_system': 'bicameral:mcp',
            'locator': {
                'system': 'remember_fact',
                'stream': source_key,
                'event_id': f'rf:{event_seed}:{secrets.token_hex(4)}',
            },
            'title': 'remember_fact input',
            'snippet': text[:240],
            'observed_at': _now_iso(),
            'retrieved_at': _now_iso(),
        }
    )


def _extract_fact(text: str, hint: dict[str, Any] | None) -> tuple[str, str, str, Any, str]:
    hint = hint or {}

    # Structured input path (JSON text)
    raw = (text or '').strip()
    if raw.startswith('{'):
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                subject = str(payload.get('subject') or hint.get('subject') or '').strip()
                predicate = str(payload.get('predicate') or hint.get('predicate') or '').strip()
                value = payload.get('value', hint.get('value'))
                fact_type = str(payload.get('fact_type') or hint.get('fact_type') or 'world_state').strip().lower()
                scope = str(payload.get('scope') or hint.get('scope') or 'private').strip()
                if subject and predicate:
                    return fact_type, subject, predicate, value, scope
        except json.JSONDecodeError:
            pass

    fact_type = str(hint.get('fact_type') or 'world_state').strip().lower() or 'world_state'
    subject = str(hint.get('subject') or '').strip()
    predicate = str(hint.get('predicate') or '').strip()
    scope = str(hint.get('scope') or 'private').strip() or 'private'

    if subject and predicate:
        value = hint.get('value', raw)
        return fact_type, subject, predicate, value, scope

    # Lightweight NL parsing: "subject predicate value"
    normalized = raw
    if ' prefers ' in normalized:
        left, _, right = normalized.partition(' prefers ')
        return 'preference', left.strip() or 'user', 'prefers', right.strip(), scope
    if ' likes ' in normalized:
        left, _, right = normalized.partition(' likes ')
        return 'preference', left.strip() or 'user', 'likes', right.strip(), scope
    if ' is ' in normalized:
        left, _, right = normalized.partition(' is ')
        return fact_type, left.strip() or 'subject', 'is', right.strip(), scope
    if '=' in normalized:
        left, _, right = normalized.partition('=')
        return fact_type, left.strip() or 'subject', predicate or 'equals', right.strip(), scope

    # Fallback: store as world_state note tied to user
    return fact_type, subject or 'user', predicate or 'note', raw, scope


def _build_state_fact(
    *,
    fact_type: str,
    subject: str,
    predicate: str,
    value: Any,
    scope: str,
    text: str,
    hint: dict[str, Any] | None,
) -> StateFact:
    created_at = _now_iso()
    object_id = f'fact_{secrets.token_hex(12)}'
    evidence = _build_evidence_ref(text=text, source_key=str((hint or {}).get('source_key') or 'remember_fact:manual'))
    return StateFact.model_validate(
        {
            'object_id': object_id,
            'root_id': object_id,
            'version': 1,
            'fact_type': fact_type if fact_type else 'world_state',
            'subject': subject,
            'predicate': predicate,
            'value': value,
            'scope': scope,
            'policy_scope': scope,
            'visibility_scope': scope,
            'evidence_refs': [evidence],
            'created_at': created_at,
            'valid_at': created_at,
            'extractor_version': 'remember_fact_v1',
            'promotion_status': 'promoted',
            'risk_level': str((hint or {}).get('risk_level') or 'medium'),
            'source_lane': str((hint or {}).get('source_lane') or 'private'),
            'source_key': str((hint or {}).get('source_key') or 'remember_fact'),
        }
    )


async def remember_fact(
    text: str,
    hint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a typed fact to the memory ledger.

    Conflict policy:
    - default: return conflict_dialog (no write)
    - hint.conflict_resolution == "supersede": write supersede event
    - hint.conflict_resolution == "parallel": write new assert event
    - hint.conflict_resolution == "cancel": skip write
    """

    if not str(text or '').strip():
        return {'error': 'text is required'}

    try:
        fact_type, subject, predicate, value, scope = _extract_fact(text, hint)
        proposed = _build_state_fact(
            fact_type=fact_type,
            subject=subject,
            predicate=predicate,
            value=value,
            scope=scope,
            text=text,
            hint=hint,
        )

        ledger = _get_ledger()

        conflict_with: StateFact | None = None
        for current in ledger.current_state_facts():
            if current.conflict_set == proposed.conflict_set:
                conflict_with = current
                break

        resolution = str((hint or {}).get('conflict_resolution') or '').strip().lower() or 'ask'
        if conflict_with is not None and resolution == 'ask':
            return {
                'status': 'conflict',
                'message': 'Conflict detected; choose supersede|parallel|cancel',
                'conflict_dialog': {
                    'candidate_fact': proposed.model_dump(mode='json'),
                    'existing_fact': conflict_with.model_dump(mode='json'),
                    'options': ['supersede', 'parallel', 'cancel'],
                    'default': 'cancel',
                },
            }

        if conflict_with is not None and resolution == 'cancel':
            return {
                'status': 'cancelled',
                'message': 'Conflict detected; write cancelled by policy',
                'candidate_fact': proposed.model_dump(mode='json'),
                'existing_fact_id': conflict_with.object_id,
            }

        event_type = 'assert'
        target_object_id: str | None = None
        if conflict_with is not None and resolution == 'supersede':
            event_type = 'supersede'
            target_object_id = conflict_with.object_id
            proposed = proposed.model_copy(
                update={
                    'root_id': conflict_with.root_id,
                    'parent_id': conflict_with.object_id,
                    'version': conflict_with.version + 1,
                }
            )

        # parallel keeps event_type assert with new root/object
        row = ledger.append_event(
            event_type,
            actor_id=str((hint or {}).get('actor_id') or 'remember_fact'),
            reason=str((hint or {}).get('reason') or 'remember_fact_write'),
            payload=proposed,
            object_id=proposed.object_id,
            object_type=proposed.object_type,
            root_id=proposed.root_id,
            target_object_id=target_object_id,
            parent_id=proposed.parent_id,
        )

        neo4j_result: dict[str, Any] | None = None
        neo4j_error: str | None = None
        disable_materialization = os.getenv('BICAMERAL_DISABLE_NEO4J_MATERIALIZATION', '').strip() == '1'
        if not disable_materialization:
            try:
                neo4j_result = await Neo4jMaterializationService().materialize_state_fact(
                    proposed,
                    supersedes_object_id=target_object_id,
                )
            except Exception as e:  # fail-open by design
                neo4j_error = f'{type(e).__name__}: {e}'
                logger.warning('Neo4j materialization failed-open: %s', neo4j_error)

        return {
            'status': 'ok',
            'message': 'Fact remembered',
            'typed_fact': proposed.model_dump(mode='json'),
            'event': {
                'event_id': row.event_id,
                'event_type': row.event_type,
                'recorded_at': row.recorded_at,
            },
            'conflict_resolution': 'supersede' if event_type == 'supersede' else 'parallel' if conflict_with is not None else 'none',
            'neo4j': {
                'materialized': neo4j_result is not None,
                'result': neo4j_result,
                'error': neo4j_error,
                'disabled': disable_materialization,
            },
        }
    except Exception as e:
        logger.exception('remember_fact failed')
        return {'error': f'remember_fact failed: {e}'}


async def get_current_state(
    subject: str,
    predicate: str | None = None,
) -> dict[str, Any]:
    """Stub for Exec 2."""
    logger.debug('get_current_state stub called for subject=%r', subject)
    return {
        'status': 'stub',
        'message': 'get_current_state not yet implemented (Phase 0 stub)',
        'subject': subject,
        'predicate': predicate,
        'facts': [],
    }


async def get_history(
    subject: str,
    predicate: str | None = None,
) -> dict[str, Any]:
    """Stub for Exec 2."""
    logger.debug('get_history stub called for subject=%r', subject)
    return {
        'status': 'stub',
        'message': 'get_history not yet implemented (Phase 0 stub)',
        'subject': subject,
        'predicate': predicate,
        'history': [],
    }


def register_tools(mcp: Any) -> None:
    """Register all memory router tools with the MCP server instance."""
    mcp.tool()(remember_fact)
    mcp.tool()(get_current_state)
    mcp.tool()(get_history)
