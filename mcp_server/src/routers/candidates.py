"""Candidate lifecycle tools for remember_fact conflict quarantine.

This branch intentionally supports the minimal follow-up path required by
`remember_fact`: list pending candidates, supersede the current fact with an
approved candidate, or reject/cancel the candidate. "parallel" promotion is not
implemented here and is rejected explicitly so the contract stays honest.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from ..models.typed_memory import StateFact
    from ..services.candidate_store import CandidateStore
    from ..services.change_ledger import ChangeLedger
except ImportError:  # pragma: no cover - top-level import fallback
    from models.typed_memory import StateFact  # type: ignore[no-redef]
    from services.candidate_store import CandidateStore  # type: ignore[no-redef]
    from services.change_ledger import ChangeLedger  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

VALID_STATUSES = frozenset({'pending', 'promoted', 'rejected'})
VALID_PROMOTION_RESOLUTIONS = frozenset({'supersede', 'cancel'})
DEFAULT_POLICY_VERSION = 'remember_fact_conflict_v1'

# Lazy singletons so importing this router does not touch SQLite.
_change_ledger: ChangeLedger | None = None
_candidate_store: CandidateStore | None = None


def _memory_router():
    try:
        from . import memory as memory_router
    except ImportError:  # pragma: no cover - top-level import fallback
        import memory as memory_router  # type: ignore[no-redef]
    return memory_router


def _get_change_ledger() -> ChangeLedger:
    global _change_ledger
    if _change_ledger is None:
        _change_ledger = _memory_router()._get_change_ledger()
    return _change_ledger


def _get_candidate_store() -> CandidateStore:
    global _candidate_store
    if _candidate_store is None:
        _candidate_store = _memory_router()._get_candidate_store()
    return _candidate_store


def _error(message: str, *, error_type: str = 'validation_error') -> dict[str, Any]:
    return {
        'status': 'error',
        'error_type': error_type,
        'message': message,
    }


def _normalize_resolution(resolution: str | None) -> str:
    return str(resolution or '').strip().lower()


def _candidate_to_fact_input(candidate: dict[str, Any]) -> dict[str, Any]:
    memory = _memory_router()
    raw_hint = candidate.get('raw_hint') if isinstance(candidate.get('raw_hint'), dict) else {}
    metadata = candidate.get('metadata') if isinstance(candidate.get('metadata'), dict) else {}
    write_context = memory._resolve_write_context(raw_hint)
    evidence_refs = metadata.get('evidence_refs')
    if not evidence_refs:
        source_key = str(candidate.get('source') or write_context['source'] or memory.DEFAULT_SOURCE)
        evidence_refs = [ref.model_dump(mode='json') for ref in memory._build_evidence_ref(source_key)]

    return {
        'assertion_type': str(candidate.get('fact_type') or '').strip(),
        'subject': str(candidate.get('subject') or '').strip(),
        'predicate': str(candidate.get('predicate') or '').strip(),
        'value': candidate.get('value'),
        'scope': (
            raw_hint.get('scope')
            or raw_hint.get('policy_scope')
            or write_context.get('scope_override')
            or metadata.get('scope')
            or memory.DEFAULT_SCOPE
        ),
        'evidence_refs': evidence_refs,
    }


async def list_candidates(status: str | None = None) -> dict[str, Any]:
    """List quarantined remember_fact candidates."""
    effective_status = str(status or '').strip().lower()
    if effective_status and effective_status not in VALID_STATUSES:
        return _error(
            f'invalid status {effective_status!r}; expected one of {sorted(VALID_STATUSES)}',
        )

    memory = _memory_router()
    candidates = _get_candidate_store().list_candidates(status=effective_status or None)
    return {
        'status': 'ok',
        'candidates': [memory._candidate_payload(candidate) for candidate in candidates],
    }


async def promote_candidate(candidate_id: str, resolution: str) -> dict[str, Any]:
    """Promote a pending candidate into the typed ledger.

    Supported resolutions on this branch:
    - supersede: write the candidate into the ledger and supersede the current fact
    - cancel: mark the candidate rejected without touching the ledger

    The previously-advertised "parallel" path is not implemented here and is
    rejected explicitly rather than pretending otherwise.
    """
    candidate_id = str(candidate_id or '').strip()
    if not candidate_id:
        return _error('candidate_id is required')

    normalized_resolution = _normalize_resolution(resolution)
    if normalized_resolution == 'parallel':
        return _error(
            'parallel resolution is not supported on this branch; use resolution="supersede" or reject_candidate(candidate_id)',
        )
    if normalized_resolution not in VALID_PROMOTION_RESOLUTIONS:
        return _error(
            f'invalid resolution {normalized_resolution!r}; expected one of {sorted(VALID_PROMOTION_RESOLUTIONS)}',
        )

    memory = _memory_router()
    store = _get_candidate_store()
    candidate = store.get_candidate(candidate_id)
    if candidate is None:
        return _error(f'candidate not found: {candidate_id}', error_type='not_found')
    if candidate.get('status') != 'pending':
        return _error(
            f'candidate {candidate_id} is already {candidate.get("status")}',
            error_type='invalid_state',
        )

    if normalized_resolution == 'cancel':
        updated = store.update_candidate_status(candidate_id, 'rejected', resolution='cancel')
        memory._log_audit('candidate_cancel', candidate_id=candidate_id, value=candidate.get('value'), result='ok')
        return {
            'status': 'ok',
            'action': 'cancelled',
            'candidate': memory._candidate_payload(updated or candidate),
        }

    raw_hint = candidate.get('raw_hint') if isinstance(candidate.get('raw_hint'), dict) else {}
    promote_context = memory._resolve_write_context(raw_hint)
    policy_version = str(raw_hint.get('policy_version') or DEFAULT_POLICY_VERSION)
    promotion_actor_id = str(promote_context.get('actor_id') or memory.DEFAULT_ACTOR_ID)
    promotion_source = str(candidate.get('source') or promote_context.get('source') or memory.DEFAULT_SOURCE)
    ledger = _get_change_ledger()

    try:
        promotion = ledger.promote_candidate_fact(
            actor_id=promotion_actor_id,
            reason='candidate_promotion:supersede',
            policy_version=policy_version,
            candidate_id=candidate_id,
            fact=_candidate_to_fact_input(candidate),
            conflict_with_fact_id=candidate.get('conflict_with_fact_id'),
        )
    except Exception as exc:
        logger.exception('promote_candidate failed for candidate_id=%s', candidate_id)
        memory._log_audit(
            'candidate_promote',
            candidate_id=candidate_id,
            error=str(exc),
            value=candidate.get('value'),
            result='error',
            actor_id=promotion_actor_id,
            source=promotion_source,
        )
        return _error(str(exc), error_type='ledger_write_error')

    promoted_obj = ledger.materialize_object(promotion.object_id)
    if not isinstance(promoted_obj, StateFact):
        message = f'candidate {candidate_id} promotion did not materialize a state fact'
        memory._log_audit(
            'candidate_promote',
            candidate_id=candidate_id,
            error=message,
            value=candidate.get('value'),
            result='error',
            actor_id=promotion_actor_id,
            source=promotion_source,
        )
        return _error(message, error_type='ledger_write_error')

    updated = store.update_candidate_status(candidate_id, 'promoted', resolution='supersede')

    materialized = False
    materialization_error = None
    try:
        materialized, materialization_error = await memory._materialize_fact(
            fact=promoted_obj,
            source=promotion_source,
            superseded_fact_id=promoted_obj.parent_id,
        )
        if not materialized and materialization_error:
            logger.warning(
                'Neo4j materialization failed (non-blocking) after candidate promotion: fact=%s err=%s',
                promoted_obj.object_id,
                materialization_error,
            )
    except Exception as exc:  # pragma: no cover - defensive parity with remember_fact
        materialization_error = str(exc)
        logger.warning(
            'Neo4j materialization threw after candidate promotion for fact=%s: %s',
            promoted_obj.object_id,
            exc,
        )

    memory._log_audit(
        'candidate_promote',
        fact=promoted_obj,
        candidate_id=candidate_id,
        value=candidate.get('value'),
        result='ok',
        actor_id=promotion_actor_id,
        source=promotion_source,
    )
    return {
        'status': 'ok',
        'action': 'promoted',
        'candidate': memory._candidate_payload(updated or candidate),
        'fact': memory._serialize_fact(promoted_obj),
        'materialized': materialized,
        'materialization_error': materialization_error,
        'promotion': {
            'object_id': promotion.object_id,
            'root_id': promotion.root_id,
            'event_id': promotion.event_id,
            'event_ids': promotion.event_ids,
        },
    }


async def reject_candidate(candidate_id: str) -> dict[str, Any]:
    """Reject a pending candidate without writing it to the ledger."""
    candidate_id = str(candidate_id or '').strip()
    if not candidate_id:
        return _error('candidate_id is required')

    memory = _memory_router()
    store = _get_candidate_store()
    candidate = store.get_candidate(candidate_id)
    if candidate is None:
        return _error(f'candidate not found: {candidate_id}', error_type='not_found')
    if candidate.get('status') != 'pending':
        return _error(
            f'candidate {candidate_id} is already {candidate.get("status")}',
            error_type='invalid_state',
        )

    updated = store.update_candidate_status(candidate_id, 'rejected', resolution='reject')
    memory._log_audit('candidate_reject', candidate_id=candidate_id, value=candidate.get('value'), result='ok')
    return {
        'status': 'ok',
        'action': 'rejected',
        'candidate': memory._candidate_payload(updated or candidate),
    }


def register_tools(mcp: Any) -> dict[str, Any]:
    mcp.tool()(list_candidates)
    mcp.tool()(promote_candidate)
    mcp.tool()(reject_candidate)

    tool_map = {
        'list_candidates': list_candidates,
        'promote_candidate': promote_candidate,
        'reject_candidate': reject_candidate,
    }

    if hasattr(mcp, '_tools') and isinstance(mcp._tools, dict):  # type: ignore[attr-defined]
        mcp._tools.update(tool_map)  # type: ignore[attr-defined]

    return tool_map
