"""Candidate lifecycle MCP tools.

Integrated candidate review surface for quarantined facts created by
``remember_fact``. This router intentionally uses the same CandidateStore and
ChangeLedger paths/models as the memory router so the conflict lifecycle is:

``remember_fact(conflict) -> list_candidates -> promote_candidate/reject_candidate``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

try:
    from fastmcp import FastMCP
except ImportError:  # pragma: no cover - optional for test envs without fastmcp installed
    class FastMCP:
        def tool(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

try:
    from mcp.server.fastmcp import Context as _McpContext
except ImportError:  # pragma: no cover - fallback for minimal test envs
    class _McpContext:  # type: ignore[no-redef]
        """Stub used when the mcp package is unavailable."""

from ..models.typed_memory import EvidenceRef, StateFact
from ..services.candidate_store import CandidateStore
from ..services.change_ledger import ChangeLedger, resolve_ledger_path
from . import memory as memory_router

logger = logging.getLogger(__name__)

# External-facing statuses aligned with Candidate.json schema; 'pending' is an internal alias for 'quarantine'
VALID_STATUSES = frozenset({'quarantine', 'promoted', 'rejected'})
# Internal set that also accepts 'pending' as an alias (used by _status_alias())
VALID_CANDIDATE_STATUSES = VALID_STATUSES | frozenset({'pending'})
VALID_RESOLUTIONS = frozenset({'supersede', 'parallel', 'cancel'})
DEFAULT_POLICY_VERSION = 'candidate_lifecycle_v1'
_MAX_REASON_LENGTH = 512

# Lazily initialized shared dependencies; tests monkeypatch these directly.
_change_ledger: ChangeLedger | None = None
_candidate_store: CandidateStore | None = None


TOOL_CONTRACTS: list[dict[str, Any]] = [
    {
        'name': 'list_candidates',
        'description': (
            'List quarantined/promoted/rejected typed-memory fact candidates for authenticated trusted reviewers. '
            'Lane-scope invariant: when the server has a configured lane (group_id), only candidates '
            'whose source_lane matches the server lane are returned. When no server lane is configured '
            '(global/unscoped deployment), all candidates are visible (global-owner review semantics). '
            'This invariant prevents cross-lane candidate metadata disclosure in lane-partitioned deployments.'
        ),
        'mode_hint': 'typed',
        'schema': {
            'inputs': {
                'status': '"quarantine" | "promoted" | "rejected" | null (default "quarantine")',
                'type_filter': 'string | null',
                'age_days': 'int | null',
                'min_confidence': 'float | null',
                'max_age_days': 'int | null',
            },
            'output': '{"status": "ok", "candidates": list[Candidate], "reviewer": string} | ErrorResponse',
        },
        'examples': [{'status': 'quarantine'}],
    },
    {
        'name': 'promote_candidate',
        'description': 'Promote a quarantined candidate into the typed ledger (supersede) or cancel it, using server-derived reviewer auth',
        'mode_hint': 'typed',
        'schema': {
            'inputs': {
                'candidate_id': 'string',
                'resolution': '"supersede" | "cancel" | "parallel" (reserved; currently rejected as unsupported)',
                'actor_id': 'string | null (informational audit hint only; never used for authorization)',
                'reason': 'string | null',
            },
            'output': '{"status": "ok", "action": string, "candidate": Candidate, "fact": TypedFact | null} | ErrorResponse',
        },
        'examples': [{'candidate_id': 'cand-001', 'resolution': 'supersede'}],
    },
    {
        'name': 'reject_candidate',
        'description': 'Reject a quarantined candidate using server-derived reviewer auth',
        'mode_hint': 'typed',
        'examples': [{'candidate_id': 'cand-002'}],
        'schema': {
            'inputs': {
                'candidate_id': 'string',
                'actor_id': 'string | null (informational audit hint only; never used for authorization)',
                'reason': 'string | null',
            },
            'output': '{"status": "ok", "action": "rejected", "candidate": Candidate, "reviewer": string, "reason": string | null} | ErrorResponse',
        },
    },
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _normalize_reason(reason: str | None) -> str | None:
    normalized = str(reason or '').strip()
    if not normalized:
        return None
    if len(normalized) > _MAX_REASON_LENGTH:
        return normalized[:_MAX_REASON_LENGTH]
    return normalized


def _get_change_ledger() -> ChangeLedger:
    global _change_ledger
    if _change_ledger is None:
        _change_ledger = ChangeLedger(resolve_ledger_path())
    return _change_ledger


def _get_candidate_store() -> CandidateStore:
    global _candidate_store
    if _candidate_store is None:
        _candidate_store = CandidateStore()
    return _candidate_store


def _status_alias(status: str | None) -> str:
    normalized = str(status or 'pending').strip().lower() or 'pending'
    if normalized == 'quarantine':
        return 'pending'
    return normalized


def _error(error_type: str, message: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'status': 'error',
        'error_type': error_type,
        'message': message,
    }
    if details is not None:
        payload['details'] = details
    return payload


def _require_reviewer(ctx: _McpContext | None) -> str | dict[str, Any]:
    server_principal = memory_router._extract_server_principal(ctx)
    trusted = memory_router._trusted_actor_ids_from_env()
    if server_principal == memory_router._ANON_PRINCIPAL or server_principal not in trusted:
        return _error(
            'unauthorized',
            'candidate review requires an authenticated trusted caller',
        )
    return server_principal


def _candidate_status_error(candidate_id: str, candidate_status: str) -> dict[str, Any]:
    return _error(
        'invalid_state',
        f'candidate {candidate_id} is already {candidate_status}',
        details={'candidate_id': candidate_id, 'status': candidate_status},
    )


def _candidate_scope(candidate: dict[str, Any]) -> str:
    raw_hint = candidate.get('raw_hint') if isinstance(candidate.get('raw_hint'), dict) else {}
    metadata = candidate.get('metadata') if isinstance(candidate.get('metadata'), dict) else {}
    return str(
        raw_hint.get('policy_scope')
        or raw_hint.get('scope')
        or metadata.get('scope')
        or 'private'
    ).strip().lower() or 'private'


def _candidate_source(candidate: dict[str, Any]) -> str:
    raw_hint = candidate.get('raw_hint') if isinstance(candidate.get('raw_hint'), dict) else {}
    trust = raw_hint.get('trust') if isinstance(raw_hint.get('trust'), dict) else {}
    return str(
        candidate.get('source')
        or trust.get('source')
        or memory_router.DEFAULT_SOURCE
    ).strip() or memory_router.DEFAULT_SOURCE


def _candidate_assertion_type(candidate: dict[str, Any]) -> str:
    raw_hint = candidate.get('raw_hint') if isinstance(candidate.get('raw_hint'), dict) else {}
    return str(
        raw_hint.get('type')
        or raw_hint.get('type_hint')
        or candidate.get('fact_type')
        or 'TypedFact'
    ).strip().lower() or 'typedfact'


def _candidate_confidence(candidate: dict[str, Any]) -> float | None:
    metadata = candidate.get('metadata') if isinstance(candidate.get('metadata'), dict) else {}
    confidence = metadata.get('confidence')
    if isinstance(confidence, (int, float)):
        return max(0.0, min(1.0, float(confidence)))
    return None


def _candidate_to_public(candidate: dict[str, Any]) -> dict[str, Any]:
    """Project internal candidate storage into the public Candidate.json contract.

    Public contract fields (Candidate.json):
        uuid, type, subject, predicate, value, status, confidence,
        created_at, updated_at, conflicting_fact_uuid, reviewed_at,
        reviewed_by, promoted_at, promoted_by, reason, resolution, metadata

    Internal storage may use different field names; this function maps them.
    """
    payload = dict(candidate)

    # Map candidate_id -> uuid (public contract primary key)
    payload['uuid'] = payload.get('candidate_id') or payload.get('uuid', '')
    # Map fact_type -> type (public contract assertion type)
    payload['type'] = _candidate_assertion_type(candidate)
    # Map conflict_with_fact_id -> conflicting_fact_uuid
    payload['conflicting_fact_uuid'] = payload.get('conflict_with_fact_id') or payload.get('conflicting_fact_uuid')

    # Ensure status uses external vocabulary (not internal 'pending')
    raw_status = str(payload.get('status') or 'quarantine').strip().lower()
    if raw_status == 'pending':
        raw_status = 'quarantine'
    payload['status'] = raw_status

    # Ensure confidence is a top-level number (public contract requires it)
    confidence = _candidate_confidence(candidate)
    payload['confidence'] = confidence if confidence is not None else 0.5

    # Ensure timestamps
    now = _now_iso()
    payload.setdefault('created_at', now)
    payload.setdefault('updated_at', now)

    # Retain backward-compat aliases for callers that may still reference them
    payload['id'] = payload['uuid']
    payload['candidate_uuid'] = payload['uuid']

    return payload


def _candidate_source_lane(candidate: dict[str, Any]) -> str | None:
    """Extract the canonical source_lane (lane/group id) from candidate storage.

    Resolution order:
    1. ``raw_hint['source_lane']`` — set by ``remember_fact`` during quarantine
       (populated from the server's configured ``group_id`` via ``_derive_source_lane()``).
    2. ``None`` — no lane identity available.

    Does NOT fall back to ``_candidate_scope()`` (visibility scope).  The two
    concepts are distinct:
      - ``scope``       = visibility/policy boundary (``'private'``, ``'public'``, etc.)
      - ``source_lane`` = canonical lane/group identifier (a real group_id).
    Conflating them causes promoted candidates to become invisible on subsequent
    lane-scoped reads when an authorized_group_id restriction is active.
    """
    raw_hint = candidate.get('raw_hint') if isinstance(candidate.get('raw_hint'), dict) else {}
    lane = str(raw_hint.get('source_lane') or '').strip()
    return lane or None


def _candidate_evidence_refs(candidate: dict[str, Any]) -> list[EvidenceRef]:
    raw_hint = candidate.get('raw_hint') if isinstance(candidate.get('raw_hint'), dict) else {}
    existing = raw_hint.get('evidence_refs')
    refs: list[EvidenceRef] = []
    if isinstance(existing, list):
        for item in existing:
            if isinstance(item, EvidenceRef):
                refs.append(item)
            elif isinstance(item, dict):
                refs.append(EvidenceRef.from_legacy_ref(item))
    if refs:
        return refs

    scope = _candidate_scope(candidate)
    source_key = _candidate_source(candidate)
    return [
        EvidenceRef.from_legacy_ref(
            {
                'source_key': source_key,
                'scope': scope,
                'source_system': 'candidate_review',
                'evidence_id': candidate.get('candidate_id') or uuid4().hex,
                'observed_at': _now_iso(),
            }
        )
    ]


def _candidate_to_fact_input(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        'assertion_type': _candidate_assertion_type(candidate),
        'subject': candidate.get('subject'),
        'predicate': candidate.get('predicate'),
        'value': candidate.get('value'),
        'scope': _candidate_scope(candidate),
        # Preserve canonical lane identity so build_object_from_candidate_fact
        # uses the real group_id, not the visibility scope.
        'source_lane': _candidate_source_lane(candidate),
        'evidence_refs': [ref.model_dump(mode='json') for ref in _candidate_evidence_refs(candidate)],
    }


def _matches_filters(
    candidate: dict[str, Any],
    *,
    type_filter: str | None,
    min_confidence: float | None,
    max_age_days: int | None,
    age_days: int | None,
) -> bool:
    if type_filter and str(candidate.get('fact_type') or '').strip().lower() != str(type_filter).strip().lower():
        return False

    confidence = _candidate_confidence(candidate)
    if min_confidence is not None:
        if confidence is None or confidence < min_confidence:
            return False

    effective_age = age_days if age_days is not None else max_age_days
    if effective_age is not None:
        created_at = str(candidate.get('created_at') or '').strip()
        if not created_at:
            return False
        normalized = created_at[:-1] + '+00:00' if created_at.endswith('Z') else created_at
        try:
            created_dt = datetime.fromisoformat(normalized)
        except ValueError:
            return False
        cutoff = datetime.now(timezone.utc) - timedelta(days=effective_age)
        if created_dt.astimezone(timezone.utc) < cutoff:
            return False

    return True


async def list_candidates(
    status: str | None = None,
    type_filter: str | None = None,
    age_days: int | None = None,
    min_confidence: float | None = None,
    max_age_days: int | None = None,
    ctx: _McpContext | None = None,
) -> dict[str, Any]:
    reviewer = _require_reviewer(ctx)
    if isinstance(reviewer, dict):
        return reviewer

    effective_status = _status_alias(status)
    if effective_status not in {'pending', 'promoted', 'rejected'}:
        return _error(
            'validation_error',
            f'invalid status {status!r}; expected one of ["pending", "promoted", "rejected", "quarantine"]',
        )

    if min_confidence is not None:
        try:
            min_confidence = float(min_confidence)
        except (TypeError, ValueError):
            return _error('validation_error', 'min_confidence must be a float between 0.0 and 1.0')
        if not (0.0 <= min_confidence <= 1.0):
            return _error('validation_error', 'min_confidence must be a float between 0.0 and 1.0')

    for field_name, value in (('age_days', age_days), ('max_age_days', max_age_days)):
        if value is None:
            continue
        try:
            value = int(value)
        except (TypeError, ValueError):
            return _error('validation_error', f'{field_name} must be a positive integer')
        if value <= 0:
            return _error('validation_error', f'{field_name} must be a positive integer')

    rows = _get_candidate_store().list_candidates(status=effective_status)

    # Lane-scope invariant: when the server has a configured lane (group_id),
    # only show candidates from the same lane to prevent cross-lane metadata
    # disclosure in lane-partitioned deployments.  When no server lane is
    # configured (global/unscoped deployment), all candidates are visible
    # (global-owner review semantics — the intended invariant for single-lane
    # or trusted-owner deployments).
    _server_lane = memory_router._derive_source_lane()
    if _server_lane is not None:
        rows = [r for r in rows if _candidate_source_lane(r) == _server_lane]

    candidates = [
        _candidate_to_public(candidate)
        for candidate in rows
        if _matches_filters(
            candidate,
            type_filter=type_filter,
            min_confidence=min_confidence,
            max_age_days=max_age_days,
            age_days=age_days,
        )
    ]
    return {
        'status': 'ok',
        'candidates': candidates,
        'reviewer': reviewer,
    }


async def promote_candidate(
    candidate_id: str,
    resolution: str,
    actor_id: str | None = None,
    reason: str | None = None,
    ctx: _McpContext | None = None,
) -> dict[str, Any]:
    del actor_id  # caller-controlled audit hint only; never the auth source

    normalized_candidate_id = str(candidate_id or '').strip()
    normalized_resolution = str(resolution or '').strip().lower()
    normalized_reason = _normalize_reason(reason)

    if not normalized_candidate_id:
        return _error('validation_error', 'candidate_id is required')
    if normalized_resolution not in VALID_RESOLUTIONS:
        return _error(
            'validation_error',
            f'invalid resolution {normalized_resolution!r}; expected one of {sorted(VALID_RESOLUTIONS)}',
        )
    if normalized_resolution == 'parallel':
        return _error('validation_error', 'parallel resolution is not supported on the integrated surface')

    reviewer = _require_reviewer(ctx)
    if isinstance(reviewer, dict):
        return reviewer

    store = _get_candidate_store()
    ledger = _get_change_ledger()
    candidate = store.get_candidate(normalized_candidate_id)
    if candidate is None:
        return _error('not_found', f'candidate not found: {normalized_candidate_id}')
    if candidate.get('status') != 'pending':
        return _candidate_status_error(normalized_candidate_id, str(candidate.get('status')))

    existing_promotion = ledger.promotion_event_for_candidate(normalized_candidate_id)
    if existing_promotion is not None:
        store.update_candidate_status(
            normalized_candidate_id,
            'promoted',
            resolution='supersede',
        )
        promoted_fact = ledger.materialize_object(str(existing_promotion.object_id or ''))
        return {
            'status': 'ok',
            'action': 'promoted',
            'candidate': _candidate_to_public(store.get_candidate(normalized_candidate_id) or candidate),
            'fact': memory_router._serialize_fact(promoted_fact) if isinstance(promoted_fact, StateFact) else None,
            'promotion': {
                'object_id': existing_promotion.object_id,
                'root_id': existing_promotion.root_id,
                'event_id': existing_promotion.event_id,
                'event_ids': [existing_promotion.event_id],
                'reconciled': True,
            },
            'reviewer': reviewer,
            'materialized': False,
        }

    if normalized_resolution == 'cancel':
        updated = store.update_candidate_status(
            normalized_candidate_id,
            'rejected',
            resolution='cancel',
        )
        if not updated:
            refreshed = store.get_candidate(normalized_candidate_id)
            if refreshed is None:
                return _error('not_found', f'candidate not found: {normalized_candidate_id}')
            return _candidate_status_error(normalized_candidate_id, str(refreshed.get('status')))
        return {
            'status': 'ok',
            'action': 'cancelled',
            'candidate': _candidate_to_public(store.get_candidate(normalized_candidate_id) or candidate),
            'reviewer': reviewer,
        }

    try:
        promotion = ledger.promote_candidate_fact(
            actor_id=reviewer,
            reason=f'candidate_promotion:{normalized_resolution}',
            policy_version=str(
                (
                    (candidate.get('raw_hint') or {}).get('policy_version')
                    if isinstance(candidate.get('raw_hint'), dict)
                    else None
                )
                or DEFAULT_POLICY_VERSION
            ),
            candidate_id=normalized_candidate_id,
            fact=_candidate_to_fact_input(candidate),
            conflict_with_fact_id=candidate.get('conflict_with_fact_id'),
            allow_parallel=False,
            require_supersede=True,
        )
    except ValueError as exc:
        return _error('validation_error', str(exc))
    except Exception as exc:
        # Detect concurrent-promotion race: a UNIQUE constraint on object_id means
        # another writer already committed a promotion for this candidate.  Re-read
        # the candidate to surface the correct invalid_state response rather than an
        # opaque operational_error.  This tightens the post-promotion store/ledger
        # skew window by giving callers a meaningful error instead of an internal
        # integrity failure.
        exc_str = str(exc)
        if 'UNIQUE constraint' in exc_str or 'unique constraint' in exc_str.lower():
            refreshed = store.get_candidate(normalized_candidate_id)
            if refreshed is not None and refreshed.get('status') not in ('pending', None):
                return _candidate_status_error(
                    normalized_candidate_id, str(refreshed.get('status', 'promoted'))
                )
        logger.exception('promote_candidate failed')
        return _error('operational_error', f'promote_candidate failed: {exc}')

    # Skew mitigation: capture the return value so we can detect the failure case
    # where the ledger succeeded but the candidate-store update did not.  The
    # existing reconcile path (promotion_event_for_candidate check at the top of
    # this function) will re-apply the store update on the next call, so this is
    # a log-and-continue rather than a hard error.
    store_updated = store.update_candidate_status(
        normalized_candidate_id,
        'promoted',
        resolution=normalized_resolution,
    )
    if store_updated is None:
        # Candidate not found in store after successful ledger promotion.
        # Possible concurrent deletion or ephemeral DB issue.  The reconcile
        # path (promotion_event_for_candidate) will heal on next call.
        logger.warning(
            'promote_candidate: ledger promotion succeeded for %r but candidate-store '
            'update returned None (skew state; candidate may have been concurrently '
            'deleted). Reconcile path will re-apply on next promote_candidate call.',
            normalized_candidate_id,
        )

    # Readback: verify store reflects the new 'promoted' status.  A stale
    # readback indicates persistent skew (e.g. DB write delayed or failed
    # after commit); log for operator visibility; reconcile will heal.
    promoted_candidate = store.get_candidate(normalized_candidate_id) or candidate
    if promoted_candidate is not candidate and promoted_candidate.get('status') in ('pending', 'quarantine'):
        logger.warning(
            'promote_candidate: readback for %r still shows status=%r after successful '
            'ledger promotion — persisted skew detected. Reconcile path will heal on '
            'next call via promotion_event_for_candidate check.',
            normalized_candidate_id, promoted_candidate.get('status'),
        )

    promoted_fact = ledger.materialize_object(promotion.object_id)

    materialized = False
    materialization_error: str | None = None
    neo4j_materialization: dict[str, Any] = {'status': 'skipped'}
    if isinstance(promoted_fact, StateFact):
        try:
            materialized, materialization_error = await memory_router._materialize_fact(
                fact=promoted_fact,
                source=_candidate_source(candidate),
                superseded_fact_id=candidate.get('conflict_with_fact_id'),
            )
            neo4j_materialization = {
                'status': 'ok' if materialized else 'failed',
                'source': _candidate_source(candidate),
            }
            if materialization_error:
                neo4j_materialization['error'] = materialization_error
        except Exception as exc:  # pragma: no cover - defensive only
            materialization_error = str(exc)
            neo4j_materialization = {'status': 'failed', 'error': materialization_error}

    response: dict[str, Any] = {
        'status': 'ok',
        'action': 'promoted',
        'candidate': _candidate_to_public(promoted_candidate),
        'fact': memory_router._serialize_fact(promoted_fact) if isinstance(promoted_fact, StateFact) else None,
        'promotion': {
            'object_id': promotion.object_id,
            'root_id': promotion.root_id,
            'event_id': promotion.event_id,
            'event_ids': promotion.event_ids,
        },
        'reviewer': reviewer,
        'materialized': materialized,
        'neo4j_materialization': neo4j_materialization,
    }
    if materialization_error:
        response['materialization_error'] = materialization_error
        response['warnings'] = [
            {
                'code': 'neo4j_materialization_failed',
                'message': materialization_error,
            }
        ]
    return response


async def reject_candidate(
    candidate_id: str,
    actor_id: str | None = None,
    reason: str | None = None,
    ctx: _McpContext | None = None,
) -> dict[str, Any]:
    del actor_id  # caller-controlled audit hint only; never the auth source
    # Preserve normalized reason for audit trail / response.  Do NOT del reason.
    normalized_reason = _normalize_reason(reason)

    normalized_candidate_id = str(candidate_id or '').strip()
    if not normalized_candidate_id:
        return _error('validation_error', 'candidate_id is required')

    reviewer = _require_reviewer(ctx)
    if isinstance(reviewer, dict):
        return reviewer

    store = _get_candidate_store()
    ledger = _get_change_ledger()
    candidate = store.get_candidate(normalized_candidate_id)
    if candidate is None:
        return _error('not_found', f'candidate not found: {normalized_candidate_id}')
    if candidate.get('status') != 'pending':
        return _candidate_status_error(normalized_candidate_id, str(candidate.get('status')))

    existing_promotion = ledger.promotion_event_for_candidate(normalized_candidate_id)
    if existing_promotion is not None:
        store.update_candidate_status(normalized_candidate_id, 'promoted', resolution='supersede')
        return _candidate_status_error(normalized_candidate_id, 'promoted')

    # Write reason into resolution so it is durably stored in candidates.db.
    # Format: 'rejected' when no reason supplied, 'rejected: <reason>' when one is.
    resolution = f'rejected: {normalized_reason}' if normalized_reason else 'rejected'
    updated = store.update_candidate_status(normalized_candidate_id, 'rejected', resolution=resolution)
    if not updated:
        refreshed = store.get_candidate(normalized_candidate_id)
        if refreshed is None:
            return _error('not_found', f'candidate not found: {normalized_candidate_id}')
        return _candidate_status_error(normalized_candidate_id, str(refreshed.get('status')))

    return {
        'status': 'ok',
        'action': 'rejected',
        'candidate': _candidate_to_public(store.get_candidate(normalized_candidate_id) or candidate),
        'reviewer': reviewer,
        'reason': normalized_reason,
    }


def register_tools(mcp: FastMCP) -> dict[str, Any]:
    @mcp.tool(name='list_candidates')
    async def list_candidates_tool(
        status: str | None = None,
        type_filter: str | None = None,
        age_days: int | None = None,
        min_confidence: float | None = None,
        max_age_days: int | None = None,
        ctx: _McpContext | None = None,
    ) -> dict[str, Any]:
        return await list_candidates(
            status=status,
            type_filter=type_filter,
            age_days=age_days,
            min_confidence=min_confidence,
            max_age_days=max_age_days,
            ctx=ctx,
        )

    @mcp.tool(name='promote_candidate')
    async def promote_candidate_tool(
        candidate_id: str,
        resolution: str,
        actor_id: str | None = None,
        reason: str | None = None,
        ctx: _McpContext | None = None,
    ) -> dict[str, Any]:
        return await promote_candidate(
            candidate_id=candidate_id,
            resolution=resolution,
            actor_id=actor_id,
            reason=reason,
            ctx=ctx,
        )

    @mcp.tool(name='reject_candidate')
    async def reject_candidate_tool(
        candidate_id: str,
        actor_id: str | None = None,
        reason: str | None = None,
        ctx: _McpContext | None = None,
    ) -> dict[str, Any]:
        return await reject_candidate(
            candidate_id=candidate_id,
            actor_id=actor_id,
            reason=reason,
            ctx=ctx,
        )

    tool_map = {
        'list_candidates': list_candidates_tool,
        'promote_candidate': promote_candidate_tool,
        'reject_candidate': reject_candidate_tool,
    }

    if hasattr(mcp, '_tools') and isinstance(mcp._tools, dict):  # type: ignore[attr-defined]
        mcp._tools.update(tool_map)  # type: ignore[attr-defined]

    return tool_map


__all__ = [
    'register_tools',
    'list_candidates',
    'promote_candidate',
    'reject_candidate',
    '_change_ledger',
    '_candidate_store',
    '_candidate_to_fact_input',
    'DEFAULT_POLICY_VERSION',
]
