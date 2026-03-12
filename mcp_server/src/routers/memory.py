"""Memory router tools for Phase 0.

`remember_fact` converts user-provided prose/hints into a typed state fact,
stores it in the change ledger and triggers deterministic Neo4j materialization.

This is the Exec 1 implementation path; non-memory tools in this router remain
lightweight placeholders and are intentionally conservative.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
from datetime import datetime, timezone
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

from ..models.typed_memory import EvidenceRef, StateFact
from ..services.candidate_store import CandidateStore
from ..services.change_ledger import ChangeLedger, _stable_object_id
from ..services.neo4j_materialization import Neo4jMaterializationService
from ..services.schema_validation import _validate_typed_object, detect_conflict

logger = logging.getLogger(__name__)

MAX_TEXT_LEN = 4096
DEFAULT_SCOPE = 'private'
DEFAULT_SOURCE = 'owner'
DEFAULT_ACTION = 'remember_fact'

# Stable module-level dependencies so tests can monkeypatch these easily.
_change_ledger = ChangeLedger(
    os.environ.get(
        'BICAMERAL_CHANGE_LEDGER_DB',
        str(ChangeLedger.__init__.__defaults__[0]),
    )
)
_candidate_store = CandidateStore()
_materializer = Neo4jMaterializationService()

# Canonicalization helpers for user-provided type hints.
_SCHEME_BY_HINT = {
    'preference': 'Preference',
    'commitment': 'Commitment',
    'operational_rule': 'OperationalRule',
    'world_state': 'TypedFact',
}

# Fast conflict resolution dialog contract.
_CONFLICT_OPTIONS = [
    {
        'id': 'A',
        'label': 'Supersede',
        'description': 'Replace existing with new fact',
    },
    {
        'id': 'B',
        'label': 'Parallel',
        'description': 'Keep both facts as parallel beliefs',
    },
    {
        'id': 'C',
        'label': 'Cancel',
        'description': 'Discard new fact, keep existing',
    },
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _truncate_value_for_log(value: Any, max_len: int = 180) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = str(value)
    text = text.strip()
    if len(text) <= max_len:
        return text
    return f'{text[: max_len - 3]}...'


def _log_audit(action: str, *, fact: StateFact | None = None, candidate_id: str | None = None, error: str | None = None, value: Any | None = None, result: str | None = None) -> None:
    payload = {
        'action': action,
        'result': result,
        'actor': DEFAULT_SOURCE,
    }
    if fact is not None:
        payload.update(
            {
                'fact_id': fact.object_id,
                'fact_type': fact.fact_type,
                'subject': fact.subject,
                'predicate': fact.predicate,
                'root_id': fact.root_id,
                'scope': fact.scope,
            }
        )
    if candidate_id:
        payload['candidate_id'] = candidate_id
    if error:
        payload['error'] = error
    if value is not None:
        payload['value_preview'] = _truncate_value_for_log(value)
    logger.info('memory.audit %s', json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _coerce_scope(raw_scope: Any) -> str:
    scope = str(raw_scope or DEFAULT_SCOPE).strip().lower()
    return scope if scope else DEFAULT_SCOPE


def _normalize_fact_type(raw: str | None) -> str:
    if raw is None:
        return 'world_state'

    normalized = re.sub(r'[^a-z0-9_]', '_', str(raw).strip().lower())
    normalized = re.sub(r'_+', '_', normalized).strip('_')

    if not normalized:
        return 'world_state'

    if normalized in {'pref', 'preference', 'prefs'}:
        return 'preference'
    if normalized in {'commit', 'commitment'}:
        return 'commitment'
    if normalized in {'rule', 'rules', 'operational_rule', 'operationalrule'}:
        return 'operational_rule'

    # Preserve explicit typed names where provided.
    return normalized


def _infer_fact_type(raw_text: str) -> tuple[str, float]:
    """Heuristic rule-based fact classifier with confidence score."""
    lower = (raw_text or '').lower()

    if re.search(r'\b(prefer|dislike|enjoy|love|like|hate)\b', lower):
        return 'preference', 0.92
    if re.search(r'\b(i\s+will|i\s+would|i\s+need\s+to|i\s+must|actually)\b', lower):
        return 'commitment', 0.84
    if re.search(r'\b(always|never|must\s+never|should\s+never|policy|rule)\b', lower):
        return 'operational_rule', 0.8

    return 'world_state', 0.55


def _coerce_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return False
    text = str(raw).strip().lower()
    return text in {'1', 'true', 'yes', 'y', 'on', 'supersede', 'parallel', 'update'}


def _clean_text(text: str) -> str:
    return ' '.join((text or '').strip().split())


def _extract_subject_from_text(text: str) -> str:
    """Try a few cheap pattern-based subject extraction heuristics."""
    if not text:
        return ''

    patterns = [
        r'\bfor\s+(.+)$',
        r'\bin\s+(.+?)\s+(?:that|and|because|when|if|,|\.|$)',
        r'\babout\s+(.+)$',
        r'\bregarding\s+(.+)$',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue

        raw = match.group(1).strip()
        raw = re.sub(r'^(the|my|a|an)\s+', '', raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r'\s+(is|are|was|were|has|have)\b.*$', '', raw, flags=re.IGNORECASE).strip()
        raw = raw.rstrip(' .,:;')
        if raw:
            return raw

    return ''


def _extract_predicate_and_value(text: str, fact_type: str, *, subject: str | None = None) -> tuple[str, str]:
    clean = _clean_text(text)

    # Fallback predicate defaults by fact type.
    default_predicate = {
        'preference': 'preference',
        'commitment': 'commitment',
        'operational_rule': 'rule',
        'world_state': 'state',
        'decision': 'decision',
    }.get(fact_type, fact_type or 'state')

    if fact_type == 'preference':
        prefixes = [
            r'\bi\s+prefer\s+',
            r'\bi\s+would\s+prefer\s+',
            r'\bi\s+like\s+',
            r'\bi\s+do\s+like\s+',
            r'\bprefer\s+',
        ]
        for prefix in prefixes:
            candidate = re.sub(prefix, '', clean, flags=re.IGNORECASE)
            candidate = candidate.strip()
            if candidate and candidate != clean:
                break
        else:
            candidate = clean

        # If subject appears in the tail phrase, treat the remaining fragment as value.
        if subject:
            candidate = re.sub(rf'\bfor\s+{re.escape(subject)}\b', '', candidate, flags=re.IGNORECASE)
            candidate = candidate.strip(' ,')
        return default_predicate, candidate or clean

    if fact_type == 'commitment':
        prefixes = [
            r'\bi\s+will\s+',
            r'\bi\s+need\s+to\s+',
            r'\bi\s+have\s+to\s+',
            r'\bi\s+must\s+',
            r'\bwill\s+',
            r'\bneed\s+to\s+',
        ]
        for prefix in prefixes:
            candidate = re.sub(prefix, '', clean, flags=re.IGNORECASE)
            candidate = candidate.strip()
            if candidate and candidate != clean:
                break
        else:
            candidate = clean
        if subject:
            candidate = re.sub(rf'\bfor\s+{re.escape(subject)}\b', '', candidate, flags=re.IGNORECASE)
            candidate = candidate.strip(' ,')
        return default_predicate, candidate or clean

    if fact_type == 'operational_rule':
        return default_predicate, clean

    return default_predicate, clean


def _resolve_schema_name(fact_type: str, raw_hint_present: bool) -> str:
    normalized = fact_type or 'world_state'
    if raw_hint_present:
        return _SCHEME_BY_HINT.get(normalized, 'TypedFact')
    return 'TypedFact'


def _build_evidence_ref() -> list[EvidenceRef]:
    return [
        EvidenceRef.from_legacy_ref(
            {
                'source_key': 'owner_asserted',
                'source_system': 'user',
                'evidence_id': uuid4().hex,
                'observed_at': _now_iso(),
            }
        )
    ]


def _serialize_fact(fact: StateFact) -> dict[str, Any]:
    payload = fact.model_dump(mode='json')
    payload['uuid'] = payload.get('object_id')
    payload['id'] = payload.get('object_id')
    payload['type'] = payload.get('fact_type')
    status = 'active'
    if not payload.get('is_current', True):
        status = 'superseded'
    payload['status'] = status
    payload.setdefault('scope', payload.get('policy_scope') or payload.get('scope', DEFAULT_SCOPE))
    return payload


def _candidate_payload(state_fact: dict[str, Any]) -> dict[str, Any]:
    payload = dict(state_fact)
    payload['candidate_uuid'] = payload.get('candidate_id')
    payload.setdefault('id', payload.get('candidate_id'))
    return payload


def _state_fact_as_dict(fact: StateFact) -> dict[str, Any]:
    data = fact.model_dump(mode='json')
    data['policy_scope'] = data.get('policy_scope', DEFAULT_SCOPE)
    data.setdefault('scope', data['policy_scope'])
    return data


def _get_conflicting_existing(current_facts: list[StateFact], candidate: dict[str, Any]) -> tuple[bool, bool, dict[str, Any] | None, StateFact | None]:
    """Detect conflict and normalize whether duplicate was requested."""
    serialized = [f.model_dump(mode='json') for f in current_facts]
    result = detect_conflict(candidate, serialized)
    if not result:
        return False, False, None, None

    existing_raw = result.get('existing_fact') or {}
    existing_id = str(existing_raw.get('object_id') or '') if isinstance(existing_raw, dict) else ''
    existing = next((f for f in current_facts if f.object_id == existing_id), None)

    if result.get('conflict'):
        return True, False, existing_raw, existing
    if result.get('reason') == 'duplicate':
        return False, True, existing_raw, existing
    return False, False, existing_raw, existing


def _current_state_facts(
    *,
    subject: str,
    predicate: str | None = None,
    scope: str | None = None,
) -> list[StateFact]:
    subject_key = str(subject or '').strip().lower()
    predicate_key = (str(predicate or '') or '').strip().lower()
    scope_key = str(scope or DEFAULT_SCOPE).strip().lower()
    facts: list[StateFact] = []
    for fact in _change_ledger.current_state_facts():
        if fact.object_type != 'state_fact' or not fact.is_current:
            continue

        fact_scope = str(fact.scope or fact.policy_scope or DEFAULT_SCOPE).strip().lower()
        if fact_scope != scope_key:
            continue

        if fact.subject.strip().lower() != subject_key:
            continue

        if predicate and fact.predicate.strip().lower() != predicate_key:
            continue

        facts.append(fact)
    return facts


def _get_graphiti_client() -> Any | None:
    """Import graphiti module lazily to avoid circular import during startup."""
    try:
        module = importlib.import_module('mcp_server.src.graphiti_mcp_server')
        return getattr(module, 'graphiti_client', None)
    except Exception:
        return None


def _materialize_fact(
    *,
    fact: StateFact,
    superseded_fact_id: str | None = None,
) -> tuple[bool, str | None]:
    graphiti_client = _get_graphiti_client()
    return _materializer.materialize_typed_fact(
        fact=fact,
        superseded_fact_id=superseded_fact_id,
        graphiti_client=graphiti_client,
    )


def _build_state_fact(
    *,
    subject: str,
    predicate: str,
    value: Any,
    fact_type: str,
    scope: str,
    parent: StateFact | None = None,
    version: int = 1,
) -> StateFact:
    object_id = _stable_object_id(str(uuid4()))
    root_id = parent.object_id if parent is not None else object_id
    parent_id = parent.object_id if parent is not None else None
    fact = StateFact.model_validate(
        {
            'object_id': object_id,
            'object_type': 'state_fact',
            'root_id': root_id,
            'parent_id': parent_id,
            'version': version,
            'subject': subject,
            'predicate': predicate,
            'value': value,
            'fact_type': fact_type,
            'scope': scope,
            'policy_scope': scope,
            'visibility_scope': scope,
            'evidence_refs': _build_evidence_ref(),
        }
    )
    return fact


async def remember_fact(text: str, hint: dict[str, Any] | None = None) -> dict[str, Any]:
    """Primary owner-asserted fact memory API.

    Returns:
        - a typed fact dict on successful ledger write, or
        - ConflictDialog dict when contradictory input is detected, or
        - ErrorResponse-style dict on validation/write failures.
    """
    clean = _clean_text(text or '')
    if not isinstance(text, str):
        return {
            'status': 'error',
            'error_type': 'validation_error',
            'message': 'text must be a string',
        }

    if not clean:
        return {
            'status': 'error',
            'error_type': 'validation_error',
            'message': 'text is required',
        }

    if len(clean) > MAX_TEXT_LEN:
        return {
            'status': 'error',
            'error_type': 'validation_error',
            'message': f'text exceeds max length {MAX_TEXT_LEN}',
        }

    if hint is None:
        hint = {}
    elif not isinstance(hint, dict):
        return {
            'status': 'error',
            'error_type': 'validation_error',
            'message': 'hint must be a dict when provided',
        }

    schema_requested = 'type' in hint or 'type_hint' in hint
    raw_type = hint.get('type') or hint.get('type_hint')

    if raw_type is not None:
        fact_type = _normalize_fact_type(str(raw_type))
        confidence = 1.0
    else:
        fact_type, confidence = _infer_fact_type(clean)

    scope = _coerce_scope(hint.get('scope') or hint.get('policy_scope'))

    subject = str(hint.get('subject') or '').strip()
    if not subject:
        subject = _extract_subject_from_text(clean)

    if not subject:
        return {
            'status': 'error',
            'error_type': 'validation_error',
            'message': 'missing required field: subject',
        }

    predicate = str(hint.get('predicate') or '').strip()
    if not predicate:
        predicate, value = _extract_predicate_and_value(clean, fact_type, subject=subject)
    else:
        value = hint.get('value') if 'value' in hint else None

    if value is None:
        # Remove obvious framing so we keep just the predicate payload.
        value = clean
        if subject:
            value = re.sub(rf'\bfor\s+{re.escape(subject)}\b', '', value, flags=re.IGNORECASE)
        if predicate:
            value = re.sub(rf'\b{re.escape(predicate)}\b', '', value, flags=re.IGNORECASE)
            value = re.sub(r'\b(i|we|you|he|she|it|they)\s+', '', value, flags=re.IGNORECASE)
            value = value.replace('  ', ' ').strip(' .,;:')
        value = value.strip() or clean

    # Build candidate for validation/diagnostics
    typed_candidate = {
        'subject': subject,
        'predicate': predicate,
        'value': value,
        'fact_type': fact_type,
        'scope': scope,
        'timestamp': _now_iso(),
    }

    schema_name = _resolve_schema_name(fact_type, raw_hint_present=schema_requested)
    valid, error = _validate_typed_object(typed_candidate, schema_name)
    if not valid:
        return {
            'status': 'error',
            'error_type': 'validation_error',
            'message': error,
        }

    # Even when specific schema validation passed, enforce core typed schema.
    valid, error = _validate_typed_object(typed_candidate, 'TypedFact')
    if not valid:
        return {
            'status': 'error',
            'error_type': 'validation_error',
            'message': error,
        }

    # Conflict / duplicate detection in current fact set.
    existing_all = _current_state_facts(subject=subject, scope=scope, predicate=None)
    is_conflict, is_duplicate, existing_payload, existing_fact = _get_conflicting_existing(
        current_facts=existing_all,
        candidate={
            'subject': subject,
            'predicate': predicate,
            'scope': scope,
            'value': value,
            'fact_type': fact_type,
        },
    )

    if is_duplicate:
        duplicate = _serialize_fact(existing_fact or StateFact.model_validate(
            {
                **typed_candidate,
                'object_type': 'state_fact',
                'object_id': _stable_object_id(str(existing_payload or '') or str(uuid4().hex)),
                'root_id': existing_payload.get('root_id', ''),
                'policy_scope': existing_payload.get('policy_scope', scope),
                'visibility_scope': existing_payload.get('visibility_scope', scope),
                'evidence_refs': _build_evidence_ref(),
                'version': int(existing_payload.get('version', 1) or 1),
            }
        ))
        return {
            'status': 'duplicate',
            'message': 'Fact already exists in current state',
            'fact': duplicate,
        }

    supersede_requested = _coerce_bool(hint.get('supersede'))

    if is_conflict and not supersede_requested:
        # quarantine + ConflictDialog (3-option).
        try:
            candidate = _candidate_store.create_candidate(
                payload={
                    'subject': subject,
                    'predicate': predicate,
                    'value': value,
                    'fact_type': fact_type,
                },
                conflict_with_fact_id=existing_payload.get('object_id') if isinstance(existing_payload, dict) else None,
                source=DEFAULT_SOURCE,
                raw_hint={**hint},
                metadata={
                    'fact_type': fact_type,
                    'confidence': confidence,
                    'input_text': clean,
                },
            )
            _log_audit('quarantine_conflict', candidate_id=candidate.get('candidate_id'), value=value, result='ok')
        except Exception as exc:
            _log_audit('quarantine_conflict', error=str(exc), value=value, result='error')
            return {
                'status': 'error',
                'error_type': 'candidate_error',
                'message': str(exc),
            }

        conflict_existing = _serialize_fact(existing_fact) if existing_fact else (_state_fact_as_dict(existing_payload) if isinstance(existing_payload, dict) else existing_payload or {})
        return {
            'type': 'ConflictDialog',
            'status': 'conflict',
            'message': 'Conflicting fact detected; quarantined for review.',
            'existing_fact': conflict_existing,
            'new_fact': _candidate_payload(candidate),
            'options': _CONFLICT_OPTIONS,
            'resolve_via': 'promote_candidate(candidate_id, resolution="supersede"|"parallel"|"cancel")',
            'candidate_id': candidate.get('candidate_id'),
            'candidate_uuid': candidate.get('candidate_id'),
        }

    parent = existing_fact if existing_fact and is_conflict and supersede_requested else None
    version = int(parent.version + 1) if parent is not None else 1
    event_type = 'supersede' if parent is not None else 'assert'

    fact = _build_state_fact(
        subject=subject,
        predicate=predicate,
        value=value,
        fact_type=fact_type,
        scope=scope,
        parent=parent,
        version=version,
    )

    try:
        _change_ledger.append_event(
            event_type,
            actor_id=DEFAULT_SOURCE,
            reason='supersede' if parent is not None else DEFAULT_ACTION,
            recorded_at=_now_iso(),
            object_type='state_fact',
            object_id=fact.object_id,
            target_object_id=parent.object_id if parent is not None else None,
            root_id=fact.root_id,
            parent_id=fact.parent_id,
            policy_version=hint.get('policy_version') if isinstance(hint, dict) else None,
            metadata={
                'source': DEFAULT_SOURCE,
                'scope': scope,
                'fact_type': fact_type,
                'input_text': clean,
                'confidence': confidence,
                'input_hint_keys': sorted(k for k in hint.keys()) if isinstance(hint, dict) else [],
                'superseded_fact_id': parent.object_id if parent is not None else None,
            },
            payload=fact,
        )
    except Exception as exc:
        _log_audit('ledger_write_failed', fact=fact, error=str(exc), value=value, result='error')
        return {
            'status': 'error',
            'error_type': 'ledger_write_error',
            'message': str(exc),
        }

    materialized = False
    materialization_error = None
    try:
        materialized, materialization_error = await _materialize_fact(
            fact=fact,
            superseded_fact_id=parent.object_id if parent is not None else None,
        )
        if not materialized and materialization_error:
            logger.warning(
                'Neo4j materialization failed (non-blocking): fact=%s err=%s',
                fact.object_id,
                materialization_error,
            )
    except Exception as exc:
        materialization_error = str(exc)
        logger.warning('Neo4j materialization threw for fact=%s: %s', fact.object_id, exc)

    _log_audit(
        'ledger_write',
        fact=fact,
        value=value,
        result='ok',
    )

    return {
        'status': 'ok',
        'fact': _serialize_fact(fact),
        'materialized': materialized,
        'materialization_error': materialization_error,
        'audit': {
            'path': 'ledger_primary',
            'action': event_type,
            'neo4j_status': 'ok' if materialized else 'deferred_or_failed',
        },
    }


async def get_current_state(subject: str, predicate: str | None = None, scope: str | None = None) -> dict[str, Any]:
    """Return currently active state facts for a subject/predicate in typed form."""
    if not subject or not str(subject).strip():
        return {'status': 'error', 'error_type': 'validation_error', 'message': 'subject is required'}

    state_facts = [
        _serialize_fact(fact)
        for fact in _current_state_facts(subject=subject, predicate=predicate, scope=_coerce_scope(scope))
    ]
    return {
        'status': 'ok',
        'facts': state_facts,
    }


async def get_history(subject: str, predicate: str | None = None, scope: str | None = None) -> dict[str, Any]:
    """Return event history for current root(s) matching subject/predicate.

    This is intentionally conservative (Phase-0 scope): it returns ledger events
    scoped to currently active roots.
    """
    if not subject or not str(subject).strip():
        return {'status': 'error', 'error_type': 'validation_error', 'message': 'subject is required'}

    scope_value = _coerce_scope(scope)
    history: list[dict[str, Any]] = []
    for fact in _current_state_facts(subject=subject, predicate=predicate, scope=scope_value):
        for row in _change_ledger.events_for_root(fact.root_id):
            history.append(
                {
                    'event_id': row.event_id,
                    'event_type': row.event_type,
                    'recorded_at': row.recorded_at,
                    'actor_id': row.actor_id,
                    'reason': row.reason,
                    'object_id': row.object_id,
                    'target_object_id': row.target_object_id,
                }
            )

    return {
        'status': 'ok',
        'history': history,
    }


def register_tools(mcp: FastMCP) -> None:
    """Register all memory tools with the MCP server instance."""

    @mcp.tool()
    async def remember_fact_tool(text: str, hint: dict[str, Any] | None = None) -> dict[str, Any]:
        return await remember_fact(text=text, hint=hint)

    @mcp.tool()
    async def get_current_state_tool(subject: str, predicate: str | None = None, scope: str | None = None) -> dict[str, Any]:
        return await get_current_state(subject=subject, predicate=predicate, scope=scope)

    @mcp.tool()
    async def get_history_tool(subject: str, predicate: str | None = None, scope: str | None = None) -> dict[str, Any]:
        return await get_history(subject=subject, predicate=predicate, scope=scope)

    # Backward compatible tool names used by external discovery.
    mcp._tools['remember_fact'] = remember_fact_tool  # type: ignore[attr-defined]
    mcp._tools['get_current_state'] = get_current_state_tool  # type: ignore[attr-defined]
    mcp._tools['get_history'] = get_history_tool  # type: ignore[attr-defined]


__all__ = [
    'register_tools',
    'remember_fact',
    'get_current_state',
    'get_history',
    '_change_ledger',
]
