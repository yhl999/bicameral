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

# Import the real FastMCP Context type so the register_tools wrapper is
# recognised by find_context_parameter and ctx is injected (not caller-supplied).
# Fallback stub is safe: find_context_parameter won't match it so ctx=None,
# and auth falls to the OAuth bearer-token contextvar (path-1 of
# _extract_server_principal) which is independent of the ctx parameter.
try:
    from mcp.server.fastmcp import Context as _McpContext
except ImportError:  # pragma: no cover - fallback for minimal test envs
    class _McpContext:  # type: ignore[no-redef]
        """Stub used when the mcp package is unavailable."""

from ..models.typed_memory import EvidenceRef, StateFact
from ..services.candidate_store import CandidateStore
from ..services.change_ledger import ChangeLedger, _stable_object_id, resolve_ledger_path
from ..services.neo4j_materialization import Neo4jMaterializationService
from ..services.schema_validation import _validate_typed_object, detect_conflict

logger = logging.getLogger(__name__)

MAX_TEXT_LEN = 4096
MAX_HINT_JSON_BYTES = 16_384
MAX_HINT_DEPTH = 4
MAX_AUX_PAYLOAD_JSON_BYTES = 8_192
MAX_AUX_PAYLOAD_DEPTH = 4
DEFAULT_SCOPE = 'private'
DEFAULT_SOURCE = 'caller_asserted_unverified'
DEFAULT_ACTOR_ID = 'caller:unverified'
DEFAULT_EVIDENCE_SOURCE_KEY = 'caller_asserted_unverified'
VERIFIED_OWNER_SOURCE = 'owner_asserted'
TRUSTED_SOURCE_ALLOWLIST = frozenset(
    {
        VERIFIED_OWNER_SOURCE,
        'delegate_asserted',
        'system_asserted',
        'caller_asserted_verified',
    }
)
DEFAULT_ACTION = 'remember_fact'

TOOL_CONTRACTS: list[dict[str, Any]] = [
    {
        'name': 'remember_fact',
        'description': 'Parse a caller-provided fact hint/text, write it into the typed ledger, or quarantine conflicts for review',
        'mode_hint': 'typed',
        'schema': {
            'inputs': {
                'text': 'string',
                'hint': 'object | null',
            },
            'output': '{"status": "ok", "fact": TypedFact} | {"status": "conflict", "candidate_id": string, "new_fact": TypedFact} | ErrorResponse',
        },
        'examples': [{'text': 'I prefer tabs over spaces', 'hint': {'fact_type': 'preference'}}],
        'phase0_behavior': 'Validates text/hint, derives trusted scope/policy from the server-side principal, writes directly to the ledger when safe, and quarantines conflicts into CandidateStore for explicit review.',
    },
    {
        'name': 'get_current_state',
        'description': 'Query the typed-memory ledger for current non-superseded facts, scoped by lane',
        'mode_hint': 'typed',
        'schema': {
            'inputs': {
                'subject': 'string',
                'predicate': 'string | null',
                'scope': 'string | null',
                'group_ids': 'list[string] | null',
                'lane_alias': 'list[string] | null',
            },
            'output': '{"status": "ok", "facts": list[TypedFact]} | ErrorResponse',
        },
        'examples': [{'subject': 'user', 'predicate': 'preferred_editor'}],
    },
    {
        'name': 'get_history',
        'description': (
            'Retrieve typed-memory change history for the root lineage(s) of currently active '
            'facts matching subject/predicate. '
            'NOTE: This method walks roots of current (non-superseded) facts only — '
            'it does NOT scan every historical root ever associated with the subject. '
            'Superseded-and-replaced lineages that are no longer active are not returned. '
            'The response includes roots_considered so callers can see which root IDs were walked. '
            'Use get_current_state to inspect only the latest fact values without history.'
        ),
        'mode_hint': 'typed',
        'schema': {
            'inputs': {
                'subject': 'string',
                'predicate': 'string | null',
                'scope': 'string | null',
                'group_ids': 'list[string] | null',
                'lane_alias': 'list[string] | null',
            },
            'output': '{"status": "ok", "history": list[dict], "roots_considered": list[string], "scope": string} | ErrorResponse',
            'semantics_note': (
                'Phase-0 semantics: walks root lineages of currently active (non-superseded) '
                'facts only. Superseded fact lineages that are no longer active are excluded '
                'even if previously associated with the same subject/predicate. '
                'roots_considered lists the root IDs that were walked to build the history list. '
                'This is a narrower scope than a full audit-log scan; it is optimised for '
                'inspecting the evolution of currently active facts.'
            ),
        },
        'examples': [{'subject': 'project-alpha', 'predicate': 'status'}],
    },
]


# Sentinel returned when no authenticated caller principal is available.
_ANON_PRINCIPAL = '__anon__'

# Server-side allowlist of actor IDs that are permitted to receive elevated
# write privileges.  Populated exclusively from the runtime environment — never
# from caller-supplied tool arguments.  An empty or absent value means NO
# caller can receive elevated trust (fail-closed).
_TRUSTED_ACTOR_IDS_ENV = 'BICAMERAL_TRUSTED_ACTOR_IDS'

_ALLOWED_HINT_KEYS = frozenset(
    {
        'type',
        'type_hint',
        'fact_type',
        'subject',
        'predicate',
        'value',
        'scope',
        'policy_scope',
        'supersede',
        'policy_version',
        'metadata',
        'trust',
    }
)
# NOTE: 'verified', 'is_owner', and 'allow_conflict_supersede' are accepted for
# schema validation and forward-compat purposes but have NO security effect on
# their own.  Privilege elevation is gated exclusively on whether the caller's
# actor_id appears in the server-side BICAMERAL_TRUSTED_ACTOR_IDS allowlist.
_ALLOWED_TRUST_KEYS = frozenset(
    {
        'verified',
        'is_owner',
        'actor_id',
        'source',
        'scope',
        'allow_conflict_supersede',
    }
)

# Lazily-initialized module-level dependencies so importing the router does not
# create SQLite files as a side effect. Tests can still monkeypatch these names.
_change_ledger: ChangeLedger | None = None
_candidate_store: CandidateStore | None = None
_materializer: Neo4jMaterializationService | None = None

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
        'description': 'Replace the current fact with the quarantined candidate',
    },
    {
        'id': 'B',
        'label': 'Cancel',
        'description': 'Reject the quarantined candidate and keep the current fact',
    },
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


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


def _get_materializer() -> Neo4jMaterializationService:
    global _materializer
    if _materializer is None:
        _materializer = Neo4jMaterializationService()
    return _materializer


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


def _log_audit(
    action: str,
    *,
    fact: StateFact | None = None,
    candidate_id: str | None = None,
    error: str | None = None,
    value: Any | None = None,
    result: str | None = None,
    actor_id: str | None = None,
    source: str | None = None,
) -> None:
    payload = {
        'action': action,
        'result': result,
        'actor': actor_id or DEFAULT_ACTOR_ID,
        'source': source or DEFAULT_SOURCE,
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


def _payload_json_size(value: Any) -> int:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    except Exception:
        encoded = str(value)
    return len(encoded.encode('utf-8'))


def _payload_depth(value: Any, _seen: set[int] | None = None) -> int:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return 0

    if _seen is None:
        _seen = set()

    value_id = id(value)
    if value_id in _seen:
        return 0
    _seen.add(value_id)

    if isinstance(value, dict):
        children = list(value.values())
    elif isinstance(value, (list, tuple)):
        children = list(value)
    else:
        _seen.discard(value_id)
        return 0

    if not children:
        _seen.discard(value_id)
        return 1

    max_child_depth = max(_payload_depth(child, _seen) for child in children)
    _seen.discard(value_id)
    return 1 + max_child_depth


def _validate_payload_bounds(
    payload: Any,
    *,
    label: str,
    max_json_bytes: int,
    max_depth: int,
) -> tuple[bool, str | None]:
    if payload is None:
        return True, None

    payload_size = _payload_json_size(payload)
    if payload_size > max_json_bytes:
        return False, f'{label} exceeds max size {max_json_bytes} bytes'

    depth = _payload_depth(payload)
    if depth > max_depth:
        return False, f'{label} exceeds max depth {max_depth}'

    return True, None


def _validate_hint_contract(hint: dict[str, Any]) -> tuple[bool, str | None]:
    unexpected_keys = sorted(set(hint) - _ALLOWED_HINT_KEYS)
    if unexpected_keys:
        return False, f'hint has unknown field(s): {unexpected_keys}'

    for field in (
        'type',
        'type_hint',
        'fact_type',
        'subject',
        'predicate',
        'scope',
        'policy_scope',
        'policy_version',
    ):
        if field in hint and hint[field] is not None and not isinstance(hint[field], str):
            return False, f'hint.{field} must be a string when provided'

    if 'metadata' in hint and hint['metadata'] is not None and not isinstance(hint['metadata'], dict):
        return False, 'hint.metadata must be an object when provided'

    trust = hint.get('trust')
    if trust is not None:
        if not isinstance(trust, dict):
            return False, 'hint.trust must be an object when provided'

        unexpected_trust_keys = sorted(set(trust) - _ALLOWED_TRUST_KEYS)
        if unexpected_trust_keys:
            return False, f'hint.trust has unknown field(s): {unexpected_trust_keys}'

        for field in ('verified', 'is_owner', 'allow_conflict_supersede'):
            if field in trust and trust[field] is not None and not isinstance(trust[field], bool):
                return False, f'hint.trust.{field} must be a boolean when provided'

        for field in ('actor_id', 'source', 'scope'):
            if field in trust and trust[field] is not None and not isinstance(trust[field], str):
                return False, f'hint.trust.{field} must be a string when provided'

    return True, None


def _trusted_actor_ids_from_env() -> frozenset[str]:
    """Return the server-configured set of trusted actor IDs.

    Reads BICAMERAL_TRUSTED_ACTOR_IDS (comma-separated list) from the runtime
    environment.  An absent or empty value returns an empty set, which causes
    _resolve_write_context to fail closed for ALL callers.

    This is the SOLE trust gate for privilege elevation.  Caller-supplied
    hint.trust fields are NOT sufficient on their own; the caller's actor_id
    must appear in this allowlist.
    """
    raw = (os.environ.get(_TRUSTED_ACTOR_IDS_ENV) or '').strip()
    if not raw:
        return frozenset()
    return frozenset(part.strip() for part in raw.split(',') if part.strip())


def _extract_server_principal(ctx: Any) -> str:
    """Extract the verified caller principal from the MCP auth context layer.

    Resolution order (most → least trusted):
    1. OAuth ``AccessToken.client_id`` from the MCP auth middleware contextvar —
       set by ``AuthContextMiddleware`` from a verified bearer token.  This path
       is only active when the server is running with OAuth configured.
    2. ``Context.client_id`` from the MCP request context meta field —
       transport-injected, NOT sourced from the raw tool-call payload.
    3. ``_ANON_PRINCIPAL`` (``'__anon__'``) sentinel — no authenticated identity
       is available.

    The returned value is **never** derived from the caller-supplied tool
    arguments, so callers cannot self-assign a principal to impersonate another
    client or claim a trusted identity they do not hold.

    Args:
        ctx: The FastMCP ``Context`` for the current request, or ``None`` when
            called outside a live MCP request (e.g. unit tests).

    Returns:
        A non-empty string identifying the authenticated caller, or ``'__anon__'``.
    """
    # 1. OAuth bearer-token contextvar (AuthContextMiddleware, verified server-side).
    try:
        from mcp.server.auth.middleware.auth_context import get_access_token  # type: ignore[import-not-found]
        access_token = get_access_token()
        if access_token is not None and access_token.client_id:
            return str(access_token.client_id)
    except Exception:
        pass

    # 2. Transport-injected request context (not from the tool-call payload).
    if ctx is not None:
        try:
            client_id = ctx.client_id
            if client_id:
                return str(client_id)
        except Exception:
            pass

    return _ANON_PRINCIPAL


def _resolve_write_context(hint: dict[str, Any], server_principal: str | None = None) -> dict[str, Any]:
    """Resolve write context from caller hint and the server-derived principal.

    Security contract
    -----------------
    Privilege elevation (verified=True, is_owner, allow_conflict_supersede) is
    NEVER derived solely from caller-supplied hint.trust fields or a raw
    ``actor_id`` string argument.  Callers MUST NOT be able to elevate their
    own privileges by supplying an actor_id that happens to match the server
    allowlist.

    Trust is only granted when the ``server_principal`` — derived from the MCP
    auth context layer (OAuth bearer token or transport client_id), **never**
    from the raw tool-call payload — appears in the server-side
    BICAMERAL_TRUSTED_ACTOR_IDS allowlist.  If the env var is unset or empty,
    ALL callers are treated as untrusted (fail-closed).

    hint.trust.actor_id and hint.trust.verified are accepted for schema
    validation and audit logging purposes only; they have NO security effect.

    Args:
        hint: The caller-supplied hint dict (tool argument — untrusted).
        server_principal: Caller identity derived from the MCP auth context
            layer via ``_extract_server_principal(ctx)``.  When ``None`` or
            ``_ANON_PRINCIPAL``, the caller is treated as untrusted.
    """
    trust = hint.get('trust') if isinstance(hint, dict) else None

    # Extract caller-supplied hint fields — informational only, NOT security gates.
    hint_actor_id = ''
    hint_source = ''
    hint_scope = None
    hint_is_owner = False
    hint_allow_supersede = False
    if isinstance(trust, dict):
        hint_actor_id = str(trust.get('actor_id') or '').strip()
        hint_source = str(trust.get('source') or '').strip().lower()
        hint_scope = str(trust.get('scope') or '').strip() or None
        hint_is_owner = bool(trust.get('is_owner') is True)
        hint_allow_supersede = bool(trust.get('allow_conflict_supersede') is True)

    # Trust gate: server-derived principal must appear in the server-configured
    # allowlist.  Caller-supplied hint.trust.actor_id is IGNORED for this check
    # — it cannot be used to impersonate a trusted principal.
    effective_principal = str(server_principal or '').strip()
    trusted_ids = _trusted_actor_ids_from_env()
    is_trusted = bool(
        effective_principal
        and effective_principal != _ANON_PRINCIPAL
        and effective_principal in trusted_ids
    )

    if not is_trusted:
        # Fail closed: no trusted server principal or not in allowlist.
        return {
            'verified': False,
            'is_owner': False,
            'actor_id': DEFAULT_ACTOR_ID,
            'source': DEFAULT_SOURCE,
            'source_key': DEFAULT_EVIDENCE_SOURCE_KEY,
            'scope_override': None,
            'allow_conflict_supersede': False,
        }

    # Trusted path: server_principal validated against the server allowlist.
    # hint.trust.is_owner and hint.trust.allow_conflict_supersede are honored
    # ONLY because the principal already passed the server-side allowlist check.
    is_owner = hint_is_owner
    source = hint_source or 'caller_asserted_verified'
    if source not in TRUSTED_SOURCE_ALLOWLIST:
        source = 'caller_asserted_verified'
    if source == VERIFIED_OWNER_SOURCE and not is_owner:
        source = 'caller_asserted_verified'

    allow_conflict_supersede = hint_allow_supersede and is_owner

    return {
        'verified': True,
        'is_owner': is_owner,
        'actor_id': effective_principal,  # server-verified identity, not hint string
        'source': source,
        'source_key': source,
        'scope_override': hint_scope,
        'allow_conflict_supersede': allow_conflict_supersede,
    }


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
    return text in {'1', 'true', 'yes', 'y', 'on'}


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


def _build_evidence_ref(source_key: str) -> list[EvidenceRef]:
    return [
        EvidenceRef.from_legacy_ref(
            {
                'source_key': source_key,
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
    """Project a quarantined candidate into the public Candidate.json contract shape."""
    payload = dict(state_fact)
    # Public contract: uuid is the primary key
    payload['uuid'] = payload.get('candidate_id') or payload.get('uuid', '')
    payload['type'] = payload.get('fact_type') or payload.get('type', 'TypedFact')
    payload['conflicting_fact_uuid'] = payload.get('conflict_with_fact_id')
    raw_status = str(payload.get('status') or 'quarantine').strip().lower()
    if raw_status == 'pending':
        raw_status = 'quarantine'
    payload['status'] = raw_status
    payload.setdefault('confidence', 0.5)
    now = _now_iso()
    payload.setdefault('created_at', now)
    payload.setdefault('updated_at', now)
    # Backward-compat aliases
    payload['candidate_uuid'] = payload['uuid']
    payload.setdefault('id', payload['uuid'])
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
    for fact in _get_change_ledger().current_state_facts():
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
    source: str,
    superseded_fact_id: str | None = None,
) -> tuple[bool, str | None]:
    graphiti_client = _get_graphiti_client()
    return _get_materializer().materialize_typed_fact(
        fact=fact,
        source=source,
        superseded_fact_id=superseded_fact_id,
        graphiti_client=graphiti_client,
    )


def _derive_source_lane() -> str | None:
    """Derive source_lane from the server's default group_id config.

    This ensures facts created via remember_fact are associated with a lane
    and remain visible to correctly scoped reads.  Returns None only when
    no default group_id is configured (backward-compatible).
    """
    try:
        from ..graphiti_mcp_server import config as _server_config
        gid = _server_config.graphiti.group_id
        if gid and isinstance(gid, str) and gid.strip():
            return gid.strip()
    except (ImportError, AttributeError, Exception):
        pass
    return None


def _build_state_fact(
    *,
    subject: str,
    predicate: str,
    value: Any,
    fact_type: str,
    scope: str,
    source_key: str,
    source_lane: str | None = None,
    parent: StateFact | None = None,
    version: int = 1,
) -> StateFact:
    object_id = _stable_object_id(str(uuid4()))
    root_id = parent.root_id if parent is not None else object_id
    parent_id = parent.object_id if parent is not None else None
    # Inherit source_lane from parent when superseding; otherwise use provided
    # or derive from server config.
    effective_lane = source_lane
    if effective_lane is None and parent is not None:
        effective_lane = parent.source_lane
    if effective_lane is None:
        effective_lane = _derive_source_lane()
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
            'source_key': source_key,
            'source_lane': effective_lane,
            'evidence_refs': _build_evidence_ref(source_key),
        }
    )
    return fact


async def remember_fact(
    text: str,
    hint: dict[str, Any] | None = None,
    *,
    _server_principal: str | None = None,
) -> dict[str, Any]:
    """Primary owner-asserted fact memory API.

    Args:
        text: Natural-language or structured fact text.
        hint: Optional extraction hints (fact_type, subject, scope, trust, etc.).
        _server_principal: Server-verified caller principal derived from the MCP
            auth context layer (set by the tool wrapper from ``ctx``).  This is
            the ONLY source of trust for privilege elevation; caller-supplied
            ``hint.trust.actor_id`` is informational only and does NOT affect
            authorization.  Not part of the public MCP tool contract.

    Returns:
        - a typed fact dict on successful ledger write, or
        - ConflictDialog dict when contradictory input is detected, or
        - ErrorResponse-style dict on validation/write failures.
    """
    if not isinstance(text, str):
        return {
            'status': 'error',
            'error_type': 'validation_error',
            'message': 'text must be a string',
        }
    clean = _clean_text(text)

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

    hint_ok, hint_error = _validate_hint_contract(hint)
    if not hint_ok:
        return {
            'status': 'error',
            'error_type': 'validation_error',
            'message': hint_error,
        }

    hint_bounds_ok, hint_bounds_error = _validate_payload_bounds(
        hint,
        label='hint',
        max_json_bytes=MAX_HINT_JSON_BYTES,
        max_depth=MAX_HINT_DEPTH,
    )
    if not hint_bounds_ok:
        return {
            'status': 'error',
            'error_type': 'validation_error',
            'message': hint_bounds_error,
        }

    write_context = _resolve_write_context(hint, server_principal=_server_principal)

    schema_requested = 'type' in hint or 'type_hint' in hint or 'fact_type' in hint
    raw_type = hint.get('type') or hint.get('type_hint') or hint.get('fact_type')

    if raw_type is not None:
        fact_type = _normalize_fact_type(str(raw_type))
        confidence = 1.0
    else:
        fact_type, confidence = _infer_fact_type(clean)

    scope = _coerce_scope(
        hint.get('scope')
        or hint.get('policy_scope')
        or write_context.get('scope_override')
    )

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
    schema_candidate = typed_candidate
    if schema_name != 'TypedFact':
        schema_candidate = {
            'subject': typed_candidate['subject'],
            'predicate': typed_candidate['predicate'],
            'value': typed_candidate['value'],
        }

    valid, error = _validate_typed_object(schema_candidate, schema_name, strict=True)
    if not valid:
        return {
            'status': 'error',
            'error_type': 'validation_error',
            'message': error,
        }

    # Even when specific schema validation passed, enforce core typed schema.
    valid, error = _validate_typed_object(typed_candidate, 'TypedFact', strict=True)
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
                'evidence_refs': _build_evidence_ref(write_context['source_key']),
                'version': int(existing_payload.get('version', 1) or 1),
            }
        ))
        return {
            'status': 'duplicate',
            'message': 'Fact already exists in current state',
            'fact': duplicate,
        }

    supersede_requested = _coerce_bool(hint.get('supersede'))
    supersede_allowed = bool(
        supersede_requested and write_context.get('allow_conflict_supersede')
    )

    if is_conflict and not supersede_allowed:
        # Conflict is quarantined by default; `hint.supersede` is gated by trusted authz.
        candidate_metadata = {
            'fact_type': fact_type,
            'confidence': confidence,
            'input_text': clean,
        }
        if isinstance(hint.get('metadata'), dict):
            candidate_metadata['hint_metadata'] = hint['metadata']

        metadata_ok, metadata_error = _validate_payload_bounds(
            candidate_metadata,
            label='candidate_metadata',
            max_json_bytes=MAX_AUX_PAYLOAD_JSON_BYTES,
            max_depth=MAX_AUX_PAYLOAD_DEPTH,
        )
        if not metadata_ok:
            return {
                'status': 'error',
                'error_type': 'validation_error',
                'message': metadata_error,
            }

        raw_hint_ok, raw_hint_error = _validate_payload_bounds(
            hint,
            label='hint',
            max_json_bytes=MAX_AUX_PAYLOAD_JSON_BYTES,
            max_depth=MAX_AUX_PAYLOAD_DEPTH,
        )
        if not raw_hint_ok:
            return {
                'status': 'error',
                'error_type': 'validation_error',
                'message': raw_hint_error,
            }

        candidate_raw_hint = {**hint}
        candidate_raw_hint['scope'] = scope
        candidate_raw_hint['policy_scope'] = scope
        # Preserve canonical lane identity so the promoted candidate is visible
        # to the same authorized lane-scoped surfaces as a direct ledger write.
        # _derive_source_lane() reads the server's configured group_id; it returns
        # None when no default group is configured (no-op for non-lane deployments).
        _candidate_source_lane = _derive_source_lane()
        if _candidate_source_lane:
            candidate_raw_hint['source_lane'] = _candidate_source_lane
        if write_context.get('verified'):
            trust_snapshot = dict(candidate_raw_hint.get('trust') or {})
            trust_snapshot.update(
                {
                    'verified': True,
                    'is_owner': bool(write_context.get('is_owner')),
                    'actor_id': write_context['actor_id'],
                    'source': write_context['source'],
                    'scope': write_context.get('scope_override') or scope,
                    'allow_conflict_supersede': bool(write_context.get('allow_conflict_supersede')),
                }
            )
            candidate_raw_hint['trust'] = trust_snapshot
        candidate_metadata['scope'] = scope

        try:
            candidate = _get_candidate_store().create_candidate(
                payload={
                    'subject': subject,
                    'predicate': predicate,
                    'value': value,
                    'fact_type': fact_type,
                },
                conflict_with_fact_id=(
                    existing_payload.get('object_id') if isinstance(existing_payload, dict) else None
                ),
                source=write_context['source'],
                raw_hint=candidate_raw_hint,
                metadata=candidate_metadata,
            )
            _log_audit(
                'quarantine_conflict',
                candidate_id=candidate.get('candidate_id'),
                value=value,
                result='ok',
                actor_id=write_context['actor_id'],
                source=write_context['source'],
            )
        except Exception as exc:
            _log_audit(
                'quarantine_conflict',
                error=str(exc),
                value=value,
                result='error',
                actor_id=write_context['actor_id'],
                source=write_context['source'],
            )
            return {
                'status': 'error',
                'error_type': 'candidate_error',
                'message': str(exc),
            }

        conflict_existing = (
            _serialize_fact(existing_fact)
            if existing_fact
            else (
                _state_fact_as_dict(existing_payload)
                if isinstance(existing_payload, dict)
                else existing_payload or {}
            )
        )
        return {
            'type': 'ConflictDialog',
            'status': 'conflict',
            'message': 'Conflicting fact detected; quarantined for review.',
            'existing_fact': conflict_existing,
            'new_fact': _candidate_payload(candidate),
            'options': _CONFLICT_OPTIONS,
            'resolve_via': (
                'promote_candidate(candidate_id, resolution="supersede") '
                'or reject_candidate(candidate_id) — '
                'authorization is derived from the MCP auth context (OAuth bearer token / transport client_id), '
                'not from the actor_id argument; actor_id is an optional audit hint only'
            ),
            'candidate_id': candidate.get('candidate_id'),
            'candidate_uuid': candidate.get('candidate_id'),
            'supersede_requested': supersede_requested,
            'supersede_allowed': supersede_allowed,
        }

    parent = existing_fact if existing_fact and is_conflict and supersede_allowed else None
    version = int(parent.version + 1) if parent is not None else 1
    event_type = 'supersede' if parent is not None else 'assert'

    fact = _build_state_fact(
        subject=subject,
        predicate=predicate,
        value=value,
        fact_type=fact_type,
        scope=scope,
        source_key=write_context['source_key'],
        parent=parent,
        version=version,
    )

    event_metadata = {
        'source': write_context['source'],
        'scope': scope,
        'fact_type': fact_type,
        'input_text': clean,
        'confidence': confidence,
        'input_hint_keys': sorted(k for k in hint) if isinstance(hint, dict) else [],
        'superseded_fact_id': parent.object_id if parent is not None else None,
        'trust': {
            'verified': bool(write_context.get('verified')),
            'is_owner': bool(write_context.get('is_owner')),
        },
    }
    if isinstance(hint.get('metadata'), dict):
        event_metadata['hint_metadata'] = hint['metadata']

    metadata_ok, metadata_error = _validate_payload_bounds(
        event_metadata,
        label='event_metadata',
        max_json_bytes=MAX_AUX_PAYLOAD_JSON_BYTES,
        max_depth=MAX_AUX_PAYLOAD_DEPTH,
    )
    if not metadata_ok:
        return {
            'status': 'error',
            'error_type': 'validation_error',
            'message': metadata_error,
        }

    try:
        _get_change_ledger().append_event(
            event_type,
            actor_id=write_context['actor_id'],
            reason='supersede' if parent is not None else DEFAULT_ACTION,
            recorded_at=_now_iso(),
            object_type='state_fact',
            object_id=fact.object_id,
            target_object_id=parent.object_id if parent is not None else None,
            root_id=fact.root_id,
            parent_id=fact.parent_id,
            policy_version=hint.get('policy_version') if isinstance(hint, dict) else None,
            metadata=event_metadata,
            payload=fact,
        )
    except Exception as exc:
        _log_audit(
            'ledger_write_failed',
            fact=fact,
            error=str(exc),
            value=value,
            result='error',
            actor_id=write_context['actor_id'],
            source=write_context['source'],
        )
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
            source=write_context['source'],
            superseded_fact_id=parent.object_id if parent is not None else None,
        )
        if not materialized and materialization_error:
            logger.warning(
                'Neo4j materialization failed (non-blocking): fact=%s err=%s',
                fact.object_id,
                materialization_error,
            )
            _log_audit(
                'neo4j_write_failed',
                fact=fact,
                error=materialization_error,
                value=value,
                result='error',
                actor_id=write_context['actor_id'],
                source=write_context['source'],
            )
        elif materialized:
            _log_audit(
                'neo4j_write',
                fact=fact,
                value=value,
                result='ok',
                actor_id=write_context['actor_id'],
                source=write_context['source'],
            )
    except Exception as exc:
        materialization_error = str(exc)
        logger.warning('Neo4j materialization threw for fact=%s: %s', fact.object_id, exc)
        _log_audit(
            'neo4j_write_failed',
            fact=fact,
            error=materialization_error,
            value=value,
            result='error',
            actor_id=write_context['actor_id'],
            source=write_context['source'],
        )

    _log_audit(
        'ledger_write',
        fact=fact,
        value=value,
        result='ok',
        actor_id=write_context['actor_id'],
        source=write_context['source'],
    )

    response = {
        'status': 'ok',
        'fact': _serialize_fact(fact),
        'materialized': materialized,
        'materialization_error': materialization_error,
        'neo4j_materialization': {
            'status': 'ok' if materialized else 'failed',
            'error': materialization_error,
        },
        'audit': {
            'path': 'ledger_primary',
            'action': event_type,
            'actor_id': write_context['actor_id'],
            'source': write_context['source'],
            'neo4j_status': 'ok' if materialized else 'deferred_or_failed',
        },
    }

    if not materialized:
        response['warnings'] = [
            {
                'code': 'neo4j_materialization_failed',
                'message': materialization_error or 'Neo4j materialization failed',
            }
        ]

    return response


def _resolve_lane_scope(
    group_ids: list[str] | None = None,
    lane_alias: list[str] | None = None,
) -> list[str] | None:
    """Resolve caller-supplied group_ids/lane_alias into an effective lane scope.

    Fail-closed semantics:
    - Returns ``None`` only when no lane scope is active AND the server
      has no ``authorized_group_ids`` restriction.
    - Returns ``[]`` when the resolved scope is empty (deny all).
    - When scope parameters are omitted but the server has
      ``authorized_group_ids`` configured, returns that list so omitted
      params do not widen access beyond the server-authorized scope.

    Delegates alias resolution to the main server's _resolve_effective_group_ids
    when available; falls back to direct group_ids pass-through otherwise.
    """
    # Always try the server-level resolver so authorized_group_ids is enforced
    # even when the caller omits explicit scope params.
    try:
        from ..graphiti_mcp_server import _resolve_effective_group_ids
        effective, invalid = _resolve_effective_group_ids(
            group_ids=group_ids,
            lane_alias=lane_alias,
        )
        if invalid:
            # Invalid aliases: treat as empty scope (no results) for safety.
            return []

        # If caller explicitly requested scope, respect the result
        if group_ids is not None or lane_alias is not None:
            return effective

        # Caller omitted scope params — check if server restricts access
        try:
            from ..graphiti_mcp_server import config as _server_config
            authorized = _server_config.graphiti.authorized_group_ids
            if authorized:
                # Server has authorized restrictions; scope to those lanes
                # so omitting params doesn't widen beyond authorized scope.
                return list(authorized)
        except (ImportError, AttributeError, Exception):
            pass

        # No server restrictions and no explicit scope → all lanes visible
        if effective:
            return effective
        return None

    except (ImportError, Exception):
        # Fallback: use group_ids directly when the server resolver is
        # unavailable (e.g. in isolated unit tests).
        if group_ids is not None:
            return list(group_ids) if group_ids else []
        if lane_alias is not None:
            return []  # can't resolve aliases without server → fail closed
        return None


def _fact_passes_lane_filter(fact: Any, lane_scope: list[str] | None) -> bool:
    """Return True if a fact's source_lane is within the allowed lane scope."""
    if lane_scope is None:
        return True
    source_lane = getattr(fact, 'source_lane', None)
    if source_lane is None:
        # Facts with no source_lane are only visible when no lane filter is active.
        # Since lane_scope is not None here, exclude them to prevent cross-lane leakage.
        return False
    return source_lane in lane_scope


async def get_current_state(
    subject: str,
    predicate: str | None = None,
    scope: str | None = None,
    group_ids: list[str] | None = None,
    lane_alias: list[str] | None = None,
) -> dict[str, Any]:
    """Return currently active state facts for a subject/predicate in typed form.

    Args:
        subject: The fact subject to filter on.
        predicate: Optional predicate filter.
        scope: Visibility scope (default 'private').
        group_ids: Optional lane scope for filtering (intersected with server-authorized scope).
        lane_alias: Optional lane aliases resolved via server config.
    """
    if not subject or not str(subject).strip():
        return {'status': 'error', 'error_type': 'validation_error', 'message': 'subject is required'}

    lane_scope = _resolve_lane_scope(group_ids=group_ids, lane_alias=lane_alias)
    state_facts = [
        _serialize_fact(fact)
        for fact in _current_state_facts(subject=subject, predicate=predicate, scope=_coerce_scope(scope))
        if _fact_passes_lane_filter(fact, lane_scope)
    ]
    return {
        'status': 'ok',
        'facts': state_facts,
    }


async def get_history(
    subject: str,
    predicate: str | None = None,
    scope: str | None = None,
    group_ids: list[str] | None = None,
    lane_alias: list[str] | None = None,
) -> dict[str, Any]:
    """Return ledger event history for the currently active root lineage(s).

    Phase-0 semantics: walks roots of currently active (non-superseded) facts only.
    It does NOT scan every historical root ever associated with a subject.  Superseded
    lineages that are no longer active are excluded; callers must not expect a full
    audit-log scan across all past versions.

    ``roots_considered`` in the response lists the root IDs that were walked, so
    callers can see the exact scope of the history returned.

    Use ``get_current_state`` to inspect only the latest fact values without history.

    Args:
        subject: The fact subject to filter on.
        predicate: Optional predicate filter.
        scope: Visibility scope (default 'private').
        group_ids: Optional lane scope for filtering (intersected with server-authorized scope).
        lane_alias: Optional lane aliases resolved via server config.
    """
    if not subject or not str(subject).strip():
        return {'status': 'error', 'error_type': 'validation_error', 'message': 'subject is required'}

    scope_value = _coerce_scope(scope)
    lane_scope = _resolve_lane_scope(group_ids=group_ids, lane_alias=lane_alias)
    ledger = _get_change_ledger()
    history: list[dict[str, Any]] = []
    seen_roots: set[str] = set()

    for fact in _current_state_facts(subject=subject, predicate=predicate, scope=scope_value):
        # Lane-scope filter: skip facts not in the caller's authorized lane scope
        if not _fact_passes_lane_filter(fact, lane_scope):
            continue
        if fact.root_id in seen_roots:
            continue
        seen_roots.add(fact.root_id)
        for row in ledger.events_for_root(fact.root_id):
            history.append(
                {
                    'event_id': row.event_id,
                    'event_type': row.event_type,
                    'recorded_at': row.recorded_at,
                    'actor_id': row.actor_id,
                    'reason': row.reason,
                    'object_id': row.object_id,
                    'target_object_id': row.target_object_id,
                    'root_id': row.root_id,
                    'parent_id': row.parent_id,
                    'candidate_id': row.candidate_id,
                }
            )

    return {
        'status': 'ok',
        'history': history,
        'scope': scope_value,
        'roots_considered': sorted(seen_roots),
    }


def register_tools(mcp: FastMCP) -> dict[str, Any]:
    """Register all memory tools with the MCP server instance."""

    # FastMCP defaults the exported tool name to the Python function name.
    # Use the public contract names explicitly so runtime discovery/invocation
    # matches get_tools() and the documented Exec 1 surface.
    #
    # ctx is annotated as _McpContext | None so FastMCP's find_context_parameter
    # recognises it as the transport-injected Context (not a caller-supplied
    # argument) and excludes it from the caller-visible tool schema.  This
    # enables path-2 auth (ctx.client_id) for non-OAuth transports.
    @mcp.tool(name='remember_fact')
    async def remember_fact_tool(text: str, hint: dict[str, Any] | None = None, ctx: _McpContext | None = None) -> dict[str, Any]:
        server_principal = _extract_server_principal(ctx)
        return await remember_fact(text=text, hint=hint, _server_principal=server_principal)

    @mcp.tool(name='get_current_state')
    async def get_current_state_tool(
        subject: str,
        predicate: str | None = None,
        scope: str | None = None,
        group_ids: list[str] | None = None,
        lane_alias: list[str] | None = None,
    ) -> dict[str, Any]:
        return await get_current_state(
            subject=subject, predicate=predicate, scope=scope,
            group_ids=group_ids, lane_alias=lane_alias,
        )

    @mcp.tool(name='get_history')
    async def get_history_tool(
        subject: str,
        predicate: str | None = None,
        scope: str | None = None,
        group_ids: list[str] | None = None,
        lane_alias: list[str] | None = None,
    ) -> dict[str, Any]:
        return await get_history(
            subject=subject, predicate=predicate, scope=scope,
            group_ids=group_ids, lane_alias=lane_alias,
        )

    tool_map = {
        'remember_fact': remember_fact_tool,
        'get_current_state': get_current_state_tool,
        'get_history': get_history_tool,
    }

    # Backward compatible tool names used by external discovery and lightweight tests.
    if hasattr(mcp, '_tools') and isinstance(mcp._tools, dict):  # type: ignore[attr-defined]
        mcp._tools.update(tool_map)  # type: ignore[attr-defined]

    return tool_map


__all__ = [
    'register_tools',
    'remember_fact',
    'get_current_state',
    'get_history',
    '_change_ledger',
    '_trusted_actor_ids_from_env',
    '_extract_server_principal',
    '_ANON_PRINCIPAL',
]
