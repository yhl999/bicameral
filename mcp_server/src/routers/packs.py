"""Pack router with materialization-backed context/workflow pack APIs."""

from __future__ import annotations

import fnmatch
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Defense-in-depth cap: materialized fact result sets are bounded regardless of
# how many matching facts exist in the ledger.  Mirrors _MAX_FACTS_CAP in
# graphiti_mcp_server.py.
MAX_PACK_MATERIALIZED_FACTS = 200

# Defense-in-depth cap for the `task` query input accepted by get_context_pack
# and get_workflow_pack.  Prevents unbounded string processing / DoS via the
# _matches_task token-scan path in a long-lived MCP server process.
MAX_TASK_QUERY_LENGTH = 2048

try:
    from ._phase0 import (
        error_response,
        require_dict,
        require_optional_dict,
        require_optional_non_empty_string,
        require_pack_id,
    )
except ImportError:  # pragma: no cover - script/top-level import fallback
    from _phase0 import (  # type: ignore[no-redef]
        error_response,
        require_dict,
        require_optional_dict,
        require_optional_non_empty_string,
        require_pack_id,
    )

from ..services.change_ledger import DB_PATH_DEFAULT, ChangeLedger
from ..services.pack_registry import (
    PackRegistryError,
    PackRegistryOperationalError,
    PackRegistryService,
)


def _to_iso_ts(value: Any) -> str:
    if isinstance(value, str) and value:
        return value
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _serialise_fact(fact: Any) -> dict[str, Any]:
    if hasattr(fact, 'model_dump'):
        return fact.model_dump(mode='json')
    if hasattr(fact, 'dict'):
        return fact.dict()
    if isinstance(fact, dict):
        return dict(fact)
    return {'value': fact}


def _to_text(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)):
        return str(value).lower()
    if value is None:
        return ''
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, sort_keys=True, ensure_ascii=False).lower()
        except (TypeError, ValueError):
            return str(value).lower()
    return str(value).lower()


def _parse_event_time(value: Any) -> float:
    if not value:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        normalized = str(value)
        if normalized.endswith('Z'):
            normalized = normalized[:-1] + '+00:00'
        return datetime.fromisoformat(normalized).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _fact_confidence(fact: Any) -> float:
    for path in (
        ('confidence',),
        ('history_meta', 'confidence'),
        ('value', 'confidence'),
    ):
        current: Any = fact
        for part in path:
            if isinstance(current, dict):
                current = current.get(part)
                continue
            if hasattr(current, part):
                current = getattr(current, part)
                continue
            current = None
            break
        if isinstance(current, (int, float)):
            return max(0.0, min(1.0, float(current)))
    return 0.5


def _predicates_for_pack(pack: dict[str, Any]) -> list[str]:
    patterns = pack.get('predicates', [])
    if not patterns:
        definition = pack.get('definition')
        if isinstance(definition, dict):
            patterns = definition.get('predicates') or definition.get('predicate_patterns', [])
    return [str(item).lower() for item in patterns if str(item).strip()]


def _matches_predicate(predicate: str, patterns: list[str]) -> bool:
    normalized = str(predicate or '').lower()
    return any(fnmatch.fnmatch(normalized, str(pattern).strip().lower()) for pattern in patterns)


def _matches_task(fact: Any, task: str | None) -> bool:
    if not task:
        return True

    tokens = [tok for tok in task.lower().split() if tok]
    if not tokens:
        return True

    serialised = [
        str(getattr(fact, 'subject', '')),
        str(getattr(fact, 'predicate', '')),
        _to_text(getattr(fact, 'value', None)),
        _to_text(_serialise_fact(fact).get('value')),
    ]
    blob = ' '.join(serialised).lower()
    return all(token in blob for token in tokens)


def _resolve_ledger_path(override: str | Path | None = None) -> Path:
    if override:
        return Path(override)
    env_override = os.getenv('BICAMERAL_CHANGE_LEDGER_PATH', '').strip()
    return Path(env_override) if env_override else Path(DB_PATH_DEFAULT)


def _materialize_pack_facts(
    *,
    pack: dict[str, Any],
    task: str | None,
    ledger_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    patterns = _predicates_for_pack(pack)
    if not patterns:
        return []

    with ChangeLedger(_resolve_ledger_path(ledger_path)) as ledger:
        current_facts = ledger.current_state_facts()

    selected: list[tuple[float, float, str, Any]] = []
    for fact in current_facts:
        if not _matches_predicate(fact.predicate, patterns):
            continue
        if not _matches_task(fact, task):
            continue

        selected_ts = _parse_event_time(
            getattr(fact, 'valid_at', None)
            or getattr(fact, 'created_at', None)
            or getattr(fact, 'updated_at', None)
            or getattr(fact, 'recorded_at', None),
        )
        selected.append((selected_ts, _fact_confidence(_serialise_fact(fact)), getattr(fact, 'object_id', ''), fact))

    selected.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    # Apply defence-in-depth cap: return only the most recent/confident facts.
    return [_serialise_fact(item[3]) for item in selected[:MAX_PACK_MATERIALIZED_FACTS]]


def _pack_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': row.get('id'),
        'scope': row.get('scope'),
        'intent': row.get('intent'),
        'description': row.get('description'),
        'consumer': row.get('consumer'),
        'version': row.get('version'),
        'predicates': row.get('predicates', []),
        'created_at': row.get('created_at'),
        'last_updated': row.get('last_updated'),
    }


def _infer_schema(definition: dict[str, Any] | None, *, scope: str | None = None) -> dict[str, Any]:
    if definition and isinstance(definition.get('schema'), dict):
        return definition['schema']

    if scope == 'workflow' or (definition and isinstance(definition.get('steps'), list) and definition['steps']):
        return {
            'type': 'array',
            'items': {
                'type': 'object',
                'required': ['step', 'action'],
                'properties': {
                    'step': {'type': 'string'},
                    'action': {'type': 'string'},
                },
            },
        }

    return {
        'type': 'object',
        'required': ['subject', 'predicate', 'value'],
        'properties': {
            'subject': {'type': 'string'},
            'predicate': {'type': 'string'},
            'value': {'type': ['string', 'number', 'boolean', 'object', 'array', 'null']},
        },
    }


def _pack_service_error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, PackRegistryOperationalError):
        return error_response('operational_error', message=str(exc))
    if isinstance(exc, PackRegistryError):
        return error_response('validation_error', message=str(exc))
    return error_response('operational_error', message=str(exc))


async def list_packs(filter: dict[str, Any] | None = None) -> list[dict[str, Any]] | dict[str, Any]:
    """List all registered packs filtered by scope/intent/consumer."""
    filter_error = require_optional_dict('filter', filter)
    if filter_error is not None:
        return filter_error

    try:
        service = PackRegistryService()
        return [_pack_metadata(pack) for pack in service.list_packs(filter=filter)]
    except Exception as exc:
        return _pack_service_error(exc)


async def get_context_pack(pack_id: str, task: str | None = None) -> dict[str, Any]:
    """Resolve context pack definition and materialized facts."""
    pack_id_error = require_pack_id('pack_id', pack_id)
    if pack_id_error is not None:
        return pack_id_error

    task_error = require_optional_non_empty_string('task', task)
    if task_error is not None:
        return task_error

    if task is not None and len(task) > MAX_TASK_QUERY_LENGTH:
        return error_response(
            'validation_error',
            message=f'task must not exceed {MAX_TASK_QUERY_LENGTH} characters',
            details={'field': 'task', 'max_length': MAX_TASK_QUERY_LENGTH},
        )

    try:
        service = PackRegistryService()
        pack = service.get_pack(pack_id)
        if not pack:
            return error_response('not_found', message=f'pack not found: {pack_id!r}')
        if pack.get('scope') != 'context':
            return error_response(
                'validation_error',
                message=f'pack {pack_id!r} is not a context pack',
            )

        facts = _materialize_pack_facts(pack=pack, task=task)
        return {
            'pack_id': pack['id'],
            'pack_metadata': _pack_metadata(pack),
            'facts': facts,
            'task_context': task,
            'materialized_at': _to_iso_ts(None),
            'fact_count': len(facts),
        }
    except Exception as exc:
        return _pack_service_error(exc)


async def get_workflow_pack(pack_id: str, task: str | None = None) -> dict[str, Any]:
    """Resolve workflow pack definition and materialized trigger facts."""
    pack_id_error = require_pack_id('pack_id', pack_id)
    if pack_id_error is not None:
        return pack_id_error

    task_error = require_optional_non_empty_string('task', task)
    if task_error is not None:
        return task_error

    if task is not None and len(task) > MAX_TASK_QUERY_LENGTH:
        return error_response(
            'validation_error',
            message=f'task must not exceed {MAX_TASK_QUERY_LENGTH} characters',
            details={'field': 'task', 'max_length': MAX_TASK_QUERY_LENGTH},
        )

    try:
        service = PackRegistryService()
        pack = service.get_pack(pack_id)
        if not pack:
            return error_response('not_found', message=f'pack not found: {pack_id!r}')
        if pack.get('scope') != 'workflow':
            return error_response(
                'validation_error',
                message=f'pack {pack_id!r} is not a workflow pack',
            )

        facts = _materialize_pack_facts(pack=pack, task=task)
        return {
            'pack_id': pack['id'],
            'pack_metadata': _pack_metadata(pack),
            'facts': facts,
            'task_context': task,
            'definition': pack.get('definition', {}),
            'materialized_at': _to_iso_ts(None),
            'fact_count': len(facts),
        }
    except Exception as exc:
        return _pack_service_error(exc)


async def describe_pack(pack_id: str) -> dict[str, Any]:
    """Return schema/definition metadata for a pack."""
    pack_id_error = require_pack_id('pack_id', pack_id)
    if pack_id_error is not None:
        return pack_id_error

    try:
        service = PackRegistryService()
        pack = service.get_pack(pack_id)
        if not pack:
            return error_response('not_found', message=f'pack not found: {pack_id!r}')

        definition = pack.get('definition')
        if not isinstance(definition, dict):
            definition = {}

        pack_registry = _pack_metadata(pack)
        return {
            'pack_id': pack['id'],
            'pack_registry': pack_registry,
            'predicates': pack.get('predicates', []),
            'schema': _infer_schema(definition, scope=pack.get('scope')),
            'examples': definition.get('examples', []),
            'instructions': definition.get('instructions'),
            'definition': definition,
        }
    except Exception as exc:
        return _pack_service_error(exc)


async def create_workflow_pack(definition: dict[str, Any]) -> dict[str, Any]:
    """Create a new workflow pack definition and persist it."""
    definition_error = require_dict('definition', definition)
    if definition_error is not None:
        return definition_error

    try:
        service = PackRegistryService()
        row = service.create_pack(definition)
        return _pack_metadata(row)
    except Exception as exc:
        return _pack_service_error(exc)


def register_tools(mcp: Any) -> dict[str, Any]:
    mcp.tool()(list_packs)
    mcp.tool()(get_context_pack)
    mcp.tool()(get_workflow_pack)
    mcp.tool()(describe_pack)
    mcp.tool()(create_workflow_pack)
    return {
        'list_packs': list_packs,
        'get_context_pack': get_context_pack,
        'get_workflow_pack': get_workflow_pack,
        'describe_pack': describe_pack,
        'create_workflow_pack': create_workflow_pack,
    }
