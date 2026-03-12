from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PACK_SCOPES = {'context', 'workflow'}
PACK_REGISTRY_SCHEMA_VERSION = '1.0.0'

MAX_PACK_ID_LENGTH = 128
MAX_INTENT_LENGTH = 512
MAX_PREDICATE_COUNT = 64
MAX_PREDICATE_LENGTH = 128
MAX_CONSUMER_LENGTH = 128
MAX_DESCRIPTION_LENGTH = 8_192
MAX_DEFINITION_DEPTH = 10
MAX_DEFINITION_NODES = 2_000
MAX_DEFINITION_ITEMS = 256
MAX_DEFINITION_STRING_LENGTH = 8_192
MAX_DEFINITION_JSON_BYTES = 128 * 1024


DEFAULT_REGISTRY: dict[str, Any] = {
    'schema_version': PACK_REGISTRY_SCHEMA_VERSION,
    'meta': {
        'created_at': '',
        'updated_at': '',
    },
    'packs': [
        {
            'id': 'context-vc-deal-brief',
            'scope': 'context',
            'intent': 'vc_deal_brief',
            'description': 'Facts that help summarize a venture context for downstream workflow',
            'consumer': 'planner',
            'version': '1.0.0',
            'predicates': ['industry', 'stage', 'founder', 'investor', 'valuation', 'risk'],
            'created_at': '',
            'last_updated': '',
            'definition': {
                'schema': {
                    'type': 'object',
                    'required': ['subject', 'predicate', 'value'],
                    'properties': {
                        'subject': {'type': 'string'},
                        'predicate': {'type': 'string'},
                        'value': {'type': ['string', 'number', 'boolean', 'object', 'array', 'null']},
                    },
                },
                'examples': [
                    {
                        'subject': 'Yuan',
                        'predicate': 'industry',
                        'value': 'financial_infrastructure',
                    },
                    {
                        'subject': 'DealX',
                        'predicate': 'stage',
                        'value': 'seed',
                    },
                ],
                'instructions': 'Use these facts to build a concise venture-readiness snapshot.',
            },
        },
        {
            'id': 'context-crypto-constraints',
            'scope': 'context',
            'intent': 'compliance',
            'description': 'Security/compliance constraints for memory materialization',
            'consumer': 'planner',
            'version': '1.0.0',
            'predicates': ['requires', 'forbidden', 'constraint', 'policy'],
            'created_at': '',
            'last_updated': '',
            'definition': {
                'schema': {
                    'type': 'object',
                    'required': ['subject', 'predicate', 'value'],
                    'properties': {
                        'subject': {'type': 'string'},
                        'predicate': {'type': 'string'},
                        'value': {'type': ['string', 'number', 'boolean', 'object', 'array', 'null']},
                    },
                },
                'examples': [
                    {
                        'subject': 'portfolio',
                        'predicate': 'requires',
                        'value': {'type': 'KYC', 'level': 'high'},
                    }
                ],
                'instructions': 'Keep only high-confidence constraints relevant to the task.',
            },
        },
        {
            'id': 'workflow-deal-review',
            'scope': 'workflow',
            'intent': 'decision_maker',
            'description': 'Workflow sequence for deal review and escalation',
            'consumer': 'planner',
            'version': '1.0.0',
            'predicates': ['decision', 'constraint', 'compliance', 'risk'],
            'created_at': '',
            'last_updated': '',
            'definition': {
                'trigger': 'new_candidate_fact',
                'steps': [
                    {'step': 'validate', 'action': 'check facts with constraints'},
                    {'step': 'prioritize', 'action': 'rank by recency/confidence'},
                    {'step': 'execute', 'action': 'build short brief'},
                ],
                'schema': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'required': ['step', 'action'],
                        'properties': {
                            'step': {'type': 'string'},
                            'action': {'type': 'string'},
                        },
                    },
                },
                'examples': [
                    {
                        'step': 'validate',
                        'action': 'check latest preference facts',
                    }
                ],
                'instructions': 'Use all gathered facts as inputs for each step decision.',
            },
        },
        {
            'id': 'workflow-standup',
            'scope': 'workflow',
            'intent': 'verifier',
            'description': 'Operational daily standup workflow',
            'consumer': 'planner',
            'version': '1.0.0',
            'predicates': ['task', 'status', 'blocker', 'priority'],
            'created_at': '',
            'last_updated': '',
            'definition': {
                'trigger': 'standup_requested',
                'steps': [
                    {'step': 'gather', 'action': 'collect open items and blockers'},
                    {'step': 'summarize', 'action': 'rank by urgency and deadline'},
                    {'step': 'report', 'action': 'emit concise status'},
                ],
                'schema': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'required': ['step', 'action'],
                    },
                },
                'instructions': 'Favor recent facts with explicit priority markers.',
            },
        },
    ],
}


class PackRegistryError(ValueError):
    """Raised when caller-supplied pack inputs fail validation."""


class PackRegistryOperationalError(PackRegistryError):
    """Raised when registry files cannot be loaded or persisted safely."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _resolved_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _public_registry_path() -> Path:
    return Path(__file__).resolve().parents[2] / 'data' / 'pack_registry.json'


def _configured_user_registry_path() -> Path | None:
    override = os.getenv('BICAMERAL_USER_PACK_REGISTRY_PATH')
    if override and override.strip():
        return Path(override.strip())
    return None


def _default_registry_path() -> Path:
    user_override = _configured_user_registry_path()
    if user_override is not None:
        return user_override

    legacy_override = os.getenv('BICAMERAL_PACK_REGISTRY_PATH')
    if legacy_override and legacy_override.strip():
        return Path(legacy_override.strip())

    return _public_registry_path()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _normalize_pack_id(value: Any) -> str:
    pack_id = (str(value or '').strip().lower()).replace(' ', '-')
    if not pack_id:
        raise PackRegistryError('pack id is required')
    if len(pack_id) > MAX_PACK_ID_LENGTH:
        raise PackRegistryError(f'pack id exceeds max length ({MAX_PACK_ID_LENGTH})')
    if not re.match(r'^[a-z0-9][a-z0-9._-]*$', pack_id):
        raise PackRegistryError(f'invalid pack id: {pack_id!r}')
    return pack_id


def _normalize_version(value: Any) -> str:
    version = str(value or '1.0.0').strip()
    if not version:
        return '1.0.0'
    if not re.match(r'^(\d+)\.(\d+)\.(\d+)$', version):
        raise PackRegistryError(f'invalid semantic version: {version!r}')
    return version


def _normalize_datetime(value: Any, *, field_name: str, default: str | None = None) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        if default is not None:
            return default
        raise PackRegistryError(f'{field_name} is required')

    if not isinstance(value, str):
        raise PackRegistryError(f'{field_name} must be a string')

    candidate = value.strip()
    if candidate.endswith('Z'):
        candidate = candidate[:-1] + '+00:00'

    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise PackRegistryError(f'{field_name} must be a valid ISO 8601 date-time: {value!r}') from exc

    if parsed.tzinfo is None:
        raise PackRegistryError(f'{field_name} must include a timezone offset')

    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _normalize_scope(value: Any) -> str:
    scope = str(value or '').strip().lower()
    if scope not in DEFAULT_PACK_SCOPES:
        raise PackRegistryError(f'invalid scope {scope!r}; expected context|workflow')
    return scope


def _normalize_predicate_filters(raw_predicates: Any, *, require_non_empty: bool) -> list[str]:
    predicates: list[str] = []
    seen: set[str] = set()
    for raw in _as_list(raw_predicates):
        predicate = str(raw).strip().lower()
        if not predicate:
            continue
        if len(predicate) > MAX_PREDICATE_LENGTH:
            raise PackRegistryError(f'predicate exceeds max length ({MAX_PREDICATE_LENGTH}): {predicate!r}')
        if predicate not in seen:
            predicates.append(predicate)
            seen.add(predicate)

    if len(predicates) > MAX_PREDICATE_COUNT:
        raise PackRegistryError(f'too many predicates; max={MAX_PREDICATE_COUNT}')
    if require_non_empty and not predicates:
        raise PackRegistryError('pack predicates must be a non-empty list')
    return predicates


def _validate_jsonish(value: Any, *, path: str, depth: int = 0, nodes: list[int] | None = None) -> None:
    if nodes is None:
        nodes = [0]

    nodes[0] += 1
    if nodes[0] > MAX_DEFINITION_NODES:
        raise PackRegistryError(f'{path} exceeds max node count ({MAX_DEFINITION_NODES})')
    if depth > MAX_DEFINITION_DEPTH:
        raise PackRegistryError(f'{path} exceeds max nesting depth ({MAX_DEFINITION_DEPTH})')

    if value is None or isinstance(value, (bool, int, float)):
        return

    if isinstance(value, str):
        if len(value) > MAX_DEFINITION_STRING_LENGTH:
            raise PackRegistryError(
                f'{path} string exceeds max length ({MAX_DEFINITION_STRING_LENGTH})'
            )
        return

    if isinstance(value, list):
        if len(value) > MAX_DEFINITION_ITEMS:
            raise PackRegistryError(f'{path} exceeds max list size ({MAX_DEFINITION_ITEMS})')
        for index, item in enumerate(value):
            _validate_jsonish(item, path=f'{path}[{index}]', depth=depth + 1, nodes=nodes)
        return

    if isinstance(value, dict):
        if len(value) > MAX_DEFINITION_ITEMS:
            raise PackRegistryError(f'{path} exceeds max object size ({MAX_DEFINITION_ITEMS})')
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise PackRegistryError(f'{path} contains a non-string or empty key')
            if len(key) > MAX_PREDICATE_LENGTH:
                raise PackRegistryError(f'{path}.{key} key exceeds max length ({MAX_PREDICATE_LENGTH})')
            _validate_jsonish(item, path=f'{path}.{key}', depth=depth + 1, nodes=nodes)
        return

    raise PackRegistryError(f'{path} contains unsupported type: {type(value).__name__}')


def _normalise_definition(raw_definition: Any, *, scope: str) -> dict[str, Any]:
    if not isinstance(raw_definition, dict):
        raise PackRegistryError('pack definition must be an object')

    _validate_jsonish(raw_definition, path='definition')

    if 'schema' in raw_definition and not isinstance(raw_definition['schema'], dict):
        raise PackRegistryError('definition.schema must be an object')

    if 'instructions' in raw_definition:
        instructions = raw_definition['instructions']
        if not isinstance(instructions, str) or not instructions.strip():
            raise PackRegistryError('definition.instructions must be a non-empty string when provided')

    if 'examples' in raw_definition and not isinstance(raw_definition['examples'], list):
        raise PackRegistryError('definition.examples must be a list when provided')

    if scope == 'workflow':
        steps = raw_definition.get('steps')
        if not isinstance(steps, list) or not steps:
            raise PackRegistryError(
                'definition.steps is required and must be a non-empty list for workflow packs'
            )
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                raise PackRegistryError(f'definition.steps[{index}] must be an object')
            step_name = str(step.get('step') or '').strip()
            action = str(step.get('action') or '').strip()
            if not step_name or not action:
                raise PackRegistryError(
                    f'definition.steps[{index}] must include non-empty step and action fields'
                )

    encoded = json.dumps(raw_definition, ensure_ascii=False, sort_keys=True)
    if len(encoded.encode('utf-8')) > MAX_DEFINITION_JSON_BYTES:
        raise PackRegistryError(f'definition exceeds max encoded size ({MAX_DEFINITION_JSON_BYTES} bytes)')

    return deepcopy(raw_definition)


def _pack_id_from_row(row: dict[str, Any]) -> str:
    raw_id = row.get('id')
    raw_pack_id = row.get('pack_id')

    if raw_id is not None and raw_pack_id is not None:
        normalized_id = _normalize_pack_id(raw_id)
        normalized_pack_id = _normalize_pack_id(raw_pack_id)
        if normalized_id != normalized_pack_id:
            raise PackRegistryError('id and pack_id must match when both are provided')
        return normalized_id

    if raw_id is not None:
        return _normalize_pack_id(raw_id)
    return _normalize_pack_id(raw_pack_id)


def _normalise_row(raw: dict[str, Any]) -> dict[str, Any]:
    row = dict(raw)
    if not row:
        raise PackRegistryError('pack record is empty')

    scope_value = row.get('scope')
    if scope_value is None and row.get('type') is not None:
        scope_value = row.get('type')
    scope = _normalize_scope(scope_value)

    pack_id = _pack_id_from_row(row)

    intent = str(row.get('intent') or '').strip().lower()
    if not intent:
        raise PackRegistryError(f'pack {pack_id} is missing required field: intent')
    if len(intent) > MAX_INTENT_LENGTH:
        raise PackRegistryError(
            f'pack {pack_id} intent exceeds max length ({MAX_INTENT_LENGTH})'
        )

    consumer = str(row.get('consumer') or '').strip().lower()
    if not consumer:
        raise PackRegistryError(f'pack {pack_id} is missing required field: consumer')
    if len(consumer) > MAX_CONSUMER_LENGTH:
        raise PackRegistryError(
            f'pack {pack_id} consumer exceeds max length ({MAX_CONSUMER_LENGTH})'
        )

    definition_raw = row.get('definition')
    if definition_raw is None:
        definition_raw = {}

    predicates = _normalize_predicate_filters(
        row.get('predicates')
        or row.get('predicate_patterns')
        or (definition_raw.get('predicates') if isinstance(definition_raw, dict) else None),
        require_non_empty=True,
    )

    definition = _normalise_definition(definition_raw, scope=scope)

    description = str(row.get('description') or '').strip()
    if not description:
        description = f'{scope} pack for {intent}'
    if len(description) > MAX_DESCRIPTION_LENGTH:
        raise PackRegistryError(
            f'pack {pack_id} description exceeds max length ({MAX_DESCRIPTION_LENGTH})'
        )

    now_iso = _now_iso()
    created_at = _normalize_datetime(
        row.get('created_at'),
        field_name=f'pack {pack_id} created_at',
        default=now_iso,
    )
    last_updated = _normalize_datetime(
        row.get('last_updated'),
        field_name=f'pack {pack_id} last_updated',
        default=created_at,
    )

    version = _normalize_version(row.get('version'))

    return {
        'id': pack_id,
        'scope': scope,
        'intent': intent,
        'description': description,
        'consumer': consumer,
        'version': version,
        'predicates': predicates,
        'created_at': created_at,
        'last_updated': last_updated,
        'definition': definition,
    }


def _normalise_filter(filter: dict[str, Any] | None) -> dict[str, str | None]:
    if not filter:
        return {}
    if not isinstance(filter, dict):
        raise PackRegistryError('filter must be an object')

    scope = filter.get('scope')
    if scope is not None:
        scope = _normalize_scope(scope)

    intent = filter.get('intent')
    intent = str(intent).strip().lower() if intent is not None else None
    consumer = filter.get('consumer')
    consumer = str(consumer).strip().lower() if consumer is not None else None

    return {
        'scope': scope,
        'intent': intent,
        'consumer': consumer,
    }


def _match_filter(row: dict[str, Any], filter: dict[str, Any]) -> bool:
    if filter.get('scope') and row.get('scope') != filter['scope']:
        return False
    if filter.get('intent') and row.get('intent') != str(filter['intent']).strip().lower():
        return False
    return not (
        filter.get('consumer') and row.get('consumer') != str(filter['consumer']).strip().lower()
    )


class PackRegistryService:
    """Persistent registry for context/workflow packs.

    Built-in packs are read from the public repo registry. User-created packs are read from and
    written to a separate overlay registry when `BICAMERAL_USER_PACK_REGISTRY_PATH` (or an
    explicit path) is provided. Reads merge both sources; writes target only the user registry.
    """

    def __init__(self, path: str | Path | None = None):
        self.public_path = _public_registry_path()
        self.path = Path(path) if path else _default_registry_path()
        self.public_path.parent.mkdir(parents=True, exist_ok=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._registry: dict[str, Any] | None = None

    def _is_public_registry(self) -> bool:
        return _resolved_path(self.path) == _resolved_path(self.public_path)

    def _empty_registry(self, *, include_builtin_packs: bool) -> dict[str, Any]:
        now = _now_iso()
        packs = DEFAULT_REGISTRY['packs'] if include_builtin_packs else []
        return {
            'schema_version': PACK_REGISTRY_SCHEMA_VERSION,
            'meta': {
                'created_at': now,
                'updated_at': now,
            },
            'packs': [_normalise_row(row) for row in packs],
        }

    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        lock_path = self.path.parent / f'.{self.path.name}.lock'
        with lock_path.open('a+', encoding='utf-8') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _write_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                'w',
                dir=path.parent,
                prefix=f'.{path.name}.',
                suffix='.tmp',
                delete=False,
                encoding='utf-8',
            ) as handle:
                tmp_name = handle.name
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        finally:
            if tmp_name and os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def _load_file(self, path: Path, *, include_builtin_packs: bool) -> dict[str, Any]:
        if not path.exists():
            payload = self._empty_registry(include_builtin_packs=include_builtin_packs)
            self._write_atomic(path, payload)
            return payload

        try:
            raw = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(raw, dict):
                raise PackRegistryError('registry root must be a JSON object')
            packs_raw = raw.get('packs', [])
            if not isinstance(packs_raw, list):
                raise PackRegistryError('registry packs must be a list')

            packs: list[dict[str, Any]] = []
            for item in packs_raw:
                if not isinstance(item, dict):
                    raise PackRegistryError('registry pack entries must be objects')
                packs.append(_normalise_row(item))

            if include_builtin_packs and not packs:
                packs = self._empty_registry(include_builtin_packs=True)['packs']

            now = _now_iso()
            return {
                'schema_version': str(raw.get('schema_version') or PACK_REGISTRY_SCHEMA_VERSION),
                'meta': {
                    'created_at': str((raw.get('meta') or {}).get('created_at') or now),
                    'updated_at': str((raw.get('meta') or {}).get('updated_at') or now),
                },
                'packs': packs,
            }
        except (OSError, json.JSONDecodeError, PackRegistryError) as exc:
            raise PackRegistryOperationalError(f'failed to load registry file {path}: {exc}') from exc

    def _merge_registries(self, base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
        merged_packs: list[dict[str, Any]] = []
        seen: set[str] = set()

        for registry in (base, overlay):
            for pack in registry.get('packs', []):
                pack_id = pack['id']
                if pack_id in seen:
                    raise PackRegistryError(f'duplicate pack id across registry sources: {pack_id}')
                merged_packs.append(dict(pack))
                seen.add(pack_id)

        return {
            'schema_version': PACK_REGISTRY_SCHEMA_VERSION,
            'meta': {
                'created_at': base.get('meta', {}).get('created_at') or overlay.get('meta', {}).get('created_at'),
                'updated_at': overlay.get('meta', {}).get('updated_at') or base.get('meta', {}).get('updated_at'),
            },
            'packs': merged_packs,
        }

    def _load(self) -> dict[str, Any]:
        try:
            if self._is_public_registry():
                return self._load_file(self.public_path, include_builtin_packs=True)

            base = self._load_file(self.public_path, include_builtin_packs=True)
            overlay = self._load_file(self.path, include_builtin_packs=False)
            return self._merge_registries(base, overlay)
        except PackRegistryOperationalError:
            raise
        except PackRegistryError as exc:
            raise PackRegistryOperationalError(str(exc)) from exc

    def _ensure_loaded(self) -> dict[str, Any]:
        if self._registry is None:
            self._registry = self._load()
        return self._registry

    def _update_meta(self, payload: dict[str, Any]) -> None:
        payload.setdefault('meta', {})
        now = _now_iso()
        payload['meta']['updated_at'] = now
        payload.setdefault('schema_version', PACK_REGISTRY_SCHEMA_VERSION)
        payload['meta'].setdefault('created_at', now)

    def list_packs(self, *, filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        filter_payload = _normalise_filter(filter)
        registry = self._ensure_loaded()
        return [dict(pack) for pack in registry.get('packs', []) if _match_filter(pack, filter_payload)]

    def get_pack(self, pack_id: str) -> dict[str, Any] | None:
        normalized_pack_id = _normalize_pack_id(pack_id)
        for pack in self._ensure_loaded().get('packs', []):
            if pack['id'] == normalized_pack_id:
                return dict(pack)
        return None

    def _assert_user_registry_writeable(self) -> None:
        if self._is_public_registry():
            raise PackRegistryError(
                'create_workflow_pack requires BICAMERAL_USER_PACK_REGISTRY_PATH (or an explicit '
                'non-public registry path); refusing to write user packs into the public repo registry'
            )

    def create_pack(self, definition: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(definition, dict):
            raise PackRegistryError('definition must be an object')

        self._assert_user_registry_writeable()

        scope_value = definition.get('scope')
        if scope_value is None and definition.get('type') is not None:
            scope_value = definition.get('type')
        scope = _normalize_scope(scope_value)
        if scope != 'workflow':
            raise PackRegistryError('create_workflow_pack requires scope=workflow')

        row = dict(definition)
        row.setdefault('id', row.get('pack_id'))
        row.setdefault('version', PACK_REGISTRY_SCHEMA_VERSION)
        normalized = _normalise_row(row)

        with self._write_lock():
            base = self._load_file(self.public_path, include_builtin_packs=True)
            overlay = self._load_file(self.path, include_builtin_packs=False)

            existing_ids = {pack['id'] for pack in base.get('packs', [])}
            existing_ids.update(pack['id'] for pack in overlay.get('packs', []))
            if normalized['id'] in existing_ids:
                raise PackRegistryError(f'pack already exists: {normalized["id"]}')

            overlay.setdefault('packs', []).append(normalized)
            self._update_meta(overlay)
            try:
                self._write_atomic(self.path, overlay)
            except OSError as exc:
                raise PackRegistryOperationalError(
                    f'failed to write registry file {self.path}: {exc}'
                ) from exc

        self.refresh()
        return dict(normalized)

    def get_schema(self) -> dict[str, Any]:
        return {
            'schema_version': PACK_REGISTRY_SCHEMA_VERSION,
            'required_fields': ['id', 'scope', 'intent', 'consumer', 'predicates', 'version', 'definition'],
            'scope_values': sorted(DEFAULT_PACK_SCOPES),
        }

    def refresh(self) -> None:
        self._registry = None
        self._ensure_loaded()
