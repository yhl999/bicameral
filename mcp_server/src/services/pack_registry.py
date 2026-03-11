from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PACK_SCOPES = {'context', 'workflow'}


PACK_REGISTRY_SCHEMA_VERSION = '1.0.0'


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
    """Raised when a registry payload cannot be normalised."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _default_registry_path() -> Path:
    override = os.getenv('BICAMERAL_USER_PACK_REGISTRY_PATH')
    if not override:
        override = os.getenv('BICAMERAL_PACK_REGISTRY_PATH')
    if override:
        return Path(override)

    return Path(__file__).resolve().parents[2] / 'data' / 'pack_registry.json'


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return [str(value)]
    return [str(item) for item in value if str(item).strip()]


def _normalize_pack_id(value: Any) -> str:
    pack_id = (str(value or '').strip().lower()).replace(' ', '-')
    if not pack_id:
        raise PackRegistryError('pack id is required')
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


def _normalize_scope(value: Any) -> str:
    scope = str(value or '').strip().lower()
    if not scope and str(value or '').strip().lower():
        scope = str(value).strip().lower()

    if scope == 'type':
        scope = 'workflow'
    if scope not in DEFAULT_PACK_SCOPES:
        raise PackRegistryError(f'invalid scope {scope!r}; expected context|workflow')
    return scope


def _normalize_predicate_filters(raw_predicates: Any) -> list[str]:
    patterns = _as_list(raw_predicates)
    if not patterns:
        return []
    # Store lower-cased to get deterministic matching for lookup.
    return [pattern.strip().lower() for pattern in patterns if pattern.strip()]


def _normalise_definition(raw_definition: Any) -> dict[str, Any]:
    if not isinstance(raw_definition, dict):
        return {}
    return deepcopy(raw_definition)


def _normalise_row(raw: dict[str, Any]) -> dict[str, Any]:
    row = dict(raw)
    if not row:
        raise PackRegistryError('pack record is empty')

    scope = row.get('scope')
    if scope is None:
        scope = row.get('type')
    scope = _normalize_scope(scope)

    pack_id = _normalize_pack_id(row.get('id') if 'id' in row else row.get('pack_id'))
    intent = str(row.get('intent') or '').strip().lower()
    if not intent:
        raise PackRegistryError(f'pack {pack_id} is missing required field: intent')

    consumer = str(row.get('consumer') or '').strip().lower()
    if not consumer:
        raise PackRegistryError(f'pack {pack_id} is missing required field: consumer')

    predicates = _normalize_predicate_filters(
        row.get('predicates')
        or row.get('predicate_patterns')
        or row.get('definition', {}).get('predicates')
        if isinstance(row.get('definition'), dict)
        else None,
    )
    if scope == 'workflow' and not predicates:
        # Workflow packs should be queryable by predicate, even if minimal.
        predicates = []
    if scope != 'workflow' and not predicates:
        raise PackRegistryError(f'pack {pack_id} is missing predicates')

    definition = _normalise_definition(row.get('definition'))

    created_at = str(row.get('created_at') or _now_iso())
    last_updated = str(row.get('last_updated') or created_at)

    description = str(row.get('description') or '').strip()
    if not description:
        description = f'{scope} pack for {intent}'

    version = _normalize_version(row.get('version'))

    normalized = {
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

    return normalized


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
    if filter.get('consumer') and row.get('consumer') != str(filter['consumer']).strip().lower():
        return False
    return True


class PackRegistryService:
    """Persistent registry for context/workflow packs.

    Backed by a JSON file so that pack definitions survive process restart.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else _default_registry_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._registry: dict[str, Any] | None = None

    def _empty_registry(self) -> dict[str, Any]:
        payload = deepcopy(DEFAULT_REGISTRY)
        now = _now_iso()
        payload['schema_version'] = PACK_REGISTRY_SCHEMA_VERSION
        payload['meta']['created_at'] = now
        payload['meta']['updated_at'] = now
        normalized: list[dict[str, Any]] = []
        for row in payload.get('packs', []):
            normalized.append(_normalise_row(row))
        payload['packs'] = normalized
        return payload

    def _write_atomic(self, payload: dict[str, Any]) -> None:
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
        tmp.replace(self.path)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            payload = self._empty_registry()
            self._write_atomic(payload)
            return payload

        try:
            raw = json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(raw, dict):
                raise PackRegistryError('registry root must be a JSON object')
            packs_raw = raw.get('packs', [])
            if not isinstance(packs_raw, list):
                raise PackRegistryError('registry packs must be a list')

            packs: list[dict[str, Any]] = []
            for item in packs_raw:
                if isinstance(item, dict):
                    packs.append(_normalise_row(item))

            now = _now_iso()
            payload: dict[str, Any] = {
                'schema_version': str(raw.get('schema_version') or PACK_REGISTRY_SCHEMA_VERSION),
                'meta': {
                    'created_at': str((raw.get('meta') or {}).get('created_at') or now),
                    'updated_at': str((raw.get('meta') or {}).get('updated_at') or now),
                },
                'packs': packs,
            }
            if not payload['packs']:
                payload['packs'] = self._empty_registry().get('packs', [])
            return payload
        except (OSError, json.JSONDecodeError, PackRegistryError) as exc:
            raise PackRegistryError(f'failed to load registry file {self.path}: {exc}')

    def _ensure_loaded(self) -> dict[str, Any]:
        if self._registry is None:
            self._registry = self._load()
        return self._registry

    def _update_meta(self, payload: dict[str, Any]) -> None:
        payload.setdefault('meta', {})
        now = _now_iso()
        payload['meta']['updated_at'] = now
        payload.setdefault('schema_version', PACK_REGISTRY_SCHEMA_VERSION)
        payload.setdefault('meta', {}).setdefault('created_at', now)

    def list_packs(self, *, filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        filter_payload = _normalise_filter(filter)
        registry = self._ensure_loaded()
        return [dict(p) for p in registry.get('packs', []) if _match_filter(p, filter_payload)]

    def get_pack(self, pack_id: str) -> dict[str, Any] | None:
        pack_id = _normalize_pack_id(pack_id)
        for pack in self._ensure_loaded().get('packs', []):
            if pack['id'] == pack_id:
                return dict(pack)
        return None

    def _upsert_pack(self, row: dict[str, Any]) -> dict[str, Any]:
        registry = self._ensure_loaded()
        if 'packs' not in registry or not isinstance(registry['packs'], list):
            registry['packs'] = []

        normalized = _normalise_row(row)

        for idx, existing in enumerate(registry['packs']):
            if existing['id'] == normalized['id']:
                normalized['created_at'] = existing.get('created_at', normalized['created_at'])
                registry['packs'][idx] = normalized
                break
        else:
            registry['packs'].append(normalized)

        self._update_meta(registry)
        self._write_atomic(registry)
        self._registry = registry
        return dict(normalized)

    def create_pack(self, definition: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(definition, dict):
            raise PackRegistryError('definition must be an object')

        row = dict(definition)
        scope = row.get('scope')
        if scope is None:
            scope = row.get('type')

        scope = _normalize_scope(scope)
        if scope != 'workflow':
            raise PackRegistryError('create_workflow_pack requires scope=workflow')

        row.setdefault('id', row.get('pack_id'))
        row.setdefault('version', PACK_REGISTRY_SCHEMA_VERSION)

        return self._upsert_pack(row)

    def get_schema(self) -> dict[str, Any]:
        return {
            'schema_version': PACK_REGISTRY_SCHEMA_VERSION,
            'required_fields': ['id', 'scope', 'intent', 'consumer', 'predicates', 'version', 'definition'],
            'scope_values': sorted(DEFAULT_PACK_SCOPES),
        }

    def refresh(self) -> None:
        """Drop local cache and reload from file."""
        self._registry = None
        self._ensure_loaded()

