from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import quote

from pydantic import BaseModel, Field, model_validator


EvidenceKind = Literal[
    'qmd_chunk',
    'message',
    'file',
    'doc_chunk',
    'event_log',
    'crm_record',
    'sql_row',
    'api_resource',
]
StateFactSubtype = Literal[
    'preference',
    'decision',
    'commitment',
    'lesson',
    'world_state',
    'operational_rule',
    'constraint',
    'relationship',
]
RiskLevel = Literal['low', 'medium', 'high']
LifecycleStatus = Literal['asserted', 'derived', 'promoted', 'invalidated', 'superseded', 'refined']
ObjectType = Literal['state_fact', 'episode', 'procedure']
PromotionStatus = Literal['proposed', 'promoted']


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _require(locator: dict[str, Any], *keys: str) -> list[str]:
    missing = [key for key in keys if locator.get(key) in (None, '')]
    if missing:
        raise ValueError(f'locator missing required keys for canonical_uri: {missing}')
    return [str(locator[key]) for key in keys]


class EvidenceRef(BaseModel):
    kind: EvidenceKind
    source_system: str
    locator: dict[str, Any]
    canonical_uri: str | None = None
    title: str | None = None
    snippet: str | None = None
    observed_at: str | None = None
    retrieved_at: str | None = None
    hash: str | None = None

    @model_validator(mode='after')
    def ensure_canonical_uri(self) -> 'EvidenceRef':
        self.canonical_uri = self.canonical_uri or self.build_canonical_uri(self.kind, self.locator)
        return self

    @staticmethod
    def build_canonical_uri(kind: EvidenceKind, locator: dict[str, Any]) -> str:
        if kind == 'qmd_chunk':
            collection, document_id, chunk_id = _require(locator, 'collection', 'document_id', 'chunk_id')
            return f'qmd://{collection}/{document_id}#chunk={chunk_id}'
        if kind == 'message':
            system, conversation_id, message_id = _require(locator, 'system', 'conversation_id', 'message_id')
            return f'msg://{system}/{conversation_id}/{message_id}'
        if kind == 'file':
            path = str(locator.get('path') or '')
            if not path:
                raise ValueError('locator missing required keys for canonical_uri: [\'path\']')
            repo = str(locator.get('repo') or 'workspace')
            start_line = int(locator.get('start_line') or 1)
            end_line = int(locator.get('end_line') or start_line)
            return f'file://{repo}/{path}#L{start_line}-L{end_line}'
        if kind == 'doc_chunk':
            system, document_id, chunk_id = _require(locator, 'system', 'document_id', 'chunk_id')
            return f'doc://{system}/{document_id}#chunk={chunk_id}'
        if kind == 'event_log':
            system, stream, event_id = _require(locator, 'system', 'stream', 'event_id')
            return f'eventlog://{system}/{stream}/{event_id}'
        if kind == 'crm_record':
            system, object_type, record_id = _require(locator, 'system', 'object_type', 'record_id')
            return f'crm://{system}/{object_type}/{record_id}'
        if kind == 'sql_row':
            system, database, table = _require(locator, 'system', 'database', 'table')
            pk_json = quote(_canonical_json(locator.get('pk_json') or {}), safe='')
            return f'sql://{system}/{database}/{table}#pk={pk_json}'
        if kind == 'api_resource':
            system, resource_type, resource_id = _require(locator, 'system', 'resource_type', 'resource_id')
            return f'api://{system}/{resource_type}/{resource_id}'
        raise ValueError(f'Unsupported evidence kind: {kind}')

    @classmethod
    def from_legacy_ref(cls, ref: dict[str, Any]) -> 'EvidenceRef':
        source_key = str(ref.get('source_key') or ref.get('scope') or ref.get('source_family') or 'legacy').strip()
        system = source_key.split(':', 1)[0].strip().lower() if source_key else 'legacy'
        evidence_id = str(
            ref.get('evidence_id')
            or ref.get('chunk_key')
            or ref.get('start_id')
            or ref.get('end_id')
            or 'unknown'
        ).strip()
        locator = {
            'system': system or 'legacy',
            'stream': source_key or system or 'legacy',
            'event_id': evidence_id,
        }
        return cls(
            kind='event_log',
            source_system=system or 'legacy',
            locator=locator,
            title=ref.get('title'),
            snippet=ref.get('snippet'),
            observed_at=ref.get('observed_at'),
            retrieved_at=ref.get('retrieved_at'),
            hash=ref.get('hash'),
        )


class TypedMemoryObjectBase(BaseModel):
    object_id: str
    object_type: ObjectType
    root_id: str
    parent_id: str | None = None
    version: int = 1
    is_current: bool = True
    source_lane: str | None = None
    source_episode_id: str | None = None
    source_message_id: str | None = None
    source_key: str | None = None
    valid_at: str | None = None
    invalid_at: str | None = None
    superseded_by: str | None = None
    policy_scope: str
    visibility_scope: str
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    extractor_version: str | None = None
    created_at: str = Field(default_factory=_now_iso)
    lifecycle_status: LifecycleStatus = 'asserted'

    @model_validator(mode='after')
    def validate_contract(self) -> 'TypedMemoryObjectBase':
        if not self.evidence_refs:
            raise ValueError('typed memory objects require at least one evidence_ref')
        if self.version < 1:
            raise ValueError('version must be >= 1')
        if not self.root_id:
            self.root_id = self.object_id
        if self.valid_at is None:
            self.valid_at = self.created_at
        return self


class StateFact(TypedMemoryObjectBase):
    object_type: Literal['state_fact'] = 'state_fact'
    fact_type: StateFactSubtype
    subject: str
    predicate: str
    value: Any
    scope: str = 'private'
    candidate_id: str | None = None
    policy_version: str | None = None
    promotion_status: PromotionStatus = 'proposed'
    risk_level: RiskLevel = 'medium'

    @property
    def conflict_set(self) -> str:
        return f'{self.subject}\n{self.predicate}\n{self.scope}'


class Episode(TypedMemoryObjectBase):
    object_type: Literal['episode'] = 'episode'
    title: str | None = None
    summary: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    annotations: list[str] = Field(default_factory=list)


class Procedure(TypedMemoryObjectBase):
    object_type: Literal['procedure'] = 'procedure'
    name: str
    trigger: str
    preconditions: list[str] = Field(default_factory=list)
    steps: list[str]
    expected_outcome: str
    success_count: int = 0
    fail_count: int = 0
    risk_level: RiskLevel = 'medium'
    promotion_status: PromotionStatus = 'proposed'


TypedMemoryObject = StateFact | Episode | Procedure


class EntityExternalId(BaseModel):
    system: str
    value: str


class EntityRegistryEntry(BaseModel):
    entity_id: str
    entity_type: str
    current_name: str
    aliases: list[str] = Field(default_factory=list)
    previous_names: list[str] = Field(default_factory=list)
    external_ids: list[EntityExternalId] = Field(default_factory=list)

    def names(self) -> set[str]:
        return {
            value.strip().lower()
            for value in [self.current_name, *self.aliases, *self.previous_names]
            if isinstance(value, str) and value.strip()
        }

    def matches_name(self, candidate: str) -> bool:
        return candidate.strip().lower() in self.names()


class EntityRegistry:
    def __init__(self, entries: list[EntityRegistryEntry] | None = None):
        self._entries: dict[str, EntityRegistryEntry] = {}
        for entry in entries or []:
            self.upsert(entry)

    def upsert(self, entry: EntityRegistryEntry) -> None:
        self._entries[entry.entity_id] = entry

    def get(self, entity_id: str) -> EntityRegistryEntry | None:
        return self._entries.get(entity_id)

    def resolve_name(self, name: str) -> EntityRegistryEntry | None:
        needle = name.strip().lower()
        for entry in self._entries.values():
            if needle in entry.names():
                return entry
        return None

    def resolve_external_id(self, system: str, value: str) -> EntityRegistryEntry | None:
        for entry in self._entries.values():
            for external_id in entry.external_ids:
                if external_id.system == system and external_id.value == value:
                    return entry
        return None

    def values(self) -> list[EntityRegistryEntry]:
        return list(self._entries.values())


OBJECT_MODEL_BY_TYPE = {
    'state_fact': StateFact,
    'episode': Episode,
    'procedure': Procedure,
}


def coerce_typed_object(payload: TypedMemoryObject | dict[str, Any]) -> TypedMemoryObject:
    if isinstance(payload, (StateFact, Episode, Procedure)):
        return payload
    object_type = str(payload.get('object_type') or '').strip().lower()
    model = OBJECT_MODEL_BY_TYPE.get(object_type)
    if model is None:
        raise ValueError(f'Unsupported object_type: {object_type!r}')
    return model.model_validate(payload)
