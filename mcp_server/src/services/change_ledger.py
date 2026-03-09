from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Relative import — no try/except needed: this module is always imported as
# part of the mcp_server.src.services package, never run as a top-level script.
from ..models.typed_memory import (
    EntityRegistry,
    EntityRegistryEntry,
    Episode,
    EvidenceRef,
    Procedure,
    StateFact,
    TypedMemoryObject,
    coerce_typed_object,
)

CANONICAL_EVENT_TYPES = frozenset(
    {
        'assert',
        'supersede',
        'invalidate',
        'refine',
        'derive',
        'promote',
        'procedure_success',
        'procedure_failure',
    }
)
CREATE_EVENT_TYPES = frozenset({'assert', 'supersede', 'refine', 'derive'})
DB_PATH_DEFAULT = Path(__file__).resolve().parents[3] / 'state' / 'change_ledger.db'


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS change_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    actor_id TEXT,
    reason TEXT,
    object_id TEXT,
    target_object_id TEXT,
    object_type TEXT,
    root_id TEXT,
    parent_id TEXT,
    candidate_id TEXT,
    policy_version TEXT,
    payload_json TEXT,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_change_events_root_recorded_at
    ON change_events(root_id, recorded_at, event_id);
CREATE INDEX IF NOT EXISTS idx_change_events_object_recorded_at
    ON change_events(object_id, recorded_at, event_id);
CREATE INDEX IF NOT EXISTS idx_change_events_target_recorded_at
    ON change_events(target_object_id, recorded_at, event_id);
"""


@dataclass(frozen=True)
class ChangeEventRow:
    event_id: str
    event_type: str
    recorded_at: str
    actor_id: str | None
    reason: str | None
    object_id: str | None
    target_object_id: str | None
    object_type: str | None
    root_id: str | None
    parent_id: str | None
    candidate_id: str | None
    policy_version: str | None
    payload_json: str | None
    metadata_json: str | None


@dataclass(frozen=True)
class CandidatePromotionResult:
    object_id: str
    root_id: str
    event_id: str
    event_ids: list[str]


class ChangeLedger:
    def __init__(self, conn_or_path: sqlite3.Connection | str | Path = DB_PATH_DEFAULT):
        if isinstance(conn_or_path, sqlite3.Connection):
            self.conn = conn_or_path
        else:
            self.conn = connect(conn_or_path)
        self.conn.row_factory = sqlite3.Row
        ensure_schema(self.conn)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _do_insert(self, row: ChangeEventRow) -> None:
        """Insert a ChangeEventRow into the DB without committing.

        Callers are responsible for calling conn.commit() when they want to
        persist.  Use append_event (autocommit=True by default) for the normal
        single-event path.
        """
        self.conn.execute(
            """
            INSERT INTO change_events(
                event_id, event_type, recorded_at, actor_id, reason,
                object_id, target_object_id, object_type, root_id, parent_id,
                candidate_id, policy_version, payload_json, metadata_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                row.event_id,
                row.event_type,
                row.recorded_at,
                row.actor_id,
                row.reason,
                row.object_id,
                row.target_object_id,
                row.object_type,
                row.root_id,
                row.parent_id,
                row.candidate_id,
                row.policy_version,
                row.payload_json,
                row.metadata_json,
            ),
        )

    def _build_event_row(
        self,
        event_type: str,
        *,
        actor_id: str | None = None,
        reason: str | None = None,
        recorded_at: str | None = None,
        object_type: str | None = None,
        object_id: str | None = None,
        target_object_id: str | None = None,
        root_id: str | None = None,
        parent_id: str | None = None,
        candidate_id: str | None = None,
        policy_version: str | None = None,
        payload: TypedMemoryObject | dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChangeEventRow:
        """Build a ChangeEventRow (no DB interaction)."""
        normalized_type = _normalize_event_type(event_type)
        if normalized_type not in CANONICAL_EVENT_TYPES:
            raise ValueError(
                f'Unsupported canonical event type {event_type!r}. Allowed: {sorted(CANONICAL_EVENT_TYPES)}'
            )

        payload_json: str | None = None
        if normalized_type in CREATE_EVENT_TYPES:
            if payload is None:
                raise ValueError(f'{normalized_type} requires payload')
            typed_object = coerce_typed_object(payload)
            typed_object = _prepare_object_for_create_event(
                typed_object,
                event_type=normalized_type,
                recorded_at=recorded_at,
                root_id=root_id,
                parent_id=parent_id or target_object_id,
            )
            object_id = typed_object.object_id
            object_type = typed_object.object_type
            root_id = typed_object.root_id
            parent_id = typed_object.parent_id
            payload_json = _canonical_json(typed_object.model_dump(mode='json'))
        else:
            object_id = object_id or target_object_id
            if object_id is None:
                raise ValueError(f'{normalized_type} requires object_id or target_object_id')

        return ChangeEventRow(
            event_id=_new_event_id(),
            event_type=normalized_type,
            recorded_at=recorded_at or _now_iso(),
            actor_id=actor_id,
            reason=reason,
            object_id=object_id,
            target_object_id=target_object_id,
            object_type=object_type,
            root_id=root_id,
            parent_id=parent_id,
            candidate_id=candidate_id,
            policy_version=policy_version,
            payload_json=payload_json,
            metadata_json=_canonical_json(metadata) if metadata is not None else None,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def append_event(
        self,
        event_type: str,
        *,
        actor_id: str | None = None,
        reason: str | None = None,
        recorded_at: str | None = None,
        object_type: str | None = None,
        object_id: str | None = None,
        target_object_id: str | None = None,
        root_id: str | None = None,
        parent_id: str | None = None,
        candidate_id: str | None = None,
        policy_version: str | None = None,
        payload: TypedMemoryObject | dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        _autocommit: bool = True,
    ) -> ChangeEventRow:
        """Append a single event to the ledger.

        _autocommit=False is an internal flag used by promote_candidate_fact to
        batch two inserts into a single atomic transaction.  External callers
        should always leave it True (the default).
        """
        row = self._build_event_row(
            event_type,
            actor_id=actor_id,
            reason=reason,
            recorded_at=recorded_at,
            object_type=object_type,
            object_id=object_id,
            target_object_id=target_object_id,
            root_id=root_id,
            parent_id=parent_id,
            candidate_id=candidate_id,
            policy_version=policy_version,
            payload=payload,
            metadata=metadata,
        )
        self._do_insert(row)
        if _autocommit:
            self.conn.commit()
        return row

    def events_for_root(self, root_id: str) -> list[ChangeEventRow]:
        rows = self.conn.execute(
            """
            SELECT *
              FROM change_events
             WHERE root_id = ?
                OR object_id IN (SELECT object_id FROM change_events WHERE root_id = ?)
                OR target_object_id IN (SELECT object_id FROM change_events WHERE root_id = ?)
             ORDER BY recorded_at, rowid
            """,
            (root_id, root_id, root_id),
        ).fetchall()
        return [_row_to_event(row) for row in rows]

    def events_for_object(self, object_id: str) -> list[ChangeEventRow]:
        root_id = self.root_id_for_object(object_id) or object_id
        return self.events_for_root(root_id)

    def root_id_for_object(self, object_id: str) -> str | None:
        row = self.conn.execute(
            """
            SELECT COALESCE(root_id, object_id) AS root_id
              FROM change_events
             WHERE object_id = ? OR target_object_id = ?
             ORDER BY recorded_at, rowid
             LIMIT 1
            """,
            (object_id, object_id),
        ).fetchone()
        if row is None:
            return None
        return str(row['root_id']) if row['root_id'] else None

    def object_id_for_event(self, event_id: str) -> str | None:
        """Return the object_id recorded for a given event_id, or None."""
        row = self.conn.execute(
            "SELECT object_id FROM change_events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        return str(row['object_id']) if row and row['object_id'] else None

    def materialize_lineage(self, root_id: str) -> list[TypedMemoryObject]:
        events = self.events_for_root(root_id)
        return project_objects(events)

    def materialize_object(self, object_id: str) -> TypedMemoryObject | None:
        objects = self.materialize_lineage(self.root_id_for_object(object_id) or object_id)
        for obj in objects:
            if obj.object_id == object_id:
                return obj
        return None

    def current_object(self, root_id: str) -> TypedMemoryObject | None:
        objects = self.materialize_lineage(root_id)
        for obj in reversed(objects):
            if obj.is_current:
                return obj
        return None

    def current_state_facts(self) -> list[StateFact]:
        roots = [str(row['root_id']) for row in self.conn.execute(
            "SELECT DISTINCT root_id FROM change_events WHERE root_id IS NOT NULL ORDER BY root_id"
        ).fetchall()]
        facts: list[StateFact] = []
        for root_id in roots:
            current = self.current_object(root_id)
            if isinstance(current, StateFact):
                facts.append(current)
        return facts

    def entity_registry(self) -> EntityRegistry:
        entries: dict[str, EntityRegistryEntry] = {}
        for fact in self.current_state_facts():
            if fact.fact_type != 'relationship':
                continue
            entity_id = str(fact.value.get('entity_id') or '') if isinstance(fact.value, dict) else ''
            if not entity_id:
                continue
            current_name = str(fact.value.get('current_name') or fact.subject)
            aliases = _ensure_str_list(fact.value.get('aliases')) if isinstance(fact.value, dict) else []
            previous_names = _ensure_str_list(fact.value.get('previous_names')) if isinstance(fact.value, dict) else []
            external_ids = []
            if isinstance(fact.value, dict):
                for item in fact.value.get('external_ids') or []:
                    if isinstance(item, dict) and item.get('system') and item.get('value'):
                        external_ids.append(item)
            entries[entity_id] = EntityRegistryEntry.model_validate(
                {
                    'entity_id': entity_id,
                    'entity_type': str(fact.value.get('entity_type') or 'entity') if isinstance(fact.value, dict) else 'entity',
                    'current_name': current_name,
                    'aliases': aliases,
                    'previous_names': previous_names,
                    'external_ids': external_ids,
                }
            )
        return EntityRegistry(list(entries.values()))

    def promote_candidate_fact(
        self,
        *,
        actor_id: str,
        reason: str,
        policy_version: str,
        candidate_id: str,
        fact: dict[str, Any],
        conflict_with_fact_id: str | None = None,
        seeded_supersede_ok: bool = False,  # kept for API compat; gate removed — manual approval always supersedes
        recorded_at: str | None = None,
    ) -> CandidatePromotionResult:
        """Promote a candidate fact into the ledger.

        Atomicity guarantee: the create/supersede event and the promote event
        are inserted in a single SQLite transaction (_autocommit=False on both
        append_event calls, followed by a single conn.commit()).  If anything
        raises between the two inserts, neither event is committed.

        One-current-object rule: when conflict_with_fact_id is provided and the
        prior object exists in the ledger, this method ALWAYS performs a supersede
        (regardless of seeded_supersede_ok).  Manual approval by a human actor is
        explicit authorization to supersede; policy auto-supersede sets
        seeded_supersede_ok=True in the trace, but the gate has been lifted for
        both paths to avoid a second "current" fact in the same conflict set.
        """
        recorded_at = recorded_at or _now_iso()
        typed_object = build_object_from_candidate_fact(
            candidate_id=candidate_id,
            fact=fact,
            policy_version=policy_version,
            recorded_at=recorded_at,
        )

        event_ids: list[str] = []
        creation_type = 'assert'
        parent_id: str | None = None
        root_id = typed_object.root_id

        # ── Central enforcement: one current object per conflict set ─────────
        # Architecture lock (Phase 0): there must be exactly one is_current=True
        # object per (subject, predicate, scope) triple at all times.
        #
        # The caller's conflict_with_fact_id is unreliable in two ways:
        #   - missing: caller omitted it entirely (upstream wiring gap)
        #   - stale:   caller provided an object_id that is no longer the
        #              current occupant of the conflict set (e.g. an already-
        #              superseded predecessor)
        #
        # In either case we must find the *actual* current occupant and use it
        # as the supersession target.  This scan runs only for StateFact
        # promotions (conflict sets are defined only on state facts).
        if isinstance(typed_object, StateFact):
            _conflict_set = typed_object.conflict_set
            for _current in self.current_state_facts():
                if (
                    _current.conflict_set == _conflict_set
                    and _current.object_id != typed_object.object_id
                ):
                    # Override whatever (or nothing) the caller passed — the
                    # actual current occupant is the authoritative target.
                    conflict_with_fact_id = _current.object_id
                    break

        # Always supersede when a conflicting prior fact is explicitly identified
        # and still exists in the ledger.  This enforces the one-current-object
        # rule regardless of whether the promotion is automated or manual.
        if conflict_with_fact_id:
            prior = self.materialize_object(conflict_with_fact_id)
            if prior is not None:
                creation_type = 'supersede'
                parent_id = prior.object_id
                root_id = prior.root_id
                typed_object = typed_object.model_copy(
                    update={
                        'root_id': prior.root_id,
                        'parent_id': prior.object_id,
                        'version': prior.version + 1,
                    }
                )

        # Build both rows first (validation happens here; no DB writes yet).
        create_row = self._build_event_row(
            creation_type,
            actor_id=actor_id,
            reason=reason,
            recorded_at=recorded_at,
            payload=typed_object,
            target_object_id=parent_id,
            candidate_id=candidate_id,
            policy_version=policy_version,
        )
        promote_row = self._build_event_row(
            'promote',
            actor_id=actor_id,
            reason=reason,
            recorded_at=recorded_at,
            object_id=typed_object.object_id,
            object_type=typed_object.object_type,
            root_id=root_id,
            candidate_id=candidate_id,
            policy_version=policy_version,
        )

        # Atomic: insert both rows then commit once.
        # If either insert raises, explicitly rollback so the first insert is
        # not left as an open (uncommitted but visible) write on this connection.
        try:
            self._do_insert(create_row)
            self._do_insert(promote_row)
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

        event_ids = [create_row.event_id, promote_row.event_id]

        return CandidatePromotionResult(
            object_id=typed_object.object_id,
            root_id=root_id,
            event_id=promote_row.event_id,
            event_ids=event_ids,
        )


def connect(path: str | Path = DB_PATH_DEFAULT) -> sqlite3.Connection:
    """Open (or create) the change_ledger SQLite DB with canonical pragmas.

    WAL mode: improves concurrent read performance and durability for the
    append-only ledger workload.
    foreign_keys: enforced as a matter of correctness hygiene; the ledger
    schema has no FK constraints today but this is defensive practice for
    future schema evolution.
    """
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    # WAL mode: better concurrency for readers; durable write-ahead log.
    conn.execute('PRAGMA journal_mode=WAL')
    # foreign_keys: ON for schema correctness hygiene.
    conn.execute('PRAGMA foreign_keys=ON')
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def project_objects(events: list[ChangeEventRow | sqlite3.Row]) -> list[TypedMemoryObject]:
    objects: dict[str, TypedMemoryObject] = {}
    ordered_events = [_row_to_event(event) if isinstance(event, sqlite3.Row) else event for event in events]
    for event in ordered_events:
        if event.event_type in CREATE_EVENT_TYPES:
            if not event.payload_json:
                raise ValueError(f'{event.event_type} event missing payload_json')
            payload = json.loads(event.payload_json)
            obj = coerce_typed_object(payload)
            if event.event_type in {'supersede', 'refine'}:
                parent_id = event.target_object_id or obj.parent_id
                if parent_id and parent_id in objects:
                    parent = objects[parent_id].model_copy(
                        update={
                            'is_current': False,
                            'superseded_by': obj.object_id,
                            'lifecycle_status': 'superseded',
                        }
                    )
                    objects[parent_id] = parent
                    obj = obj.model_copy(
                        update={
                            'parent_id': obj.parent_id or parent_id,
                            'root_id': parent.root_id,
                            'version': max(obj.version, parent.version + 1),
                            'lifecycle_status': 'refined' if event.event_type == 'refine' else obj.lifecycle_status,
                        }
                    )
            elif event.event_type == 'derive':
                obj = obj.model_copy(update={'lifecycle_status': 'derived'})
            objects[obj.object_id] = obj
            continue

        target_id = event.target_object_id or event.object_id
        if not target_id or target_id not in objects:
            continue
        current = objects[target_id]
        if event.event_type == 'promote':
            updates: dict[str, Any] = {'lifecycle_status': 'promoted'}
            if isinstance(current, (StateFact, Procedure)):
                updates['promotion_status'] = 'promoted'
            objects[target_id] = current.model_copy(update=updates)
        elif event.event_type == 'invalidate':
            objects[target_id] = current.model_copy(
                update={
                    'is_current': False,
                    'invalid_at': event.recorded_at,
                    'lifecycle_status': 'invalidated',
                }
            )
        elif event.event_type == 'procedure_success' and isinstance(current, Procedure):
            objects[target_id] = current.model_copy(update={'success_count': current.success_count + 1})
        elif event.event_type == 'procedure_failure' and isinstance(current, Procedure):
            objects[target_id] = current.model_copy(update={'fail_count': current.fail_count + 1})
    return sorted(objects.values(), key=lambda obj: (obj.root_id, obj.version, obj.object_id))


def build_object_from_candidate_fact(
    *,
    candidate_id: str,
    fact: dict[str, Any],
    policy_version: str,
    recorded_at: str,
) -> TypedMemoryObject:
    assertion_type = str(fact.get('assertion_type') or '').strip().lower()
    evidence_refs = [
        item if isinstance(item, EvidenceRef) else EvidenceRef.from_legacy_ref(item)
        for item in (fact.get('evidence_refs') or [])
    ]
    if not evidence_refs:
        raise ValueError('candidate promotion requires evidence_refs')

    source_lane = _source_lane_from_legacy_refs(fact.get('evidence_refs') or [])
    source_key = _source_key_from_legacy_refs(fact.get('evidence_refs') or [])
    base = {
        'object_id': _stable_object_id(candidate_id),
        'root_id': _stable_object_id(candidate_id),
        'version': 1,
        'source_lane': source_lane,
        'source_key': source_key,
        'policy_scope': str(fact.get('scope') or 'private'),
        'visibility_scope': str(fact.get('scope') or 'private'),
        'evidence_refs': evidence_refs,
        'created_at': recorded_at,
        'valid_at': recorded_at,
    }

    if assertion_type == 'episode':
        value = fact.get('value')
        summary = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
        return Episode.model_validate(
            {
                **base,
                'title': str(fact.get('predicate') or 'episode'),
                'summary': summary,
            }
        )

    return StateFact.model_validate(
        {
            **base,
            'fact_type': _state_fact_type(assertion_type, str(fact.get('predicate') or '')),
            'subject': str(fact.get('subject') or ''),
            'predicate': str(fact.get('predicate') or ''),
            'value': fact.get('value'),
            'scope': str(fact.get('scope') or 'private'),
            'candidate_id': candidate_id,
            'policy_version': policy_version,
        }
    )


def _prepare_object_for_create_event(
    obj: TypedMemoryObject,
    *,
    event_type: str,
    recorded_at: str | None,
    root_id: str | None,
    parent_id: str | None,
) -> TypedMemoryObject:
    updates: dict[str, Any] = {}
    if root_id:
        updates['root_id'] = root_id
    if parent_id:
        updates['parent_id'] = parent_id
    if recorded_at:
        updates['created_at'] = recorded_at
        updates['valid_at'] = recorded_at
    if event_type == 'derive':
        updates['lifecycle_status'] = 'derived'
    elif event_type == 'refine':
        updates['lifecycle_status'] = 'refined'
    return obj.model_copy(update=updates) if updates else obj


def _row_to_event(row: sqlite3.Row) -> ChangeEventRow:
    return ChangeEventRow(
        event_id=str(row['event_id']),
        event_type=str(row['event_type']),
        recorded_at=str(row['recorded_at']),
        actor_id=row['actor_id'],
        reason=row['reason'],
        object_id=row['object_id'],
        target_object_id=row['target_object_id'],
        object_type=row['object_type'],
        root_id=row['root_id'],
        parent_id=row['parent_id'],
        candidate_id=row['candidate_id'],
        policy_version=row['policy_version'],
        payload_json=row['payload_json'],
        metadata_json=row['metadata_json'],
    )


def _normalize_event_type(event_type: str) -> str:
    normalized = str(event_type or '').strip().lower()
    return normalized


def _stable_object_id(seed: str) -> str:
    digest = hashlib.sha256(f'object|{seed}'.encode('utf-8')).hexdigest()
    return f'obj_{digest[:24]}'


def _new_event_id() -> str:
    """Generate a collision-safe event ID using cryptographic randomness.

    Uses 96 bits (12 bytes) of randomness from os.urandom, giving essentially
    zero collision probability even under concurrent writes across processes.
    The PRIMARY KEY constraint on event_id provides a hard DB-level guard.
    """
    return f'evt_{secrets.token_hex(12)}'


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _ensure_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def _state_fact_type(assertion_type: str, predicate: str) -> str:
    if assertion_type == 'preference':
        return 'preference'
    if assertion_type == 'decision':
        return 'decision'
    normalized = predicate.strip().lower()
    mapping = {
        'commitment.': 'commitment',
        'lesson.': 'lesson',
        'relationship.': 'relationship',
        'constraint.': 'constraint',
        'rule.': 'operational_rule',
        'operational_rule.': 'operational_rule',
    }
    for prefix, fact_type in mapping.items():
        if normalized.startswith(prefix):
            return fact_type
    return 'world_state'


def _source_lane_from_legacy_refs(refs: list[dict[str, Any]]) -> str | None:
    for ref in refs:
        scope = ref.get('scope')
        if isinstance(scope, str) and scope.strip():
            return scope.strip()
    return None


def _source_key_from_legacy_refs(refs: list[dict[str, Any]]) -> str | None:
    for ref in refs:
        source_key = ref.get('source_key')
        if isinstance(source_key, str) and source_key.strip():
            return source_key.strip()
    return None
