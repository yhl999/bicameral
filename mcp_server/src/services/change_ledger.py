from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
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
except ImportError:  # pragma: no cover - top-level import fallback
    from models.typed_memory import (
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


def resolve_ledger_path(override: str | Path | None = None) -> Path:
    """Return the effective ledger DB path, honoring env overrides.

    Resolution order:
    1. Explicit *override* argument (from caller / test fixture).
    2. ``BICAMERAL_CHANGE_LEDGER_DB`` env var.
    3. ``BICAMERAL_CHANGE_LEDGER_PATH`` env var (legacy alias).
    4. ``DB_PATH_DEFAULT`` (repo-relative ``state/change_ledger.db``).

    All integrated routers MUST use this helper (or pass the result of it)
    so that a non-default ledger path is honoured consistently across
    memory, candidates, episodes/procedures, and packs.
    """
    import os

    if override:
        return Path(override)
    env_db = (os.environ.get('BICAMERAL_CHANGE_LEDGER_DB') or '').strip()
    if env_db:
        return Path(env_db)
    env_path = (os.environ.get('BICAMERAL_CHANGE_LEDGER_PATH') or '').strip()
    if env_path:
        return Path(env_path)
    return Path(DB_PATH_DEFAULT)


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
CREATE UNIQUE INDEX IF NOT EXISTS ux_change_events_create_object_id
    ON change_events(object_id)
    WHERE object_id IS NOT NULL
      AND event_type IN ('assert', 'supersede', 'refine', 'derive');
CREATE TABLE IF NOT EXISTS typed_roots (
    root_id TEXT PRIMARY KEY,
    latest_recorded_at TEXT NOT NULL,
    object_type TEXT,
    source_lane TEXT,
    current_object_id TEXT,
    current_version INTEGER NOT NULL DEFAULT 0,
    current_payload_json TEXT,
    search_text TEXT NOT NULL DEFAULT '',
    lineage_event_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_typed_roots_latest_recorded_at
    ON typed_roots(latest_recorded_at DESC, root_id);
CREATE INDEX IF NOT EXISTS idx_typed_roots_object_type_latest
    ON typed_roots(object_type, latest_recorded_at DESC, root_id);
CREATE INDEX IF NOT EXISTS idx_typed_roots_source_lane_latest
    ON typed_roots(source_lane, latest_recorded_at DESC, root_id);
CREATE INDEX IF NOT EXISTS idx_typed_roots_source_lane_object_type_latest
    ON typed_roots(source_lane, object_type, latest_recorded_at DESC, root_id);
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
        root_id = self._root_id_for_event_row(row)
        if root_id:
            _refresh_typed_root_row(self.conn, root_id)

    def _root_id_for_event_row(self, row: ChangeEventRow) -> str | None:
        if row.root_id:
            return str(row.root_id)
        for object_id in (row.object_id, row.target_object_id):
            if not object_id:
                continue
            root_id = self.root_id_for_object(str(object_id))
            if root_id:
                return root_id
        return None

    def typed_root_snapshot(self, root_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM typed_roots WHERE root_id = ?",
            (root_id,),
        ).fetchone()

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

    def promotion_event_for_candidate(self, candidate_id: str) -> ChangeEventRow | None:
        """Return the first promote event for a candidate_id, or None.

        Used by promote_candidate() to reconcile after a partial failure where
        the ledger write succeeded but the candidates DB update did not commit.
        Querying by candidate_id instead of re-writing prevents duplicate
        promote events on retry.
        """
        row = self.conn.execute(
            """
            SELECT * FROM change_events
             WHERE candidate_id = ? AND event_type = 'promote'
             ORDER BY recorded_at, rowid
             LIMIT 1
            """,
            (candidate_id,),
        ).fetchone()
        return _row_to_event(row) if row else None

    def invalidate_event_for_object(self, object_id: str) -> ChangeEventRow | None:
        """Return the most recent invalidate event for object_id, or None.

        Used by deny_candidate() to reconcile after a partial failure where
        the ledger invalidate write succeeded but the candidates DB update did
        not commit.  Checking before writing prevents duplicate invalidate
        events on retry.
        """
        row = self.conn.execute(
            """
            SELECT * FROM change_events
             WHERE object_id = ? AND event_type = 'invalidate'
             ORDER BY recorded_at DESC, rowid DESC
             LIMIT 1
            """,
            (object_id,),
        ).fetchone()
        return _row_to_event(row) if row else None

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

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying SQLite connection.

        Safe to call multiple times; subsequent calls after the first are no-ops.
        Callers that open a ledger for a short-lived read (e.g. pack
        materialization inside the MCP server process) should call this when done,
        or use the context-manager form::

            with ChangeLedger(path) as ledger:
                facts = ledger.current_state_facts()
        """
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self) -> ChangeLedger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def promote_candidate_fact(
        self,
        *,
        actor_id: str,
        reason: str,
        policy_version: str,
        candidate_id: str,
        fact: dict[str, Any],
        conflict_with_fact_id: str | None = None,
        seeded_supersede_ok: bool = False,  # kept for API compat
        allow_parallel: bool = False,
        require_supersede: bool = False,
        recorded_at: str | None = None,
        manage_transaction: bool = True,
    ) -> CandidatePromotionResult:
        """Promote a candidate fact into the ledger.

        Atomicity guarantee: the create/supersede event and the promote event
        are inserted in a single SQLite transaction. If anything raises between
        the two inserts, neither event is committed.

        For historical compatibility, ``seeded_supersede_ok`` is accepted but no
        longer changes behavior by itself. ``require_supersede`` fails closed
        instead of silently degrading an explicit supersede request into a plain
        assert when no valid target can be materialized. ``manage_transaction``
        allows callers to compose candidate-row status transitions and ledger
        writes inside one outer transaction.
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
        requested_conflict_with_fact_id = str(conflict_with_fact_id or '').strip() or None
        resolved_conflict_with_fact_id = requested_conflict_with_fact_id

        # ── Serialize the read-currentness + write-events critical section ───
        # BEGIN IMMEDIATE acquires the write reservation (RESERVED lock) before
        # we read current state.  In WAL mode a plain DEFERRED BEGIN allows
        # another writer to slip in between our conflict-set scan and our commit,
        # which can yield two is_current=True facts in the same conflict set.
        # IMMEDIATE prevents that by forcing concurrent promotions to serialize
        # here.
        try:
            if manage_transaction:
                self.conn.execute('BEGIN IMMEDIATE')

            # When parallel resolution is requested, explicitly skip the
            # legacy one-current-object enforcement.
            if not allow_parallel and isinstance(typed_object, StateFact):
                _conflict_set = typed_object.conflict_set
                for _current in self.current_state_facts():
                    if (
                        _current.conflict_set == _conflict_set
                        and _current.object_id != typed_object.object_id
                        # Lane isolation: skip facts that cannot be auto-superseded
                        # by this candidate.
                        #
                        # When the incoming candidate has a source_lane (lane isolation
                        # is active), it must only auto-supersede facts from the same
                        # lane.  This covers two cases:
                        #   1. _current.source_lane differs (cross-lane): skip.
                        #   2. _current.source_lane is None (legacy unscoped fact):
                        #      also skip — hard-failing here is cleaner and safer than
                        #      silently allowing a scoped candidate to absorb an
                        #      unscoped lineage tree (which could span multiple lanes).
                        # When the candidate has no source_lane (unscoped/global
                        # deployment), no lane filter is applied.
                        and not (
                            typed_object.source_lane is not None
                            and (
                                _current.source_lane is None
                                or _current.source_lane != typed_object.source_lane
                            )
                        )
                    ):
                        resolved_conflict_with_fact_id = _current.object_id
                        break

            if not allow_parallel and resolved_conflict_with_fact_id:
                prior = self.materialize_object(resolved_conflict_with_fact_id)
                if prior is not None:
                    _validate_supersede_target(candidate=typed_object, prior=prior)
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

            if require_supersede and creation_type != 'supersede':
                if requested_conflict_with_fact_id:
                    raise ValueError(
                        'explicit supersede requested but no valid supersede target '
                        f'was materialized for {requested_conflict_with_fact_id!r}'
                    )
                raise ValueError(
                    'explicit supersede requested but no current conflict-set occupant '
                    'was found to supersede'
                )

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

            self._do_insert(create_row)
            self._do_insert(promote_row)
            if manage_transaction:
                self.conn.commit()
        except Exception:
            if manage_transaction:
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
    busy_timeout: required for BEGIN IMMEDIATE serialization to work
    gracefully under concurrent promotions.  Without it, a second writer that
    races to acquire the RESERVED lock gets SQLITE_BUSY immediately instead of
    retrying.  5 s is generous for a local file-backed DB; adjust if needed.
    """
    db_path = Path(path)
    if db_path.name != ':memory:':
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # WAL mode: better concurrency for readers; durable write-ahead log.
    conn.execute('PRAGMA journal_mode=WAL')
    # foreign_keys: ON for schema correctness hygiene.
    conn.execute('PRAGMA foreign_keys=ON')
    # busy_timeout: wait up to 5 s before surfacing SQLITE_BUSY on a write lock.
    conn.execute('PRAGMA busy_timeout=5000')
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    _ensure_typed_root_index(conn)
    conn.commit()



def _typed_root_ids_from_events(conn: sqlite3.Connection) -> list[str]:
    root_rows = conn.execute(
        """
        SELECT DISTINCT root_id
          FROM change_events
         WHERE payload_json IS NOT NULL
           AND root_id IS NOT NULL
         ORDER BY root_id
        """
    ).fetchall()
    root_ids: list[str] = []
    for row in root_rows:
        root_id = str(row['root_id']) if isinstance(row, sqlite3.Row) else str(row[0])
        if root_id:
            root_ids.append(root_id)
    return root_ids


def _typed_root_needs_refresh(conn: sqlite3.Connection, root_id: str) -> bool:
    snapshot = conn.execute(
        """
        SELECT latest_recorded_at, lineage_event_count
          FROM typed_roots
         WHERE root_id = ?
        """,
        (root_id,),
    ).fetchone()
    if snapshot is None:
        return True

    lineage_stats = conn.execute(
        """
        SELECT max(recorded_at) AS latest_recorded_at,
               count(*) AS lineage_event_count
          FROM change_events
         WHERE root_id = ?
            OR object_id IN (SELECT object_id FROM change_events WHERE root_id = ?)
            OR target_object_id IN (SELECT object_id FROM change_events WHERE root_id = ?)
        """,
        (root_id, root_id, root_id),
    ).fetchone()
    if lineage_stats is None or lineage_stats['latest_recorded_at'] is None:
        return True

    return (
        str(snapshot['latest_recorded_at']) != str(lineage_stats['latest_recorded_at'])
        or int(snapshot['lineage_event_count'] or 0)
        != int(lineage_stats['lineage_event_count'] or 0)
    )


def _ensure_typed_root_index(conn: sqlite3.Connection) -> None:
    event_root_ids = _typed_root_ids_from_events(conn)
    root_count = int(conn.execute('SELECT count(*) FROM typed_roots').fetchone()[0])
    if root_count != len(event_root_ids):
        conn.execute('DELETE FROM typed_roots')
        for root_id in event_root_ids:
            _refresh_typed_root_row(conn, root_id)
        return

    stale_root_ids = [
        root_id for root_id in event_root_ids if _typed_root_needs_refresh(conn, root_id)
    ]
    for root_id in stale_root_ids:
        _refresh_typed_root_row(conn, root_id)


def _refresh_typed_root_row(conn: sqlite3.Connection, root_id: str) -> None:
    rows = conn.execute(
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
    if not rows:
        conn.execute('DELETE FROM typed_roots WHERE root_id = ?', (root_id,))
        return

    objects = project_objects(rows)
    if not objects:
        conn.execute('DELETE FROM typed_roots WHERE root_id = ?', (root_id,))
        return

    current = next((obj for obj in reversed(objects) if obj.is_current), objects[-1])
    latest_recorded_at = str(rows[-1]['recorded_at'])
    current_payload_json = _canonical_json(current.model_dump(mode='json'))
    search_text = '\n'.join(
        _canonical_json(obj.model_dump(mode='json'))
        for obj in objects
    ).lower()

    conn.execute(
        """
        INSERT INTO typed_roots(
            root_id, latest_recorded_at, object_type, source_lane,
            current_object_id, current_version, current_payload_json,
            search_text, lineage_event_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(root_id) DO UPDATE SET
            latest_recorded_at = excluded.latest_recorded_at,
            object_type = excluded.object_type,
            source_lane = excluded.source_lane,
            current_object_id = excluded.current_object_id,
            current_version = excluded.current_version,
            current_payload_json = excluded.current_payload_json,
            search_text = excluded.search_text,
            lineage_event_count = excluded.lineage_event_count
        """,
        (
            root_id,
            latest_recorded_at,
            current.object_type,
            current.source_lane,
            current.object_id,
            int(current.version),
            current_payload_json,
            search_text,
            len(rows),
        ),
    )


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
            if _is_trusted_feedback_event(event.metadata_json):
                objects[target_id] = current.model_copy(update={'success_count': current.success_count + 1})
        elif event.event_type == 'procedure_failure' and isinstance(current, Procedure):
            if _is_trusted_feedback_event(event.metadata_json):
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

    # Prefer explicit source_lane from the fact dict (set during candidate creation
    # in remember_fact); fall back to evidence refs for legacy callers.
    source_lane = str(fact.get('source_lane') or '').strip() or None
    if source_lane is None:
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

    if assertion_type == 'procedure':
        value = fact.get('value') if isinstance(fact.get('value'), dict) else {}
        raw_steps = value.get('steps') or []
        if isinstance(raw_steps, str):
            raw_steps = [raw_steps]
        steps = [str(s).strip() for s in raw_steps if str(s).strip()]
        if not steps:
            # Synthesise a single placeholder step from the value description or subject
            fallback = (
                str(value.get('description') or '').strip()
                or str(fact.get('subject') or '').strip()
                or 'extracted from source'
            )
            steps = [fallback]
        raw_preconditions = value.get('preconditions') or []
        if isinstance(raw_preconditions, str):
            raw_preconditions = [raw_preconditions]
        return Procedure.model_validate(
            {
                **base,
                'name': str(value.get('name') or fact.get('subject') or 'unnamed_procedure').strip() or 'unnamed_procedure',
                'trigger': str(value.get('trigger') or fact.get('predicate') or '').strip(),
                'preconditions': [str(p).strip() for p in raw_preconditions if str(p).strip()],
                'steps': steps,
                'expected_outcome': str(value.get('expected_outcome') or '').strip(),
                'risk_level': str(value.get('risk_level') or 'medium').strip() or 'medium',
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


def _validate_supersede_target(*, candidate: TypedMemoryObject, prior: TypedMemoryObject) -> None:
    if candidate.object_type != prior.object_type:
        raise ValueError(
            'incompatible supersede target: '
            f'{candidate.object_type} candidate cannot supersede '
            f'{prior.object_type} object {prior.object_id}'
        )

    if (
        isinstance(candidate, StateFact)
        and isinstance(prior, StateFact)
        and candidate.conflict_set != prior.conflict_set
    ):
        raise ValueError(
            'incompatible supersede target: '
            f'state_fact conflict set mismatch for target {prior.object_id}'
        )

    # Lane isolation: when the candidate has a source_lane (lane isolation is
    # active), the supersede target must belong to the exact same lane.
    #
    # Policy (hard-fail on unscoped legacy facts):
    #   - prior.source_lane is None  → reject: a scoped candidate must not absorb
    #     an unscoped lineage tree; the prior predates lane awareness and its
    #     multi-lane provenance is unknown.
    #   - prior.source_lane != candidate.source_lane → reject: classic cross-lane
    #     supersede.
    # When the candidate itself has no source_lane (global/unscoped deployment),
    # no lane check is performed (backward-compatible).
    if candidate.source_lane is not None:
        if prior.source_lane is None:
            raise ValueError(
                'unscoped-fact supersede rejected: '
                f'candidate (source_lane={candidate.source_lane!r}) cannot supersede '
                f'an unscoped legacy fact (source_lane=None, object_id={prior.object_id!r}); '
                f'when lane isolation is active the supersede target must belong to the '
                f'same lane as the candidate'
            )
        if candidate.source_lane != prior.source_lane:
            raise ValueError(
                'cross-lane supersede rejected: '
                f'candidate (source_lane={candidate.source_lane!r}) cannot supersede '
                f'or adopt lineage from fact in a different lane '
                f'(source_lane={prior.source_lane!r}, object_id={prior.object_id!r})'
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
    digest = hashlib.sha256(f'object|{seed}'.encode()).hexdigest()
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


def _is_trusted_feedback_event(metadata_json: str | None) -> bool:
    if not metadata_json:
        return False
    try:
        metadata = json.loads(metadata_json)
    except Exception:
        return False
    return bool(isinstance(metadata, dict) and metadata.get('trusted_feedback'))


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
    """Return source_lane from legacy evidence ref dicts.

    Reads the ``scope`` field from each ref as a lane/group identifier.  This
    is correct for the ``truth.candidates`` path where evidence refs store the
    real group_id in ``scope`` (e.g., ``'s1_observational_memory'``).

    For the ``mcp_server`` router path, callers that need to preserve a real
    group_id should pass it as ``source_lane`` in the top-level fact dict
    instead, since there the ``scope`` field carries visibility policy
    (``'private'``, ``'public'``) rather than a group identifier.
    ``build_object_from_candidate_fact`` therefore checks ``fact['source_lane']``
    before calling this function.

    Note: serialised ``EvidenceRef`` model dicts (from ``.model_dump(mode='json')``)
    do not carry a ``scope`` field, so this function correctly returns ``None``
    for those callers and the top-level check in
    ``build_object_from_candidate_fact`` takes priority.
    """
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
