#!/usr/bin/env python3
"""OM convergence runner.

Foundational implementation for:
- lock-scoped convergence watermark semantics
- dead-letter queue reconciliation between Neo4j and candidates sqlite
- monitoring_started_at backfill contract
- optional GC safety gates for Message/Episode cleanup
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from truth import candidates as candidates_store

LOCK_PATH = Path("/tmp/om_graph_write.lock")
STATE_ID = "singleton"


@dataclass
class ConvergenceState:
    last_convergence_at: str
    next_node_cursor: str | None
    cycle_started_at: str | None


class ConvergenceError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso(text: str | None) -> datetime | None:
    if not text:
        return None
    value = str(text).strip()
    if not value:
        return None
    try:
        if value.endswith("Z"):
            return datetime.fromisoformat(value[:-1]).replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, "timestamp": now_iso(), **payload}, ensure_ascii=True))


def neo4j_driver() -> Any:
    try:
        from neo4j import GraphDatabase  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ConvergenceError("neo4j driver is required") from exc

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise ConvergenceError("NEO4J_PASSWORD is required")

    return GraphDatabase.driver(uri, auth=(user, password))


def _ensure_state(session: Any, run_started_at: str) -> ConvergenceState:
    row = session.run(
        """
        MERGE (s:OMConvergenceState {state_id:$state_id})
        ON CREATE SET
          s.last_convergence_at = $run_started_at,
          s.next_node_cursor = NULL,
          s.cycle_started_at = NULL
        RETURN s.last_convergence_at AS last_convergence_at,
               s.next_node_cursor AS next_node_cursor,
               s.cycle_started_at AS cycle_started_at
        """,
        {"state_id": STATE_ID, "run_started_at": run_started_at},
    ).single()

    if row is None:
        raise ConvergenceError("failed to bootstrap OMConvergenceState")

    return ConvergenceState(
        last_convergence_at=str(row["last_convergence_at"]),
        next_node_cursor=str(row["next_node_cursor"]) if row.get("next_node_cursor") else None,
        cycle_started_at=str(row["cycle_started_at"]) if row.get("cycle_started_at") else None,
    )


def _backfill_monitoring_started_at(session: Any) -> int:
    row = session.run(
        """
        MATCH (n:OMNode)
        WHERE coalesce(n.status, '') = 'monitoring'
          AND n.monitoring_started_at IS NULL
        SET n.monitoring_started_at = coalesce(n.status_changed_at, n.created_at, $now_iso)
        RETURN count(n) AS updated
        """,
        {"now_iso": now_iso()},
    ).single()
    return int(row["updated"] if row else 0)


def _query_dead_letter_messages(session: Any) -> list[dict[str, Any]]:
    rows = session.run(
        """
        MATCH (m:Message)
        WHERE coalesce(m.om_dead_letter, false) = true
        RETURN m.message_id AS message_id,
               coalesce(m.source_session_id, 'unknown') AS source_session_id,
               coalesce(m.om_extract_attempts, 0) AS attempts,
               coalesce(m.created_at, $now_iso) AS created_at,
               m.om_chunk_id AS om_chunk_id
        """,
        {"now_iso": now_iso()},
    ).data()
    return rows


def _message_is_dead_letter(session: Any, message_id: str) -> bool:
    row = session.run(
        """
        MATCH (m:Message {message_id:$message_id})
        RETURN coalesce(m.om_dead_letter, false) AS dead
        """,
        {"message_id": message_id},
    ).single()
    return bool(row and row["dead"])


def reconcile_dead_letter_queue(session: Any) -> tuple[int, int, int]:
    """Return tuple: (upserted_rows, deleted_rows, queue_size)."""

    conn = candidates_store.connect(candidates_store.DB_PATH_DEFAULT)
    upserted = 0
    deleted = 0
    try:
        graph_rows = _query_dead_letter_messages(session)
        for row in graph_rows:
            created_at = str(row.get("created_at") or now_iso())
            candidates_store.upsert_om_dead_letter(
                conn,
                message_id=str(row["message_id"]),
                source_session_id=str(row.get("source_session_id") or "unknown"),
                attempts=int(row.get("attempts") or 0),
                last_error="reconciliation_backfill",
                first_failed_at=created_at,
                last_failed_at=created_at,
                last_chunk_id=str(row.get("om_chunk_id")) if row.get("om_chunk_id") else None,
            )
            upserted += 1

        for record in candidates_store.list_om_dead_letters(conn):
            message_id = str(record.get("message_id") or "")
            if not message_id:
                continue
            if not _message_is_dead_letter(session, message_id):
                deleted += candidates_store.remove_om_dead_letter(conn, message_id)

        queue_size = len(candidates_store.list_om_dead_letters(conn))
    finally:
        conn.close()

    return upserted, deleted, queue_size


def run_gc(session: Any, days: int) -> tuple[int, int]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max(1, int(days)))).replace(microsecond=0)
    cutoff_iso = cutoff.isoformat().replace("+00:00", "Z")

    message_row = session.run(
        """
        MATCH (m:Message)
        WHERE coalesce(m.om_extracted, false) = true
          AND coalesce(m.created_at, '') < $cutoff_iso
          AND NOT EXISTS {
            MATCH (m)-[:EVIDENCE_FOR]->(n:OMNode)
            WHERE coalesce(n.status, '') IN ['open', 'monitoring', 'reopened']
          }
        WITH m LIMIT 2000
        DETACH DELETE m
        RETURN count(*) AS deleted_messages
        """,
        {"cutoff_iso": cutoff_iso},
    ).single()
    deleted_messages = int(message_row["deleted_messages"] if message_row else 0)

    episode_row = session.run(
        """
        MATCH (e:Episode)
        WHERE NOT EXISTS { MATCH (e)-[:HAS_MESSAGE]->(:Message) }
        WITH e LIMIT 2000
        DETACH DELETE e
        RETURN count(*) AS deleted_episodes
        """
    ).single()
    deleted_episodes = int(episode_row["deleted_episodes"] if episode_row else 0)

    return deleted_messages, deleted_episodes


def update_watermark(session: Any, run_started_at: str) -> None:
    session.run(
        """
        MATCH (s:OMConvergenceState {state_id:$state_id})
        SET s.last_convergence_at = $run_started_at
        """,
        {"state_id": STATE_ID, "run_started_at": run_started_at},
    ).consume()


def run(args: argparse.Namespace) -> int:
    run_started_at = now_iso()

    driver = neo4j_driver()
    with driver, driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
        state = _ensure_state(session, run_started_at)

        emit(
            "OM_CONVERGENCE_WINDOW",
            last_convergence_at=state.last_convergence_at,
            run_started_at=run_started_at,
        )

        updated_monitoring = _backfill_monitoring_started_at(session)
        upserted, deleted, dead_letter_queue_size = reconcile_dead_letter_queue(session)

        deleted_messages = 0
        deleted_episodes = 0
        if args.run_gc:
            deleted_messages, deleted_episodes = run_gc(session, args.gc_days)

        update_watermark(session, run_started_at)

        emit(
            "OM_CONVERGENCE_DONE",
            monitoring_backfilled=updated_monitoring,
            dead_letter_upserted=upserted,
            dead_letter_deleted=deleted,
            dead_letter_queue_size=dead_letter_queue_size,
            deleted_messages=deleted_messages,
            deleted_episodes=deleted_episodes,
        )

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OM convergence")
    parser.add_argument("--run-gc", action="store_true", help="execute message/episode GC")
    parser.add_argument("--gc-days", type=int, default=90)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            return run(args)
        except Exception as exc:
            emit("OM_CONVERGENCE_FAILED", error=str(exc))
            return 1
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    raise SystemExit(main())
