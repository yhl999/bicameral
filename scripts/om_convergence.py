#!/usr/bin/env python3
"""OM convergence runner.

Implements:
- lock-scoped convergence watermark semantics (bounded pass / cursor)
- deterministic transition precedence (at most one transition per node per pass)
- dead-letter queue reconciliation between Neo4j and candidates sqlite
- monitoring_started_at backfill contract
- GC safety gates for Message/Episode cleanup (+ optional dry-run evidence mode)
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from truth import candidates as candidates_store

LOCK_PATH = Path("/tmp/om_graph_write.lock")
STATE_ID = "singleton"
MAX_NODES_PER_CONVERGENCE_PASS = 500
SIMILARITY_THRESHOLD = 0.85
NEO4J_ENV_FALLBACK_FILE = Path.home() / ".clawdbot" / "credentials" / "neo4j.env"


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


def _load_neo4j_env_fallback() -> None:
    if os.environ.get("NEO4J_PASSWORD"):
        return
    if not NEO4J_ENV_FALLBACK_FILE.exists():
        return

    for raw_line in NEO4J_ENV_FALLBACK_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in {"NEO4J_PASSWORD", "NEO4J_USER", "NEO4J_URI", "NEO4J_DATABASE"} and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def neo4j_driver() -> Any:
    try:
        from neo4j import GraphDatabase  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ConvergenceError("neo4j driver is required") from exc

    _load_neo4j_env_fallback()

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


def _open_cycle_if_needed(session: Any, state: ConvergenceState, now_utc: str) -> ConvergenceState:
    if state.next_node_cursor is not None:
        cycle_started = state.cycle_started_at or now_utc
        if state.cycle_started_at is None:
            session.run(
                """
                MATCH (s:OMConvergenceState {state_id:$state_id})
                SET s.cycle_started_at = $cycle_started_at
                """,
                {"state_id": STATE_ID, "cycle_started_at": cycle_started},
            ).consume()
        return ConvergenceState(
            last_convergence_at=state.last_convergence_at,
            next_node_cursor=state.next_node_cursor,
            cycle_started_at=cycle_started,
        )

    session.run(
        """
        MATCH (s:OMConvergenceState {state_id:$state_id})
        SET s.cycle_started_at = $cycle_started_at,
            s.next_node_cursor = NULL
        """,
        {"state_id": STATE_ID, "cycle_started_at": now_utc},
    ).consume()
    return ConvergenceState(
        last_convergence_at=state.last_convergence_at,
        next_node_cursor=None,
        cycle_started_at=now_utc,
    )


def _finalize_cycle(
    session: Any,
    *,
    cycle_started_at: str,
    last_processed_node_id: str | None,
    has_more_nodes: bool,
) -> None:
    if has_more_nodes and last_processed_node_id:
        session.run(
            """
            MATCH (s:OMConvergenceState {state_id:$state_id})
            SET s.next_node_cursor = $next_node_cursor,
                s.cycle_started_at = $cycle_started_at
            """,
            {
                "state_id": STATE_ID,
                "next_node_cursor": last_processed_node_id,
                "cycle_started_at": cycle_started_at,
            },
        ).consume()
        return

    session.run(
        """
        MATCH (s:OMConvergenceState {state_id:$state_id})
        SET s.next_node_cursor = NULL,
            s.last_convergence_at = $cycle_started_at,
            s.cycle_started_at = NULL
        """,
        {"state_id": STATE_ID, "cycle_started_at": cycle_started_at},
    ).consume()


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


def _fetch_nodes_for_pass(session: Any, cursor: str | None) -> tuple[list[dict[str, Any]], bool]:
    rows = session.run(
        """
        MATCH (n:OMNode)
        WHERE ($cursor IS NULL OR n.node_id > $cursor)
        RETURN n.node_id AS node_id,
               coalesce(n.node_type, '') AS node_type,
               coalesce(n.status, 'open') AS status,
               coalesce(n.semantic_domain, '') AS semantic_domain,
               coalesce(n.status_changed_at, n.created_at, $now_iso) AS status_changed_at,
               coalesce(n.monitoring_started_at, n.status_changed_at, n.created_at, $now_iso) AS monitoring_started_at,
               coalesce(n.created_at, $now_iso) AS created_at,
               coalesce(n.content_embedding, []) AS content_embedding
        ORDER BY n.node_id ASC
        LIMIT $limit
        """,
        {
            "cursor": cursor,
            "limit": MAX_NODES_PER_CONVERGENCE_PASS,
            "now_iso": now_iso(),
        },
    ).data()

    has_more = False
    if rows:
        last_id = str(rows[-1].get("node_id") or "")
        if last_id:
            probe = session.run(
                """
                MATCH (n:OMNode)
                WHERE n.node_id > $last_id
                RETURN n.node_id AS node_id
                ORDER BY n.node_id ASC
                LIMIT 1
                """,
                {"last_id": last_id},
            ).single()
            has_more = probe is not None

    return rows, has_more


def _fetch_events_in_window(session: Any, start_iso: str, end_iso: str) -> list[dict[str, Any]]:
    return session.run(
        """
        MATCH (e:OMExtractionEvent)-[:EMITTED]->(n:OMNode)
        WHERE e.emitted_at > $start_iso
          AND e.emitted_at <= $end_iso
        RETURN e.event_id AS event_id,
               e.emitted_at AS emitted_at,
               coalesce(e.semantic_domain, '') AS semantic_domain,
               coalesce(e.content_embedding, []) AS content_embedding,
               coalesce(n.node_type, '') AS emitted_node_type
        """,
        {"start_iso": start_iso, "end_iso": end_iso},
    ).data()


def _vector(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    out: list[float] = []
    for item in values:
        try:
            out.append(float(item))
        except Exception:
            return []
    return out


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    aa = 0.0
    bb = 0.0
    for av, bv in zip(a, b, strict=False):
        dot += av * bv
        aa += av * av
        bb += bv * bv
    if aa <= 0.0 or bb <= 0.0:
        return 0.0
    return dot / (math.sqrt(aa) * math.sqrt(bb))


def _event_matches_node(node: dict[str, Any], event: dict[str, Any]) -> bool:
    node_domain = str(node.get("semantic_domain") or "")
    event_domain = str(event.get("semantic_domain") or "")
    if node_domain and event_domain and node_domain == event_domain:
        return True

    node_embedding = _vector(node.get("content_embedding"))
    event_embedding = _vector(event.get("content_embedding"))
    return _cosine_similarity(node_embedding, event_embedding) >= SIMILARITY_THRESHOLD


def _between_open_left(emitted_at: str | None, start: datetime | None, end: datetime) -> bool:
    when = parse_iso(emitted_at)
    if when is None:
        return False
    if start is None:
        return when <= end
    return when > start and when <= end


def _count_status_window_mentions(node: dict[str, Any], events: list[dict[str, Any]], now_dt: datetime) -> int:
    status_start = parse_iso(str(node.get("status_changed_at") or ""))
    count = 0
    for event in events:
        if not _between_open_left(str(event.get("emitted_at") or ""), status_start, now_dt):
            continue
        if _event_matches_node(node, event):
            count += 1
    return count


def _count_monitoring_mentions(node: dict[str, Any], events: list[dict[str, Any]], now_dt: datetime) -> int:
    monitoring_start = parse_iso(str(node.get("monitoring_started_at") or ""))
    count = 0
    for event in events:
        if not _between_open_left(str(event.get("emitted_at") or ""), monitoring_start, now_dt):
            continue
        if _event_matches_node(node, event):
            count += 1
    return count


def _has_related_reappearance(node: dict[str, Any], events: list[dict[str, Any]]) -> bool:
    for event in events:
        emitted_node_type = str(event.get("emitted_node_type") or "")
        if emitted_node_type not in {"Friction", "Commitment"}:
            continue
        if _event_matches_node(node, event):
            return True
    return False


def _has_addresses(session: Any, node_id: str) -> bool:
    row = session.run(
        """
        MATCH (:Judgment)-[:ADDRESSES]->(n:OMNode {node_id:$node_id})
        RETURN count(*) > 0 AS has_link
        """,
        {"node_id": node_id},
    ).single()
    return bool(row and row["has_link"])


def _has_fresh_addresses(session: Any, node_id: str, since_iso: str, now_utc: str) -> bool:
    row = session.run(
        """
        MATCH (:Judgment)-[r:ADDRESSES]->(n:OMNode {node_id:$node_id})
        WHERE coalesce(r.linked_at, '') > $since_iso
          AND coalesce(r.linked_at, '') <= $now_iso
        RETURN count(r) > 0 AS has_fresh
        """,
        {"node_id": node_id, "since_iso": since_iso, "now_iso": now_utc},
    ).single()
    return bool(row and row["has_fresh"])


def _apply_transition(
    session: Any,
    *,
    node_id: str,
    node_type: str,
    target_status: str,
    now_utc: str,
) -> None:
    if target_status == "monitoring":
        session.run(
            """
            MATCH (n:OMNode {node_id:$node_id})
            SET n.status = $target_status,
                n.status_changed_at = $now_iso,
                n.monitoring_started_at = $now_iso
            """,
            {"node_id": node_id, "target_status": target_status, "now_iso": now_utc},
        ).consume()
    else:
        session.run(
            """
            MATCH (n:OMNode {node_id:$node_id})
            SET n.status = $target_status,
                n.status_changed_at = $now_iso
            """,
            {"node_id": node_id, "target_status": target_status, "now_iso": now_utc},
        ).consume()

    if target_status == "closed" and node_type in {"Friction", "Commitment"}:
        session.run(
            """
            MATCH (j:Judgment)-[:ADDRESSES]->(n:OMNode {node_id:$node_id})
            MERGE (j)-[r:RESOLVES]->(n)
            ON CREATE SET r.linked_at = $now_iso
            """,
            {"node_id": node_id, "now_iso": now_utc},
        ).consume()


def _determine_transition(
    session: Any,
    *,
    node: dict[str, Any],
    convergence_events: list[dict[str, Any]],
    status_window_events: list[dict[str, Any]],
    monitoring_window_events: list[dict[str, Any]],
    now_utc: str,
) -> tuple[str | None, str | None]:
    status = str(node.get("status") or "open").strip().lower()
    node_id = str(node.get("node_id") or "")
    if not node_id:
        return None, None

    now_dt = parse_iso(now_utc) or datetime.now(timezone.utc)
    status_changed_at = str(node.get("status_changed_at") or now_utc)
    status_changed_dt = parse_iso(status_changed_at) or now_dt

    monitoring_started_at = str(node.get("monitoring_started_at") or status_changed_at)
    monitoring_started_dt = parse_iso(monitoring_started_at) or now_dt

    monitoring_duration_days = max(0, (now_dt - monitoring_started_dt).days)
    status_age_days = max(0, (now_dt - status_changed_dt).days)

    status_window_mentions = _count_status_window_mentions(node, status_window_events, now_dt)
    monitoring_mentions = _count_monitoring_mentions(node, monitoring_window_events, now_dt)

    convergence_reappearance = _has_related_reappearance(node, convergence_events)
    monitoring_reappearance = _has_related_reappearance(node, monitoring_window_events)

    # Precedence 1: MONITORING|CLOSED|ABANDONED -> REOPENED
    if status in {"monitoring", "closed", "abandoned"} and convergence_reappearance:
        return "reopened", "reappearance_in_convergence_window"

    # Precedence 2: OPEN|REOPENED -> MONITORING
    if status == "open" and _has_addresses(session, node_id):
        return "monitoring", "addresses_link_detected"
    if status == "reopened" and _has_fresh_addresses(session, node_id, status_changed_at, now_utc):
        return "monitoring", "fresh_addresses_since_reopen"

    # Precedence 3: MONITORING -> ABANDONED|CLOSED
    if status == "monitoring" and monitoring_duration_days >= 14:
        if monitoring_mentions == 0:
            return "abandoned", "monitoring_aged_without_mentions"
        if monitoring_mentions > 0 and not monitoring_reappearance:
            return "closed", "monitoring_aged_without_reappearance"

    # Precedence 4: OPEN|REOPENED -> ABANDONED
    if status == "open" and status_age_days >= 30 and status_window_mentions == 0:
        return "abandoned", "open_aged_without_mentions"
    if (
        status == "reopened"
        and status_age_days >= 30
        and status_window_mentions == 0
        and not _has_fresh_addresses(session, node_id, status_changed_at, now_utc)
    ):
        return "abandoned", "reopened_aged_without_mentions_or_addresses"

    return None, None


def run_gc(session: Any, days: int, *, dry_run: bool = False) -> dict[str, int | str | bool]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max(1, int(days)))).replace(microsecond=0)
    cutoff_iso = cutoff.isoformat().replace("+00:00", "Z")

    gc_base_predicate = """
        m.graphiti_extracted_at IS NOT NULL
        AND coalesce(m.om_extracted, false) = true
        AND coalesce(m.om_dead_letter, false) = false
        AND coalesce(m.created_at, '') < $cutoff_iso
        AND NOT EXISTS {
          MATCH (m)-[:EVIDENCE_FOR]->(n:OMNode)
          WHERE coalesce(n.status, '') IN ['open', 'monitoring', 'reopened']
        }
        AND NOT EXISTS {
          MATCH (m)-[:SUPPORTS_CORE]->(c:CoreMemory)
          WHERE coalesce(c.retention_status, '') = 'active'
        }
    """

    eligible_row = session.run(
        f"""
        MATCH (m:Message)
        WHERE {gc_base_predicate}
        RETURN count(m) AS eligible_messages
        """,
        {"cutoff_iso": cutoff_iso},
    ).single()
    eligible_messages = int(eligible_row["eligible_messages"] if eligible_row else 0)

    deleted_messages = 0
    if not dry_run:
        deleted_row = session.run(
            f"""
            MATCH (m:Message)
            WHERE {gc_base_predicate}
            WITH m LIMIT 2000
            DETACH DELETE m
            RETURN count(*) AS deleted_messages
            """,
            {"cutoff_iso": cutoff_iso},
        ).single()
        deleted_messages = int(deleted_row["deleted_messages"] if deleted_row else 0)

    eligible_episode_row = session.run(
        """
        MATCH (e:Episode)
        WHERE coalesce(e.started_at, '') < $cutoff_iso
          AND NOT EXISTS { MATCH (e)-[:HAS_MESSAGE]->(:Message) }
        RETURN count(e) AS eligible_episodes
        """,
        {"cutoff_iso": cutoff_iso},
    ).single()
    eligible_episodes = int(eligible_episode_row["eligible_episodes"] if eligible_episode_row else 0)

    deleted_episodes = 0
    if not dry_run:
        deleted_episode_row = session.run(
            """
            MATCH (e:Episode)
            WHERE coalesce(e.started_at, '') < $cutoff_iso
              AND NOT EXISTS { MATCH (e)-[:HAS_MESSAGE]->(:Message) }
            WITH e LIMIT 2000
            DETACH DELETE e
            RETURN count(*) AS deleted_episodes
            """,
            {"cutoff_iso": cutoff_iso},
        ).single()
        deleted_episodes = int(deleted_episode_row["deleted_episodes"] if deleted_episode_row else 0)

    return {
        "dry_run": dry_run,
        "cutoff_iso": cutoff_iso,
        "eligible_messages": eligible_messages,
        "deleted_messages": deleted_messages,
        "eligible_episodes": eligible_episodes,
        "deleted_episodes": deleted_episodes,
    }


def run(args: argparse.Namespace) -> int:
    run_now = now_iso()

    driver = neo4j_driver()
    with driver, driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
        state = _ensure_state(session, run_now)
        state = _open_cycle_if_needed(session, state, run_now)

        run_started_at = state.cycle_started_at or run_now
        now_utc = run_now

        emit(
            "OM_CONVERGENCE_WINDOW",
            last_convergence_at=state.last_convergence_at,
            run_started_at=run_started_at,
            next_node_cursor=state.next_node_cursor,
            max_nodes_per_pass=MAX_NODES_PER_CONVERGENCE_PASS,
        )

        updated_monitoring = _backfill_monitoring_started_at(session)

        nodes, has_more_nodes = _fetch_nodes_for_pass(session, state.next_node_cursor)
        convergence_events = _fetch_events_in_window(session, state.last_convergence_at, run_started_at)

        transitions = 0
        for node in nodes:
            node_status_changed_at = str(node.get("status_changed_at") or run_started_at)
            node_monitoring_started_at = str(node.get("monitoring_started_at") or node_status_changed_at)

            status_window_events = [
                event
                for event in convergence_events
                if _between_open_left(str(event.get("emitted_at") or ""), parse_iso(node_status_changed_at), parse_iso(now_utc) or datetime.now(timezone.utc))
            ]
            monitoring_window_events = [
                event
                for event in convergence_events
                if _between_open_left(str(event.get("emitted_at") or ""), parse_iso(node_monitoring_started_at), parse_iso(now_utc) or datetime.now(timezone.utc))
            ]

            target_status, reason = _determine_transition(
                session,
                node=node,
                convergence_events=convergence_events,
                status_window_events=status_window_events,
                monitoring_window_events=monitoring_window_events,
                now_utc=now_utc,
            )
            if not target_status:
                continue

            from_status = str(node.get("status") or "open")
            _apply_transition(
                session,
                node_id=str(node.get("node_id") or ""),
                node_type=str(node.get("node_type") or ""),
                target_status=target_status,
                now_utc=now_utc,
            )
            transitions += 1
            emit(
                "OM_CONVERGENCE_TRANSITION",
                node_id=str(node.get("node_id") or ""),
                from_status=from_status,
                to_status=target_status,
                reason=reason,
            )

        upserted, deleted, dead_letter_queue_size = reconcile_dead_letter_queue(session)

        gc_summary: dict[str, Any] = {
            "dry_run": bool(args.gc_dry_run),
            "cutoff_iso": "",
            "eligible_messages": 0,
            "deleted_messages": 0,
            "eligible_episodes": 0,
            "deleted_episodes": 0,
        }
        if args.run_gc:
            gc_summary = run_gc(session, args.gc_days, dry_run=bool(args.gc_dry_run))

        last_processed_node_id = str(nodes[-1].get("node_id") or "") if nodes else None
        _finalize_cycle(
            session,
            cycle_started_at=run_started_at,
            last_processed_node_id=last_processed_node_id,
            has_more_nodes=has_more_nodes,
        )

        cycle_complete = not (has_more_nodes and last_processed_node_id)
        emit(
            "OM_CONVERGENCE_DONE",
            monitoring_backfilled=updated_monitoring,
            dead_letter_upserted=upserted,
            dead_letter_deleted=deleted,
            dead_letter_queue_size=dead_letter_queue_size,
            nodes_scanned=len(nodes),
            nodes_transitioned=transitions,
            next_node_cursor=(last_processed_node_id if has_more_nodes else None),
            cycle_complete=cycle_complete,
            gc=gc_summary,
        )

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OM convergence")
    parser.add_argument("--run-gc", action="store_true", help="execute message/episode GC")
    parser.add_argument("--gc-days", type=int, default=90)
    parser.add_argument("--gc-dry-run", action="store_true", help="compute GC candidates without deletion")
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
