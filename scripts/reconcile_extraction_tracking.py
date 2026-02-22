#!/usr/bin/env python3
"""Reconcile extraction_tracking lifecycle rows against Neo4j episodic reality.

Why this exists:
- `mcp_ingest_sessions.py` records `extraction_tracking` rows at enqueue time using a
  deterministic local episode UUID.
- Graphiti queue processing writes Episodic nodes with server-side UUIDs.
- Without reconciliation, large portions of extraction debt stay forever `queued`
  even when episodes are present in Neo4j.

This script resolves that drift for session-based lanes by:
1) deriving an expected episodic `name` from registry chunk metadata,
2) finding matching Episodic nodes in Neo4j,
3) marking matching rows `succeeded` (and rewriting episode_uuid to canonical Neo4j uuid).

Default mode is dry-run; pass `--apply` to write updates.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "state" / "ingest_registry.db"
DEFAULT_GROUPS = ["s1_sessions_main", "s1_chatgpt_history", "s1_memory_day1"]


@dataclass
class ReconcileRow:
    group_id: str
    episode_uuid: str
    chunk_uuid: str | None
    chunk_key: str | None
    source_key: str | None
    status: str
    evidence_id: str | None
    content_hash: str | None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def _ensure_neo4j_env() -> tuple[str, str, str]:
    # Prefer process env; fall back to local credentials file when available.
    _load_env_file(Path.home() / ".clawdbot" / "credentials" / "neo4j.env")

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError("NEO4J_PASSWORD missing (set env or ~/.clawdbot/credentials/neo4j.env)")
    return uri, user, password


def _expected_episode_name(row: ReconcileRow) -> str | None:
    chunk_key = (row.chunk_key or "").strip()
    if not chunk_key:
        return None

    evidence_id = (row.evidence_id or "").strip()
    content_hash = (row.content_hash or "").strip()

    # mcp_ingest_sessions.py naming rule:
    # ep_name = f"{sub_key}:{evidence_id[:8] or content_hash[:8]}"
    # record_chunk stores '(missing)' when evidence_id absent, but ep_name falls back to hash.
    if evidence_id and evidence_id != "(missing)":
        suffix = evidence_id[:8]
    elif content_hash:
        suffix = content_hash[:8]
    else:
        return None

    return f"{chunk_key}:{suffix}"


def _resolve_groups(conn: sqlite3.Connection, groups_arg: list[str] | None) -> list[str]:
    if groups_arg:
        out: list[str] = []
        seen: set[str] = set()
        for g in groups_arg:
            gg = (g or "").strip()
            if gg and gg not in seen:
                out.append(gg)
                seen.add(gg)
        return out

    rows = conn.execute(
        """
        SELECT DISTINCT group_id
        FROM extraction_tracking
        WHERE status IN ('queued', 'failed')
          AND (
            source_key LIKE 'sessions:%'
            OR group_id IN (?, ?, ?)
          )
        ORDER BY group_id
        """,
        tuple(DEFAULT_GROUPS),
    ).fetchall()
    return [str(r[0]) for r in rows]


def _load_rows(conn: sqlite3.Connection, *, group_id: str, limit: int) -> list[ReconcileRow]:
    query = (
        "SELECT "
        "t.group_id, t.episode_uuid, t.chunk_uuid, t.chunk_key, t.source_key, t.status, "
        "c.evidence_id, c.content_hash "
        "FROM extraction_tracking t "
        "LEFT JOIN chunks c ON c.chunk_uuid = t.chunk_uuid "
        "WHERE t.group_id = ? AND t.status IN ('queued', 'failed') "
        "ORDER BY t.updated_at DESC "
    )
    params: list[Any] = [group_id]
    if limit > 0:
        query += "LIMIT ?"
        params.append(limit)

    rows = conn.execute(query, params).fetchall()
    out: list[ReconcileRow] = []
    for r in rows:
        out.append(
            ReconcileRow(
                group_id=str(r[0]),
                episode_uuid=str(r[1]),
                chunk_uuid=(str(r[2]) if r[2] is not None else None),
                chunk_key=(str(r[3]) if r[3] is not None else None),
                source_key=(str(r[4]) if r[4] is not None else None),
                status=str(r[5]),
                evidence_id=(str(r[6]) if r[6] is not None else None),
                content_hash=(str(r[7]) if r[7] is not None else None),
            )
        )
    return out


def _neo4j_episode_index(driver: Any, *, group_id: str) -> dict[str, tuple[str, str | None]]:
    """Return {episode_name: (episode_uuid, created_at_iso)} for session-chunk episodes.

    If multiple nodes share the same name, keep the most recent created_at.
    """

    by_name: dict[str, tuple[str, str | None]] = {}
    with driver.session() as session:
        records = session.run(
            """
            MATCH (e:Episodic {group_id: $group_id})
            WHERE e.name IS NOT NULL
              AND e.source_description STARTS WITH 'session chunk: '
            RETURN e.name AS name, e.uuid AS uuid, toString(e.created_at) AS created_at
            """,
            group_id=group_id,
        )
        for rec in records:
            name = str(rec.get("name") or "").strip()
            uuid = str(rec.get("uuid") or "").strip()
            created = rec.get("created_at")
            created_s = str(created).strip() if created is not None else None
            if not name or not uuid:
                continue

            prev = by_name.get(name)
            if prev is None:
                by_name[name] = (uuid, created_s)
                continue

            # Keep newest if timestamps are available and comparable as ISO strings.
            prev_created = prev[1]
            if created_s and (not prev_created or created_s > prev_created):
                by_name[name] = (uuid, created_s)

    return by_name


def _apply_success(
    conn: sqlite3.Connection,
    *,
    group_id: str,
    old_episode_uuid: str,
    new_episode_uuid: str,
    succeeded_at: str,
) -> bool:
    """Apply success update; return True if row updated.

    Prefer rewriting episode_uuid to canonical Neo4j UUID. If that conflicts,
    fall back to status-only update on the existing row.
    """

    now_iso = _utc_now_iso()

    try:
        cur = conn.execute(
            """
            UPDATE extraction_tracking
            SET episode_uuid = ?,
                status = 'succeeded',
                last_succeeded_at = ?,
                last_failure_reason = NULL,
                success_count = COALESCE(success_count, 0) + 1,
                updated_at = ?
            WHERE group_id = ?
              AND episode_uuid = ?
              AND status IN ('queued', 'failed')
            """,
            (new_episode_uuid, succeeded_at, now_iso, group_id, old_episode_uuid),
        )
        if int(cur.rowcount or 0) > 0:
            return True
    except sqlite3.IntegrityError:
        # Target UUID already exists for this group; keep old key but mark succeeded.
        print(
            f"  [WARN] UUID conflict: {new_episode_uuid!r} already exists in extraction_tracking "
            f"for group={group_id!r}; marking old row succeeded without UUID rewrite."
        )

    cur = conn.execute(
        """
        UPDATE extraction_tracking
        SET status = 'succeeded',
            last_succeeded_at = ?,
            last_failure_reason = NULL,
            success_count = COALESCE(success_count, 0) + 1,
            updated_at = ?
        WHERE group_id = ?
          AND episode_uuid = ?
          AND status IN ('queued', 'failed')
        """,
        (succeeded_at, now_iso, group_id, old_episode_uuid),
    )
    return int(cur.rowcount or 0) > 0


def run(args: argparse.Namespace) -> int:
    db_path = Path(args.db_path).expanduser().resolve()
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row

    groups = _resolve_groups(conn, args.group_id)
    if not groups:
        print("No queued/failed extraction rows found for reconciliation.")
        return 0

    uri, user, password = _ensure_neo4j_env()
    driver = GraphDatabase.driver(uri, auth=(user, password))

    total_rows = 0
    total_matchable = 0
    total_matched = 0
    total_updated = 0

    print(f"Reconcile mode: {'APPLY' if args.apply else 'DRY_RUN'}")
    print(f"Registry DB: {db_path}")
    print(f"Groups: {', '.join(groups)}")

    try:
        for group_id in groups:
            rows = _load_rows(conn, group_id=group_id, limit=int(args.limit or 0))
            total_rows += len(rows)
            if not rows:
                print(f"\n[{group_id}] rows=0")
                continue

            index = _neo4j_episode_index(driver, group_id=group_id)

            matched_rows: list[tuple[ReconcileRow, str, str | None]] = []
            unmatchable = 0
            for row in rows:
                expected_name = _expected_episode_name(row)
                if not expected_name:
                    unmatchable += 1
                    continue
                hit = index.get(expected_name)
                if hit is None:
                    continue
                neo_uuid, created_at = hit
                matched_rows.append((row, neo_uuid, created_at))

            matchable = len(rows) - unmatchable
            total_matchable += matchable
            total_matched += len(matched_rows)

            updated = 0
            if args.apply and matched_rows:
                for row, neo_uuid, created_at in matched_rows:
                    succeeded_at = created_at or _utc_now_iso()
                    if _apply_success(
                        conn,
                        group_id=group_id,
                        old_episode_uuid=row.episode_uuid,
                        new_episode_uuid=neo_uuid,
                        succeeded_at=succeeded_at,
                    ):
                        updated += 1
                conn.commit()

            total_updated += updated

            print(
                f"\n[{group_id}] rows={len(rows)} matchable={matchable} matched={len(matched_rows)} "
                f"unmatchable={unmatchable} updated={updated} neo_index={len(index)}"
            )

            if args.sample > 0 and matched_rows:
                print("  sample matches:")
                for row, neo_uuid, _created in matched_rows[: args.sample]:
                    exp = _expected_episode_name(row)
                    print(
                        f"    old={row.episode_uuid} -> neo={neo_uuid} "
                        f"chunk={row.chunk_key} name={exp}"
                    )

        print("\nSUMMARY")
        print(
            f"rows={total_rows} matchable={total_matchable} matched={total_matched} "
            f"updated={total_updated} mode={'apply' if args.apply else 'dry-run'}"
        )

        # Return non-zero only for hard failures; unmatched rows are expected.
        return 0
    finally:
        driver.close()
        conn.close()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Reconcile extraction_tracking rows against Neo4j episodes")
    ap.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help="Path to ingest_registry.db (default: state/ingest_registry.db)",
    )
    ap.add_argument(
        "--group-id",
        action="append",
        help="Group id(s) to reconcile (repeatable). Default: auto-detect session groups with queued/failed rows.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional per-group row limit (0 = no limit).",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=3,
        help="Print up to N sample matches per group (default: 3).",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Apply updates (default is dry-run).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
