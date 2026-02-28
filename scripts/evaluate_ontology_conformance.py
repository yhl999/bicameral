#!/usr/bin/env python3
"""Observe-only ontology conformance evaluator.

Computes conformance metrics for extracted graph data and reports warnings when
below configured thresholds.

IMPORTANT: This script is OBSERVE-ONLY.
- It NEVER blocks ingestion, extraction, or episode processing.
- It NEVER drops episodes, edges, or nodes.
- Exit code is 0 unless an operational failure occurs (DB/connection error).
- All findings are surfaced as warnings in the JSON output and human summary.

Metrics computed:
  typed_entity_rate     Fraction of Entity nodes that have a type matching the
                        lane's allowed entity types.
  allowed_relation_rate Fraction of relationships whose type is in the allow-list.
  out_of_schema_count   Absolute count of relationships not in the allow-list.

Usage:
  uv run python scripts/evaluate_ontology_conformance.py \\
      --group-id s1_sessions_main \\
      [--allow-rel RELATES_TO PREFERS REQUIRES ...] \\
      [--typed-entity-threshold 0.5] \\
      [--allowed-relation-threshold 0.5] \\
      [--observe-only]   # default: True; kept for explicitness \\
      [--backend neo4j|falkordb] \\
      [--dry-run]        # skip DB queries; use synthetic fixture data

Output:
  JSON report to stdout + human-readable summary to stderr.
  Exit code 0 always (unless DB connection failure).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)

# ── Default thresholds ────────────────────────────────────────────────────────

DEFAULT_TYPED_ENTITY_THRESHOLD = 0.5
DEFAULT_ALLOWED_RELATION_THRESHOLD = 0.5

# ── Fixture data for dry-run / unit-test mode ─────────────────────────────────

_DRY_RUN_ENTITIES: list[dict[str, Any]] = [
    {"entity_type": "Preference", "name": "prefers bullet summaries"},
    {"entity_type": "Requirement", "name": "no meetings before 10:30am"},
    {"entity_type": "Organization", "name": "Blockchain Capital"},
    {"entity_type": "UNKNOWN_TYPE", "name": "some unlabeled entity"},
]

_DRY_RUN_RELATIONS: list[dict[str, Any]] = [
    {"relation_type": "RELATES_TO"},
    {"relation_type": "PREFERS"},
    {"relation_type": "REQUIRES"},
    {"relation_type": "OFF_SCHEMA_REL"},
]


# ── DB query helpers ──────────────────────────────────────────────────────────


def _query_neo4j(group_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Query Neo4j for entity and relationship data for the given group.

    Returns:
        (entities, relations) — lists of dicts with 'entity_type'/'relation_type' keys.

    Raises:
        RuntimeError: If the Neo4j driver is unavailable or query fails.
    """
    try:
        import neo4j  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "neo4j driver not installed; run: uv add neo4j"
        ) from exc

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    if not password:
        raise RuntimeError(
            "NEO4J_PASSWORD environment variable must be set for Neo4j backend."
        )

    driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []

    try:
        with driver.session(database=database) as session:
            # Entity nodes — prefer the 'name' label property; fall back to labels.
            entity_result = session.run(
                """
                MATCH (n:Entity)
                WHERE n.group_id = $group_id
                RETURN
                    coalesce(n.entity_type, 'UNKNOWN') AS entity_type,
                    coalesce(n.name, '') AS name
                """,
                group_id=group_id,
            )
            for record in entity_result:
                entities.append({
                    "entity_type": record["entity_type"],
                    "name": record["name"],
                })

            # Relationships — type comes from the Neo4j relationship type.
            relation_result = session.run(
                """
                MATCH (a)-[r]->(b)
                WHERE r.group_id = $group_id
                RETURN type(r) AS relation_type
                """,
                group_id=group_id,
            )
            for record in relation_result:
                relations.append({"relation_type": record["relation_type"]})
    finally:
        driver.close()

    return entities, relations


def _query_falkordb(group_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Query FalkorDB for entity and relationship data.

    Raises:
        RuntimeError: If the FalkorDB client is unavailable or query fails.
    """
    try:
        import falkordb  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "falkordb driver not installed; run: uv add 'falkordb>=1.1.2,<2.0.0'"
        ) from exc

    host = os.environ.get("REDIS_HOST", "localhost")
    port = int(os.environ.get("REDIS_PORT", "6379"))

    client = falkordb.FalkorDB(host=host, port=port)
    graph = client.select_graph(group_id)

    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []

    try:
        entity_result = graph.query(
            "MATCH (n:Entity) RETURN coalesce(n.entity_type, 'UNKNOWN') AS entity_type, "
            "coalesce(n.name, '') AS name"
        )
        for row in entity_result.result_set:
            entities.append({"entity_type": row[0], "name": row[1]})

        relation_result = graph.query(
            "MATCH (a)-[r]->(b) RETURN type(r) AS relation_type"
        )
        for row in relation_result.result_set:
            relations.append({"relation_type": row[0]})
    finally:
        pass  # FalkorDB client has no explicit close method

    return entities, relations


# ── Metric computation ────────────────────────────────────────────────────────


def compute_conformance_metrics(
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    allowed_entity_types: set[str],
    allowed_relation_types: set[str],
) -> dict[str, Any]:
    """Compute conformance metrics from raw entity/relation lists.

    This function is pure — no side effects, no DB calls.  Suitable for
    unit testing with fixture data.

    Args:
        entities:             List of dicts with 'entity_type' key.
        relations:            List of dicts with 'relation_type' key.
        allowed_entity_types: Set of entity type names from the lane ontology.
        allowed_relation_types: Set of relationship type names from the lane ontology.

    Returns:
        Dict with keys:
            total_entities          int
            typed_entities          int
            typed_entity_rate       float   (0.0–1.0; 1.0 if no entities)
            total_relations         int
            allowed_relations       int
            allowed_relation_rate   float   (0.0–1.0; 1.0 if no relations)
            out_of_schema_count     int
            out_of_schema_types     list[str]  (unique off-schema relation types)
            off_schema_entity_types list[str]  (unique off-schema entity types)
    """
    total_entities = len(entities)
    typed_count = sum(
        1 for e in entities if e.get("entity_type", "UNKNOWN") in allowed_entity_types
    )
    typed_entity_rate = typed_count / total_entities if total_entities > 0 else 1.0

    total_relations = len(relations)
    allowed_count = sum(
        1 for r in relations if r.get("relation_type", "") in allowed_relation_types
    )
    out_of_schema_count = total_relations - allowed_count
    allowed_relation_rate = allowed_count / total_relations if total_relations > 0 else 1.0

    # Collect unique off-schema types for diagnostics
    out_of_schema_types = sorted(
        {r["relation_type"] for r in relations if r.get("relation_type", "") not in allowed_relation_types}
    )
    off_schema_entity_types = sorted(
        {e["entity_type"] for e in entities if e.get("entity_type", "UNKNOWN") not in allowed_entity_types}
    )

    return {
        "total_entities": total_entities,
        "typed_entities": typed_count,
        "typed_entity_rate": typed_entity_rate,
        "total_relations": total_relations,
        "allowed_relations": allowed_count,
        "allowed_relation_rate": allowed_relation_rate,
        "out_of_schema_count": out_of_schema_count,
        "out_of_schema_types": out_of_schema_types,
        "off_schema_entity_types": off_schema_entity_types,
    }


# ── Ontology profile loader ───────────────────────────────────────────────────


def _load_allowed_types_from_ontology(
    group_id: str,
    config_path: Path | None = None,
) -> tuple[set[str], set[str]]:
    """Load allowed entity and relationship types from the ontology config.

    Returns:
        (allowed_entity_types, allowed_relation_types) — both as sets of strings.
        Returns empty sets if the group is not in the config (permissive fallback).
    """
    if config_path is None:
        # Search relative to this file's repo root
        repo_root = Path(__file__).resolve().parents[1]
        candidates = [
            repo_root / "mcp_server" / "config" / "extraction_ontologies.yaml",
            repo_root / "config" / "extraction_ontologies.yaml",
        ]
        for c in candidates:
            if c.exists():
                config_path = c
                break

    if config_path is None or not config_path.exists():
        logger.warning("extraction_ontologies.yaml not found; using empty type sets")
        return set(), set()

    try:
        import yaml  # type: ignore
        raw = yaml.safe_load(config_path.read_text()) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load ontology config: %s", exc)
        return set(), set()

    definition = raw.get(group_id, {})
    if not isinstance(definition, dict):
        return set(), set()

    entity_types = {e["name"] for e in definition.get("entity_types", []) if "name" in e}
    relation_types = {r["name"] for r in definition.get("relationship_types", []) if "name" in r}
    return entity_types, relation_types


# ── Report generation ─────────────────────────────────────────────────────────


def build_report(
    group_id: str,
    metrics: dict[str, Any],
    typed_entity_threshold: float,
    allowed_relation_threshold: float,
    observe_only: bool,
    dry_run: bool,
    warnings: list[str],
) -> dict[str, Any]:
    """Build the full JSON report dict."""
    passed = (
        metrics["typed_entity_rate"] >= typed_entity_threshold
        and metrics["allowed_relation_rate"] >= allowed_relation_threshold
    )

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "group_id": group_id,
        "observe_only": observe_only,
        "dry_run": dry_run,
        "thresholds": {
            "typed_entity_rate": typed_entity_threshold,
            "allowed_relation_rate": allowed_relation_threshold,
        },
        "metrics": metrics,
        "conformance_passed": passed,
        "warnings": warnings,
        "note": (
            "OBSERVE-ONLY: no episodes were dropped or blocked regardless of conformance status."
            if observe_only
            else "observe_only=false: thresholds are informational only in this implementation."
        ),
    }


def print_human_summary(report: dict[str, Any]) -> None:
    """Print a human-readable conformance summary to stderr."""
    m = report["metrics"]
    group_id = report["group_id"]
    passed = report["conformance_passed"]
    status_symbol = "✓" if passed else "⚠"

    lines = [
        "",
        f"=== Ontology Conformance Report: {group_id} ===",
        f"  Generated at : {report['generated_at']}",
        f"  Mode         : {'DRY-RUN' if report['dry_run'] else 'LIVE'} | OBSERVE-ONLY={report['observe_only']}",
        "",
        f"  Entities     : {m['typed_entities']}/{m['total_entities']} typed"
        f" → rate={m['typed_entity_rate']:.3f}"
        f" (threshold {report['thresholds']['typed_entity_rate']:.2f})",
        f"  Relations    : {m['allowed_relations']}/{m['total_relations']} allowed"
        f" → rate={m['allowed_relation_rate']:.3f}"
        f" (threshold {report['thresholds']['allowed_relation_rate']:.2f})",
        f"  Out-of-schema: {m['out_of_schema_count']} relations",
    ]

    if m["out_of_schema_types"]:
        lines.append(f"  Off-schema relation types: {', '.join(m['out_of_schema_types'])}")
    if m["off_schema_entity_types"]:
        lines.append(f"  Off-schema entity  types : {', '.join(m['off_schema_entity_types'])}")

    lines += [
        "",
        f"  {status_symbol} Conformance: {'PASS' if passed else 'WARN (below threshold)'}",
        "",
    ]

    if report["warnings"]:
        lines.append("  Warnings:")
        for w in report["warnings"]:
            lines.append(f"    - {w}")
        lines.append("")

    lines.append(
        "  ⚑  OBSERVE-ONLY — no episodes were dropped or blocked."
        if report["observe_only"]
        else "  (observe_only=false — thresholds are informational only)"
    )
    lines.append("")

    print("\n".join(lines), file=sys.stderr)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--group-id",
        required=True,
        help="Graph group_id to evaluate (e.g. s1_sessions_main)",
    )
    p.add_argument(
        "--allow-rel",
        nargs="*",
        metavar="REL_TYPE",
        help=(
            "Override allowed relationship types. If not set, loads from "
            "extraction_ontologies.yaml for the given group-id."
        ),
    )
    p.add_argument(
        "--allow-entity",
        nargs="*",
        metavar="ENTITY_TYPE",
        help=(
            "Override allowed entity types. If not set, loads from "
            "extraction_ontologies.yaml for the given group-id."
        ),
    )
    p.add_argument(
        "--typed-entity-threshold",
        type=float,
        default=DEFAULT_TYPED_ENTITY_THRESHOLD,
        help=f"Minimum typed_entity_rate before warning (default: {DEFAULT_TYPED_ENTITY_THRESHOLD})",
    )
    p.add_argument(
        "--allowed-relation-threshold",
        type=float,
        default=DEFAULT_ALLOWED_RELATION_THRESHOLD,
        help=f"Minimum allowed_relation_rate before warning (default: {DEFAULT_ALLOWED_RELATION_THRESHOLD})",
    )
    p.add_argument(
        "--observe-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Observe-only mode (default: True). Never blocks ingestion.",
    )
    p.add_argument(
        "--backend",
        choices=["neo4j", "falkordb"],
        default="neo4j",
        help="Database backend (default: neo4j)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Skip DB queries; use synthetic fixture data for testing.",
    )
    p.add_argument(
        "--ontology-config",
        type=Path,
        default=None,
        help="Path to extraction_ontologies.yaml (auto-detected if not set)",
    )
    p.add_argument(
        "--output",
        choices=["json", "summary", "both"],
        default="both",
        help="Output format (default: both — JSON to stdout + summary to stderr)",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns exit code (0 = success, 1 = operational failure)."""
    args = _parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")

    warnings: list[str] = []

    # ── Resolve allowed types ─────────────────────────────────────────────────
    allowed_entity_types: set[str]
    allowed_relation_types: set[str]

    if args.allow_rel is not None or args.allow_entity is not None:
        # CLI overrides take precedence
        allowed_entity_types = set(args.allow_entity or [])
        allowed_relation_types = set(args.allow_rel or [])
        logger.debug(
            "Using CLI-supplied allowed types: entities=%s, relations=%s",
            allowed_entity_types,
            allowed_relation_types,
        )
    else:
        # Auto-load from ontology config
        allowed_entity_types, allowed_relation_types = _load_allowed_types_from_ontology(
            args.group_id, args.ontology_config
        )
        if not allowed_entity_types and not allowed_relation_types:
            warnings.append(
                f"No ontology profile found for group_id={args.group_id!r}. "
                "All types will be considered off-schema. "
                "Add a profile to extraction_ontologies.yaml or use --allow-rel / --allow-entity."
            )

    logger.debug(
        "Allowed entity types (%d): %s", len(allowed_entity_types), sorted(allowed_entity_types)
    )
    logger.debug(
        "Allowed relation types (%d): %s", len(allowed_relation_types), sorted(allowed_relation_types)
    )

    # ── Fetch data ────────────────────────────────────────────────────────────
    if args.dry_run:
        logger.debug("DRY-RUN: using fixture data, skipping DB queries")
        entities = _DRY_RUN_ENTITIES
        relations = _DRY_RUN_RELATIONS
    else:
        try:
            if args.backend == "neo4j":
                entities, relations = _query_neo4j(args.group_id)
            else:
                entities, relations = _query_falkordb(args.group_id)
        except RuntimeError as exc:
            # Operational failure (DB unavailable, auth error, etc.)
            # Emit to stderr and exit 1 — the only non-zero exit case.
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: Unexpected failure querying {args.backend}: {exc}", file=sys.stderr)
            return 1

    logger.debug(
        "Fetched %d entities and %d relations for group_id=%s",
        len(entities),
        len(relations),
        args.group_id,
    )

    # ── Compute metrics ───────────────────────────────────────────────────────
    metrics = compute_conformance_metrics(
        entities, relations, allowed_entity_types, allowed_relation_types
    )

    # ── Threshold warnings (OBSERVE-ONLY — never blocks) ─────────────────────
    if metrics["typed_entity_rate"] < args.typed_entity_threshold:
        warnings.append(
            f"typed_entity_rate={metrics['typed_entity_rate']:.3f} is below threshold "
            f"{args.typed_entity_threshold:.2f}. "
            f"Off-schema entity types: {metrics['off_schema_entity_types']}. "
            "OBSERVE-ONLY: no data was dropped."
        )
    if metrics["allowed_relation_rate"] < args.allowed_relation_threshold:
        warnings.append(
            f"allowed_relation_rate={metrics['allowed_relation_rate']:.3f} is below threshold "
            f"{args.allowed_relation_threshold:.2f}. "
            f"Off-schema relation types: {metrics['out_of_schema_types']}. "
            "OBSERVE-ONLY: no data was dropped."
        )

    # Emit warnings via logging so they show up in operational logs
    for w in warnings:
        logger.warning("[conformance] %s", w)

    # ── Build report ──────────────────────────────────────────────────────────
    report = build_report(
        group_id=args.group_id,
        metrics=metrics,
        typed_entity_threshold=args.typed_entity_threshold,
        allowed_relation_threshold=args.allowed_relation_threshold,
        observe_only=args.observe_only,
        dry_run=args.dry_run,
        warnings=warnings,
    )

    # ── Output ────────────────────────────────────────────────────────────────
    if args.output in ("json", "both"):
        print(json.dumps(report, indent=2))

    if args.output in ("summary", "both"):
        print_human_summary(report)

    # Exit 0 always — observe-only, no episode drops
    return 0


if __name__ == "__main__":
    sys.exit(main())
