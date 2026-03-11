#!/usr/bin/env python3
"""Import Graphiti-derived candidates into the derived candidates queue.

Goal: populate ``state/candidates.db`` with *suggested* facts derived from
Graphiti extraction outputs.

Current importer query paths:
- Entity RELATES_TO path (anchor-rooted; default)
- Entity RELATES_TO path (unanchored content lanes)
- OM-native path for ``s1_observational_memory`` (OMNode→OMNode typed edges)

Conservative mapping policy:
- Anchor/rooted entity imports preserve the existing v1 mapping behavior.
- OM-native imports are lane-isolated and mapped into ``graphiti.om.*`` namespace
  at sub-recommendation confidence to avoid noisy auto-promotion.

Idempotent: uses ``candidates.upsert_candidate`` fingerprinting.

Anchor config:
- Pass ``--anchors-config <path>`` to load anchor entities from a JSON file.
- If omitted, runs in anchor-less mode: all RELATES_TO edges are imported
  without subject-id scoping (anchor_name="content_unanchored").
- See ``config/anchors.example.json`` for the schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from graph_driver import add_backend_args, get_sync_client

# Add repo root to sys.path so truth.* packages are importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from truth.candidates import IneligibleAssertionTypeError, connect, upsert_candidate  # noqa: E402
from truth.entity_type_routing import route_entity_edge  # noqa: E402


def _neo4j_to_python(val: Any) -> Any:
    """Convert neo4j temporal types to Python-native equivalents for JSON serialization."""
    if val is None:
        return None
    type_name = type(val).__name__
    # neo4j.time.DateTime, neo4j.time.Date, neo4j.time.Time
    if type_name in ("DateTime", "Date", "Time") and hasattr(val, "iso_format"):
        return val.iso_format()
    # neo4j.time.Duration
    if type_name == "Duration" and hasattr(val, "iso_format"):
        return val.iso_format()
    # Lists (e.g. episodes array)
    if isinstance(val, list):
        return [_neo4j_to_python(v) for v in val]
    return val


# Module-level graph client and backend, set in main()
_graph_client = None
_backend = None

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PACK_REGISTRY_PATH = REPO_ROOT / "config" / "runtime_pack_registry.json"


# ---------------------------------------------------------------------------
# Lane type + query strategy contract
# ---------------------------------------------------------------------------

LANE_KIND_ENTITY = "entity"
LANE_KIND_OM_NATIVE = "om_native"

QUERY_STRATEGY_ENTITY_ANCHORED = "entity_relates_to_anchored"
QUERY_STRATEGY_ENTITY_CONTENT_UNANCHORED = "entity_relates_to_content_unanchored"
QUERY_STRATEGY_OM_NATIVE = "om_native"

# Candidate-generating lane contract: every candidate lane must declare a lane
# kind (OM-native vs Entity), and each lane kind must map to an importer query
# strategy.
CANDIDATE_LANE_KIND_BY_GROUP: Dict[str, str] = {
    "s1_sessions_main": LANE_KIND_ENTITY,
    "s1_chatgpt_history": LANE_KIND_ENTITY,
    "s1_observational_memory": LANE_KIND_OM_NATIVE,
    # Legacy candidate lane kept for compatibility with historical/manual runs.
    "s1_memory_day1": LANE_KIND_ENTITY,
}

QUERY_STRATEGY_BY_CANDIDATE_LANE_KIND: Dict[str, str] = {
    LANE_KIND_ENTITY: QUERY_STRATEGY_ENTITY_ANCHORED,
    LANE_KIND_OM_NATIVE: QUERY_STRATEGY_OM_NATIVE,
}

DEFAULT_CANDIDATE_GENERATING_LANES_FALLBACK: List[str] = [
    "s1_sessions_main",
    "s1_observational_memory",
    "s1_chatgpt_history",
]

# OM-native canonical edge types for candidate extraction.
OM_NATIVE_REL_TYPES: Tuple[str, ...] = (
    "MOTIVATES",
    "GENERATES",
    "SUPERSEDES",
    "ADDRESSES",
    "RESOLVES",
)

# Bound OM traversal/query workload to avoid unbounded scans in one call.
# Deterministic ordering + pagination preserves idempotent candidate upserts.
OM_NATIVE_PAGE_SIZE = 500
OM_NATIVE_MAX_PAGES = 2000


def _ordered_unique(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for v in values:
        vv = (v or "").strip()
        if not vv or vv in seen:
            continue
        out.append(vv)
        seen.add(vv)
    return out


def load_candidate_generating_lanes_from_registry(
    registry_path: Path = RUNTIME_PACK_REGISTRY_PATH,
) -> List[str]:
    """Load lane_policy.candidate_generating list from runtime registry.

    Falls back to a safe static set when registry is missing/malformed.
    """
    try:
        data = json.loads(registry_path.read_text())
        lane_policy = data.get("lane_policy", {})
        candidate_generating = lane_policy.get("candidate_generating")
        if isinstance(candidate_generating, list):
            normalized = _ordered_unique(str(x) for x in candidate_generating)
            if normalized:
                return normalized
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass

    return list(DEFAULT_CANDIDATE_GENERATING_LANES_FALLBACK)


def missing_candidate_lane_query_paths(candidate_lanes: Iterable[str]) -> List[str]:
    """Return candidate lanes that lack an importer query strategy contract."""
    missing: List[str] = []
    for lane in _ordered_unique(candidate_lanes):
        lane_kind = CANDIDATE_LANE_KIND_BY_GROUP.get(lane)
        if lane_kind is None:
            missing.append(lane)
            continue
        if lane_kind not in QUERY_STRATEGY_BY_CANDIDATE_LANE_KIND:
            missing.append(lane)
    return missing


def validate_candidate_lane_query_contract(candidate_lanes: Iterable[str]) -> None:
    """Enforce candidate-lane -> lane-kind -> query-strategy contract."""
    missing = missing_candidate_lane_query_paths(candidate_lanes)
    if missing:
        raise ValueError(
            "Missing importer query strategy for candidate-generating lane(s): "
            f"{sorted(missing)}. "
            "Update CANDIDATE_LANE_KIND_BY_GROUP and/or "
            "QUERY_STRATEGY_BY_CANDIDATE_LANE_KIND."
        )


def strategy_for_candidate_lane(graph: str) -> str:
    """Resolve the query strategy for a candidate-generating lane."""
    lane_kind = CANDIDATE_LANE_KIND_BY_GROUP.get(graph)
    if lane_kind is None:
        raise KeyError(f"Unknown candidate lane: {graph}")
    strategy = QUERY_STRATEGY_BY_CANDIDATE_LANE_KIND.get(lane_kind)
    if strategy is None:
        raise KeyError(f"No query strategy for candidate lane kind: {lane_kind}")
    return strategy


# ---------------------------------------------------------------------------
# Shared query helpers
# ---------------------------------------------------------------------------


def _now_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


_SAFE_CYPHER_RE = re.compile(r"^[A-Za-z0-9 _.\-]+$")


def cypher_quote(s: str) -> str:
    """Safely quote a string for Cypher interpolation.

    Only allows alphanumeric characters, spaces, underscores, dots, and hyphens.
    Raises ValueError for anything else to prevent Cypher injection.
    """
    if not _SAFE_CYPHER_RE.match(s):
        raise ValueError(f"Refusing to interpolate unsafe Cypher value: {s!r}")
    return f"'{s}'"


def run_graph_query(
    graph: str,
    query: str,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[List[Any]], List[str]]:
    """Run a Cypher query via the configured graph client."""
    return _graph_client.query(graph, query, params=params)


# ---------------------------------------------------------------------------
# Anchor model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Anchor:
    canonical_name: str
    subject_id: str
    aliases: Tuple[str, ...]


def load_anchors_from_config(path: str | None) -> List[Anchor]:
    """Load anchors from a JSON config file, or return [] for anchor-less mode.

    Args:
        path: Path to a JSON file following the anchors.example.json schema,
              or None to run without subject-id scoping.

    Returns:
        List of Anchor objects, or [] if path is None or anchors list is empty.
    """
    if not path:
        return []
    with open(path) as f:
        raw = json.load(f)
    return [
        Anchor(
            canonical_name=entry["canonical_name"],
            subject_id=entry["subject_id"],
            aliases=tuple(entry.get("aliases", [])),
        )
        for entry in raw.get("anchors", [])
    ]


# Content graphs don't reliably emit anchor-sourced edges, so we import
# RELATES_TO edges in a separate mode using source-entity-derived subjects.
CONTENT_GRAPH_PREFIXES: Tuple[str, ...] = ("s1_inspiration_", "s1_content_strategy")


def is_content_graph(graph: str) -> bool:
    return any(graph.startswith(p) for p in CONTENT_GRAPH_PREFIXES)


def to_content_subject_id(source_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (source_name or "unknown").lower()).strip("_")
    if not slug:
        slug = "unknown"
    return f"content.source:{slug}"


def iter_anchor_aliases(anchors: Iterable[Anchor]) -> Iterable[Tuple[str, str]]:
    """Yield (name, subject_id) pairs for every lookup name of each anchor.

    Yields the canonical_name first, then each alias.  Previously only aliases
    were yielded, so an anchor whose graph node uses canonical_name would
    silently produce zero query results.
    """
    for a in anchors:
        yield a.canonical_name, a.subject_id
        for alias in a.aliases:
            yield alias, a.subject_id


# ---------------------------------------------------------------------------
# Predicate mapping (entity RELATES_TO path)
# ---------------------------------------------------------------------------


_INT_RE = re.compile(r"\b(\d{6,})\b")


def parse_int_from_fact(fact: str) -> Optional[int]:
    if not fact:
        return None
    m = _INT_RE.search(fact)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def parse_optimizing_for_list(fact: str) -> Optional[List[str]]:
    """Heuristic parse for '... optimizing for X and Y ...'."""
    if not fact:
        return None

    lower = fact.lower()
    if "optimizing for" not in lower:
        return None

    # Take substring after 'optimizing for'
    after = fact[lower.index("optimizing for") + len("optimizing for") :].strip()
    after = after.rstrip(".!")

    # Split on ' and ' / commas.
    parts: List[str] = []
    for chunk in re.split(r"\band\b|,", after):
        c = chunk.strip()
        if not c:
            continue
        parts.append(c)

    # De-duplicate while preserving order
    seen = set()
    out: List[str] = []
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)

    return out or None


def map_relation_to_candidate(
    *,
    source_name: str,
    target_name: str,
    rel_name: str,
    fact: str,
    is_procedure_edge: bool = False,
    a_labels: "list[str] | None" = None,
    b_labels: "list[str] | None" = None,
) -> Tuple[str, str, Any, float]:
    """Return (predicate, assertion_type, value_obj, confidence).

    Routing priority:
    1. Label-driven routing via public helper route_entity_edge() (a_labels/b_labels).
    2. Legacy boolean fallback: is_procedure_edge=True delegates to the helper
       with a_labels=["Procedure"] — kept for backwards compat, prefer passing
       a_labels/b_labels directly.
    3. rel_name dispatch (existing logic unchanged).
    """
    # Try label-driven routing first (delegates to public helper)
    routed = route_entity_edge(
        source_name=source_name,
        target_name=target_name,
        rel_name=rel_name,
        fact=fact,
        a_labels=a_labels,
        b_labels=b_labels,
    )
    if routed is not None:
        return routed

    # Legacy boolean fallback (deprecated — prefer passing a_labels/b_labels)
    if is_procedure_edge:
        return route_entity_edge(
            source_name=source_name,
            target_name=target_name,
            rel_name=rel_name,
            fact=fact,
            a_labels=["Procedure"],
            b_labels=[],
        )

    rn = (rel_name or "").strip().upper()

    # High-signal / core truth examples
    if rn in {"HAS_CHAT_ID", "HAS_TELEGRAM_CHAT_ID"}:
        v = parse_int_from_fact(fact)
        if v is None:
            # Fall back to textual value.
            return (
                "security.telegram.chat_id",
                "factual_assertion",
                {"text": fact.strip()} if fact else {"object": target_name},
                0.85,
            )
        return ("security.telegram.chat_id", "factual_assertion", v, 0.92)

    # Low-risk / preferences
    if rn == "IS_OPTIMIZING_FOR":
        vlist = parse_optimizing_for_list(fact)
        if vlist:
            return ("pref.goals.optimizing_for", "preference", vlist, 0.88)
        return ("pref.goals.optimizing_for", "preference", {"text": fact.strip()}, 0.80)

    if rn == "TRUST":
        return ("style.security.trust_boundary", "decision", {"text": fact.strip()}, 0.82)

    # Default: keep Graphiti predicate in a separate namespace.
    # Use LOW confidence so it doesn't auto-recommend.
    pred = f"graphiti.{rn.lower()}" if rn else "graphiti.relates_to"
    return (pred, "factual_assertion", {"object": target_name}, 0.70)


# ---------------------------------------------------------------------------
# Predicate mapping (OM-native path)
# ---------------------------------------------------------------------------


def _safe_trim(text: Any, limit: int = 280) -> str:
    value = str(text or "").strip()
    return value[:limit]


def _om_subject_id(graph: str, src_node_id: str) -> str:
    """Lane-isolated synthetic subject id for OM-native candidates."""
    return f"om.node:{graph}:{src_node_id}"


def _om_edge_evidence_id(
    *,
    graph: str,
    src_node_id: str,
    rel_type: str,
    dst_node_id: str,
    rel_chunk_id: str,
) -> str:
    raw = "|".join([
        graph,
        src_node_id,
        rel_type,
        dst_node_id,
        rel_chunk_id,
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def map_om_relation_to_candidate(
    *,
    rel_type: str,
    src_node_type: str,
    dst_node_id: str,
    dst_node_type: str,
    dst_content: str,
    dst_domain: str,
) -> Tuple[str, str, Dict[str, Any], float]:
    """Conservative mapping for OM-native edges.

    Returns (predicate, assertion_type, value_obj, confidence).
    """
    rel = (rel_type or "").strip().upper()
    predicate = f"graphiti.om.{rel.lower()}" if rel else "graphiti.om.related"

    if rel == "SUPERSEDES":
        assertion_type = "decision"
        confidence = 0.78
    elif rel in {"MOTIVATES", "GENERATES", "ADDRESSES", "RESOLVES"}:
        assertion_type = "decision"
        confidence = 0.76
    else:
        # Unknown OM relation types stay conservative.
        assertion_type = "episode"
        confidence = 0.72

    value: Dict[str, Any] = {
        "source_node_type": (src_node_type or "OMNode"),
        "target_node_id": dst_node_id,
        "target_node_type": (dst_node_type or "OMNode"),
        "target_content": _safe_trim(dst_content, limit=280),
    }
    if dst_domain:
        value["target_semantic_domain"] = dst_domain

    return predicate, assertion_type, value, confidence


# ---------------------------------------------------------------------------
# Import loop
# ---------------------------------------------------------------------------


def resolve_query_strategy_for_graph(graph: str) -> str:
    """Resolve importer query strategy for a graph lane.

    Candidate-generating lanes are resolved via the explicit lane-type contract.
    Non-candidate lanes keep existing behavior for compatibility.
    """
    if graph in CANDIDATE_LANE_KIND_BY_GROUP:
        return strategy_for_candidate_lane(graph)
    if is_content_graph(graph):
        return QUERY_STRATEGY_ENTITY_CONTENT_UNANCHORED
    return QUERY_STRATEGY_ENTITY_ANCHORED


def import_graph(
    *,
    conn,
    graph: str,
    anchors: List[Anchor],
    dry_run: bool,
) -> Dict[str, int]:
    stats = {
        "rows": 0,
        "upserts": 0,
        "created": 0,
        "skipped_ineligible": 0,
        "errors": 0,
    }

    strategy = resolve_query_strategy_for_graph(graph)

    def process_entity_rows(rows: List[List[Any]], idx: Dict[str, int], *, anchor_name: str, subject_id: str) -> None:
        for r in rows:
            stats["rows"] += 1
            try:
                source = str(r[idx.get("a.name", 0)])
                target = str(r[idx.get("b.name", 1)])
                rel_uuid = str(r[idx.get("r.uuid", 2)])
                rel_name = str(r[idx.get("r.name", 3)])
                raw_fact = r[idx.get("r.fact", 4)]
                fact = str(raw_fact) if raw_fact is not None else ""
                valid_at = _neo4j_to_python(r[idx.get("r.valid_at", 5)]) if "r.valid_at" in idx else None
                created_at = _neo4j_to_python(r[idx.get("r.created_at", 6)]) if "r.created_at" in idx else None
                episodes = _neo4j_to_python(r[idx.get("r.episodes", 7)]) if "r.episodes" in idx else None

                a_labels = list(r[idx["a_labels"]]) if "a_labels" in idx else []
                b_labels = list(r[idx["b_labels"]]) if "b_labels" in idx else []
                is_procedure_edge = "Procedure" in a_labels or "Procedure" in b_labels

                predicate, assertion_type, value_obj, confidence = map_relation_to_candidate(
                    source_name=source,
                    target_name=target,
                    rel_name=rel_name,
                    fact=fact,
                    is_procedure_edge=is_procedure_edge,
                    a_labels=a_labels,
                    b_labels=b_labels,
                )

                effective_subject = subject_id
                if anchor_name == "content_unanchored":
                    effective_subject = to_content_subject_id(source)

                evidence_ref: Dict[str, Any] = {
                    "source_key": f"graphiti:{graph}",
                    "chunk_key": f"rel:{rel_uuid}",
                    "evidence_id": rel_uuid,
                    "observed_at": valid_at or created_at,
                    "scope": "private",
                    "extra": {
                        "graph": graph,
                        "anchor": anchor_name,
                        "source_entity": source,
                        "target_entity": target,
                        "rel_name": rel_name,
                        "episodes": episodes,
                    },
                }

                payload = {
                    "subject": effective_subject,
                    "predicate": predicate,
                    "scope": "private",
                    "assertion_type": assertion_type,
                    "value": value_obj,
                    "evidence_refs": [evidence_ref],
                    "evidence_quote": (str(raw_fact).strip()[:200] if raw_fact is not None else None) or None,
                    "speaker_id": None,
                    "confidence": float(confidence),
                    "source_trust": None,
                    "conflict_with_fact_id": None,
                    "origin": "graphiti",
                    "reason": f"graphiti:{graph}",
                    "explicit_update": False,
                }

                if dry_run:
                    continue

                res = upsert_candidate(conn, **payload)
                stats["upserts"] += 1
                if res.created:
                    stats["created"] += 1

            except IneligibleAssertionTypeError:
                stats["skipped_ineligible"] += 1
            except Exception as e:
                stats["errors"] += 1
                print(f"ERROR row graph={graph} anchor={anchor_name}: {type(e).__name__}: {e}")

    def process_om_rows(rows: List[List[Any]], idx: Dict[str, int]) -> None:
        for r in rows:
            stats["rows"] += 1
            try:
                src_node_id = str(r[idx["src_node_id"]])
                src_node_type = str(r[idx.get("src_node_type", idx["src_node_id"])])
                src_content = str(r[idx.get("src_content", idx["src_node_id"])])
                src_domain = str(r[idx.get("src_domain", idx["src_node_id"])])

                dst_node_id = str(r[idx["dst_node_id"]])
                dst_node_type = str(r[idx.get("dst_node_type", idx["dst_node_id"])])
                dst_content = str(r[idx.get("dst_content", idx["dst_node_id"])])
                dst_domain = str(r[idx.get("dst_domain", idx["dst_node_id"])])

                rel_type = str(r[idx["rel_type"]]).strip().upper()
                rel_chunk_id = str(r[idx.get("rel_chunk_id", idx["rel_type"])])

                linked_at = _neo4j_to_python(r[idx.get("linked_at")]) if "linked_at" in idx else None
                src_last_observed_at = (
                    _neo4j_to_python(r[idx.get("src_last_observed_at")])
                    if "src_last_observed_at" in idx
                    else None
                )
                src_created_at = _neo4j_to_python(r[idx.get("src_created_at")]) if "src_created_at" in idx else None
                dst_last_observed_at = (
                    _neo4j_to_python(r[idx.get("dst_last_observed_at")])
                    if "dst_last_observed_at" in idx
                    else None
                )
                dst_created_at = _neo4j_to_python(r[idx.get("dst_created_at")]) if "dst_created_at" in idx else None

                observed_at = (
                    linked_at
                    or src_last_observed_at
                    or src_created_at
                    or dst_last_observed_at
                    or dst_created_at
                    or _now_iso_z()
                )

                evidence_id = _om_edge_evidence_id(
                    graph=graph,
                    src_node_id=src_node_id,
                    rel_type=rel_type,
                    dst_node_id=dst_node_id,
                    rel_chunk_id=rel_chunk_id,
                )

                predicate, assertion_type, value_obj, confidence = map_om_relation_to_candidate(
                    rel_type=rel_type,
                    src_node_type=src_node_type,
                    dst_node_id=dst_node_id,
                    dst_node_type=dst_node_type,
                    dst_content=dst_content,
                    dst_domain=dst_domain,
                )

                evidence_ref: Dict[str, Any] = {
                    "source_key": f"graphiti:{graph}",
                    "chunk_key": f"om_rel:{evidence_id}",
                    "evidence_id": evidence_id,
                    "observed_at": observed_at,
                    "scope": "private",
                    "extra": {
                        "graph": graph,
                        "lane_type": LANE_KIND_OM_NATIVE,
                        "source_node_id": src_node_id,
                        "source_node_type": src_node_type,
                        "source_semantic_domain": src_domain,
                        "target_node_id": dst_node_id,
                        "target_node_type": dst_node_type,
                        "target_semantic_domain": dst_domain,
                        "rel_type": rel_type,
                        "rel_chunk_id": rel_chunk_id,
                    },
                }

                evidence_quote = (
                    f"{_safe_trim(src_content, 90)} --{rel_type}--> {_safe_trim(dst_content, 90)}"
                ).strip()
                if not evidence_quote:
                    evidence_quote = None

                payload = {
                    "subject": _om_subject_id(graph, src_node_id),
                    "predicate": predicate,
                    "scope": "private",
                    "assertion_type": assertion_type,
                    "value": value_obj,
                    "evidence_refs": [evidence_ref],
                    "evidence_quote": evidence_quote,
                    "speaker_id": None,
                    "confidence": float(confidence),
                    "source_trust": None,
                    "conflict_with_fact_id": None,
                    "origin": "graphiti",
                    "reason": f"graphiti:{graph}:om_native",
                    "explicit_update": False,
                }

                if dry_run:
                    continue

                res = upsert_candidate(conn, **payload)
                stats["upserts"] += 1
                if res.created:
                    stats["created"] += 1

            except IneligibleAssertionTypeError:
                stats["skipped_ineligible"] += 1
            except Exception as e:
                stats["errors"] += 1
                print(f"ERROR row graph={graph} strategy={strategy}: {type(e).__name__}: {e}")

    # OM-native query path.
    if strategy == QUERY_STRATEGY_OM_NATIVE:
        om_return = (
            "RETURN "
            "a.node_id AS src_node_id, "
            "coalesce(a.node_type, 'OMNode') AS src_node_type, "
            "coalesce(a.content, '') AS src_content, "
            "coalesce(a.semantic_domain, '') AS src_domain, "
            "b.node_id AS dst_node_id, "
            "coalesce(b.node_type, 'OMNode') AS dst_node_type, "
            "coalesce(b.content, '') AS dst_content, "
            "coalesce(b.semantic_domain, '') AS dst_domain, "
            "type(r) AS rel_type, "
            "coalesce(r.chunk_id, '') AS rel_chunk_id, "
            "r.linked_at AS linked_at, "
            "a.last_observed_at AS src_last_observed_at, "
            "a.created_at AS src_created_at, "
            "b.last_observed_at AS dst_last_observed_at, "
            "b.created_at AS dst_created_at "
            "ORDER BY src_node_id, dst_node_id, rel_type, rel_chunk_id "
        )

        if _backend == "neo4j":
            q = (
                "MATCH (a:OMNode)-[r]->(b:OMNode) "
                "WHERE a.group_id = $group_id "
                "AND b.group_id = $group_id "
                "AND coalesce(r.group_id, $group_id) = $group_id "
                "AND type(r) IN $rel_types "
                + om_return
                + "SKIP $skip LIMIT $limit"
            )
            base_params: Optional[Dict[str, Any]] = {
                "group_id": graph,
                "rel_types": list(OM_NATIVE_REL_TYPES),
            }
        else:
            rel_filter = ", ".join(cypher_quote(rt) for rt in OM_NATIVE_REL_TYPES)
            q = (
                "MATCH (a:OMNode)-[r]->(b:OMNode) "
                "WHERE type(r) IN [" + rel_filter + "] "
                + om_return
                + "SKIP {skip} LIMIT {limit}"
            )
            base_params = None

        pages = 0
        offset_rows = 0
        while pages < OM_NATIVE_MAX_PAGES:
            try:
                if _backend == "neo4j":
                    q_params = dict(base_params or {})
                    q_params.update({"skip": offset_rows, "limit": OM_NATIVE_PAGE_SIZE})
                    header, rows, _ = run_graph_query(graph, q, params=q_params)
                else:
                    paged_query = q.format(skip=offset_rows, limit=OM_NATIVE_PAGE_SIZE)
                    header, rows, _ = run_graph_query(graph, paged_query, params=None)
            except Exception as e:
                stats["errors"] += 1
                print(f"ERROR graph={graph} strategy={strategy}: {e}")
                return stats

            if not rows:
                break

            idx = {name: i for i, name in enumerate(header)}
            process_om_rows(rows, idx)

            pages += 1
            if len(rows) < OM_NATIVE_PAGE_SIZE:
                break
            offset_rows += OM_NATIVE_PAGE_SIZE

        if pages >= OM_NATIVE_MAX_PAGES:
            print(
                f"WARN graph={graph} strategy={strategy}: reached OM pagination safety cap "
                f"(pages={OM_NATIVE_MAX_PAGES}, page_size={OM_NATIVE_PAGE_SIZE})"
            )

        return stats

    # Content graphs: import all RELATES_TO edges (not only anchor-rooted edges)
    # and map source entity names into content-scoped synthetic subjects.
    if strategy == QUERY_STRATEGY_ENTITY_CONTENT_UNANCHORED:
        if _backend == "neo4j":
            q = (
                "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
                "WHERE a.group_id = $group_id AND b.group_id = $group_id "
                "RETURN a.name, b.name, r.uuid, r.name, r.fact, r.valid_at, r.created_at, r.episodes, labels(a) AS a_labels, labels(b) AS b_labels"
            )
            q_params = {"group_id": graph}
        else:
            q = (
                "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
                "RETURN a.name, b.name, r.uuid, r.name, r.fact, r.valid_at, r.created_at, r.episodes, labels(a) AS a_labels, labels(b) AS b_labels"
            )
            q_params = None

        try:
            header, rows, _ = run_graph_query(graph, q, params=q_params)
        except Exception as e:
            stats["errors"] += 1
            print(f"ERROR graph={graph} anchor=content_unanchored: {e}")
            return stats

        if rows:
            idx = {name: i for i, name in enumerate(header)}
            process_entity_rows(rows, idx, anchor_name="content_unanchored", subject_id="content.source:unknown")
        return stats

    # Default anchored entity import mode.
    # When no anchors are configured (anchor-less mode), fall back to
    # content-unanchored import to import all RELATES_TO edges.
    if not anchors:
        if _backend == "neo4j":
            q = (
                "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
                "WHERE a.group_id = $group_id AND b.group_id = $group_id "
                "RETURN a.name, b.name, r.uuid, r.name, r.fact, r.valid_at, r.created_at, r.episodes, labels(a) AS a_labels, labels(b) AS b_labels"
            )
            q_params = {"group_id": graph}
        else:
            q = (
                "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
                "RETURN a.name, b.name, r.uuid, r.name, r.fact, r.valid_at, r.created_at, r.episodes, labels(a) AS a_labels, labels(b) AS b_labels"
            )
            q_params = None

        try:
            header, rows, _ = run_graph_query(graph, q, params=q_params)
        except Exception as e:
            stats["errors"] += 1
            print(f"ERROR graph={graph} anchor=anchorless: {e}")
            return stats

        if rows:
            idx = {name: i for i, name in enumerate(header)}
            process_entity_rows(rows, idx, anchor_name="anchorless", subject_id="subject:unknown")
        return stats

    for anchor_name, subject_id in iter_anchor_aliases(anchors):
        if _backend == "neo4j":
            q = (
                "MATCH (a:Entity {name: $anchor_name})"
                "-[r:RELATES_TO]->(b:Entity) "
                "WHERE a.group_id = $group_id AND b.group_id = $group_id "
                "RETURN a.name, b.name, r.uuid, r.name, r.fact, r.valid_at, r.created_at, r.episodes, labels(a) AS a_labels, labels(b) AS b_labels"
            )
            q_params = {"anchor_name": anchor_name, "group_id": graph}
        else:
            q = (
                "MATCH (a:Entity {name:" + cypher_quote(anchor_name) + "})"
                "-[r:RELATES_TO]->(b:Entity) "
                "RETURN a.name, b.name, r.uuid, r.name, r.fact, r.valid_at, r.created_at, r.episodes, labels(a) AS a_labels, labels(b) AS b_labels"
            )
            q_params = None

        try:
            header, rows, _ = run_graph_query(graph, q, params=q_params)
        except Exception as e:
            stats["errors"] += 1
            print(f"ERROR graph={graph} anchor={anchor_name}: {e}")
            continue

        if not rows:
            continue

        idx = {name: i for i, name in enumerate(header)}
        process_entity_rows(rows, idx, anchor_name=anchor_name, subject_id=subject_id)

    return stats


def main() -> None:
    global _graph_client, _backend

    candidate_generating_lanes = load_candidate_generating_lanes_from_registry()
    validate_candidate_lane_query_contract(candidate_generating_lanes)

    ap = argparse.ArgumentParser()
    add_backend_args(ap)
    ap.add_argument(
        "--db",
        default=str(REPO_ROOT / "state" / "candidates.db"),
        help="Path to derived candidates.db",
    )
    ap.add_argument(
        "--graphs",
        # Default to lane-policy candidate_generating lanes from runtime registry.
        # This keeps importer defaults aligned with policy wiring.
        default=",".join(candidate_generating_lanes),
        help="Comma-separated graph names to import from",
    )
    ap.add_argument(
        "--anchors-config",
        default=None,
        help=(
            "Path to JSON file defining anchor entities. "
            "If omitted, runs in anchor-less mode (all edges imported without subject-id scoping). "
            "See config/anchors.example.json for the schema."
        ),
    )
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()

    graphs = [g.strip() for g in (args.graphs or "").split(",") if g.strip()]
    if not graphs:
        raise SystemExit("No graphs specified")

    db_path = Path(args.db)

    anchors = load_anchors_from_config(args.anchors_config)

    _graph_client = get_sync_client(args.backend)
    _backend = args.backend

    conn = None
    if not args.dry_run:
        conn = connect(db_path)

    total = {"rows": 0, "upserts": 0, "created": 0, "skipped_ineligible": 0, "errors": 0}

    print(f"Importing Graphiti edges → candidates.db  (backend: {args.backend})")
    print(f"  db: {db_path}")
    print(f"  graphs: {graphs}")
    if anchors:
        print(f"  anchors: {[a.canonical_name for a in anchors]}")
    else:
        print("  anchors: (anchor-less mode — all edges imported without subject-id scoping)")
    print(f"  content_graph_prefixes(unanchored): {list(CONTENT_GRAPH_PREFIXES)}")
    print(f"  om_native_rel_types: {list(OM_NATIVE_REL_TYPES)}")
    print(f"  dry_run: {args.dry_run}")
    print()

    try:
        for g in graphs:
            strategy = resolve_query_strategy_for_graph(g)
            print(f"== graph: {g} (strategy={strategy}) ==")
            st = import_graph(conn=conn, graph=g, anchors=anchors, dry_run=args.dry_run)
            for k in total:
                total[k] += int(st.get(k, 0))
            print(
                f"  rows={st['rows']} upserts={st['upserts']} created={st['created']} "
                f"skipped_ineligible={st['skipped_ineligible']} errors={st['errors']}"
            )
            print()
    finally:
        if _graph_client is not None:
            _graph_client.close()
        if conn is not None:
            conn.close()

    print("DONE")
    print(
        f"  total_rows={total['rows']} total_upserts={total['upserts']} created={total['created']} "
        f"skipped_ineligible={total['skipped_ineligible']} errors={total['errors']}"
    )


if __name__ == "__main__":
    main()
