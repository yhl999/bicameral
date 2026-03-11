"""Entity-label-to-assertion-type routing helpers.

Importers (private or public) call ``route_entity_edge`` to determine the
canonical assertion_type, predicate, value dict, and confidence for a
RELATES_TO edge based on the Neo4j entity labels attached to its endpoints.

This keeps routing logic in one place so private importers stay thin wrappers.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Label sets that trigger non-default routing
# ---------------------------------------------------------------------------

#: Entity labels that map to assertion_type='procedure'
PROCEDURE_ENTITY_LABELS: frozenset[str] = frozenset({"Procedure"})

#: Entity labels that map to assertion_type='episode'
EPISODE_ENTITY_LABELS: frozenset[str] = frozenset({"Episode", "EngineeringIncident", "AuditEpisode"})

#: Default confidence when routing by entity label (conservative; candidate
#: review is expected before promotion).
DEFAULT_LABEL_CONFIDENCE: float = 0.72


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def route_entity_edge(
    *,
    source_name: str,
    target_name: str,
    rel_name: str,
    fact: str,
    a_labels: list[str] | None = None,
    b_labels: list[str] | None = None,
) -> tuple[str, str, Any, float] | None:
    """Map a RELATES_TO edge to a typed (predicate, assertion_type, value, confidence).

    Returns a 4-tuple when entity labels indicate a specific typed routing, or
    ``None`` when no label-driven routing applies (caller falls back to its own
    rel_name dispatch).

    Parameters
    ----------
    source_name:
        Name of the source entity node.
    target_name:
        Name of the target entity node.
    rel_name:
        Relationship type / name extracted from the graph edge.
    fact:
        Free-text fact string on the edge.
    a_labels:
        Neo4j labels on the source entity node (may be ``None`` when not
        fetched).
    b_labels:
        Neo4j labels on the target entity node (may be ``None`` when not
        fetched).

    Returns
    -------
    tuple[str, str, Any, float] | None
        ``(predicate, assertion_type, value_dict, confidence)`` when a
        label-driven route is found, otherwise ``None``.
    """
    a_set = frozenset(a_labels or [])
    b_set = frozenset(b_labels or [])
    all_labels = a_set | b_set

    # --- Procedure routing ---------------------------------------------------
    if all_labels & PROCEDURE_ENTITY_LABELS:
        return (
            "procedure.steps",
            "procedure",
            {
                "name": source_name,
                "trigger": (rel_name or "").strip() or "relates_to",
                "description": fact.strip() if fact else "",
            },
            DEFAULT_LABEL_CONFIDENCE,
        )

    # --- Episode routing -----------------------------------------------------
    if all_labels & EPISODE_ENTITY_LABELS:
        return (
            "episode.description",
            "episode",
            {
                "name": source_name,
                "description": fact.strip() if fact else "",
                "participants": [source_name, target_name],
            },
            DEFAULT_LABEL_CONFIDENCE,
        )

    # No label-driven route found — caller uses its own dispatch
    return None
