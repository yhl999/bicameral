"""
HybridRetrievalService — production-surface hybrid retrieval.

Provides typed state/procedure candidate extraction and Reciprocal Rank
Fusion (RRF) merge for the hybrid default surface of search_memory_facts.

Design goals:
- No benchmark-script imports; all fusion logic is self-contained.
- Deterministic and testable: same inputs → same output.
- Narrow scope: state + procedure candidates only (not episodes).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from .typed_retrieval import TypedRetrievalService
except ImportError:  # pragma: no cover — top-level import fallback
    from services.typed_retrieval import TypedRetrievalService

# ── RRF tuning constants ──────────────────────────────────────────────────────
# k=60: standard RRF constant matching experiment calibration in lane_fair_merge
# and reranker blend (reranker uses merge_score * 60 at rrf_blend_weight=0.2).
# Larger k softens rank differences — less top-heavy than k=1, more forgiving
# for mid-rank candidates that appear in multiple sources.
_HYBRID_RRF_K: float = 60.0
# Graph-edge recall gets full weight; typed candidates get a slight discount so
# that a weak typed match does not dominate a strong graph hit.
_HYBRID_WEIGHT_GRAPH: float = 1.0
_HYBRID_WEIGHT_TYPED: float = 0.85


# ── Key helpers for deduplication ────────────────────────────────────────────

def _graph_fact_id(fact: dict[str, Any]) -> str:
    """Stable deduplication key for a graph-edge fact."""
    uuid = str(fact.get("uuid", "") or "").strip()
    if uuid:
        return f"graph:{uuid}"
    name = str(fact.get("name", "") or "").strip().lower()[:64]
    return f"graph:name:{name}"


def _typed_item_id(item: dict[str, Any], *, bucket: str) -> str:
    """Stable deduplication key for a typed ledger object."""
    oid = str(item.get("object_id", "") or "").strip()
    return f"typed:{bucket}:{oid}" if oid else f"typed:{bucket}:unknown"


# ── Typed → hybrid-entry conversion ──────────────────────────────────────────

def _typed_to_hybrid_entry(item: dict[str, Any], *, source_label: str) -> dict[str, Any]:
    """Convert a typed state/procedure object to a hybrid-response entry.

    The hybrid-fact shape:
    - ``_source``: provenance tag ('typed_state' | 'typed_procedure')
    - ``_hybrid_score``: RRF score (annotated after merge)
    - ``_object_type``: original typed object type
    - ``fact``: human-readable summary text
    - ``uuid``: stable object identifier (from ``object_id``)
    - ``_original``: full original typed object payload for downstream use
    """
    oid = str(item.get("object_id", "") or "")
    if source_label == "typed_state":
        parts = [
            str(item.get("subject", "") or ""),
            str(item.get("predicate", "") or ""),
            str(item.get("value", "") or ""),
        ]
        fact_text = " ".join(p for p in parts if p)
    elif source_label == "typed_procedure":
        parts = [
            str(item.get("name", "") or ""),
            str(item.get("trigger", "") or ""),
        ]
        fact_text = " ".join(p for p in parts if p) or oid
    else:
        fact_text = oid

    return {
        "uuid": oid,
        "fact": fact_text or oid,
        "_source": source_label,
        "_object_type": str(item.get("object_type", "") or ""),
        "_hybrid_score": 0.0,
        "_original": item,
    }


# ── RRF merge ─────────────────────────────────────────────────────────────────

def rrf_merge_hybrid(
    *,
    graph_facts: list[dict[str, Any]],
    typed_state: list[dict[str, Any]],
    typed_procedures: list[dict[str, Any]],
    max_facts: int,
    rrf_k: float = _HYBRID_RRF_K,
    weight_graph: float = _HYBRID_WEIGHT_GRAPH,
    weight_typed: float = _HYBRID_WEIGHT_TYPED,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion over graph facts + typed state/procedure candidates.

    Each source is ranked independently and weighted. Items that appear in
    multiple sources accumulate their RRF scores. The merged list is returned
    in score-descending order, capped at ``max_facts``.

    Args:
        graph_facts: Ranked list of graph-edge facts (index 0 = best).
        typed_state: Ranked list of typed state-fact objects.
        typed_procedures: Ranked list of typed procedure objects.
        max_facts: Maximum items to return.
        rrf_k: RRF smoothing constant.
        weight_graph: Multiplicative weight for graph-edge source.
        weight_typed: Multiplicative weight for typed-ledger source.

    Returns:
        Merged list of hybrid-entry dicts, annotated with ``_source`` and
        ``_hybrid_score``, sorted by descending RRF score.
    """
    if max_facts <= 0:
        return []

    scores: dict[str, float] = {}
    registry: dict[str, dict[str, Any]] = {}

    for rank, fact in enumerate(graph_facts, start=1):
        key = _graph_fact_id(fact)
        scores[key] = scores.get(key, 0.0) + weight_graph / (rrf_k + rank)
        if key not in registry:
            entry = dict(fact)
            entry.setdefault("_source", "graph")
            entry["_hybrid_score"] = 0.0
            registry[key] = entry

    for rank, item in enumerate(typed_state, start=1):
        key = _typed_item_id(item, bucket="state")
        scores[key] = scores.get(key, 0.0) + weight_typed / (rrf_k + rank)
        if key not in registry:
            registry[key] = _typed_to_hybrid_entry(item, source_label="typed_state")

    for rank, item in enumerate(typed_procedures, start=1):
        key = _typed_item_id(item, bucket="procedure")
        scores[key] = scores.get(key, 0.0) + weight_typed / (rrf_k + rank)
        if key not in registry:
            registry[key] = _typed_to_hybrid_entry(item, source_label="typed_procedure")

    # Annotate with final RRF scores.
    for key, score in scores.items():
        if key in registry:
            registry[key]["_hybrid_score"] = round(score, 6)

    ordered_keys = sorted(scores, key=lambda k: (-scores[k], k))
    return [registry[k] for k in ordered_keys[:max_facts]]


# ── Service class ─────────────────────────────────────────────────────────────

@dataclass
class HybridRetrievalService:
    """Production-surface service for the hybrid default retrieval path.

    Fetches typed state + procedure candidates from the change ledger, then
    merges them with graph-recall facts via Reciprocal Rank Fusion.

    Used by ``search_memory_facts`` when ``retrieval_mode='hybrid'``.

    Attributes:
        om_projection_service: Optional OM projection service passed through
            to the underlying ``TypedRetrievalService``.
    """

    om_projection_service: Any | None = None
    _typed_service: TypedRetrievalService | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._typed_service is None:
            self._typed_service = TypedRetrievalService(
                om_projection_service=self.om_projection_service,
            )

    async def get_typed_candidates(
        self,
        *,
        query: str,
        effective_group_ids: list[str] | None = None,
        max_candidates: int = 10,
        metadata_filters: dict[str, Any] | None = None,
        history_mode: str = "auto",
        current_only: bool | None = None,
    ) -> dict[str, Any]:
        """Run typed retrieval (state + procedures only) for the hybrid candidate pool.

        Episodes are excluded from the candidate pool; they belong in the
        episodic/graph recall path rather than the fact-surface hybrid merge.

        Args:
            query: The retrieval query.
            effective_group_ids: Lane/group scope, resolved before this call.
            max_candidates: Cap on typed results (mirrors max_facts).
            metadata_filters: Optional metadata filter map (already intersected
                with group scope by the caller).
            history_mode: Typed retrieval mode forwarded from the caller
                (``'auto'`` | ``'current'`` | ``'history'`` | ``'all'``).
            current_only: Optional explicit override for current-only typed
                retrieval, forwarded from the caller.

        Returns:
            Typed retrieval result dict (``state``, ``procedures``, ``counts``, …).
        """
        assert self._typed_service is not None
        return await self._typed_service.search(
            query=query,
            object_types=["state", "procedure"],
            metadata_filters=metadata_filters or {},
            history_mode=history_mode,
            current_only=current_only,
            max_results=max_candidates,
            max_evidence=20,
            effective_group_ids=effective_group_ids,
        )

    def merge(
        self,
        *,
        graph_facts: list[dict[str, Any]],
        typed_results: dict[str, Any],
        max_facts: int,
    ) -> list[dict[str, Any]]:
        """Merge graph-recall facts with typed state/procedure candidates via RRF.

        Args:
            graph_facts: Ranked graph-edge facts from the graph recall path.
            typed_results: Result dict from :meth:`get_typed_candidates`.
            max_facts: Maximum items in the returned merged list.

        Returns:
            RRF-merged list of hybrid entries, score-descending.
        """
        state_items: list[dict[str, Any]] = typed_results.get("state", []) or []
        procedure_items: list[dict[str, Any]] = typed_results.get("procedures", []) or []
        return rrf_merge_hybrid(
            graph_facts=graph_facts,
            typed_state=state_items,
            typed_procedures=procedure_items,
            max_facts=max_facts,
        )
