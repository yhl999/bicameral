"""
HybridRetrievalService — production-surface hybrid retrieval.

Provides typed state/procedure candidate extraction and Reciprocal Rank
Fusion (RRF) merge for the hybrid default surface of search_memory_facts.

Design goals:
- No benchmark-script imports; all fusion logic is self-contained.
- Deterministic and testable: same inputs → same output.
- Narrow scope: state + procedure candidates only (not episodes).

Contract notes:
- object_types filtering is not supported in the hybrid typed-candidate subpath.
  Hybrid always fetches state + procedure regardless of any caller-supplied
  object_types. Callers who need object_types filtering should use
  retrieval_mode='typed' directly.
- center_node_uuid only affects the graph-recall subpath of hybrid retrieval.
  It is not forwarded to the typed-candidate subpath (typed retrieval has no
  concept of a graph-center node). This is intentional.
- Cross-source deduplication is not performed across graph and typed results.
  Graph facts and typed objects are distinct entity types (graph edges vs.
  ledger objects) and are not deduplicated against each other by design. The
  RRF merge produces a unified ranked list, but does not collapse duplicates
  across source types.
"""
from __future__ import annotations

import hashlib
import json
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
    """Stable deduplication key for a typed ledger object.

    When ``object_id`` is present it is used directly (fast path).
    When ``object_id`` is absent or empty a SHA-1 content hash is used as
    a fallback to avoid the collision that would occur if multiple ID-less
    items were all mapped to the same ``"typed:<bucket>:unknown"`` key.
    """
    oid = str(item.get("object_id", "") or "").strip()
    if oid:
        return f"typed:{bucket}:{oid}"
    # Fallback: derive a collision-resistant key from item content so that
    # distinct ID-less items do not overwrite each other in the RRF registry.
    try:
        content = json.dumps(item, sort_keys=True, default=str)
    except Exception:
        content = str(item)
    h = hashlib.sha1(content.encode(), usedforsecurity=False).hexdigest()[:16]  # noqa: S324
    return f"typed:{bucket}:anon:{h}"


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
        max_evidence: int = 20,
    ) -> dict[str, Any]:
        """Run typed retrieval (state + procedures only) for the hybrid candidate pool.

        Episodes are excluded from the candidate pool; they belong in the
        episodic/graph recall path rather than the fact-surface hybrid merge.

        Note: ``object_types`` is intentionally not exposed here. Hybrid always
        fetches ``["state", "procedure"]`` regardless of any caller-supplied
        ``object_types``. Callers who need explicit object-type filtering should
        use ``retrieval_mode='typed'`` directly.

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
            max_evidence: Maximum evidence items per typed object. Forwarded
                directly from the caller's ``max_evidence`` parameter.

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
            max_evidence=max_evidence,
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


# ── Provenance materialization ────────────────────────────────────────────────


def build_provenance(
    merged_results: list[dict[str, Any]],
    typed_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build provenance section for the hybrid response.

    Produces:
    - ``refs``: One provenance ref per merged result, containing source type,
      source id, object type, and evidence keys for typed items.
    - ``resolved_evidence``: A flat dict keyed by stable evidence keys,
      containing resolved evidence detail for typed-backed items.

    This enables callers to dereference typed-backed results without
    spelunking raw ``_original`` internals. Graph-backed items get
    simple refs with no evidence resolution (graph evidence lives
    in the graph edge itself).

    Args:
        merged_results: The RRF-merged (or reranked) candidate list.
        typed_results: Optional typed retrieval result dict (for evidence data).

    Returns:
        Dict with ``refs`` and ``resolved_evidence`` keys.
    """
    refs: list[dict[str, Any]] = []
    resolved_evidence: dict[str, dict[str, Any]] = {}

    # Build evidence index from typed results
    evidence_index: dict[str, list[dict[str, Any]]] = {}
    if typed_results:
        raw_evidence = typed_results.get("evidence", []) or []
        for ev in raw_evidence:
            parent_id = str(ev.get("parent_object_id", "") or "")
            if parent_id:
                evidence_index.setdefault(parent_id, []).append(ev)

    for item in merged_results:
        source = str(item.get("_source", "unknown"))
        ref: dict[str, Any] = {"source": source}

        if source == "graph":
            ref["source_id"] = str(item.get("uuid", "") or "")
            ref["object_type"] = "entity_edge"
            ref["lane"] = str(item.get("group_id", "") or "")
            ref["evidence_keys"] = []
        else:
            # Typed item — extract from _original if available
            original = item.get("_original", {}) or {}
            object_id = str(
                original.get("object_id", "")
                or item.get("uuid", "")
                or ""
            )
            ref["source_id"] = object_id
            ref["object_type"] = str(
                original.get("object_type", "")
                or item.get("_object_type", "")
                or source
            )
            ref["lane"] = str(
                original.get("source_lane", "")
                or original.get("group_id", "")
                or ""
            )

            # Resolve evidence for this typed item
            evidence_keys: list[str] = []
            if object_id:
                # Check for evidence in the typed results evidence index
                item_evidence = evidence_index.get(object_id, [])
                # Also check for inline evidence_refs in the original
                inline_refs = original.get("evidence_refs", []) or []

                for ev_idx, ev in enumerate(item_evidence):
                    ev_key = f"{object_id}:ev:{ev_idx}"
                    evidence_keys.append(ev_key)
                    resolved_evidence[ev_key] = _resolve_evidence_entry(ev)

                for er_idx, er in enumerate(inline_refs):
                    er_key = f"{object_id}:ref:{er_idx}"
                    if er_key not in resolved_evidence:
                        evidence_keys.append(er_key)
                        resolved_evidence[er_key] = _resolve_evidence_ref(er)

            ref["evidence_keys"] = evidence_keys

        refs.append(ref)

    return {
        "refs": refs,
        "resolved_evidence": resolved_evidence,
    }


def _resolve_evidence_entry(ev: dict[str, Any]) -> dict[str, Any]:
    """Resolve a typed evidence entry into a flat provenance detail dict."""
    return {
        "kind": str(ev.get("kind", "") or ev.get("type", "") or "evidence"),
        "source_system": str(ev.get("source_system", "") or ""),
        "canonical_uri": str(ev.get("canonical_uri", "") or ""),
        "snippet": str(ev.get("snippet", "") or ev.get("content", "") or "")[:500],
        "observed_at": str(ev.get("observed_at", "") or ""),
        "status": "resolved",
    }


def _resolve_evidence_ref(er: Any) -> dict[str, Any]:
    """Resolve an inline evidence_ref into a flat provenance detail dict."""
    if isinstance(er, dict):
        return {
            "kind": str(er.get("kind", "") or "ref"),
            "source_system": str(er.get("source_system", "") or ""),
            "canonical_uri": str(er.get("canonical_uri", "") or ""),
            "snippet": str(er.get("snippet", "") or er.get("title", "") or "")[:500],
            "observed_at": str(er.get("observed_at", "") or ""),
            "status": "resolved",
        }
    # Scalar ref (string URI, etc.)
    return {
        "kind": "ref",
        "canonical_uri": str(er)[:500],
        "status": "resolved",
    }
