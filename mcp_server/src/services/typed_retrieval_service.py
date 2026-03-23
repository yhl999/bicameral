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
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

try:
    from .typed_retrieval import TypedRetrievalService
except ImportError:  # pragma: no cover — top-level import fallback
    from services.typed_retrieval import TypedRetrievalService

logger = logging.getLogger(__name__)

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


# ── Phase 2A: Query-Intent Lane Filter ────────────────────────────────────────

# Suppression matrix: intent → list of lane labels to suppress (zero-score).
SUPPRESSION_MATRIX: dict[str, list[str]] = {
    "persona": ["engineering", "technical"],
    "preference": ["engineering", "incident", "operational"],
    "operational": [],
    "technical": ["persona", "preference"],
    "decision": ["persona", "preference"],
    "engineering": [],
    "incident": [],
    "generic": [],
}

# Intent classification patterns: list of (compiled_regex, intent_label).
# Order matters — first match wins.
_INTENT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # decision — check early; "alternative … considered" can have words between
    (re.compile(
        r"\b(?:why\s+did\s+(?:we|you|i)|trade[\s-]?off|"
        r"alternative(?:s)?(?:\s+\w+)*\s+considered|"
        r"decision|decided|chose|rationale|reasoning\s+behind|"
        r"pros?\s+(?:and|&)\s+cons?|compare|comparison|"
        r"evaluation|evaluated|assessed|picked|selected\s+(?:over|instead))",
        re.IGNORECASE,
    ), "decision"),
    # operational — before persona to avoid "schedule" false positives;
    #   "pipeline" is qualified to avoid stealing technical queries.
    (re.compile(
        r"\b(?:interrupt\s+vs|batch\s+update|operational\s+(?:rule|procedure|mode)|"
        r"communication\s+guard|guard\s+server|cron\s+(?:job|schedule|audit)|"
        r"heartbeat|workflow|(?:deploy(?:ment)?|ci|cd|build)\s+pipeline|"
        r"deploy(?:ment)?|rollback|runbook|"
        r"procedure|playbook|escalat(?:e|ion)|on[\s-]?call|"
        r"incident\s+(?:response|management|review)|post[\s-]?mortem)",
        re.IGNORECASE,
    ), "operational"),
    # technical / engineering — before persona/preference to catch specific terms
    (re.compile(
        r"\b(?:how\s+does\s+(?:it|the|this)\s+work|architecture|spec(?:s|ification)?|"
        r"implement(?:ation|ed)?|code|function|class|method|module|"
        r"api|endpoint|schema|database|query|index|"
        r"bug|fix(?:ed)?|error|exception|stack\s+trace|debug|"
        r"performance|latency|throughput|benchmark|pipeline|"
        r"config(?:uration)?|infrastructure|service|server|"
        r"graph(?:iti)?|neo4j|falkordb|rrf|rerank|retrieval|"
        r"vector|embedding|semantic|hybrid|fusion|"
        r"typed\s+retrieval|change\s+ledger|om\s+projection|"
        r"feature\s+flag|migration|refactor)",
        re.IGNORECASE,
    ), "technical"),
    # persona — "schedule" qualified to avoid stealing operational/technical
    (re.compile(
        r"\b(?:favorite|favourite|prefer(?:s|red|ence)?|personality|character|"
        r"style|habit|routine|who\s+(?:is|am)|about\s+(?:yuan|me|him|her|them)|"
        r"schedul(?:e|ing)\s+(?:preference|default|block|habit)|"
        r"scheduling|calendar|availability|morning|workout|"
        r"background|family|hometown|grew\s+up|boarding\s+school|"
        r"communication\s+(?:style|rules|preference)|"
        r"decision\s+style|working\s+hours|meeting\s+default|buffer\s+time)",
        re.IGNORECASE,
    ), "persona"),
    # preference
    (re.compile(
        r"\b(?:opinion\s+on|think\s+about|taste\s+in|like(?:s)?\s+(?:to|about)?|"
        r"dislike|enjoy|love(?:s)?|hate(?:s)?|"
        r"recommend(?:ation)?|suggestion|what\s+(?:do\s+(?:you|i)\s+think|should\s+i)|"
        r"cuisine|restaurant|wine|food|favorite\s+(?:food|movie|book|song|color|music)|"
        r"best\s+(?:restaurant|bar|place)|ranking|rated)",
        re.IGNORECASE,
    ), "preference"),
]


class QueryIntentClassifier:
    """Rule-based query-intent classifier with TTL cache.

    Classifies a query string into one of the intent labels used by the
    suppression matrix.  Results are cached for 1 hour keyed on SHA-256
    of the query to avoid redundant regex passes on repeated queries.
    """

    _CACHE_TTL_SECONDS: int = 3600  # 1 hour

    def __init__(self) -> None:
        self._cache: dict[str, tuple[str, float]] = {}

    def classify(self, query: str) -> str:
        """Classify *query* into an intent label.

        Returns one of: persona, preference, decision, operational,
        technical, engineering, incident, generic.
        """
        key = hashlib.sha256(query.encode()).hexdigest()
        now = time.monotonic()

        # Check cache
        if key in self._cache:
            intent, ts = self._cache[key]
            if now - ts < self._CACHE_TTL_SECONDS:
                return intent
            # Expired — fall through to re-classify.

        intent = self._classify_uncached(query)
        self._cache[key] = (intent, now)
        return intent

    @staticmethod
    def _classify_uncached(query: str) -> str:
        """Run pattern matching against *query*; return intent label."""
        for pattern, label in _INTENT_PATTERNS:
            if pattern.search(query):
                return label
        return "generic"

    def clear_cache(self) -> None:
        """Clear the classification cache (useful for testing)."""
        self._cache.clear()


# Module-level singleton for lightweight reuse across calls.
_query_intent_classifier = QueryIntentClassifier()


def _lane_label(candidate: dict[str, Any]) -> str:
    """Derive a lane label from candidate metadata.

    Inspects ``_source``, entity type, tags, bucket, and object type to
    return one of: persona, preference, operational, technical, decision,
    engineering, incident, generic.
    """
    source = str(candidate.get("_source", "") or "").lower()

    # ── Typed candidates (from ChangeLedger) ──────────────────────────────
    if source in ("typed_state", "typed_procedure"):
        original = candidate.get("_original", {}) or {}
        obj_type = str(original.get("object_type", "") or "").lower()
        bucket = str(original.get("bucket", "") or "").lower()
        subject = str(original.get("subject", "") or "").lower()
        tags = [str(t).lower() for t in (original.get("tags", []) or [])]

        # Explicit bucket match
        if bucket in ("persona",):
            return "persona"
        if bucket in ("preference", "preferences"):
            return "preference"
        if bucket in ("operational",):
            return "operational"
        if bucket in ("decision", "decisions"):
            return "decision"

        # Object-type / tag heuristics
        if obj_type in ("persona", "identity"):
            return "persona"
        if obj_type in ("preference",):
            return "preference"
        if obj_type in ("decision", "decision_framework"):
            return "decision"
        if obj_type in ("procedure",) or "decision_framework" in tags:
            return "decision"
        if any(t in tags for t in ("engineering", "ops", "architecture", "technical")):
            return "engineering"

        # Subject-based heuristics
        if any(kw in subject for kw in ("favorite", "preference", "taste", "opinion")):
            return "preference"
        if any(kw in subject for kw in ("schedule", "calendar", "routine", "habit")):
            return "persona"

        return "generic"

    # ── Graph candidates ──────────────────────────────────────────────────
    if source == "graph":
        entity_type = str(candidate.get("entity_type", "") or "").lower()
        name = str(candidate.get("name", "") or "").lower()
        fact = str(candidate.get("fact", "") or "").lower()
        tags = [str(t).lower() for t in (candidate.get("tags", []) or [])]
        group_id = str(candidate.get("group_id", "") or "").lower()

        # Incident detection
        if entity_type == "incident" or "incident" in tags:
            return "incident"

        # Engineering / technical signals
        engineering_keywords = (
            "feature flag", "migration", "refactor", "bug", "fix",
            "scanner", "runtime", "version", "config", "deploy",
            "pipeline", "ci/cd", "test", "spec",
        )
        if any(kw in name or kw in fact for kw in engineering_keywords):
            return "engineering"
        if any(t in ("engineering", "ops", "architecture", "technical") for t in tags):
            return "engineering"

        # Persona / preference signals from graph
        persona_keywords = (
            "favorite", "preference", "schedule", "routine",
            "personality", "background", "family",
        )
        if any(kw in name or kw in fact for kw in persona_keywords):
            return "persona"

        # Decision signals
        if "decision" in entity_type or "decision" in name or "trade-off" in fact:
            return "decision"

        return "generic"

    # Fallback
    return "generic"


# ── Phase 2B: Candidate Diversity Helpers ─────────────────────────────────────

# Env-var overrides for tuning knobs
def _parse_float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        logger.warning("Invalid value for %s; using default %.2f", name, default)
        return default

_DEDUP_THRESHOLD: float = _parse_float_env("BICAMERAL_DEDUP_THRESHOLD", 0.85)
_MMR_WEIGHT_DIVERSITY: float = _parse_float_env("BICAMERAL_MMR_WEIGHT_DIVERSITY", 0.3)


def _candidate_fact_text(candidate: dict[str, Any]) -> str:
    """Extract canonical text from a candidate for similarity comparison."""
    parts: list[str] = []

    # Primary: fact text
    fact = str(candidate.get("fact", "") or "")
    if fact:
        parts.append(fact)

    # Secondary: name
    name = str(candidate.get("name", "") or "")
    if name and name != fact:
        parts.append(name)

    # For typed items, pull subject/predicate/value from _original
    original = candidate.get("_original", {}) or {}
    for key in ("subject", "predicate", "value"):
        v = str(original.get(key, "") or "")
        if v and v not in parts:
            parts.append(v)

    return " ".join(parts).strip() or str(candidate.get("uuid", ""))


def _tokenize(text: str) -> set[str]:
    """Simple whitespace + lowercased tokenizer for Jaccard similarity."""
    return set(re.findall(r"\w+", text.lower()))


def _text_similarity(a: str, b: str) -> float:
    """Jaccard token overlap similarity (0.0–1.0)."""
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0


def _dedup_candidates(
    candidates: list[dict[str, Any]],
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Remove near-duplicate candidates.

    Keeps the highest-scoring variant (assumes *candidates* is sorted by
    descending ``_hybrid_score`` already).

    Args:
        candidates: Score-sorted candidate list.
        threshold: Jaccard similarity threshold (0.0–1.0).  Defaults to
            ``_DEDUP_THRESHOLD`` (env-overridable).

    Returns:
        De-duplicated list preserving original order.
    """
    if threshold is None:
        threshold = _DEDUP_THRESHOLD

    kept: list[dict[str, Any]] = []
    for candidate in candidates:
        fact_text = _candidate_fact_text(candidate)
        is_dup = False
        for kept_cand in kept:
            kept_text = _candidate_fact_text(kept_cand)
            if _text_similarity(fact_text, kept_text) >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(candidate)
    return kept


def _apply_mmr_diversity(
    candidates: list[dict[str, Any]],
    mmr_weight: float | None = None,
    max_items: int = 10,
) -> list[dict[str, Any]]:
    """Re-rank candidates using Maximal Marginal Relevance.

    Balances relevance (``_hybrid_score``) with diversity (Jaccard distance
    from already-selected candidates).

    Args:
        candidates: Input candidate list (post-dedup).
        mmr_weight: Diversity weight λ (0 = pure relevance, 1 = pure
            diversity).  Defaults to ``_MMR_WEIGHT_DIVERSITY``.
        max_items: Maximum items to select.

    Returns:
        MMR-reranked list, length ≤ min(len(candidates), max_items).
    """
    if mmr_weight is None:
        mmr_weight = _MMR_WEIGHT_DIVERSITY

    if not candidates:
        return []

    selected: list[dict[str, Any]] = []
    remaining = list(candidates)

    while remaining and len(selected) < max_items:
        best_idx = 0
        best_mmr = float("-inf")

        for i, cand in enumerate(remaining):
            relevance = cand.get("_hybrid_score", 0.0)

            # Diversity penalty: max similarity to any already-selected
            if selected:
                cand_text = _candidate_fact_text(cand)
                max_sim = max(
                    _text_similarity(cand_text, _candidate_fact_text(s))
                    for s in selected
                )
            else:
                max_sim = 0.0

            mmr_score = (1.0 - mmr_weight) * relevance - mmr_weight * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


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
    query_intent: str | None = None,
    apply_diversity: bool = True,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion over graph facts + typed state/procedure candidates.

    Each source is ranked independently and weighted. Items that appear in
    multiple sources accumulate their RRF scores. The merged list is returned
    in score-descending order, capped at ``max_facts``.

    Phase 2A: When *query_intent* is provided, candidates whose lane label
    is in the suppression matrix for that intent are zero-scored before
    final ranking.

    Phase 2B: After scoring and suppression, near-duplicate candidates are
    removed and MMR diversity re-ranking is applied (when *apply_diversity*
    is True).

    Args:
        graph_facts: Ranked list of graph-edge facts (index 0 = best).
        typed_state: Ranked list of typed state-fact objects.
        typed_procedures: Ranked list of typed procedure objects.
        max_facts: Maximum items to return.
        rrf_k: RRF smoothing constant.
        weight_graph: Multiplicative weight for graph-edge source.
        weight_typed: Multiplicative weight for typed-ledger source.
        query_intent: Optional intent label from QueryIntentClassifier.
            When provided, candidates in suppressed lanes are zero-scored.
        apply_diversity: When True (default), apply dedup + MMR diversity
            after RRF merge.  Set to False when an LLM reranker will run
            downstream.

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

    # ── Phase 2A: Lane suppression ────────────────────────────────────────
    if query_intent:
        suppressed_lanes = SUPPRESSION_MATRIX.get(query_intent, [])
        if suppressed_lanes:
            for key in list(scores.keys()):
                candidate = registry.get(key)
                if candidate is not None:
                    lane = _lane_label(candidate)
                    if lane in suppressed_lanes:
                        scores[key] = 0.0

    # Annotate with final RRF scores.
    for key, score in scores.items():
        if key in registry:
            registry[key]["_hybrid_score"] = round(score, 6)

    ordered_keys = sorted(scores, key=lambda k: (-scores[k], k))
    merged = [registry[k] for k in ordered_keys]

    # ── Phase 2B: Dedup + MMR diversity ───────────────────────────────────
    if apply_diversity and merged:
        try:
            merged = _dedup_candidates(merged)
        except Exception:
            logger.warning("Candidate dedup failed; returning undeduped pool", exc_info=True)
        try:
            merged = _apply_mmr_diversity(merged, max_items=max_facts)
        except Exception:
            logger.warning("MMR diversity failed; returning pool without MMR", exc_info=True)

    return merged[:max_facts]


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
        query: str | None = None,
        apply_diversity: bool = True,
    ) -> list[dict[str, Any]]:
        """Merge graph-recall facts with typed state/procedure candidates via RRF.

        Args:
            graph_facts: Ranked graph-edge facts from the graph recall path.
            typed_results: Result dict from :meth:`get_typed_candidates`.
            max_facts: Maximum items in the returned merged list.
            query: Original query string.  When provided, the query-intent
                classifier runs and passes the intent to the RRF merge for
                lane suppression.
            apply_diversity: When True (default), dedup + MMR diversity is
                applied.  Set to False when an LLM reranker runs downstream.

        Returns:
            RRF-merged list of hybrid entries, score-descending.
        """
        state_items: list[dict[str, Any]] = typed_results.get("state", []) or []
        procedure_items: list[dict[str, Any]] = typed_results.get("procedures", []) or []

        # Phase 2A: classify query intent for lane suppression
        intent: str | None = None
        if query:
            intent = _query_intent_classifier.classify(query)

        return rrf_merge_hybrid(
            graph_facts=graph_facts,
            typed_state=state_items,
            typed_procedures=procedure_items,
            max_facts=max_facts,
            query_intent=intent,
            apply_diversity=apply_diversity,
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
