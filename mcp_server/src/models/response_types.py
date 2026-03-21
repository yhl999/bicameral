"""Response type definitions for Graphiti MCP Server."""

from typing import Any

from typing_extensions import NotRequired, TypedDict


class ErrorResponse(TypedDict):
    error: str
    message: NotRequired[str]
    details: NotRequired[Any]


class SuccessResponse(TypedDict):
    message: str
    details: NotRequired[Any]


class NodeResult(TypedDict):
    uuid: str
    name: str
    labels: list[str]
    created_at: str | None
    summary: str | None
    group_id: str
    attributes: dict[str, Any]


class NodeSearchResponse(TypedDict):
    message: str
    nodes: list[NodeResult]


class FactSearchResponse(TypedDict):
    message: str
    facts: list[dict[str, Any]]


class HybridTypedCandidates(TypedDict):
    """Typed state/procedure candidates surfaced by the hybrid retrieval path."""
    state: list[dict[str, Any]]
    procedures: list[dict[str, Any]]
    counts: dict[str, int]


class HybridDiagnostics(TypedDict, total=False):
    """Degradation/diagnostic signals for the hybrid retrieval path.

    Present only when the retrieval path degraded in some way; absent on clean runs.
    """
    typed_retrieval_failed: bool
    fallback: str  # e.g. 'graph_only'
    error: str     # str(exception) from the failed typed retrieval call


class HybridRerankMetadata(TypedDict, total=False):
    """Metadata about the LLM reranking pass applied to the hybrid pool.

    Present on all hybrid responses. ``method`` indicates the rerank path:
    - ``'llm'``: LLM reranking was applied successfully.
    - ``'passthrough'``: LLM reranking was skipped (disabled or no API key).
    - ``'fallback'``: LLM reranking failed; RRF order was preserved.
    """
    method: str           # 'llm' | 'passthrough' | 'fallback'
    model: str            # model id used (only when method='llm')
    total_scored: int     # number of candidates scored
    diagnostics: dict[str, Any]  # only when degraded


class HybridProvenanceRef(TypedDict, total=False):
    """Per-result provenance reference for hybrid retrieval results.

    Enables callers to trace each result back to its source without
    spelunking raw internals.
    """
    source: str           # 'graph' | 'typed_state' | 'typed_procedure'
    source_id: str        # uuid (graph) or object_id (typed)
    object_type: str      # e.g. 'state_fact', 'procedure', 'entity_edge'
    lane: str             # source lane / group_id
    evidence_keys: list[str]  # keys into resolved_evidence (typed items only)


class HybridProvenance(TypedDict, total=False):
    """Top-level provenance section for the hybrid retrieval response.

    Contains per-result references and resolved evidence for typed-backed items.
    """
    refs: list[HybridProvenanceRef]                 # one per merged_result item
    resolved_evidence: dict[str, dict[str, Any]]    # key → evidence detail


class HybridResponse(TypedDict, total=False):
    """Typed envelope for retrieval_mode='hybrid' responses from search_memory_facts.

    All fields except ``message``, ``retrieval_mode``, ``facts``,
    ``typed_candidates``, ``merged_results``, and ``result_count`` are
    conditionally present (``total=False``).
    """
    message: str
    retrieval_mode: str          # always 'hybrid'
    facts: list[dict[str, Any]]  # raw graph-recall facts (pre-merge input)
    typed_candidates: HybridTypedCandidates
    merged_results: list[dict[str, Any]]
    result_count: int
    candidate_rows: list[dict[str, Any]]   # only when OM lane in scope
    diagnostics: HybridDiagnostics         # only when typed retrieval degraded
    rerank: HybridRerankMetadata           # LLM reranking metadata (always present)
    provenance: HybridProvenance           # per-result provenance refs


class EpisodeSearchResponse(TypedDict, total=False):
    """Integrated episodes/procedures router search response.

    The integrated surface (routers/episodes_procedures.py) always returns
    'episodes', 'limit', 'offset', 'total', 'has_more', 'next_offset'.
    It does NOT return 'message'.

    The legacy graphiti surface (graphiti_mcp_server.py) returns 'message'
    and 'episodes' only (no pagination fields).  Both surfaces share this
    type name for contract-string continuity.
    """
    episodes: list[dict[str, Any]]
    limit: int
    offset: int
    total: int
    has_more: bool
    next_offset: int | None


class ProcedureSearchResponse(TypedDict, total=False):
    """Integrated episodes/procedures router search response.

    The integrated surface (routers/episodes_procedures.py) always returns
    'procedures', 'limit', 'offset', 'total', 'has_more', 'next_offset'.
    It does NOT return 'message'.
    """
    procedures: list[dict[str, Any]]
    limit: int
    offset: int
    total: int
    has_more: bool
    next_offset: int | None


class TypedMemoryQueryMetadata(TypedDict):
    subject: str
    predicate: str | None
    scope: str | None
    group_ids: list[str]
    lane_alias: list[str] | None
    limit: int
    result_count: int
    truncated: bool


class CurrentStateResponse(TypedDict):
    """Runtime envelope for get_current_state: {'status': 'ok', 'facts': [...]}."""
    status: str
    facts: list[dict[str, Any]]
    metadata: NotRequired[TypedMemoryQueryMetadata]


class HistoryResponse(TypedDict):
    """Runtime envelope for get_history: {'status': 'ok', 'history': [...], 'scope': ..., 'roots_considered': [...]}."""
    status: str
    history: list[dict[str, Any]]
    scope: str
    roots_considered: list[str]
    metadata: NotRequired[TypedMemoryQueryMetadata]


class StatusResponse(TypedDict):
    status: str
    message: str


class PackMetadata(TypedDict):
    """Pack metadata entry — shape returned by _pack_metadata() / list_packs / create_workflow_pack."""
    id: NotRequired[str | None]
    scope: NotRequired[str | None]
    intent: NotRequired[str | None]
    description: NotRequired[str | None]
    consumer: NotRequired[str | None]
    version: NotRequired[str | None]
    predicates: NotRequired[list]
    created_at: NotRequired[str | None]
    last_updated: NotRequired[str | None]


class PackMaterialized(TypedDict):
    """Runtime envelope for get_context_pack."""
    pack_id: str
    pack_metadata: PackMetadata
    facts: list[dict[str, Any]]
    task_context: NotRequired[str | None]
    materialized_at: str
    fact_count: int


class WorkflowPackMaterialized(TypedDict):
    """Runtime envelope for get_workflow_pack."""
    pack_id: str
    pack_metadata: PackMetadata
    facts: list[dict[str, Any]]
    task_context: NotRequired[str | None]
    definition: NotRequired[dict[str, Any]]
    materialized_at: str
    fact_count: int
