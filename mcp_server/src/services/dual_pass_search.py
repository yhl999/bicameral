"""Dual-pass retrieval: text search + center-node-anchored search.

Proven by Phase 2b eval (2026-03-16) to improve relevant_rate by +2.4pp over
single-pass text-only search when evaluating against a personal knowledge graph
(chatgpt_history vNext benchmark, constrained mode).

How it works:
  1. Pass 1 — Normal text search (hybrid BM25+cosine, N/2 facts)
  2. Pass 2 — Center-node search from a known entity UUID (hybrid + node_distance, N/2 facts)
  3. Merge both passes, deduplicate by UUID
  4. Return up to N unique facts

The center-node pass guarantees facts from the target entity's graph neighborhood
appear in the candidate pool, even when BM25/embedding text matching fails to
surface them. This addresses graph disconnection for personal knowledge queries.

Usage (from MCP caller):
    from services.dual_pass_search import dual_pass_fact_search

    results = await dual_pass_fact_search(
        search_fn=search_memory_facts,
        query="What are Yuan's niche interests?",
        group_ids=["my_group"],
        center_node_uuid="<yuan-entity-uuid>",
        max_facts=20,
    )

Note: This module depends on the ``search_memory_facts`` callable having the
standard MCP tool signature.  It does NOT modify the search pipeline internals
or touch embeddings/reranking — it is purely a multi-call orchestration pattern.
"""

from __future__ import annotations

from typing import Any, Callable, Coroutine


async def dual_pass_fact_search(
    *,
    search_fn: Callable[..., Coroutine[Any, Any, dict[str, Any]]],
    query: str,
    group_ids: list[str] | None = None,
    center_node_uuid: str | None = None,
    search_mode: str = 'hybrid',
    max_facts: int = 20,
    text_ratio: float = 0.5,
    **extra_kwargs: Any,
) -> dict[str, Any]:
    """Execute dual-pass retrieval and merge results.

    Args:
        search_fn: Async callable with the ``search_memory_facts`` signature.
        query: The search query.
        group_ids: Group IDs to scope the search.
        center_node_uuid: Entity node UUID for the center-node pass.
            If None, falls back to single-pass text search.
        search_mode: Retrieval mode for both passes (hybrid|semantic|keyword).
        max_facts: Total maximum facts to return after merge.
        text_ratio: Fraction of max_facts allocated to text pass (default 0.5).
        **extra_kwargs: Passed through to both search_fn calls.

    Returns:
        Dict with 'facts' list (deduplicated, capped at max_facts) and
        metadata about the dual-pass execution.
    """
    text_n = max(1, int(max_facts * text_ratio))
    center_n = max(1, max_facts - text_n)

    # Pass 1: text search
    text_res = await search_fn(
        query=query,
        group_ids=group_ids,
        search_mode=search_mode,
        max_facts=text_n,
        **extra_kwargs,
    )
    text_facts = text_res.get('facts', []) if isinstance(text_res, dict) else []

    # Pass 2: center-node search (skip if no UUID provided)
    center_facts: list[dict[str, Any]] = []
    if center_node_uuid:
        try:
            center_res = await search_fn(
                query=query,
                group_ids=group_ids,
                search_mode=search_mode,
                max_facts=center_n,
                center_node_uuid=center_node_uuid,
                **extra_kwargs,
            )
            center_facts = center_res.get('facts', []) if isinstance(center_res, dict) else []
        except Exception:
            # Fail open: if center-node search errors, we still have text results
            center_facts = []

    # Merge + deduplicate
    seen_uuids: set[str] = set()
    merged: list[dict[str, Any]] = []

    for fact in text_facts:
        uuid = fact.get('uuid', '') if isinstance(fact, dict) else ''
        if uuid and uuid not in seen_uuids:
            seen_uuids.add(uuid)
            merged.append(fact)

    center_new = 0
    for fact in center_facts:
        uuid = fact.get('uuid', '') if isinstance(fact, dict) else ''
        if uuid and uuid not in seen_uuids:
            seen_uuids.add(uuid)
            merged.append(fact)
            center_new += 1

    return {
        'facts': merged[:max_facts],
        'message': 'Facts retrieved via dual-pass search',
        '_dual_pass_meta': {
            'text_count': len(text_facts),
            'center_node_new': center_new,
            'total_merged': len(merged),
            'center_node_uuid': center_node_uuid,
        },
    }
