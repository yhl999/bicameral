# Graphiti Core Patch Classification (2026-02-22)

Goal: keep upstream syncability while preserving only intentional local behavior.

## Approved keep-list (C: keep as explicit patch stack)
These are intentional local behavior changes and may remain as explicit patch commits:

1. `graphiti_core/search/search.py`
2. `graphiti_core/search/search_config.py`
3. `graphiti_core/search/search_config_recipes.py`
4. `graphiti_core/search/search_utils.py`
5. `graphiti_core/edges.py`
6. `graphiti_core/graphiti.py`
7. `graphiti_core/utils/maintenance/edge_operations.py`

Rationale:
- Trust-aware retrieval scoring hooks (migration/runtime behavior)
- defensive malformed edge handling
- migration-only deterministic dedupe mode controls

## B: externalize over time (target)
Behavior currently in keep-list that should migrate to runtime layer over time:
- Trust boost/ranking overlays → runtime retrieval adapter (`mcp_server`/runtime layer)
- Migration-only dedupe switches → migration scripts / maintenance wrappers

## A: drop from local core drift during sync conflict resolution
Any `graphiti_core/**` change not in keep-list above defaults to upstream.

## Conflict policy
- During upstream sync merge/rebase:
  - `graphiti_core/**`: upstream-first by default
  - reapply only keep-list patches
- Everything else: resolve case-by-case with preference for preserving local runtime/docs infra.

## Temporary freeze rule
Until sync PR merges: no new direct `graphiti_core/**` edits on `main` except allowlisted keep-list patches.
