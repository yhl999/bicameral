# Local Hotfix Registry (Graphiti Core)

This repository (`bicameral`) maintains intentional local behavioral deviations from upstream `graphiti_core` inside the `patches/` directory.

Because the velocity of upstream `getzep/graphiti` is high, we track explicit patch files rather than relying solely on git rebase/merge history to survive conflict resolution.

## Active hotfixes (Graphiti Core)

### 1) Deterministic Migration Dedupe Mode
- Purpose: Prevent semantic duplicate resolution instability (`invalid duplicate_facts idx`) during bulk backfill by disabling semantic (LLM-based) re-evaluation for exact node matches.
- Files: `graphiti_core/graphiti.py`
- Rationale: Mandatory for Gate 3 reliable curation loading; migration-only parameter thread.

### 2) Malformed `RELATES_TO` Edge Guarding
- Purpose: Prevent `ValidationError` hydration failures downstream when legacy edges miss `uuid`, `group_id`, or `episodes`.
- Files: `graphiti_core/edges.py`, `graphiti_core/utils/maintenance/edge_operations.py`
- Rationale: Schema divergence existed dynamically in legacy data; defensive fallback required inside the core hydration paths.

### 3) Trust-Aware Retrieval & Ranking Additions
- Purpose: Allow overlay scores/thresholds to boost canonical facts (`ingest_curated_facts` vs LLM extractions).
- Files:
  - `graphiti_core/search/search.py`
  - `graphiti_core/search/search_config.py`
  - `graphiti_core/search/search_config_recipes.py`
  - `graphiti_core/search/search_utils.py`
- Rationale: Essential behavior layer for `bicameral` trust topology; cannot be pushed to runtime solely via hooks due to hardcoded score aggregation in `graphiti_core/search`. Will be re-evaluated as upstream search APIs mature.

## How to Sync Upstream

To safely absorb upstream updates while keeping these hotfixes:

1. **Start Sync Branch:** Create a branch `sync/upstream-YYYYMMDD` from `origin/main`.
2. **Merge Upstream:** `git pull upstream main`
3. **Resolve Conflicts (Upstream Wins in core):** If conflicts arise in `graphiti_core/**`, accept upstream's version. You do not manually rebuild the logic during conflict resolution.
4. **Re-Apply Patches:** Run `git apply patches/graphiti_core/*` to neatly apply our explicit hotfixes over the fresh upstream baseline.
5. **Re-Export Patches:** If upstream structural changes occurred (e.g. they moved code blocks around), run `./scripts/export-core-patches.sh` post-validation to update the line numbers in the stored patch files for the next release.
6. **Guard Check:** The CI run will verify no other files in `graphiti_core/**` were modified besides what is listed in `config/graphiti_core_allowlist.txt`.
