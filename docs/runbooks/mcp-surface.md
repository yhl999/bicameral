# MCP Surface Runbook

Operational reference for the Bicameral typed-memory MCP surface — introduced in PR #184 (merge commit `1f23912`, 2026-03-14).

---

## Overview

The MCP surface exposes Bicameral's typed-memory system to agents and users via a unified, schema-validated API. All 16+ methods are discoverable at runtime via `get_tools()`.

**Key design principles:**

- **Ledger-first for typed facts** — `remember_fact()` writes to `change_ledger.db` first (canonical), then materializes to Neo4j (derived projection). This makes typed facts supersedeable and auditable without relying on LLM extraction.
- **Lane isolation enforced at every write boundary** — facts, candidates, and lineage are strictly scoped to their origin lane. Cross-lane access is hard-failed, not silently allowed.
- **Conflict-first writes** — contradicting an existing fact produces a `ConflictDialog` and a quarantined candidate; it does not silently overwrite. Resolution is explicit via `promote_candidate`.
- **Fail closed** — schema failures, lane mismatches, and authorization failures all return structured error types; no silent failures.

---

## MCP Methods

### Typed Memory Writes

| Method | Purpose |
|--------|---------|
| `remember_fact(text, hint?)` | Assert a typed fact. Writes ledger-first, then Neo4j. Returns typed fact or ConflictDialog. |
| `promote_candidate(candidate_id, resolution)` | Resolve a quarantined conflict candidate. Resolutions: `supersede`, `cancel`, `reject`. |
| `reject_candidate(candidate_id)` | Mark a candidate as rejected (no promotion). |

### Typed Memory Reads

| Method | Purpose |
|--------|---------|
| `get_current_state(subject, predicate?)` | Point query for the current non-superseded fact for a subject/predicate. |
| `get_history(subject, predicate?)` | Walk the supersession lineage chain (current root backwards). Not a full audit log. |
| `list_candidates(lane?)` | List quarantined candidates awaiting review. Scoped to server lane when lane-bound. |
| `search_memory_facts(query, result_format?)` | Semantic search. `result_format="typed"` uses ledger-backed typed objects; `result_format="facts"` uses legacy graph edges. |
| `search_nodes(query)` | Semantic node search (for entity discovery). |
| `search_episodes(query)` | Episode retrieval stub (returns placeholder for post-MVP implementation). |
| `get_episode(episode_id)` | Episode fetch stub. |
| `search_procedures(query)` | Procedure retrieval. Uses heuristic fetch window; deep offsets may miss edge cases. |
| `get_procedure(procedure_id)` | Procedure fetch. |

### Pack Surface

| Method | Purpose |
|--------|---------|
| `list_packs(lane?)` | List available context and workflow packs. |
| `get_context_pack(pack_id, task?)` | Load and materialize a context pack (with optional task hint for materialization). |
| `get_workflow_pack(pack_id)` | Load a workflow pack. |
| `describe_pack(pack_id)` | Inspect pack schema, materialization metadata, and version. |
| `create_workflow_pack(...)` | Create a new workflow pack (schema-validated before persistence). |

### Self-Description

| Method | Purpose |
|--------|---------|
| `get_tools()` | Returns runtime schema for all methods with mode-distinguishing tooltips (typed vs legacy). |

---

## Lane Isolation Model

Lane isolation is **enforced at write boundaries, not just read boundaries**. Key invariants:

### remember_fact conflict disclosure
- Conflict detection is lane-aware. A fact in lane A cannot produce a `ConflictDialog` that exposes `existing_fact` metadata from lane B.
- If two lanes share `subject + predicate + scope`, conflicts are resolved per-lane only.

### Candidate lifecycle (promote / reject / cancel)
- Lane ownership is checked immediately after loading the candidate, before any side effects.
- Error type: `unauthorized` — messages explicitly name candidate lane, server lane, and the disallowed action.
- Cross-lane promotion, rejection, and cancellation are all hard-failed.
- Regression tests: `test_lane_isolation_regression.py::TestEndpointLaneOwnershipEnforcement`

### Supersede targeting
- A scoped-lane candidate **cannot** supersede a fact from another lane, even when `subject + predicate + scope` match.
- A scoped-lane candidate **cannot** auto-supersede an unscoped (legacy, `source_lane=None`) fact.
- Unscoped candidates can still supersede unscoped facts (backward compatibility).
- Enforced in: `ChangeLedger._validate_supersede_target` and `ChangeLedger.promote_candidate_fact` auto-resolution scan.
- Regression tests: `test_lane_isolation_regression.py::TestCrossLaneSupersede` + `TestUnscopedLegacyFactSupersede`

### list_candidates
- Scoped server returns only candidates belonging to its own lane.
- Cross-lane candidate metadata is not disclosed.
- Regression tests: `test_lane_isolation_regression.py::TestLaneBoundedCandidateReview`

---

## Error Types

All MCP methods return structured errors. Common types:

| Error type | Meaning |
|------------|---------|
| `validation_error` | Schema or input validation failure. |
| `not_found` | Requested resource does not exist. |
| `ambiguous` | Multiple matching results; caller must narrow the query. |
| `conflict` | Write collides with existing fact; ConflictDialog returned. |
| `unauthorized` | Lane ownership mismatch on promote/reject/cancel. |
| `duplicate` | Fact already exists with identical content. |

---

## Database Layout

| Store | Path | Purpose |
|-------|------|---------|
| Change ledger | `change_ledger.db` (repo root by default) | Canonical source of truth for all typed fact writes, candidates, and lineage. |
| Candidate store | `change_ledger.db` (same file, candidates table) | Quarantined conflict candidates awaiting review. |
| Neo4j | `NEO4J_URI` (env) | Derived projection for legacy facts mode compatibility + OM-native retrieval. |

---

## Running Tests

All tests are in `mcp_server/tests/`. Run from the `mcp_server/` directory.

```bash
# Security-critical paths (lane isolation, remember_fact, candidates)
uv run pytest tests/test_lane_isolation_regression.py tests/test_candidate_contract_surfaces.py tests/test_remember_fact.py -v --tb=short

# Full fast suite (all non-integration tests)
uv run pytest tests/ -v --tb=short

# Lane isolation regression only
uv run pytest tests/test_lane_isolation_regression.py -v --tb=short
```

**Note:** Some test files (`test_async_operations.py`, `test_graphiti_service_preflight.py`, `test_rate_limiter_caller_principal.py`, `test_search_lane_isolation.py`) have path-resolution issues when run from git worktrees. Run from the main checkout.

**Baseline:** 548 tests passing, 56 skipped, 0 failures (as of 2026-03-14, commit `669273f`).

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `NEO4J_URI` | Optional (defaults to bolt://localhost:7687) | Neo4j connection URI |
| `NEO4J_USER` | Optional (defaults to `neo4j`) | Neo4j username |
| `NEO4J_PASSWORD` | Required for live Neo4j | Neo4j password |
| `NEO4J_DATABASE` | Optional | Neo4j database name |

Load from `~/.clawdbot/credentials/neo4j.env` in operator environments.

---

## Known Gaps / Post-MVP Work

1. **Episode/Procedure implementation** — `search_episodes`, `get_episode` return stubs. Full implementation requires post-rescope session transcript ingestion.
2. **Response envelope consistency** — most methods use `status: ok|error`; some legacy episode/procedure error paths emit `{"error": ..., "message": ...}`. Cosmetic, not a logic bug.
3. **`get_history()` scope** — walks current-root lineages only, not a full audit-log scan. Name is slightly misleading; documented in method docstring.
4. **`search_procedures()` fetch window** — heuristic deep offsets; lane-shadowing bug is fixed but edge-case recall at very large offsets is not guaranteed.
5. **Post-promotion ledger/store skew** — `promote_candidate` reconciles post-promotion, but there is a brief window where candidate-store readback can lag. Reconciliation exists; cosmetic ops concern only.
6. **OM promotion ledger-write** — OM facts are still projected at read time via `OMTypedProjectionService`. Full ledger-backed OM promotion is tracked as a separate workstream (`task-bicameral-om-closeout-v2`). When that lands, `OMTypedProjectionService` can be deprecated.

---

## Related Docs

- `docs/MEMORY-RUNTIME-WIRING.md` — backend switch contract (qmd_primary vs graphiti_primary)
- `docs/scope-policy.md` — lane scope policy definitions
- `docs/runbooks/om-operations.md` — OM operations runbook
- `docs/public/SECURITY-BOUNDARIES.md` — public threat model and boundary contract
- `mcp_server/tests/test_lane_isolation_regression.py` — authoritative regression test suite for lane isolation
