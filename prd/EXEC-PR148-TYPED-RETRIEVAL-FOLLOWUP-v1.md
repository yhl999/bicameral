# PRD: PR #148 typed retrieval follow-up fix pass v1

## Objective
Clear the remaining P1/P2 review findings on PR #148 without changing the public retrieval surface.

## Scope
- Keep `search_memory_facts` as the single public retrieval API
- Preserve facts-mode backward compatibility
- Fix:
  1. non-matching / weak-token typed queries returning unrelated recent objects
  2. candidate-root selection scanning/group-sorting `change_events` on each typed query
  3. lineage-cap invisibility for long-lived mutated roots
  4. unbounded QMD subprocess stdout buffering before cap enforcement

## Owned Paths
- `prd/EXEC-PR148-TYPED-RETRIEVAL-FOLLOWUP-v1.md`
- `mcp_server/src/services/change_ledger.py`
- `mcp_server/src/services/typed_retrieval.py`
- `mcp_server/src/services/evidence_callback.py`
- `tests/test_typed_retrieval_service.py`
- `tests/test_evidence_callback.py`

## Design
- Add a root-level retrieval index/snapshot so typed candidate-root selection can query one row per root instead of grouping over `change_events` each time.
- Use that same root snapshot to surface the current object when a root exceeds the lineage materialization cap, instead of silently dropping the root.
- For non-empty queries, require lexical evidence before applying recency/current/version boosts; do not fall back to recent roots when tokenized queries have no root matches.
- Read QMD subprocess stdout incrementally with an explicit byte cap and avoid buffering stderr in memory.

## DoD
- [ ] Non-empty weak/non-matching typed queries fail closed instead of returning unrelated recent objects
- [ ] Candidate-root selection uses root-level index/snapshot rather than per-query `change_events` group-by scan
- [ ] Roots over the lineage event cap still surface their current object
- [ ] QMD stdout cap is enforced during read, not after unbounded buffering
- [ ] Focused regression tests cover each fix path
- [ ] Targeted tests pass
