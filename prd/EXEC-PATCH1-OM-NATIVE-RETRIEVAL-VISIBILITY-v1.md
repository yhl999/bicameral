# PRD: Patch 1 OM-native retrieval visibility rescue v1

## Objective
Rescue retrieval visibility for OM-native experimental groups without implementing the full typed projection follow-up.

## Scope
- Keep the existing OM adapter architecture.
- Expand OM adapter scoping from canonical `s1_observational_memory` to OM-native experimental groups.
- Let benchmark tooling override fixture scope with direct group IDs / lane aliases for bakeoffs.
- Preserve canonical OM behavior and mixed-lane fusion.
- Keep fail-closed semantics only for canonical `s1_observational_memory`; let explicit experimental OM groups fall back to Graphiti retrieval when the OM adapter returns no hits.

## Non-Goals
- No OMNode → Entity/Episodic conversion.
- No Patch 2/3 typed-contract redesign.
- No broad lane-policy rewrite.

## Owned Paths
- `prd/EXEC-PATCH1-OM-NATIVE-RETRIEVAL-VISIBILITY-v1.md`
- `mcp_server/src/services/om_group_scope.py`
- `mcp_server/src/services/search_service.py`
- `mcp_server/src/graphiti_mcp_server.py`
- `scripts/run_retrieval_benchmark.py`
- `mcp_server/tests/test_search_om_lane.py`
- `tests/test_retrieval_benchmark.py`
- `docs/runbooks/om-operations.md`

## Design
- Add a single OM-native group-scope helper so retrieval-side recognition is centralized.
- Treat canonical `s1_observational_memory` plus explicit experimental `_om_` groups as OM-native.
- Preserve all-lanes (`[]`) behavior by probing only the canonical OM lane unless callers explicitly target an experimental OM-native group.
- Keep canonical OM lane requests fail-closed, but allow single explicit experimental OM groups to fall back to Graphiti retrieval when OM primitive search returns zero rows.
- Add benchmark CLI scope overrides so bakeoffs can target arbitrary OM-native groups without editing fixture rows.

## DoD
- [ ] `search_nodes(... group_ids=["ontbk15batch_20260310_om_f"])` can enter the OM adapter path.
- [ ] `search_memory_facts(... group_ids=["ontbk15batch_20260310_om_f"])` can enter the OM adapter path.
- [ ] Canonical `s1_observational_memory` behavior remains unchanged.
- [ ] Explicit experimental OM groups with OM primitive hits still return through the OM adapter path.
- [ ] Explicit experimental OM groups without OM primitive hits fall back to Graphiti retrieval instead of failing closed.
- [ ] Mixed-lane fusion still works.
- [ ] Benchmark harness supports direct group / alias override for OM bakeoffs.
- [ ] Focused regression tests pass.

## Validation
```bash
cd /Users/archibald/clawd/projects/bicameral
pytest mcp_server/tests/test_search_om_lane.py tests/test_retrieval_benchmark.py tests/test_om_only_query_returns_om_evidence.py tests/test_mixed_lane_query_returns_fused_results_with_lane_provenance.py -q
```