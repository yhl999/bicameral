# PRD: Patch 2 + 3 OM shared typed retrieval projection v1

## Objective
Project OM-native retrieval into the shared typed retrieval contract so OM-visible groups can surface through the same `state` / `episodes` / `procedures` / `evidence` interface used by ledger-backed typed memory, then add benchmark/test parity so this path can be exercised directly.

## Scope
- Keep OM native at the substrate and write path.
- Add a retrieval-layer OM projection that feeds the shared typed contract instead of inventing fake Graphiti storage shapes.
- Preserve the existing ledger-backed typed retrieval path for non-OM objects.
- Extend benchmark/test tooling so OM-native lanes can be validated through `result_format="typed"` rather than only through ad hoc facts/nodes probes.

## Non-Goals
- No OMNode / OM edges rewritten into Graphiti Entity / Episodic storage.
- No changes to Patch 1 branch intent beyond stacking on it if needed.
- No substrate rewrite away from OM-native storage.
- No new parallel user-facing retrieval surface.

Owned Paths:
- `prd/EXEC-PATCH2-3-OM-SHARED-TYPED-RETRIEVAL-v1.md`
- `mcp_server/src/services/typed_retrieval.py`
- `mcp_server/src/services/om_typed_projection.py`
- `mcp_server/src/graphiti_mcp_server.py`
- `scripts/run_retrieval_benchmark.py`
- `scripts/typed_retrieval_smoke.py`
- `mcp_server/tests/test_search_om_lane.py`
- `tests/test_typed_retrieval_service.py`
- `tests/test_search_memory_facts_typed_mode.py`
- `tests/test_retrieval_benchmark.py`

## Design
- Introduce a small OM typed-projection adapter that reuses the existing OM retrieval adapter and projects OM-native results into typed read-model objects at retrieval time.
- Project OM nodes into typed `Episode` objects and OM relations into typed `StateFact` objects with stable synthetic IDs, lane provenance, and pass-through evidence pointers back to OM-native artifacts.
- Teach the shared typed retrieval service to merge ledger-backed typed objects with OM projected objects under the same ranking, filtering, bucketing, and evidence-resolution flow.
- Extend the benchmark harness with a typed-contract mode so OM bakeoffs can target `search_memory_facts(..., result_format="typed")` directly.

DoD checklist
- [x] Typed retrieval can surface OM-native canonical lane content through shared typed buckets.
- [x] Typed retrieval can surface OM-native experimental-group content through shared typed buckets when explicitly scoped.
- [x] OM remains retrieval-projected only; no fake Graphiti storage entities are introduced.
- [x] Typed retrieval evidence bucket includes deterministic OM provenance pointers for projected OM results.
- [x] Benchmark tooling can exercise Bicameral typed retrieval directly.
- [x] Focused regression tests cover typed OM projection behavior and benchmark typed-mode wiring.

Validation commands
```bash
cd projects/bicameral
PYTHONPATH=. pytest tests/test_typed_retrieval_service.py tests/test_search_memory_facts_typed_mode.py tests/test_retrieval_benchmark.py -q
pytest mcp_server/tests/test_search_om_lane.py -q
python3 scripts/typed_retrieval_smoke.py
```
