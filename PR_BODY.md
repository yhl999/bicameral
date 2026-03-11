## Summary
- switch `runtime_pack_router.py` content-pack materialization from the dead HTTP `/search` assumption to Graphiti MCP HTTP (`initialize` + `tools/call(search_memory_facts)`)
- update the router test stub to speak MCP and assert the new transport path
- document the runtime-pack overlay materialization contract in the runbook
- relax the Phase 3C short-text anti-meta gate for `s1_writing_samples` craft markers so concise style signals are not auto-dropped purely for being short

## Validation
- `uv run pytest -q tests/test_runtime_pack_router.py`
- `python3 -m py_compile scripts/runtime_pack_router.py graphiti_core/graphiti.py`
- reran content-pack canaries from the active runtime overlay against live MCP-backed data:
  - Step 6 lane utility report regenerated (`average_score = 4.167`)
  - content-pack quality gate passes for both long-form and tweet probes
  - `content_voice_style` still falls back to static guidance because live `s1_writing_samples` currently returns `0` facts

## Known follow-up
- I attempted a bounded `s1_writing_samples` re-ingest against a throwaway MCP server to validate the Phase 3C repair end-to-end, but that temp server hit an OpenRouter auth-header failure during extraction. So this PR fixes the transport bug and lands the writing-samples gate repair, but the live lane still needs a clean re-ingest/restart pass to prove recovery.
