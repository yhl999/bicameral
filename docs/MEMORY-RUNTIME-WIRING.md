# Memory Runtime Wiring

This document defines the runtime backend contract, evidence plane integration, OM fast-write wiring, and pack injection policy.

---

## Runtime Model

### Backend Profiles

Two backend profiles are supported:

- `qmd_primary` (production default) — QMD handles retrieval; Graphiti runs in governed shadow mode
- `graphiti_primary` (operator opt-in) — Graphiti handles primary retrieval; QMD as evidence callback

Profiles are declared in `config/runtime_memory_backend_profiles.json`.
Current active state is stored in `config/.runtime_memory_backend_state.json`.

### Typed Retrieval Contract

The typed memory rescope adds a **typed retrieval contract** to the established `search_memory_facts` surface. Set `result_format='typed'` to receive four buckets:

| Bucket | Contents |
|---|---|
| `state` | Current StateFact objects (preferences, decisions, lessons, rules, etc.) |
| `episodes` | Time-bounded Episode objects with evidence links |
| `procedures` | Active Procedure objects (trigger → steps → outcome) |
| `evidence` | Raw evidence resolved through pluggable evidence callback |

Legacy callers continue to receive `facts` format by default. Typed mode does not become the default until legacy callers are safely migrated or backward-compatible support is validated.

### Evidence Plane

Raw evidence (transcripts, notes, docs, artifacts) is accessed through a **pluggable evidence plane**. Current adapters:

| Adapter | Scope | Implementation |
|---|---|---|
| QMD | File/log evidence chunks | `QMDEvidenceCallback` in `evidence_callback.py` |
| Passthrough | Non-QMD structured references | `PassThroughEvidenceCallback` in `evidence_callback.py` |

Every surfaced memory object carries `EvidenceRef` pointers with `canonical_uri` and structured `locator` objects. The `evidence` bucket in typed retrieval resolves these through the callback registry.

The evidence plane is not the canonical memory plane — it's the proof layer. Bicameral holds governed typed memory objects; the evidence plane holds the raw source material those objects were derived from.

### Operator Commands

```bash
python3 scripts/runtime_memory_backend_status.py
python3 scripts/runtime_memory_backend_switch.py --target graphiti_primary --dry-run
python3 scripts/runtime_memory_backend_switch.py --target qmd_primary --dry-run
python3 scripts/runtime_memory_backend_switch.py --target graphiti_primary --execute
python3 scripts/runtime_memory_backend_switch.py --revert --execute
```

### Guardrails

- Group-safe gating must stay enabled in all active profiles
- Shadow compare should remain enabled during cutover
- One-command revert must always be available after a switch

### Public/Private Split

Public repo contains the generic switch/status framework and example/default profile config. Private operational overlays may replace profile values at deploy time but should not change the switch contract surface.

---

## Security Considerations

The ingest pipeline processes raw session transcripts, memory files, and conversation data that inherently contain personally identifiable information (PII). This is by design — personal memory is the core use case.

**Key security constraints:**
- The `sanitize_content()` pipeline strips channel metadata, tool routing noise, and wrapper blocks before LLM extraction. It does NOT strip PII — that would defeat the purpose.
- The `evidence_callback` resolvers never send raw evidence to external services. Resolution is local-only.
- Group-safe gating prevents private facts from surfacing in group-chat retrieval contexts.
- The private overlay repo holds all deployment-specific config. The public repo contains zero private data.

---

## OM Fast-Write Wiring

The OM fast-write path bypasses the MCP server's ingestion queue for sub-second writes from live transcript streams.

### Enabling Fast-Write

```bash
# Wire the state file
python3 scripts/om_fast_write.py set-state \
  --runtime-repo /path/to/runtime-repo \
  --enabled --reason "hook_wired"

# Write messages directly
python3 scripts/om_fast_write.py write \
  --session-id "<session_id>" --role user \
  --content "<message text>" --created-at "2026-02-26T12:00:00Z"
```

### Integration Points

| Hook | How |
|---|---|
| **OpenClaw plugin** | Plugin invokes fast-write on each inbound message |
| **Standalone** | CLI for manual/test writes |

Fast-write creates `Message` and `Episode` nodes directly in Neo4j. Fail-closed: if embedding fails, the write is skipped entirely (no partial state). Dedup: content-hash on (session_id, message content, created_at) prevents double-writes.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `https://api.openai.com/v1` | LLM chat-completions endpoint. Priority: `OM_COMPRESSOR_LLM_BASE_URL` > `LLM_BASE_URL` > `OPENAI_BASE_URL` > default. |
| `EMBEDDER_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible embedding endpoint. Priority: `EMBEDDER_BASE_URL` > `OPENAI_BASE_URL` > default. |
| `OM_EMBEDDING_MODEL` | `embeddinggemma` | Embedding model |
| `OM_EMBEDDING_DIM` | `768` | Expected vector dimension |
| `RUNTIME_REPO_ROOT` | (none) | If set, fast-write updates the state file automatically |

---

## Runtime Pack Injection Policy

Runtime pack routing is performed by `scripts/runtime_pack_router.py`.

### Policy Controls

- **Multi-group retrieval matrix** per pack via `retrieval.group_ids_by_mode`
- **ChatGPT lane gating** via per-profile `chatgpt_mode = off|scoped|global`
- **Scoped** is the intended safe default when mode is omitted
- **Engineering learnings** can be materialized at runtime (`--materialize`) from latest loop artifacts

### Runtime Checkout

Operational runtime should execute from the canonical runtime checkout:
```bash
tools/graphiti -> ../projects/graphiti-openclaw-runtime
```

Apply private overlay before operations:
```bash
/path/to/private-repo/scripts/apply-overlay.sh /path/to/runtime-checkout
```

---

## See Also

- [The Dual-Brain Architecture](DUAL-BRAIN-ARCHITECTURE.md) — Why two systems, how trust works
- [OM Operations Runbook](runbooks/om-operations.md) — Fast-write, compressor, convergence
- [Retrieval Trust Scoring](retrieval-trust-scoring.md) — Trust multiplier mechanics
- [Runtime Pack Overlay](runbooks/runtime-pack-overlay.md) — How private packs map to agents
