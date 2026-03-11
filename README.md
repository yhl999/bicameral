# Bicameral — Dual-Brain Typed Memory Runtime for Agents

Bicameral is a **ledger-first, typed memory system** built as a production delta layer on top of [upstream Graphiti](https://github.com/getzep/graphiti). It turns Graphiti from a graph-memory library into a **dual-brain, policy-governed memory runtime** with first-class semantic, episodic, and procedural memory.

The key insight: **A single LLM-powered brain can't be trusted with your memories.** We added a second brain — a strict, append-only **ChangeLedger** — that acts as a thermostat to keep the semantic engine honest. On top of this dual-brain substrate, we layer **typed memory objects** that give agents currentness, supersession, temporal precision, and governed truth.

If you're looking for the core Graphiti framework docs:
- Upstream repo: <https://github.com/getzep/graphiti>
- Upstream docs: <https://help.getzep.com/graphiti>

---

## Why This Exists

Plain vector search (RAG) is good at recall ("find documents about X") but terrible at **truth management**. Even Graphiti — a breakthrough temporal knowledge graph — relies 100% on an LLM to decide what's true. When your AI manages your calendar, drafts your deals, and handles your relationships, "let the LLM figure it out" is a recipe for operational failure.

Bicameral solves three problems vanilla Graphiti can't:

- **No silent truth mutations.** Every promotion, supersession, and invalidation is recorded in the ChangeLedger with full provenance.
- **No LLM hallucinations becoming doctrine.** Unverified claims stay quarantined until approved through a governed promotion policy.
- **Agents that learn and remember correctly.** Typed memory objects track what is *current*, what *changed*, and what *procedure* to follow — not just what was mentioned once.

---

## Architecture Overview

### The Dual Brain

| | Brain 1 — Semantic Engine (Neo4j) | Brain 2 — ChangeLedger (SQLite) |
|---|---|---|
| **Purpose** | Semantic richness, relationships, embeddings | Deterministic truth, mutation history, governance |
| **Characteristics** | Non-deterministic, LLM-extracted, fast retrieval | Append-only, hash-chained, auditable |
| **Source of truth?** | No. Derived. | **Yes. Canonical.** |

Brain 1 holds all the messy, probabilistic semantic richness. Brain 2 holds deterministic truth — every assertion, supersession, and invalidation is recorded as an immutable ledger event. If the two disagree, **Brain 2 wins**.

The bridge between them is a **trust multiplier**: promoted facts get `trust_score = 1.0` in Brain 1, which biases retrieval ranking toward verified truth. The formula: `final_score = rrf_score + (trust_score × trust_weight)`.

For the full architecture deep-dive, see [The Dual-Brain Architecture](docs/DUAL-BRAIN-ARCHITECTURE.md).

### The Four-Layer Stack

```
┌──────────────────────────────────────────────────────────────┐
│  L4 - Retrieval / Answer Assembly                            │
│  Typed retrieval contract returning state + episodes +       │
│  procedures + evidence. Mixed-mode routing for               │
│  currentness queries vs raw evidence recall.                 │
├──────────────────────────────────────────────────────────────┤
│  L3 - Derived Read Models / Projections                      │
│  Current-state view. Supersession history. Episode timeline. │
│  Procedure registry. Graph projection. Search indexes.       │
│  Context/workflow packs for scoped assembly.                 │
├──────────────────────────────────────────────────────────────┤
│  L2 - Typed Memory Objects                                   │
│  StateFact (preferences, decisions, lessons, rules…)         │
│  Episode (time-bounded events with evidence links)           │
│  Procedure (trigger → steps → outcome, versioned)            │
│  Entity registry. EvidenceRef pointers.                      │
├──────────────────────────────────────────────────────────────┤
│  L1 - ChangeLedger + Semantic Engine                         │
│  Append-only event stream (assert, supersede, invalidate,    │
│  refine, derive, promote, procedure_success/failure).        │
│  Neo4j graph as derived semantic index.                      │
│  Pluggable evidence plane (QMD as current adapter).          │
└──────────────────────────────────────────────────────────────┘
```

### Layer 0 — Raw Evidence Plane

Immutable source references (transcripts, notes, docs, artifacts, structured records) accessed through a **pluggable evidence plane**. This is not the canonical memory plane — it's the evidence plane. QMD is the current file/log adapter. Every surfaced memory object carries `EvidenceRef` pointers back to raw source material.

---

## Typed Memory Objects

Bicameral organizes memory into three first-class typed object families, all governed by the ChangeLedger and sharing a common base contract:

### StateFact — Semantic Memory

State-bearing truth about the world: preferences, decisions, commitments, lessons, operational rules, constraints, relationships, world-state observations.

Each StateFact has a **subtype** that determines its promotion policy and risk tier. Subtypes: `preference`, `decision`, `commitment`, `lesson`, `world_state`, `operational_rule`, `constraint`, `relationship`.

**Currentness model:** One current object per conflict set. When a preference changes, the old StateFact is explicitly `supersede`d — not silently overwritten.

### Episode — Episodic Memory

Time-bounded events with evidence links. Episodes are derived from session transcripts and carry immutable core fields (time span, participants, source refs) plus editable annotations/labels.

Episodes serve as the **evidence anchor** for StateFacts and Procedures. A promoted fact always traces back to the episode(s) that sourced it.

### Procedure — Procedural Memory

Reusable action plans with trigger conditions, preconditions, ordered steps, expected outcomes, success/failure counters, version lineage, and `is_current` status.

Procedures evolve through feedback: successful executions increment counters and may trigger auto-promotion of candidates; failures trigger version evolution or warnings.

**Promotion gate:** `ProcedureCandidate` objects promote to active `Procedure` via human approval or a conservative repeated-success threshold (low-risk: 3 successes across 2 distinct episodes; medium-risk: 5 across 3; high-risk: never auto-promotes).

### Shared Base Contract

All typed objects carry:
```
object_id, object_type
root_id, parent_id, version, is_current
source_lane, source_episode_id, source_message_id, source_key
valid_at, invalid_at, superseded_by
policy_scope, visibility_scope
evidence_refs[]
extractor_version, created_at
```

---

## The ChangeLedger

The ChangeLedger is an append-only, hash-chained event stream that records every mutation to Bicameral's typed memory objects. It is the **canonical source of truth** — the graph database and all read models are derived from it.

### Event Vocabulary (v1)

| Event | When |
|---|---|
| `assert` | New object introduced or new state-bearing claim made |
| `supersede` | Current object replaced by a newer one in the same conflict set |
| `invalidate` | Object marked no longer valid without replacement |
| `refine` | Object improved/tightened without changing essential identity |
| `derive` | Object synthesized from other evidence or objects |
| `promote` | Object moves from candidate/proposed to authoritative status |
| `procedure_success` | Procedure executed successfully (increments counters) |
| `procedure_failure` | Procedure execution failed (triggers evolution/warning) |

Each event is immutable, hash-chained, and carries full provenance (actor, timestamp, source evidence, reasoning).

---

## Promotion Policy

Not everything ingested becomes truth. The Promotion Policy governs how candidates move from "observed" to "promoted."

### Key Rules

- **Trust boundary:** Only owner-authored evidence is eligible for auto-promotion. Non-owner content can create entities but facts stay quarantined.
- **Assertion gating:** Only decisions, preferences, and factual assertions can promote. Questions, hypotheticals, and quotes are blocked.
- **Risk tiers:** Low-risk facts (preferences) auto-promote at high confidence (>0.90). Medium-risk requires stricter evidence. High-risk always requires human approval.
- **Conflict detection:** Contradictions are flagged. The system never silently overwrites existing truth.
- **Corroboration:** Independent evidence from multiple sources carries more weight. Same-lineage evidence gets bounded boost caps to prevent confidence inflation.

### Policy v3 (Unified Default)

Policy v3 is the unified default for all code paths (`POLICY_VERSION_DEFAULT = "promotion-v3"`). Both OM-derived candidates and graph-lane candidates use the same v3 decision evaluator.

---

## Observational Memory (OM)

OM is the runtime synthesis/control ("metabolism") loop for the Dual Brain stack. It closes the velocity gap between raw transcript intake and the full MCP extraction pipeline.

```
Live transcript → om_fast_write.py → Neo4j Message/Episode nodes
                                              │
                                     om_compressor.py → OMNode observations
                                              │
                                     om_convergence.py → Lifecycle state machine
                                              │
                                     promotion_policy_v3.py → CoreMemory (on corroboration)
```

OM types: `WorldState`, `Judgment`, `OperationalRule`, `Commitment`, `Friction`. Nodes progress through `OPEN → MONITORING → CLOSED/ABANDONED` with optional `REOPENED` paths.

For operations details, see [OM Operations Runbook](docs/runbooks/om-operations.md).

---

## Graph Lanes & Custom Ontologies

Each domain of knowledge lives in its own isolated graph (`group_id`). Data never leaks between lanes.

Each lane can define its own **extraction ontology** — domain-specific entity and relationship types. A content-inspiration lane extracts `RhetoricalMove`, `HookPattern`, `VoiceQuality`; an engineering lane extracts `FailurePattern`, `ToolApiBehavior`, `ArchitectureDecision`.

Ontologies are defined in YAML config (`config/extraction_ontologies.yaml`). Adding a new lane requires only a YAML block — zero code changes.

### Lane Matrix (v3)

| Lane (`group_id`) | Retrieval | Candidate-Generating | Notes |
|---|---|---|---|
| `s1_sessions_main` | ✅ Global | ✅ Yes | Primary conversational memory |
| `s1_observational_memory` | ✅ Global | ✅ Yes | OM synthesis nodes |
| `s1_chatgpt_history` | ✅ Global | ✅ Yes | Imported conversation history |
| `s1_memory_day1` | ❌ Corroboration-only | ✅ Yes | Bootstrap memory (not surfaced directly) |
| `s1_curated_refs` | ✅ Global | ❌ No | Reference snapshots |
| `s1_inspiration_*`, `s1_writing_samples`, `s1_content_strategy` | ❌ Pack injection | ❌ No | Craft patterns, injected via content packs |
| `engineering_learnings`, `learning_self_audit` | ❌ Separate domain | ❌ (own pipeline) | Own ledger pipelines, different trust semantics |

For ontology details, see [Custom Ontologies](docs/custom-ontologies.md).

---

## Context & Workflow Packs

### Content Packs (L3 Read Models)

Scoped read-assembly bundles. Each pack defines:
- **Retrieval matrix:** Which graph lanes to query, by mode
- **Token budget tier:** A (600/10), B (1200/20, default), C (2400/40)
- **Scope policy:** `private` (DMs) or `group_safe` (group chats, auto-selected)
- **ChatGPT lane policy:** `off`, `scoped`, `global`

### Workflow Packs

Multi-step orchestration on top of content packs:
- Declares required context pack dependencies
- Defines tool steps with explicit permission boundaries
- Execution modes: `draft_only` or `execute_with_approval`
- Supports dual-lane ingest: artifacts (few-shot examples) + extracted claims (policy-gated)

The **Runtime Pack Router** (`scripts/runtime_pack_router.py`) resolves which packs are active for a given agent + intent, applies policy, and returns formatted context.

---

## Trust-Aware Retrieval

Promotion status feeds back into retrieval quality:

| Status | `trust_score` | Effect |
|---|---|---|
| Promoted (in ChangeLedger) | 1.0 | Strongest boost |
| Corroborated (≥2 independent sources) | 0.6 | Moderate boost |
| Standard candidate (single source) | 0.25 | Minimal signal |
| Not in pipeline | NULL | Neutral baseline |

Boost: `final_score = rrf_score + (trust_score × trust_weight)`. Default `trust_weight` is 0.15, configurable via `GRAPHITI_TRUST_WEIGHT`.

For details, see [Retrieval Trust Scoring](docs/retrieval-trust-scoring.md).

---

## Public/Private Split Model

This fork operates on an **Engine/Fuel** split:

- **Public repo (this one):** Runtime architecture, routing logic, policy enforcement, ontology framework, tooling. Knows *how* to route requests but contains no private data.
- **Private overlay repo (yours):** Your `runtime_pack_registry.json`, `runtime_consumer_profiles.json`, `plugin_intent_rules.json`, `workflows/*.pack.yaml`, extraction ontology config, and graph state.

```bash
# Apply private overlay into your runtime checkout
./path/to/private-repo/scripts/apply-overlay.sh /path/to/runtime-checkout
```

---

## Installation & Setup

### Prerequisites
- Python 3.13+
- Neo4j (default) or FalkorDB (legacy) — graph database backend
- OpenAI API key (for LLM extraction + embeddings)
- **Endpoint split (recommended):** set `LLM_BASE_URL` for LLM routing and `EMBEDDER_BASE_URL` for embedding routing separately

### Quick Start

```bash
git clone https://github.com/yhl999/bicameral.git
cd bicameral
uv sync                    # OR: pip install -e ".[neo4j]"
cp config/config.example.yaml config/config.yaml
# Edit config.yaml: Neo4j connection, OpenAI key, etc.

python3 scripts/delta_tool.py list-commands        # Verify delta tooling
python3 scripts/runtime_pack_router.py --verify-only  # Verify pack config
python3 mcp_server/main.py                         # Start MCP server (HTTP)
```

### Ingesting Data

Data flows through a 7-stage pipeline: Source Material → Evidence → Ingest Registry → Graphiti MCP → Candidates DB → Promotion Policy → ChangeLedger.

```bash
# Bootstrap Neo4j with historical transcripts
python3 scripts/import_transcripts_to_neo4j.py \
  --sessions-dir path/to/session_transcripts/ --dry-run
python3 scripts/import_transcripts_to_neo4j.py \
  --sessions-dir path/to/session_transcripts/

# Ingest sessions (Neo4j source mode, production path)
python3 scripts/mcp_ingest_sessions.py \
  --group-id s1_sessions_main --source-mode neo4j \
  --mcp-url http://localhost:8000/mcp

# Check ingestion status
python3 scripts/registry_status.py

# Verify adapter contract compliance
python3 scripts/ingest_adapter_contract_check.py --strict
```

Ingestion is idempotent (content-hash dedup), incremental (delta since last watermark), and supports sub-chunking for large evidence (>10k chars).

### Ingest Sanitization

Before evidence reaches the LLM extractor, a sanitizer strips wrapper noise: `<graphiti-context>` blocks, untrusted metadata wrappers, and channel routing metadata. Raw message content is preserved unchanged. Token savings: ~20–60% per episode.

---

## Keeping Up with Upstream

This fork tracks `getzep/graphiti` via a deterministic PR-based sync lane with an explicit patch stack for `graphiti_core` hotfixes.

- **Default cadence:** Weekly (Monday) via GitHub Action
- **Conflict policy:** Upstream wins. Re-apply local patch stack from `patches/graphiti_core/`
- **Core guardrail:** CI enforces no undocumented `graphiti_core` drift

See [Upstream Sync Runbook](docs/runbooks/upstream-sync-openclaw.md) and [`HOTFIXES.md`](HOTFIXES.md).

---

## Documentation Index

### Architecture & Concepts
- [The Dual-Brain Architecture](docs/DUAL-BRAIN-ARCHITECTURE.md) — Why two systems, how trust works, OM synthesis
- [Custom Ontologies](docs/custom-ontologies.md) — Per-lane extraction entity types
- [Retrieval Trust Scoring](docs/retrieval-trust-scoring.md) — Trust multiplier mechanics
- [Memory Runtime Wiring](docs/MEMORY-RUNTIME-WIRING.md) — Backend profiles, evidence plane, OM wiring
- [Scope Policy](docs/scope-policy.md) — Ingestion scope: message-only default, toolResult opt-in
- [Runtime Pack Overlay](docs/runbooks/runtime-pack-overlay.md) — How private packs map to agents

### Operations
- [OM Operations](docs/runbooks/om-operations.md) — Fast-write, compressor, convergence, GC, promotion, dedupe
- [Sessions Ingestion](docs/runbooks/sessions-ingestion.md) — Batch & steady-state config, sub-chunking, recall gate
- [Dual-Brain Operators Guide](docs/runbooks/dual-brain-operators-guide.md) — Approving facts, handling conflicts, debugging
- [Adding Data Sources](docs/runbooks/adding-data-sources.md) — Onboarding new content
- [Upstream Sync Runbook](docs/runbooks/upstream-sync-openclaw.md)
- [State Migration Runbook](docs/runbooks/state-migration.md)
- [OpenClaw Plugin Troubleshooting](docs/runbooks/openclaw-plugin-troubleshooting.md)

### Technical Contracts
- [Boundary Contract](docs/public/BOUNDARY-CONTRACT.md)
- [Migration Sync Toolkit](docs/public/MIGRATION-SYNC-TOOLKIT.md)
- [Release Checklist](docs/public/RELEASE-CHECKLIST.md)

---

## Current Status

### What's Shipped
- **Dual-brain architecture** with Neo4j (Brain 1) + SQLite ChangeLedger (Brain 2) operational
- **Promotion policy v3** unified across all code paths (OM + graph-lane candidates)
- **Lane matrix v3** locked and codified in config + code
- **OM pipeline** operational: fast-write → compressor → convergence → CoreMemory promotion
- **Trust-aware retrieval** with RRF + trust multiplier live
- **Ingest sanitization** stripping wrapper noise before extraction (~20-60% token savings)
- **Custom ontologies** with per-lane YAML config, zero-code lane addition
- **Public/private split** with overlay model, deterministic rebuild, security rails

### In Progress (Typed Memory Rescope)
The typed memory rescope epic is actively shipping the post-rescope architecture described above:
- **Phase 0 (Schema + Interface Decision):** ✅ Locked — schema, event vocabulary, must-win suite defined
- **Phase 1 (ChangeLedger + Typed Object Base):** Building canonical change model and StateFact/Episode/Procedure implementations
- **Phases 2-4 (Lane Unification, Typed Retrieval, Procedural Memory):** Parallel fan-out after Phase 1
- **Phase 5 (OM Closeout):** Carry-forward under rescoped architecture
- **Integration:** Must-win proof/kill suite (6 blind questions, 5/6 win/tie required)

### Operator Note
Runtime retrieval is currently QMD-primary with Graphiti in governed shadow mode. The typed memory rescope will gate the flip to Bicameral-primary retrieval on passing the must-win evaluation suite. See private repo `docs/runbooks/operator-rollout-runbook.md` for the decision flow.

## CI Policy

Canonical PR gates:
- `.github/workflows/ci.yml`
- `.github/workflows/migration-sync-tooling.yml`
