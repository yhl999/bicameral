# The Dual-Brain Architecture: Why Two Systems Are Better Than One

## The Problem With "Magic" AI Memory

Most AI memory systems rely on a single brain: an LLM that reads new information, looks at old information, and decides what's true. Graphiti (the upstream library) does this — it uses a prompt to ask an LLM whether new observations contradict old ones.

This works for a demo. But for a Chief-of-Staff AI managing your calendar, drafting your deals, and handling your relationships, it breaks down.

**The LLM has no concept of authority.** If someone in a group chat jokingly says "Yuan loves eating glass," the LLM might silently invalidate your real diet preference. No audit log. No rollback. Just hallucinated metadata you'll never know was deleted.

**The LLM struggles with context limits.** Auto-invalidation requires loading old facts into the prompt. With thousands of facts, the LLM misses contradictions — leaving you with a graph that simultaneously asserts conflicting truths.

**There's no trace of truth decisions.** When a fact is invalidated, it's gone. No history of why. No way to revert.

This is what happens when you have only **Brain 1** — the semantic, LLM-powered, non-deterministic brain.

---

## The Dual-Brain Solution

We built **Brain 2** — the ChangeLedger: a strict, append-only event stream backed by SQLite.

### Brain 1: The Semantic Engine (Neo4j)
- Holds all the semantic richness — entities, relationships, embeddings, nuance
- Non-deterministic, probabilistic, LLM-extracted
- Good at: finding things that *feel* relevant to a query
- Bad at: knowing what's true

### Brain 2: The ChangeLedger (SQLite)
- Records every mutation as an immutable, hash-chained event
- Event types: `assert`, `supersede`, `invalidate`, `refine`, `derive`, `promote`, `procedure_success`, `procedure_failure`
- Good at: truth governance, audit trails, rollback, conflict resolution
- Bad at: semantic understanding

### The Bridge: Trust as a Thermostat

When a fact is promoted in Brain 2, the system stamps the corresponding Brain 1 edge with `trust_score = 1.0`. At retrieval time, trust scores act as a reranking multiplier:

```
final_score = rrf_semantic_score + (trust_score × trust_weight)
```

The joke about eating glass gets low semantic relevance. Your actual preference gets `trust_score = 1.0`. Truth crushes noise at retrieval time.

---

## Typed Memory: What the Dual Brain Governs

The dual-brain substrate governs three families of **typed memory objects**:

### StateFact — Semantic Memory
State-bearing truth: preferences, decisions, commitments, lessons, operational rules, constraints, relationships, world-state observations. Each has a subtype that determines promotion policy and risk tier.

**Currentness model:** One current object per conflict set. Supersession is explicit, not silent.

### Episode — Episodic Memory
Time-bounded events with evidence links. Derived from session transcripts with immutable core fields (time span, participants, source refs) plus editable annotations.

Episodes are the **evidence anchor** — every promoted StateFact traces back to its source episode(s).

### Procedure — Procedural Memory
Reusable action plans: trigger conditions, preconditions, ordered steps, expected outcomes, success/failure counters, version lineage, `is_current` status.

Procedures evolve through feedback. Successful executions increment counters and may trigger auto-promotion of candidates; failures trigger version evolution.

### Shared Base Contract

All typed objects carry: `object_id`, `object_type`, `root_id`, `parent_id`, `version`, `is_current`, source provenance (`source_lane`, `source_episode_id`, `source_key`), temporal bounds (`valid_at`, `invalid_at`, `superseded_by`), scope (`policy_scope`, `visibility_scope`), and `evidence_refs[]`.

---

## Why Not Just Use One Brain?

**Option A: Brain 1 only (vanilla Graphiti approach)**
- Pros: Simpler. No extra database.
- Cons: Unreliable truth. Hallucinating LLM. No audit trail. Silent mutations. Unacceptable for high-stakes decisions.

**Option B: Brain 2 only (pure ledger)**
- Pros: Perfect auditability.
- Cons: No semantic understanding. No vector search. No relationship traversal.

**Option C: Dual Brain with Typed Memory**
- Pros: Semantic richness of Brain 1 + deterministic truth of Brain 2. Typed objects for currentness, supersession, and procedural learning. Evidence traceability.
- Cons: Operational complexity. Two systems to maintain and sync.

We chose C because the alternative is an AI that eventually believes it's a die-hard vegan living in Portland.

---

## How the Dual Brain Actually Works

### The Flow

1. **Extraction (Brain 1):** Transcript arrives. LLM extracts entities and relationships into Neo4j. Fast, messy, non-deterministic.

2. **Candidate Generation (Bridge):** A subset of extractions become "promotion candidates" — potential facts worthy of truth status. Logged into `candidates.db` with source, confidence, and Neo4j edge UUID.

3. **Promotion Decision (Brain 2):** Candidates wait for approval. Owner-authored facts may auto-promote at high confidence. Third-party facts stay quarantined. Risky facts require human review.

4. **Conflict Detection (Brain 2):** Before promotion, the system checks for contradictions with existing promoted facts. Conflicts are flagged, never silently resolved.

5. **ChangeLedger Event (Brain 2):** Promotion writes an immutable `promote` event. Supersession writes a `supersede` event with explicit chain references. The old fact is not deleted — it's marked as superseded with a timestamp and pointer to its replacement.

6. **Trust Sync (Bridge):** After promotion, sync stamps `trust_score = 1.0` on corresponding Brain 1 edges. Unpromoted edges get `0.25` or NULL.

7. **Retrieval (Bridge):** Queries use RRF with trust multiplier. Promoted facts surface higher. Noise sinks.

### Supersession Without Silent Deletion

When a fact changes (e.g., "Yuan is 25" → "Yuan is 26"):

```json
{
  "event_type": "supersede",
  "object_id": "fact-uuid-26",
  "content": "Yuan is 26",
  "supersedes": "fact-uuid-25",
  "valid_at": "2026-02-21",
  "ledger_hash": "abc123..."
}
```

The old fact is not deleted. It's superseded with full provenance. You can audit the entire history and revert if needed.

Brain 1 supports a **closure semantics pass** (`scripts/apply_closure_semantics.py`) for graph-level housekeeping: `RESOLVES`/`SUPERSEDES` edges auto-invalidate target facts at the closure event's timestamp. Idempotent, offline, no LLM calls. Brain 2 handles policy; the closure pass handles graph consistency.

---

## Observational Memory (OM): Synthesis + Control Layer

OM is the runtime synthesis/control ("metabolism") loop for the Dual Brain. It is **not** an independent third authority on truth — it coordinates rapid intake into Brain 1 with governed promotion pathways into Brain 2.

The Dual Brain governs *what to trust*. But the MCP server's ingestion pipeline has non-trivial latency (LLM extraction per episode). In a high-throughput agent runtime, messages accumulate faster than the pipeline can drain. OM closes this gap:

### Stage 1 — Fast-Write (`scripts/om_fast_write.py`)
Transcript messages written directly into Neo4j as `Message`/`Episode` nodes without MCP extraction queue. Fail-closed: no embedding = no write.

### Stage 2 — Compressor (`scripts/om_compressor.py`)
Background process drains unextracted `Message` backlog into `OMNode` observations. Trigger: backlog ≥ 50 messages or oldest ≥ 48h. Types: `WorldState`, `Judgment`, `OperationalRule`, `Commitment`, `Friction`.

### Stage 3 — Convergence (`scripts/om_convergence.py`)
OMNode lifecycle state machine: `OPEN → MONITORING → CLOSED/ABANDONED`, with `REOPENED` paths. Dead-letter reconciliation. Optional GC (90-day TTL with retention gates).

### Stage 4 — CoreMemory Promotion (`truth/promotion_policy_v3.py`)
Corroborated OMNodes promote to `CoreMemory` nodes (exempt from GC), linked to source evidence via `SUPPORTS_CORE` edges.

### Key Numbers

| Parameter | Value |
|---|---|
| Compressor trigger (backlog) | ≥ 50 messages |
| Compressor trigger (age) | ≥ 48 hours |
| Convergence pass limit | 500 nodes |
| GC TTL (default) | 90 days |

For operations: [OM Operations Runbook](runbooks/om-operations.md).

---

## Why This Matters for Agents

The Dual Brain with typed memory unlocks capabilities vanilla Graphiti can't offer:

**Agents that know what's current.** "What's my coffee preference?" returns the latest supersession chain, not a random historical mention. The ChangeLedger tracks exactly when and why preferences changed.

**Agents that learn procedures.** After repeated successful coding patterns, the system extracts `ProcedureCandidate` objects. Once corroborated across multiple episodes, they promote to active `Procedure` objects with versioned evolution. The next agent run gets them injected as context — knowledge compounds instead of evaporating.

**Agents that can explain their answers.** Every surfaced memory object carries `evidence_refs[]` pointers back to source episodes and raw evidence. The agent can always say *why* it believes something and *where* it learned it.

**Agents that handle conflict honestly.** When facts contradict, the system flags it rather than silently picking a winner. The operator (or policy) resolves explicitly, and the resolution is auditable.

---

## The Tradeoff

The Dual Brain with typed memory is operationally complex:

- You maintain **two systems** (Neo4j + SQLite ChangeLedger) that must stay in sync.
- You write and tune **promotion policies** that determine which observations become truth.
- You monitor the **candidates queue** and handle conflicts when the system flags contradictions.
- You run **typed object lifecycle management** — tracking currentness, supersession chains, and procedure evolution.

Vanilla Graphiti is simpler — dump transcripts in and let the LLM figure it out.

But if your AI is managing anything high-stakes — calendar, deals, relationships, operational memory — "letting the LLM figure it out" is a recipe for silent data corruption you'll never notice until it causes real harm. We bet the complexity was worth it.

---

## See Also

- [Retrieval Trust Scoring](retrieval-trust-scoring.md) — How the trust multiplier works in code
- [Memory Runtime Wiring](MEMORY-RUNTIME-WIRING.md) — Backend profiles, evidence plane, OM wiring
- [Custom Ontologies](custom-ontologies.md) — Teaching each graph lane what to extract
- [Scope Policy](scope-policy.md) — Ingestion scope controls
- [OM Operations Runbook](runbooks/om-operations.md) — Trigger math, lock ordering, GC, convergence
- [Dual-Brain Operators Guide](runbooks/dual-brain-operators-guide.md) — Day-to-day operations
- [Runtime Pack Overlay](runbooks/runtime-pack-overlay.md) — How private packs map to agents
