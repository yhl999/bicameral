# PRD: Lane-Specific Extraction Ontologies + Dynamic Content Pack Materialization

## PRD Metadata
- Type: Execution
- Kanban Task: `task-lane-specific-ontologies-v0`
- Parent: `task-post-graphiti-capability-roadmap`
- Cross-references:
  - `EXEC-CONTENT-CREATION-EPIC-v0` (content workflow architecture)
  - `PRD-ENGINEERING-MEMORY-ANTFARM-CLR-EPIC` (engineering memory architecture)
  - `PRD-RUNTIME-SELECTION-INJECTION-ORCHESTRATION-v0` (runtime pack router)
  - Private repo issue #9 (replace synthetic content voice/writing packs with real corpus)
- Interview decisions locked: 2026-02-18 (Yuan)

---

## 1) Problem Statement

Graphiti extraction currently uses **one global `entity_types` set** for all group_ids. Every episode — whether it's a tweet from an admired writer, Yuan's own investment memo, a content strategy guide, or an engineering postmortem — goes through the same generic extraction pipeline.

This produces:
- **Low-signal entities** for content lanes (generic nouns instead of rhetorical moves, voice patterns, hook structures)
- **Missing domain-specific relationships** (no "INSPIRED_BY author style", no "FIXED_BY remediation")
- **No metadata-driven filtering** at retrieval time (can't ask "give me casual-register voice patterns" or "Hayes-style narrative hooks")

Meanwhile, the `content_voice_style` and `content_writing_samples` runtime packs are still **synthetic placeholders** (explicitly marked `STATUS: SYNTHETIC PLACEHOLDER (v1)`), even though real extraction data exists in the graphs.

### What this PRD delivers
1. **Per-lane extraction ontologies** (entity types + relationship types + extraction emphasis) for 6 graph lanes
2. **Extraction routing** so each `group_id` uses its lane-specific ontology
3. **Dynamic content pack materialization** replacing synthetic placeholders with graph-backed output
4. **Public repo documentation** showing users how to define custom ontologies for their own use cases
5. **Metadata corrections** for writing samples evidence (incorrect subtypes, missing AI/compliance tags)

---

## 2) Interview Decisions (locked 2026-02-18)

### Content Inspiration (short-form + long-form)
- **Priority order:** rhetorical moves ≈ voice/tone > audience reaction > author style > topic/thesis > specific lines
- **Short-form emphasis:** hooks, compression, punch, first-line formulas
- **Long-form emphasis:** argument structure, narrative arc, transitions, closers
- Rhetorical move types and voice/tone qualities differ between formats
- **Author style archetype** should be captured as an entity for optional retrieval filtering ("give me Hayes-style hooks") but retrieval must also work without specifying an author

### Writing Samples (Yuan's own writing)
- **One graph** (`s1_writing_samples`) with register tags — not two separate graphs
- **Three registers:** `casual` | `semiformal` | `formal`
  - casual: tweets, tweet threads/replies, iMessage threads, stream of consciousness
  - semiformal: Medium articles, personal blog, thesis drafts, pre-compliance "Why We Invested"
  - formal: compliance-approved "Why We Invested" pieces
- **Primary extraction focus:** what makes Yuan's writing *his* (voice fingerprint, structural patterns, register variation)
- **Light extraction** of topic/domain (future content may differ from past topics)
- **AI assistance metadata:** binary `chatgpt_assisted: true|false`, default `false` unless explicitly stated by Yuan
- **Compliance metadata:** binary `compliance_approved: true|false`, default `false` unless explicitly stated
- **Metadata correction required:** current evidence tags `investment_memo` subtype incorrectly — Yuan did not share investment memos. Correct subtypes: `tweet`, `tweet_thread`, `tweet_reply`, `medium_article`, `imessage_thread`, `why_we_invested`, `stream_of_consciousness`, `thesis_draft`
- Extraction should account for AI-assistance and compliance-editing levels (weight unassisted/unedited samples higher for voice authenticity)

### Content Strategy
- **All entity types:** platform mechanics, hook patterns, distribution tactics, anti-patterns, audience psychology
- **Staleness/shelf-life tuning:** deferred to Twitter Radar epic (needs engagement analytics first)

### Sessions Main + ChatGPT History
- **No ontology changes.** Keep current generic extraction.

### Engineering Learnings
- **Expand entity types:** add `ToolApiBehavior`, `ArchitectureDecision`, `PerformancePattern`, `DependencyCompatibility` to existing `failure_pattern`, `review_pattern`, `security`, `success_pattern`

### Self-Audit (Operational Learnings)
- **Separate ontology** from engineering learnings (different domain, different consumers)
- **Weight MISSes higher** than HITs for entity creation priority

### Public Repo
- **Schema/contract documentation** with minimal examples (not opinionated worked examples)
- Custom ontology definitions live in private repo only; public repo shows the contract + how to define your own

---

## 3) Ontology Definitions

### 3.1 Content Inspiration — Short-Form (`s1_inspiration_short_form`)

**Extraction emphasis:** hooks, compression, punch, engagement triggers

| Entity Type | Description | Example |
|---|---|---|
| `RhetoricalMove` | A specific technique used in the piece | "cold open with absurd analogy", "one-liner callback" |
| `HookPattern` | First-line/opening strategy | "provocative question", "contrarian statement", "unexpected juxtaposition" |
| `VoiceQuality` | Tone/register characteristic | "irreverent-casual", "deadpan-authoritative", "erudite-compressed" |
| `EngagementSignal` | What made it resonate/go viral | "ratio bait", "tribal identity signal", "counterintuitive data point" |
| `AuthorStyle` | The writer's broader style archetype | "Tyler Cowen — compressed erudition", "Balaji — thread-as-essay" |
| `CompressionTechnique` | How complex ideas are compressed | "metaphor substitution", "assumed shared context", "implication over statement" |

| Relationship Type | Meaning |
|---|---|
| `USES_MOVE` | Piece → RhetoricalMove |
| `OPENS_WITH` | Piece → HookPattern |
| `EXHIBITS` | Piece → VoiceQuality |
| `TRIGGERED` | Piece → EngagementSignal |
| `AUTHORED_BY` | Piece → AuthorStyle |
| `COMPRESSES_VIA` | Piece → CompressionTechnique |

### 3.2 Content Inspiration — Long-Form (`s1_inspiration_long_form`)

**Extraction emphasis:** argument structure, narrative arc, transitions, sustained voice

| Entity Type | Description | Example |
|---|---|---|
| `RhetoricalMove` | Technique (long-form variants) | "extended analogy across sections", "dialectical pivot", "callback close" |
| `NarrativeArc` | Overall structural pattern | "thesis-antithesis-synthesis", "chronological buildup to reveal", "nested stories" |
| `ArgumentStructure` | How the argument is built | "evidence stacking", "concession-then-reframe", "first-principles derivation" |
| `VoiceQuality` | Tone/register (long-form variants) | "essayistic-intimate", "academic-accessible", "storyteller-with-data" |
| `TransitionTechnique` | How sections/ideas connect | "rhetorical question bridge", "anecdote pivot", "tempo shift" |
| `EngagementSignal` | What made it resonate | "shareable thesis statement", "quotable paragraph", "unexpected conclusion" |
| `AuthorStyle` | Writer's archetype | "Arthur Hayes — storytelling + macro thesis", "Paul Graham — simple clarity" |
| `OpeningStrategy` | Long-form hook approach | "in-medias-res anecdote", "bold claim first sentence", "scene-setting" |

| Relationship Type | Meaning |
|---|---|
| `USES_MOVE` | Piece → RhetoricalMove |
| `STRUCTURED_AS` | Piece → NarrativeArc / ArgumentStructure |
| `EXHIBITS` | Piece → VoiceQuality |
| `TRANSITIONS_VIA` | Section → TransitionTechnique |
| `AUTHORED_BY` | Piece → AuthorStyle |
| `OPENS_WITH` | Piece → OpeningStrategy |
| `TRIGGERED` | Piece → EngagementSignal |

### 3.3 Writing Samples — Yuan's Own (`s1_writing_samples`)

**Extraction emphasis:** voice fingerprint, structural patterns, register variation. Light on topic.

| Entity Type | Description | Example |
|---|---|---|
| `VoiceFingerprint` | Characteristic writing move unique to Yuan | "Machiavelli-meets-f-bomb register shift", "parenthetical self-deprecation" |
| `StructuralPattern` | How Yuan builds/organizes | "hook → context → thesis → evidence → callback", "list-with-commentary" |
| `RegisterMarker` | Signal of register level | "slang density", "citation formality", "sentence length variation" |
| `SignaturePhrase` | Recurring distinctive language | specific phrases, characteristic word choices, verbal tics |
| `ToneShift` | Where/how register changes within a piece | "opens casual, pivots to technical, closes with joke" |

| Relationship Type | Meaning |
|---|---|
| `DEMONSTRATES` | Sample → VoiceFingerprint |
| `FOLLOWS` | Sample → StructuralPattern |
| `SIGNALS` | Sample → RegisterMarker |
| `CONTAINS` | Sample → SignaturePhrase |
| `SHIFTS_VIA` | Sample → ToneShift |

**Episode metadata (passed through, not extracted):**
- `register`: `casual` | `semiformal` | `formal`
- `chatgpt_assisted`: `true` | `false`
- `compliance_approved`: `true` | `false`
- `subtype`: `tweet` | `tweet_thread` | `tweet_reply` | `medium_article` | `imessage_thread` | `why_we_invested` | `stream_of_consciousness` | `thesis_draft`

**Extraction weighting:** Episodes with `chatgpt_assisted=false` and `compliance_approved=false` are weighted higher for voice authenticity extraction.

### 3.4 Content Strategy (`s1_content_strategy`)

**Extraction emphasis:** actionable platform knowledge, engagement mechanics

| Entity Type | Description | Example |
|---|---|---|
| `PlatformMechanic` | How a platform's algo/system works | "Twitter rewards early engagement velocity", "quote tweets boost less than replies" |
| `HookTemplate` | Proven first-line formula | "Thread: Here's what nobody tells you about X", "'Unpopular opinion:' framing" |
| `DistributionTactic` | Strategy for reach/engagement | "reply to high-follower accounts within 15 min", "thread with standalone first tweet" |
| `AntiPattern` | What kills engagement or looks bad | "hashtag spam", "self-promotional tone", "engagement bait without substance" |
| `AudiencePsychology` | What specific audiences respond to | "crypto audience rewards contrarian takes with evidence", "VC audience shares 'builder insight' content" |

| Relationship Type | Meaning |
|---|---|
| `EXPLOITS` | Tactic → PlatformMechanic |
| `TARGETS` | Tactic → AudiencePsychology |
| `COUNTERS` | AntiPattern → what it prevents |
| `USES_HOOK` | Tactic → HookTemplate |
| `SUPERSEDED_BY` | Old mechanic → new mechanic (staleness chain) |

### 3.5 Engineering Learnings (`engineering_learnings`)

**Extraction emphasis:** what worked, what failed, why, and how to prevent recurrence

| Entity Type | Description | Example |
|---|---|---|
| `FailurePattern` | What went wrong and why | "FalkorDB GRAPH.QUERY on empty graph returns no rows, not error" |
| `SuccessPattern` | What worked well | "nohup + log redirect prevents CLR output-buffer SIGKILL" |
| `ReviewPattern` | What reviewers consistently catch | "missing error handling on async graph operations" |
| `SecurityFinding` | Security-relevant discovery | "content scanner missed canary phrase injection pattern" |
| `ToolApiBehavior` | Specific tool/API quirk or gotcha | "gh issue comment with backticks in zsh requires heredoc" |
| `ArchitectureDecision` | Why we chose X over Y | "chose per-group FalkorDB graphs over namespace-only segregation" |
| `PerformancePattern` | What's slow/fast and why | "long-form extraction ~3 min/episode due to dense chunk size" |
| `DependencyCompatibility` | Library/version interaction | "FalkorDB v4.16.3 doesn't support EXISTS subquery" |

| Relationship Type | Meaning |
|---|---|
| `CAUSED_BY` | Failure → root cause |
| `FIXED_BY` | Failure → remediation/success pattern |
| `PREVENTS` | Pattern → future failure |
| `AFFECTS` | ToolApiBehavior → specific tool/service |
| `SUPERSEDES` | New learning → stale old learning |
| `DECIDED_BECAUSE` | ArchitectureDecision → tradeoff/rationale |
| `CONSTRAINS` | DependencyCompatibility → what it limits |

### 3.6 Self-Audit / Operational Learnings (`learning_self_audit`)

**Extraction emphasis:** what went wrong in assistant operations, why, and how to prevent it

| Entity Type | Description | Example |
|---|---|---|
| `ToolBehavior` | Tool-specific operational quirk | "TFL releases 2 months ahead, not 1 — Feb drop covers through March" |
| `OperationalRule` | Process/policy that should be followed | "Always use update-with-hotfixes.sh for OpenClaw updates" |
| `SecurityGap` | Discovered security weakness in operations | "Content scanner missed canary phrase injection" |
| `PreferenceMiss` | User preference was wrong or missed | "Yuan prefers pushy instant pings, not daily digests" |
| `Remediation` | The fix applied to prevent recurrence | "Added canary pattern to content-scanner rules" |

| Relationship Type | Meaning |
|---|---|
| `CAUSED_BY` | Gap/Miss → root cause |
| `FIXED_BY` | Gap/Miss → Remediation |
| `PREVENTS` | Rule/Remediation → future gap |
| `AFFECTS` | ToolBehavior → specific tool/service |
| `SUPERSEDES` | New learning → old stale learning |

**Extraction weighting:** MISS entries are weighted higher than HIT entries for entity creation.

---

## 4) Implementation

### Slice A: Ontology Config + Extraction Routing

**Owned Paths (private repo: `yhl999/graphiti-openclaw`):**
- `config/extraction_ontologies.yaml` (new — per-group ontology definitions)
- `mcp_server/src/graphiti_mcp_server.py` (modify — resolve ontology per group_id)
- `mcp_server/src/services/queue_service.py` (modify — pass ontology to extraction)

**What it does:**
1. Add `config/extraction_ontologies.yaml` mapping each `group_id` to its entity types, relationship types, and extraction emphasis prompt.
2. Modify `GraphitiService` to load ontology config and resolve per-group entity types (extending the per-group client routing from PR #50).
3. Modify `QueueService.add_episode()` to accept and pass per-group entity types instead of using the global default.
4. Keep sessions_main and chatgpt_history on the existing generic ontology (no change).

### Slice B: Writing Samples Metadata Correction

**Owned Paths (private repo):**
- `scripts/fix_writing_samples_metadata.py` (new — one-time metadata correction)
- Evidence files as needed

**What it does:**
1. Fix incorrect `investment_memo` subtype tags → correct subtypes per Yuan's actual inputs.
2. Add `register`, `chatgpt_assisted`, `compliance_approved` metadata fields based on Yuan's original notes + defaults (false unless explicitly stated).
3. This is a one-time data correction script, not ongoing pipeline logic.

### Slice C: Dynamic Content Pack Materialization

**Owned Paths (private repo):**
- `workflows/content_voice_style.pack.yaml` (modify — replace synthetic placeholder)
- `workflows/content_writing_samples.pack.yaml` (modify — replace synthetic placeholder)
- `scripts/runtime_pack_router.py` (modify — add graph-backed materializers)
- `config/runtime_pack_registry.json` (modify — add materialization config)

**What it does:**
1. Add materializer functions for `content_voice_style` and `content_writing_samples` packs that query real graph data.
2. Query `s1_writing_samples` for voice fingerprints, structural patterns, register markers filtered by register tag.
3. Query `s1_inspiration_short_form` / `s1_inspiration_long_form` for rhetorical moves and author styles (as supplementary context).
4. Fallback: if graph coverage is below a minimum threshold, fall back to current static guidance.
5. Remove `STATUS: SYNTHETIC PLACEHOLDER` markers once graph-backed output is validated.

### Slice D: Re-extraction with New Ontologies

**What it does:**
1. After Slice A lands, re-run extraction for the 4 content graphs + engineering + self-audit using lane-specific ontologies.
2. This requires clearing existing generic entities from those graphs and re-extracting from the same episodes.
3. Sessions_main and chatgpt_history are **not re-extracted** (no ontology change).

**Risk:** Re-extraction is expensive (same ~3 min/episode for long-form). Plan: run as background batch, prioritize short-form + writing samples first (smallest, fastest wins), then content strategy, then long-form.

### Slice E: Public Repo Documentation

**Owned Paths (public repo: `yhl999/graphiti-openclaw`):**
- `docs/custom-ontologies.md` (new — schema/contract for defining custom ontologies)
- `README.md` (modify — add pointer to custom ontologies doc)
- `config/extraction_ontologies.example.yaml` (new — example config showing the contract)

**What it does:**
1. Document the `extraction_ontologies.yaml` schema: how to map `group_id` → entity types + relationship types + extraction emphasis.
2. Show a minimal example (not our actual production ontologies — those stay private).
3. Reference existing example workflow/content packs as complementary docs.
4. Keep it concise: schema definition, one example, pointer to workflow pack docs.

---

## 5) Definition of Done

- [ ] `config/extraction_ontologies.yaml` exists with definitions for all 6 lanes
- [ ] MCP extraction routes by `group_id` to lane-specific entity types
- [ ] Writing samples metadata corrected (subtypes, register, AI-assistance, compliance flags)
- [ ] `content_voice_style` pack is graph-backed with register-aware queries
- [ ] `content_writing_samples` pack is graph-backed with register-aware queries
- [ ] `SYNTHETIC PLACEHOLDER` markers removed from both packs
- [ ] Engineering learnings ontology expanded with 4 new entity types
- [ ] Self-audit has its own dedicated ontology (separate from engineering)
- [ ] 4 content graphs + engineering + self-audit re-extracted with new ontologies
- [ ] Contamination check: 0 misplaced episodes, 0 foreign entities post re-extraction
- [ ] Public repo has `docs/custom-ontologies.md` + example config
- [ ] Sessions_main and chatgpt_history untouched (no re-extraction)

## 6) Validation Commands

```bash
set -euo pipefail
cd /Users/archibald/clawd/projects/graphiti-openclaw-runtime

# Ontology config validates
python3 -c "import yaml; yaml.safe_load(open('config/extraction_ontologies.yaml'))"

# MCP compiles
python3 -m py_compile mcp_server/src/graphiti_mcp_server.py
python3 -m py_compile mcp_server/src/services/queue_service.py

# Contamination check
python3 scripts/scan_misplacements_all_graphs.py -o /tmp/misplacements.json

# Content packs no longer synthetic
! grep -q "SYNTHETIC PLACEHOLDER" workflows/content_voice_style.pack.yaml
! grep -q "SYNTHETIC PLACEHOLDER" workflows/content_writing_samples.pack.yaml
```

---

## 7) Execution Order

1. **Slice A** (ontology config + routing) — prerequisite for everything
2. **Slice B** (metadata correction) — can parallel with A
3. **Slice D** (re-extraction) — after A lands
4. **Slice C** (dynamic packs) — after D has enough coverage
5. **Slice E** (public docs) — can parallel with C/D

Estimated: Slice A is ~half day of implementation; Slice D (re-extraction) is wall-clock heavy but unattended; Slice C is ~half day; Slice E is a few hours.

---

## 8) What Changed From Old Decisions

| Old Decision | Status | Change |
|---|---|---|
| Content epic marked BLOCKED on writing samples | **Unblocked** — samples were ingested; extraction just needs ontology | Proceed |
| Engineering wiring marked "manual, Yuan-in-loop" | **Done** — Yuan confirmed engineering injection/ingestion loops are wired | Close `task-engineering-wiring-antfarm-clr-injection` |
| Synthetic content packs as placeholder | **Replace** — graph has enough data for initial materialization | This PRD |
| Single global entity_types for all groups | **Replace** — per-lane ontologies | This PRD |
| `investment_memo` subtype in writing samples | **Incorrect** — Yuan did not share investment memos; fix metadata | Slice B |
| Content creation epic on Kanban | **Missing** — folded into later tracks; resurface as this PRD's parent reference | Track via `task-lane-specific-ontologies-v0` |
