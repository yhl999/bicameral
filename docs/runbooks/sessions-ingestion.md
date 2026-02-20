# Sessions Ingestion Runbook

## Overview

This runbook covers the full lifecycle of Graphiti extraction: batch ingestion for initial/re-extraction, steady-state configuration for ongoing operations, and the post-processing pipeline that keeps the graph clean.

**Default backend (2026-02-19): Neo4j.** FalkorDB is legacy-only.

**If you're an agent with no context:** read this end-to-end. It contains hard-won operational knowledge from running a 7,000+ episode extraction across 9 graph groups on a Mac mini M-series.

---

## Table of Contents

1. [Architecture: How Extraction Works](#architecture-how-extraction-works)
2. [Graph Groups & Ontologies](#graph-groups--ontologies)
3. [Batch Ingestion (Initial / Re-Extraction)](#batch-ingestion)
4. [Steady-State Configuration](#steady-state-configuration)
5. [Post-Processing Pipeline](#post-processing-pipeline)
6. [Adding New Data Sources](#adding-new-data-sources)
7. [Troubleshooting & Failure Modes](#troubleshooting--failure-modes)
8. [Operational Learnings](#operational-learnings)

---

## Architecture: How Extraction Works

```
Evidence (JSON files)
    ↓  enqueue via ingest scripts
MCP Server (add_memory endpoint)
    ↓  per-group async queue (QueueService)
Graphiti Core (add_episode)
    ↓  LLM extraction → entity resolution → embedding → graph write
Neo4j (nodes, relationships, episodic timeline)
```

### Key Components

| Component | Role | Config |
|-----------|------|--------|
| **Ingest scripts** | Parse source data → evidence JSON → enqueue to MCP | `scripts/mcp_ingest_sessions.py`, `scripts/ingest_content_groups.py`, etc. |
| **MCP Server** | Hosts `add_memory` tool, manages per-group queues | `mcp_server/main.py`, port 8000 (default) |
| **QueueService** | Async episode processing with per-group serial queues | `mcp_server/src/services/queue_service.py` |
| **OntologyRegistry** | Per-group entity type resolution | `mcp_server/config/extraction_ontologies.yaml` |
| **Graphiti Core** | LLM-powered entity/relationship extraction + graph writes | `graphiti_core/` |
| **Neo4j** | Graph database (single DB, group_id property scoping) | `bolt://localhost:7687` |

### Per-Group Queue Isolation

The MCP server maintains **separate async queues per `group_id`**. This means:

- Episodes within the same group are processed according to `GRAPHITI_QUEUE_CONCURRENCY` (serial by default)
- Different groups process in parallel (bounded by `SEMAPHORE_LIMIT`)
- Ontology profiles are resolved per-call, not per-shard — a single MCP instance handles all groups correctly
- No risk of cross-group contamination: `group_id` is stamped on every node/edge at write time

### Evidence → Episode Flow

1. Ingest script reads evidence JSON (or generates from source DB/files)
2. Large evidence is **sub-chunked** at enqueue time (>10k chars → split into `:p0`, `:p1`, ... parts)
3. Each chunk is sent to MCP `add_memory` with: `name`, `episode_body`, `group_id`, `source_description`
4. MCP queues it in the group's async queue
5. Graphiti Core processes: LLM entity extraction → entity resolution (dedup against existing graph) → embedding → Neo4j write
6. Each episode creates: 1 Episodic node + N Entity nodes + M Relationship edges + episodic edges

### Important: Workers vs MCP Drain

Ingest worker processes exit 0 when **enqueuing** is done, NOT when extraction is complete. The MCP server drains queued episodes asynchronously. To know when extraction is truly finished, **poll the Neo4j episode count** — not the worker exit state.

---

## Graph Groups & Ontologies

### Active Groups

| Group ID | Source | Ontology | Ingest Script | ~Episodes |
|----------|--------|----------|---------------|-----------|
| `s1_sessions_main` | Session transcripts | Custom (9 entity types) | `mcp_ingest_sessions.py` | ~5,000+ |
| `s1_chatgpt_history` | ChatGPT export | Default | `mcp_ingest_sessions.py` | ~543 |
| `s1_memory_day1` | `memory/*.md` files | Default | `mcp_ingest_sessions.py` | ~604 |
| `s1_inspiration_long_form` | Long-form writing samples | Custom | `ingest_content_groups.py` | ~491 |
| `s1_inspiration_short_form` | Short-form/tweet samples | Custom | `ingest_content_groups.py` | ~190 |
| `s1_writing_samples` | Writing style examples | Custom | `ingest_content_groups.py` | ~48 |
| `s1_content_strategy` | Strategy docs | Custom | `ingest_content_groups.py` | ~6 |
| `engineering_learnings` | CLR compound notes | Custom | `ingest_compound_notes.py` | ~80+ |
| `learning_self_audit` | Nightly self-audit | Custom | `mcp_ingest_self_audit.py` | ~22+ |
| `s1_curated_refs` | MEMORY.md, preferences.md, etc. | Default | `curated_snapshot_ingest.py` | ~4 |

### Ontology Configuration

Custom entity types are defined in `mcp_server/config/extraction_ontologies.yaml`. Groups without an explicit ontology entry fall back to the global default entity types from `config.yaml`.

The ontology resolver is called **per-episode** at extraction time (not per-shard), so a single MCP instance correctly handles all groups with their respective ontologies.

### ChatGPT History: Constantine Filtering

The ChatGPT export (`conversations.json`) contains conversations from a shared account. The parser (`ingest/parse_chatgpt.py`) filters out contaminated conversations:

- Finds the conversation titled "Leadership: Follower Focus" as the cutoff
- Excludes ALL conversations with `create_time ≤ cutoff`, except 3 allowlisted titles
- Generates `evidence/chatgpt/all_evidence.json` + `evidence/chatgpt/filter_report.json`

Always verify the filter report after regenerating evidence.

---

## Batch Ingestion

Use batch mode for initial extraction, re-extraction after provider/embedding model changes, or disaster recovery.

### When You Need Batch Mode

- Switching embedding models (dimension change makes old vectors incompatible → full re-extract)
- Switching LLM providers (different extraction quality → may want fresh graph)
- Neo4j database wipe/recovery
- First-time setup

### Pre-Flight Checklist

1. **Wipe Neo4j** (if re-extracting): `MATCH (n) DETACH DELETE n` — all groups share one DB
2. **Reset ingest registry**: `DELETE FROM extraction_tracking WHERE group_id = '<group>'`
3. **Reset cursors** (for cursor-based scripts):
   - Self-audit: `DELETE FROM kv WHERE key LIKE 'self_audit%'` in `state/registry.db`
   - Curated refs: `DELETE FROM curated_files` in `state/ingest_registry.db`
4. **Verify evidence files exist** for all groups
5. **Verify LLM provider** has sufficient credits/quota
6. **Verify Ollama** is running with the correct embedding model loaded

### Batch Configuration: Maximize Throughput

```bash
# Launch 4 MCP shards for parallel extraction
SEMAPHORE_LIMIT=20          # 20 concurrent extractions per shard
GRAPHITI_QUEUE_CONCURRENCY=20  # parallel within each group (batch only!)
GRAPHITI_MAX_EPISODE_BODY_CHARS=12000

for port in 8000 8001 8002 8003; do
  SEMAPHORE_LIMIT=20 \
  ./start-graphiti-mcp-openrouter.sh --port $port &
done
```

### Launching Batch Workers

```bash
# sessions_main: 20 workers across 4 shards
for shard in $(seq 0 19); do
  port=$((8000 + shard / 5))
  nohup bash tools/falkordb/extraction-worker.sh sessions \
    --mcp-url http://localhost:${port}/mcp \
    --group-id s1_sessions_main \
    --shards 20 --shard-index $shard \
    --force \
    > logs/graphiti/worker-${shard}.log 2>&1 &
done

# chatgpt_history
nohup bash tools/falkordb/extraction-worker.sh sessions \
  --mcp-url http://localhost:8002/mcp \
  --group-id s1_chatgpt_history \
  --evidence evidence/chatgpt/all_evidence.json \
  --force \
  > logs/graphiti/chatgpt-history.log 2>&1 &

# memory_day1 (generate evidence first)
python3 ingest/parse_memory.py --output evidence/memory --memory-dir ~/clawd/memory
nohup bash tools/falkordb/extraction-worker.sh sessions \
  --mcp-url http://localhost:8003/mcp \
  --group-id s1_memory_day1 \
  --evidence evidence/memory/memory/all_memory_evidence.json \
  --force \
  > logs/graphiti/memory-day1.log 2>&1 &

# content_groups (sequential with drain-waiting, needs --backend neo4j)
nohup python3 scripts/ingest_content_groups.py \
  --backend neo4j --mcp-url http://localhost:8001/mcp \
  --force --sleep 0.1 --poll 15 --stable-checks 15 --max-wait 7200 \
  > logs/graphiti/content-groups.log 2>&1 &

# engineering_learnings
nohup python3 scripts/ingest_compound_notes.py \
  --mcp-url http://localhost:8000/mcp \
  > logs/graphiti/engineering-learnings.log 2>&1 &

# self_audit (reset cursor first)
sqlite3 state/registry.db "DELETE FROM kv WHERE key LIKE 'self_audit%';"
nohup python3 scripts/mcp_ingest_self_audit.py \
  --mcp-url http://localhost:8001/mcp \
  --group-id learning_self_audit \
  --input ~/clawd/memory/self-audit.jsonl \
  > logs/graphiti/self-audit.log 2>&1 &

# curated_refs (reset hash state first)
sqlite3 state/ingest_registry.db "DELETE FROM curated_files;"
nohup python3 scripts/curated_snapshot_ingest.py \
  --mcp-url http://localhost:8002/mcp \
  > logs/graphiti/curated-refs.log 2>&1 &
```

### Monitoring Batch Progress

```bash
# Episode counts by group
curl -s -u neo4j:"$NEO4J_PASSWORD" http://localhost:7474/db/neo4j/tx/commit \
  -H "Content-Type: application/json" \
  -d '{"statements":[{"statement":"MATCH (e:Episodic) RETURN e.group_id as gid, count(e) as cnt ORDER BY cnt DESC"}]}' \
  | python3 -c "import sys,json; [print(f'{r[\"row\"][0]}: {r[\"row\"][1]}') for r in json.load(sys.stdin)['results'][0]['data']]"

# Throughput (last 5 min)
# ... same query with: WHERE e.created_at > datetime() - duration({minutes: 5})

# System load
uptime

# OpenRouter spend
curl -s https://openrouter.ai/api/v1/auth/key \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['usage_daily'])"
```

### Batch Throughput Reference (Mac mini M-series, 24GB RAM)

These numbers are from a real 9-group batch extraction in Feb 2026:

| Configuration | Throughput | Notes |
|---------------|-----------|-------|
| 4 shards × SEMAPHORE=20, 1 group | ~2,340 episodes/hr | Peak, small graph (<2k nodes) |
| 4 shards × SEMAPHORE=20, 9 groups | ~1,488 episodes/hr | Larger graph (12k+ nodes), Neo4j entity resolution scaling |
| Single shard × SEMAPHORE=10 | ~500 episodes/hr | Conservative, good dedup quality |

**Why throughput degrades with graph size:** Entity resolution during `add_episode` performs similarity searches against existing nodes. More nodes = more comparisons per episode. This is O(n) per extraction. A graph with 12k nodes is measurably slower per episode than one with 2k nodes.

**Bottleneck hierarchy (Mac mini):**
1. **Ollama embeddings** — 476% CPU (5 cores) for embeddinggemma-300M under load
2. **Neo4j entity resolution** — scales from 23% → 105% CPU as graph grows
3. **OpenRouter LLM latency** — ~2s round-trip per API call, ~15 calls per episode
4. **RAM** — Ollama (6.7GB) + Neo4j (2.9GB) + MCP shards (2GB) ≈ 12GB minimum

---

## Steady-State Configuration

After batch ingestion completes, switch to a configuration that prioritizes **extraction quality over throughput**.

### Why Quality > Speed for Steady-State

- **Serial per-group processing** means each episode sees the latest graph state → entity resolution catches duplicates in real-time (instead of creating 128 copies of "Archibald")
- **Timeline integrity** is maintained when episodes are processed in chronological order within each group
- **Incremental volume is low** — ~50-100 new episodes/day for sessions_main, a handful for other groups — serial handles this in minutes
- **Multiple groups process in parallel** — the semaphore allows N groups to extract simultaneously

### Steady-State MCP Config

```bash
# Single MCP shard (port 8000)
SEMAPHORE_LIMIT=8              # max 8 groups extracting simultaneously
GRAPHITI_QUEUE_CONCURRENCY=1   # SERIAL within each group (key for dedup + timeline)
GRAPHITI_MAX_EPISODE_BODY_CHARS=12000
```

**Why `CONCURRENCY=1`:** Each episode's entity resolution compares against the current graph state. With concurrency=1, episode N+1 sees episode N's entities. With concurrency=20, episodes N+1 through N+20 all see the same stale snapshot → 20x duplicate entities.

**Why `SEMAPHORE_LIMIT=8`:** This is the total concurrent extraction slots across all groups. With 10+ active groups, 8 means most groups can process simultaneously. If you add more groups, bump to 10-12.

### Steady-State Cron Schedule

| Cron | Script | Frequency | Notes |
|------|--------|-----------|-------|
| sessions_main | `mcp_ingest_sessions.py --incremental --group-id s1_sessions_main` | Every 30 min | Delta since last watermark + overlap window |
| engineering_learnings | `ingest_compound_notes.py` | After CLR runs (or hourly) | New compound notes |
| learning_self_audit | `mcp_ingest_self_audit.py` | Daily (nightly) | From `memory/self-audit.jsonl` |
| curated_refs | `curated_snapshot_ingest.py` | Daily | SHA-256 hash-gated — skips if files unchanged |
| dedupe_nodes | `dedupe_nodes.py --backend neo4j` | Daily (2 AM) | Merge duplicate entities within each group |
| repair_timeline | `repair_timeline.py --backend neo4j` | Daily (after dedup) | Rebuild NEXT_EPISODE chains per group |
| import_candidates | `import_graphiti_candidates.py --backend neo4j` | Weekly (or on-demand) | Promote graph facts → candidates.db |

### Switching from Batch to Steady-State

1. **Verify batch is complete** — poll Neo4j episode counts against expected totals
2. **Stop all batch shards and workers**
3. **Run post-processing** (see next section)
4. **Start single production shard** with steady-state config
5. **Update launchd plist** for the production MCP service
6. **Enable cron jobs** for incremental ingest + daily maintenance

---

## Post-Processing Pipeline

Run these **after batch ingestion completes** (or periodically during steady-state).

### 1. Deduplicate Entities

High-concurrency batch extraction creates duplicate entity nodes (e.g., "Archibald" × 128).

```bash
# Dry run first
python3 scripts/dedupe_nodes.py --backend neo4j --group-id s1_sessions_main --dry-run

# Execute (destructive — merges duplicate nodes)
python3 scripts/dedupe_nodes.py --backend neo4j --group-id s1_sessions_main --confirm-destructive

# Run for all groups
for gid in s1_sessions_main s1_chatgpt_history s1_memory_day1 engineering_learnings \
           s1_inspiration_long_form s1_inspiration_short_form s1_writing_samples \
           s1_content_strategy learning_self_audit s1_curated_refs; do
  python3 scripts/dedupe_nodes.py --backend neo4j --group-id $gid --confirm-destructive
done
```

### 2. Repair Timeline

Batch extraction creates isolated episodes (0% NEXT_EPISODE linkage). The repair script rebuilds chronological chains.

```bash
python3 scripts/repair_timeline.py --backend neo4j --group-id s1_sessions_main --confirm-destructive

# All groups
for gid in s1_sessions_main s1_chatgpt_history s1_memory_day1 engineering_learnings \
           s1_inspiration_long_form s1_inspiration_short_form s1_writing_samples \
           s1_content_strategy learning_self_audit s1_curated_refs; do
  python3 scripts/repair_timeline.py --backend neo4j --group-id $gid --confirm-destructive
done
```

### 3. Graph Health Check

```bash
# Contamination check (should be zero)
MATCH (a)-[r:RELATES_TO]->(b) WHERE a.group_id <> b.group_id RETURN count(r)

# Duplicate count per group
MATCH (n:Entity) WITH n.group_id as gid, toLower(n.name) as name, count(n) as cnt
WHERE cnt > 1 RETURN gid, count(name) as duped_names, sum(cnt) as total_dupes

# Timeline coverage
MATCH (e:Episodic)-[:NEXT_EPISODE]->() RETURN e.group_id as gid, count(*) as linked
```

---

## Adding New Data Sources

Each new data source needs exactly three things:

### 1. A `group_id`

Choose a descriptive, snake_case identifier: `s1_taste_wine`, `s1_taste_restaurants`, `s1_crm_people`, `s1_gtd_tasks`.

### 2. An Ontology (optional but recommended)

Add a block to `mcp_server/config/extraction_ontologies.yaml`:

```yaml
s1_taste_wine:
  extraction_emphasis: >-
    Focus on tasting notes, producer style, terroir, vintage quality,
    food pairings, and price-to-quality assessments.
  entity_types:
    - name: Wine
      description: "Specific wine (producer + cuvée + vintage)"
    - name: Producer
      description: "Winery or négociant"
    - name: Region
      description: "Appellation or geographic terroir"
    - name: TastingNote
      description: "Sensory descriptor or quality assessment"
  relationship_types:
    - name: PRODUCED_BY
    - name: FROM_REGION
    - name: PAIRS_WITH
```

Groups without an explicit ontology entry fall back to the global default entity types. Custom ontologies dramatically improve extraction quality for domain-specific data.

### 3. An Ingest Adapter Script

The pattern for any SQLite DB source:

```python
#!/usr/bin/env python3
"""Ingest <source> into Graphiti via MCP add_memory."""

import json, sqlite3, urllib.request
from uuid import uuid5, NAMESPACE_URL

GROUP_ID = "s1_taste_wine"
MCP_URL = "http://localhost:8000/mcp"

def add_memory(name, body, group_id, source_desc):
    """Send one episode to MCP."""
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "add_memory", "arguments": {
            "name": name, "episode_body": body,
            "group_id": group_id, "source_description": source_desc,
        }}
    }
    req = urllib.request.Request(MCP_URL, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=120)

def main():
    db = sqlite3.connect("/path/to/wine.db")
    rows = db.execute("SELECT id, name, notes, rating FROM wines").fetchall()
    for wine_id, name, notes, rating in rows:
        body = f"Wine: {name}\nNotes: {notes}\nRating: {rating}"
        # Deterministic UUID for idempotent re-runs
        uuid = str(uuid5(NAMESPACE_URL, f"wine:{wine_id}"))
        add_memory(name=name, body=body, group_id=GROUP_ID,
                   source_desc=f"wine_db:{wine_id}")
        print(f"Queued: {name}")

if __name__ == "__main__":
    main()
```

Key principles:
- **Deterministic UUIDs** from source IDs → idempotent re-runs
- **Sub-chunk large content** (>10k chars) before sending to MCP
- **Track cursor/watermark** for incremental ingest
- **Validate body size** against `GRAPHITI_MAX_EPISODE_BODY_CHARS` (default 12k)

### 4. Add to Cron Schedule

Add the new ingest to the cron schedule with appropriate frequency:

- **Taste DBs:** daily (data changes infrequently)
- **CRM:** hourly or on-interaction (captures meeting notes promptly)
- **GTD:** every 30 min (task status changes are time-sensitive)

---

## Troubleshooting & Failure Modes

### LLM Provider Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `402 Insufficient credits` | OpenRouter balance depleted | Top up at https://openrouter.ai/settings/credits |
| `404 No endpoints matching data policy` | ZDR (zero data retention) filtering out all providers | Disable ZDR at https://openrouter.ai/settings/privacy or switch model |
| `429 Rate limited` | Provider rate limit exceeded | Reduce `SEMAPHORE_LIMIT` |
| `context_length_exceeded` | Episode too large for LLM context | Reduce `--subchunk-size` or `GRAPHITI_MAX_EPISODE_BODY_CHARS` |

**OpenRouter ZDR note:** When enabled, OpenRouter only routes to providers that guarantee zero data retention. If the ZDR-compatible endpoint for your model goes down, ALL requests 404 — even though the model exists on other providers. Disable ZDR temporarily during batch extraction, re-enable after.

### Database Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `Max pending queries exceeded` | FalkorDB queue full | Raise `MAX_QUEUED_QUERIES` or reduce concurrency |
| `ServiceUnavailable` | Neo4j connection pool exhausted | Raise `max_connection_pool_size` in driver config |
| `episode_body too large` | Content exceeds `GRAPHITI_MAX_EPISODE_BODY_CHARS` | Sub-chunk upstream or raise the limit |

### Embedding Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `503 Service Unavailable` from Ollama | Ollama overloaded with concurrent requests | Reduce `SEMAPHORE_LIMIT` or wait (self-recovers) |
| Dimension mismatch | Old embeddings (e.g., 1024d) mixed with new (768d) | Full re-extraction required — wipe Neo4j + registry |

**Critical: Embedding model changes require full re-extraction.** You cannot mix embeddings from different models in the same graph. If you switch embedding models, wipe Neo4j completely and re-extract all groups.

### Stale Cursors / Registry State

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Done: 0/0 ingested` | Cursor thinks everything is already processed | Reset cursor (see Pre-Flight Checklist) |
| `changed=0 ingested=0` | Hash-gated script sees no file changes | Delete hash state from `curated_files` table |
| Workers skip all chunks | Registry marks chunks as already queued | Delete from `extraction_tracking` or use `--force` |

---

## Operational Learnings

Hard-won knowledge from production operation. Read these before modifying the pipeline.

### Throughput vs Quality Tradeoff

**Batch mode** (high concurrency): Optimized for speed. Creates significant duplicates (e.g., "Archibald" × 128) and zero timeline links. Requires post-processing (dedup + timeline repair). Use for initial/re-extraction only.

**Steady-state** (serial per-group): Optimized for quality. Entity resolution sees latest graph state → minimal duplicates. Timeline maintained naturally by chronological processing order. Use for ongoing operations.

**Never run batch-mode concurrency in steady-state.** The cleanup cost (dedup, timeline repair, wasted LLM calls on duplicate resolution) exceeds the time saved.

### FalkorDB → Neo4j Migration (Feb 2026)

FalkorDB's single-threaded write model couldn't handle >2 concurrent extraction pipelines without constant timeouts. Neo4j handles 80+ concurrent extractions without issue. The migration required:

- Full re-extraction (embedding dimension change: 1024d OpenAI → 768d Ollama)
- Dual-backend driver abstraction (`scripts/graph_driver.py`)
- All scripts updated for `--backend neo4j` flag
- FalkorDB still runs on port 6379 with legacy data (read-only reference)

### OpenAI → OpenRouter Switch (Feb 2026)

OpenAI quota exhaustion (BCAP key) forced a provider switch mid-extraction. OpenRouter was chosen for:
- OpenAI-compatible API (drop-in replacement)
- Crypto payments, instant top-up
- No hard quota wall (credits-based)
- Zero data retention option

**Gotcha:** OpenRouter's ZDR policy can block all requests if the ZDR-compatible provider for your model goes down. This is an OpenRouter-wide issue, not per-model.

### Ollama Embeddings (Local)

`embeddinggemma-300M` (768 dimensions) via Ollama provides:
- Zero cost, unlimited RPM
- 72-97ms latency per embedding
- Adequate quality for entity dedup/similarity (MTEB ~62 vs OpenAI's ~65)

**Resource warning:** Under high concurrency (80+ parallel extractions), Ollama consumes 5+ CPU cores and 6.7GB RAM for a 300M parameter model. This is due to concurrent inference contexts, not model size.

### Sub-Chunking is Essential

Without sub-chunking, large evidence (sessions can be 25k+ chars) causes:
1. `context_length_exceeded` errors from the LLM
2. Retry-with-shrink loops that waste API calls
3. Truncated content that loses information

Sub-chunking at enqueue time (default 10k chars) is deterministic, lossless, and idempotent. **Enable it for all groups**, not just sessions_main.

### Cross-Contamination: Not a Risk on Neo4j

FalkorDB used separate graph databases per group. Episodes could land in the wrong graph if the MCP shard's `--group-id` didn't match the episode's group_id. This was a real production incident.

Neo4j uses a single database with `group_id` property scoping. The shard's `--group-id` flag is only a fallback default — ingest scripts always pass `group_id` explicitly per-call. Cross-contamination is architecturally impossible as long as ingest scripts set `group_id` correctly.

### Graph Size Impacts Throughput

Entity resolution during `add_episode` performs similarity searches against existing nodes. Benchmarked on Mac mini M-series:

| Graph Size | Neo4j CPU | Approx Throughput |
|-----------|-----------|-------------------|
| 2k nodes | 23% | ~2,340 ep/hr |
| 6k nodes | 60% | ~1,800 ep/hr |
| 12k nodes | 105% | ~1,488 ep/hr |

This is expected and unavoidable — it's the cost of quality entity resolution. Post-extraction dedup (which merges duplicates) can reduce node count and improve subsequent extraction performance.
