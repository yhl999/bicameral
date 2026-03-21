#!/usr/bin/env python3
"""chatgpt_history Lock-Push Experiment Harness.

Tests multiple retrieval-shaping strategies on the current representative slice
to push MRR from 0.525 toward the 0.75 lock bar.

Phase 1: Retrieve full merged+reranked pools per query (one-time)
Phase 2: Apply strategy variants to select top-5 per query
Phase 3: Judge any queries where top-5 changed vs baseline
Phase 4: Compute metrics + comparison

Strategies:
  A: baseline — current system (no changes)
  B: text-dedup — deduplicate facts with identical fact text
  C: text-dedup + synth-cap-2 — max 2 synthesized facts in top-5
  D: text-dedup + synth-cap-2 + first-order-boost — +0.1 for first-order facts
  E: text-dedup + synth-cap-3 — more permissive synth cap

Must run from CWD without graphiti_core/ directory:
  cd /tmp && /Users/archibald/clawd/projects/bicameral-runtime/.venv/bin/python3 \
    /Users/archibald/clawd/projects/bicameral/scripts/chatgpt_history_lock_push.py \
    --run-dir /path/to/run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
import urllib.request
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BATCH_DIR = Path("/Users/archibald/clawd/projects/bicameral-runtime/state/chatgpt_history_vnext_20260316_batch")
QUERY_BANK_PATH = BATCH_DIR / "query_bank.json"
BASELINE_METRICS_PATH = BATCH_DIR / "phase2b_metrics_constrained.json"
BASELINE_SCORED_PATH = BATCH_DIR / "phase2b_scored_results_constrained.json"
SYNTH_METRICS_PATH = Path("/Users/archibald/clawd/projects/bicameral-runtime/state/chatgpt_history_synthesis_pass_20260317/run/synth_eval_metrics.json")

REPO = Path("/Users/archibald/clawd/projects/bicameral-runtime")
sys.path.insert(0, str(REPO / "mcp_server" / "src"))

logger = logging.getLogger("lock_push")

# Retrieval settings
TEXT_SEARCH_WINDOW = 20
CENTER_NODE_WINDOW = 20
RERANK_TOPK = 20
FINAL_TOPK = 5
RERANKER_MODEL = "openai/gpt-4o-mini"
JUDGE_MODEL = "openai/gpt-4o-mini"

BUCKET_TYPE_PRIORS = {
    "bio": {"boost_terms": ["person", "personal", "background", "education", "university",
                            "school", "family", "language", "age", "passport", "immigration",
                            "location", "city", "born", "grew up", "workout", "physical",
                            "career", "history", "famous", "figure", "identify"], "boost_weight": 0.15},
    "rel": {"boost_terms": ["relationship", "dating", "friend", "friendship", "social",
                            "mentor", "colleague", "partner", "hinge", "raya", "app",
                            "sibling", "grandmother", "sister", "family"], "boost_weight": 0.15},
    "work": {"boost_terms": ["blockchain", "capital", "venture", "investment", "fund",
                             "portfolio", "deal", "crypto", "token", "protocol",
                             "professional", "company", "firm", "carry", "partner",
                             "eigenlayer", "BCAP", "organization"], "boost_weight": 0.15},
    "pref": {"boost_terms": ["preference", "like", "enjoy", "favorite", "taste",
                             "wine", "food", "restaurant", "film", "movie", "music",
                             "cooking", "travel", "hobby", "interest", "style",
                             "GLP", "semaglutide", "health", "luxury", "watch",
                             "fashion", "champagne", "book", "reading"], "boost_weight": 0.15},
    "heur": {"boost_terms": ["writing", "communication", "decision", "thinking",
                             "strategy", "pattern", "heuristic", "approach", "style",
                             "method", "process", "habit", "tendency", "belief",
                             "philosophy", "principle", "theory", "geopolitical",
                             "curate", "calibrate", "skepticism", "evaluate",
                             "navigate", "collaborate", "refine", "personalize"], "boost_weight": 0.15},
}

QUERY_CLASS = {
    "bio_01": "fragmented", "bio_02": "direct", "bio_03": "direct",
    "bio_04": "fragmented", "bio_05": "fragmented", "bio_06": "direct",
    "bio_07": "fragmented", "bio_08": "inferential", "bio_09": "direct",
    "rel_01": "fragmented", "rel_02": "direct", "rel_03": "direct",
    "rel_04": "direct", "rel_05": "fragmented", "rel_06": "direct",
    "rel_07": "fragmented", "rel_08": "fragmented",
    "work_01": "direct", "work_02": "fragmented", "work_03": "inferential",
    "work_04": "direct", "work_05": "fragmented", "work_06": "fragmented",
    "work_07": "inferential", "work_08": "fragmented", "work_09": "fragmented",
    "pref_01": "fragmented", "pref_02": "fragmented", "pref_03": "direct",
    "pref_04": "fragmented", "pref_05": "direct", "pref_05b": "fragmented",
    "pref_06": "direct", "pref_07": "direct", "pref_08": "direct",
    "pref_09": "fragmented",
    "heur_01": "inferential", "heur_02": "inferential", "heur_03": "inferential",
    "heur_04": "inferential", "heur_05": "fragmented", "heur_06": "direct",
    "heur_07": "inferential",
    "neg_01": "negative", "neg_02": "negative", "neg_03": "negative",
    "neg_04": "negative", "neg_05": "negative", "neg_06": "negative",
}

# ─── Strategy definitions ────────────────────────────────────────────────────

STRATEGIES = {
    "A_baseline": {
        "dedup_text": False,
        "synth_cap": None,  # no cap
        "first_order_boost": 0.0,
        "pref_boost_weight": 0.15,
    },
    "B_text_dedup": {
        "dedup_text": True,
        "synth_cap": None,
        "first_order_boost": 0.0,
        "pref_boost_weight": 0.15,
    },
    "C_dedup_synthcap2": {
        "dedup_text": True,
        "synth_cap": 2,
        "first_order_boost": 0.0,
        "pref_boost_weight": 0.15,
    },
    "D_dedup_synthcap2_foboost": {
        "dedup_text": True,
        "synth_cap": 2,
        "first_order_boost": 0.1,
        "pref_boost_weight": 0.15,
    },
    "E_dedup_synthcap3": {
        "dedup_text": True,
        "synth_cap": 3,
        "first_order_boost": 0.0,
        "pref_boost_weight": 0.15,
    },
}


def configure_env():
    os.environ["CONFIG_PATH"] = str(REPO / "mcp_server" / "config" / "config-docker-neo4j.yaml")
    os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    os.environ["NEO4J_DATABASE"] = "neo4j"
    neo4j_env = Path.home() / ".clawdbot" / "credentials" / "neo4j.env"
    if neo4j_env.exists():
        for line in neo4j_env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not openrouter_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")
    os.environ["OPENAI_API_URL"] = "https://openrouter.ai/api/v1"
    os.environ["OPENAI_API_KEY"] = openrouter_key
    os.environ["LLM__MODEL"] = "openai/gpt-4o-mini"
    os.environ["LLM__PROVIDERS__OPENAI__API_KEY"] = openrouter_key
    os.environ["LLM__PROVIDERS__OPENAI__API_URL"] = "https://openrouter.ai/api/v1"
    os.environ["EMBEDDER__PROVIDER"] = "openai"
    os.environ["EMBEDDER__MODEL"] = "embeddinggemma"
    os.environ["EMBEDDER__DIMENSIONS"] = "768"
    os.environ["EMBEDDER__PROVIDERS__OPENAI__API_KEY"] = "ollama"
    os.environ["EMBEDDER__PROVIDERS__OPENAI__API_URL"] = "http://localhost:11434/v1"


def _fact_text(f: Any) -> str:
    if isinstance(f, dict):
        return f.get("fact", f.get("name", f.get("content", str(f))))
    return str(f)


def _fact_enriched_text(f: Any) -> str:
    if not isinstance(f, dict):
        return str(f)
    parts = []
    fact = f.get("fact", "")
    if fact:
        parts.append(fact)
    name = f.get("name", "")
    if name and name != fact:
        parts.append(f"[relation: {name}]")
    return " | ".join(parts) if parts else str(f)


def _is_synthesized(f: Any) -> bool:
    if not isinstance(f, dict):
        return False
    attrs = f.get("attributes", {})
    if isinstance(attrs, dict) and attrs.get("is_synthesized"):
        return True
    # Also check if episodes is empty (synthesized facts have episodes=[])
    episodes = f.get("episodes", None)
    if episodes is not None and len(episodes) == 0:
        return True
    return False


def _type_aware_boost(fact: Any, bucket: str, boost_weight: float = 0.15) -> float:
    priors = BUCKET_TYPE_PRIORS.get(bucket)
    if not priors:
        return 0.0
    fact_text_lower = _fact_text(fact).lower()
    enriched_lower = _fact_enriched_text(fact).lower() if isinstance(fact, dict) else fact_text_lower
    matched = sum(1 for term in priors["boost_terms"] if term.lower() in fact_text_lower or term.lower() in enriched_lower)
    return boost_weight * math.sqrt(min(matched, 5)) if matched else 0.0


def _openrouter_call(prompt: str, system: str, api_key: str, model: str = JUDGE_MODEL) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return json.loads(data["choices"][0]["message"]["content"])
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + 1)
            else:
                return {"score": 0, "rationale": f"Error: {e}"}
    return {"score": 0, "rationale": "Exhausted retries"}


def _call_reranker_sync(query: str, facts: list[dict], api_key: str) -> list[tuple[int, float]]:
    if not facts:
        return []
    fact_list = [f"{i+1}. {_fact_enriched_text(f)}" for i, f in enumerate(facts)]
    prompt = f"""Score each fact's relevance to the query on a 0.0 to 1.0 scale.

Query: {query}

Facts:
{chr(10).join(fact_list)}

Return ONLY a JSON object mapping fact numbers to relevance scores.
Example: {{"1": 0.8, "2": 0.1, "3": 0.95}}

Score 0.9-1.0: Directly answers the query with specific, useful information.
Score 0.6-0.8: Partially relevant, right topic but missing specifics.
Score 0.3-0.5: Tangentially related, same general domain.
Score 0.0-0.2: Irrelevant or wrong domain entirely."""

    result = _openrouter_call(prompt, "You are a retrieval relevance scorer. Return only valid JSON.", api_key, RERANKER_MODEL)
    results = []
    for i in range(len(facts)):
        score = result.get(str(i + 1), result.get(i + 1, 0.5))
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.5
        results.append((i, score))
    results.sort(key=lambda x: -x[1])
    return results


# ─── Phase 1: Retrieval ────────────────────────────────────────────────────────

async def phase1_retrieve(group_id: str, run_dir: Path) -> list[dict]:
    """Retrieve full merged pools and rerank scores for all queries."""
    from config.schema import GraphitiConfig
    import graphiti_mcp_server as gms
    from graphiti_mcp_server import GraphitiService, search_memory_facts
    from neo4j import AsyncGraphDatabase

    queries = json.loads(QUERY_BANK_PATH.read_text())
    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    # Find Yuan UUIDs
    neo4j_creds = {}
    cred_path = Path.home() / ".clawdbot" / "credentials" / "neo4j.env"
    if cred_path.exists():
        for line in cred_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                neo4j_creds[k] = v

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = neo4j_creds.get("NEO4J_USER", os.environ.get("NEO4J_USER", "neo4j"))
    pw = neo4j_creds.get("NEO4J_PASSWORD", os.environ.get("NEO4J_PASSWORD", ""))

    async_driver = AsyncGraphDatabase.driver(uri, auth=(user, pw))
    try:
        async with async_driver.session() as neo_session:
            result = await neo_session.run(
                "MATCH (e:Entity) WHERE e.group_id = $gid AND (e.name CONTAINS 'Yuan Han Li' OR e.name = 'Yuan') "
                "RETURN e.uuid AS uuid ORDER BY size(e.name) DESC", gid=group_id)
            yuan_uuids = [r.data()["uuid"] async for r in result]
    finally:
        await async_driver.close()

    primary_yuan_uuid = yuan_uuids[0] if yuan_uuids else None
    logger.info(f"Yuan UUID: {primary_yuan_uuid}")

    # Init Graphiti
    cfg = GraphitiConfig()
    gms.config = cfg
    service = GraphitiService(cfg, semaphore_limit=1)
    await service.initialize()
    gms.graphiti_service = service

    pools = []
    for i, q in enumerate(queries):
        logger.info(f"[{i+1}/{len(queries)}] Retrieving {q['id']}: {q['query'][:60]}...")
        bucket = q["bucket"]

        try:
            # Text search
            text_res = await search_memory_facts(
                query=q["query"], group_ids=[group_id],
                search_mode="hybrid", max_facts=TEXT_SEARCH_WINDOW)
            text_facts = text_res.get("facts", []) if isinstance(text_res, dict) else []

            # Center-node search
            center_facts = []
            if bucket != "neg" and primary_yuan_uuid:
                try:
                    center_res = await search_memory_facts(
                        query=q["query"], group_ids=[group_id],
                        search_mode="hybrid", max_facts=CENTER_NODE_WINDOW,
                        center_node_uuid=primary_yuan_uuid)
                    center_facts = center_res.get("facts", []) if isinstance(center_res, dict) else []
                except Exception as e:
                    logger.warning(f"Center-node search failed for {q['id']}: {e}")

            # Merge and deduplicate by UUID
            seen_uuids = set()
            merged = []
            for f in text_facts + center_facts:
                uid = f.get("uuid", "") if isinstance(f, dict) else ""
                if uid and uid not in seen_uuids:
                    seen_uuids.add(uid)
                    merged.append(f)

            # Rerank the full pool (up to RERANK_TOPK)
            rerank_pool = merged[:RERANK_TOPK]
            rerank_scores = {}
            if bucket != "neg" and len(rerank_pool) > 1:
                raw_scores = _call_reranker_sync(q["query"], rerank_pool, api_key)
                for orig_idx, score in raw_scores:
                    fact_uuid = rerank_pool[orig_idx].get("uuid", f"idx_{orig_idx}")
                    rerank_scores[fact_uuid] = score
                    if isinstance(rerank_pool[orig_idx], dict):
                        rerank_pool[orig_idx]["_rerank_score"] = round(score, 4)

            pools.append({
                "query_id": q["id"],
                "bucket": bucket,
                "query": q["query"],
                "expected_sketch": q["expected_sketch"],
                "merged_facts": rerank_pool,  # full pool with rerank scores
                "total_merged": len(merged),
                "rerank_scores": rerank_scores,
            })
        except Exception as e:
            logger.error(f"Error retrieving {q['id']}: {e}")
            pools.append({
                "query_id": q["id"], "bucket": bucket,
                "query": q["query"], "expected_sketch": q.get("expected_sketch", ""),
                "merged_facts": [], "total_merged": 0, "rerank_scores": {},
                "error": str(e),
            })

        if i < len(queries) - 1:
            await asyncio.sleep(0.8)

    # Cleanup
    client = await service.get_client()
    await client.driver.close()

    # Save pools
    pool_path = run_dir / "retrieval_pools.json"
    pool_path.write_text(json.dumps(pools, indent=2, default=str))
    logger.info(f"Saved {len(pools)} retrieval pools to {pool_path}")
    return pools


# ─── Phase 2: Strategy application ────────────────────────────────────────────

def apply_strategy(pools: list[dict], strategy_name: str, strategy_cfg: dict) -> list[dict]:
    """Apply a strategy to the retrieval pools and return top-5 selections."""
    results = []
    for pool in pools:
        bucket = pool["bucket"]
        facts = deepcopy(pool["merged_facts"])

        if not facts:
            results.append({
                "query_id": pool["query_id"], "bucket": bucket,
                "query": pool["query"], "expected_sketch": pool["expected_sketch"],
                "facts": [], "fact_count": 0, "strategy": strategy_name,
            })
            continue

        # Step 1: Text deduplication
        if strategy_cfg["dedup_text"]:
            seen_texts = set()
            deduped = []
            for f in facts:
                text = _fact_text(f).strip()
                if text not in seen_texts:
                    seen_texts.add(text)
                    deduped.append(f)
            facts = deduped

        # Step 2: Score computation
        if bucket != "neg":
            pref_bw = strategy_cfg.get("pref_boost_weight", 0.15)
            fo_boost = strategy_cfg.get("first_order_boost", 0.0)

            for f in facts:
                if not isinstance(f, dict):
                    continue
                rerank = f.get("_rerank_score", 0.5)
                bw = pref_bw if bucket == "pref" else 0.15
                type_boost = _type_aware_boost(f, bucket, bw)
                # First-order boost: non-synthesized facts get a bonus
                fo = fo_boost if (fo_boost > 0 and not _is_synthesized(f)) else 0.0
                f["_combined_score"] = round(rerank + type_boost + fo, 4)

            # Sort by combined score
            facts.sort(key=lambda f: -(f.get("_combined_score", 0) if isinstance(f, dict) else 0))

        # Step 3: Synth cap enforcement
        synth_cap = strategy_cfg.get("synth_cap")
        if synth_cap is not None and bucket != "neg":
            selected = []
            synth_count = 0
            for f in facts:
                if _is_synthesized(f):
                    if synth_count < synth_cap:
                        selected.append(f)
                        synth_count += 1
                    # else: skip this synthesized fact
                else:
                    selected.append(f)
                if len(selected) >= FINAL_TOPK:
                    break
            facts = selected
        else:
            facts = facts[:FINAL_TOPK]

        # Clean internal scoring keys for output
        clean = []
        for f in facts[:FINAL_TOPK]:
            if isinstance(f, dict):
                clean.append({k: v for k, v in f.items() if not k.startswith("_")})
            else:
                clean.append(f)

        results.append({
            "query_id": pool["query_id"], "bucket": bucket,
            "query": pool["query"], "expected_sketch": pool["expected_sketch"],
            "facts": clean, "fact_count": len(clean), "strategy": strategy_name,
            "total_merged": pool["total_merged"],
        })

    return results


# ─── Phase 3: Judging ──────────────────────────────────────────────────────────

def judge_results(strategy_results: list[dict], api_key: str,
                  existing_judgments: dict[str, dict] | None = None) -> list[dict]:
    """Judge strategy results. Reuse existing judgments for identical top-5."""
    scored = []
    for r in strategy_results:
        qid = r["query_id"]
        bucket = r["bucket"]
        facts = r.get("facts", [])

        # Check if we can reuse existing judgment
        if existing_judgments and qid in existing_judgments:
            existing = existing_judgments[qid]
            existing_top_facts = existing.get("_top_fact_texts", [])
            current_top_facts = [_fact_text(f) for f in facts[:1]]
            if existing_top_facts == current_top_facts:
                scored.append(existing)
                continue

        if bucket == "neg":
            if not facts:
                scored.append({"query_id": qid, "bucket": bucket, "score": 0,
                               "rationale": "No results — correct for negative control.",
                               "is_negative": True, "false_positive": False,
                               "_top_fact_texts": []})
            else:
                facts_text = "\n".join([f"- {_fact_text(f)}" for f in facts[:3]])
                prompt = f"""This is a NEGATIVE CONTROL query — the retrieval lane should NOT have relevant results.

Query: {r['query']}
Retrieved results:
{facts_text}

Score false-positive severity:
0 = NOT relevant (correct), 1 = tangentially related, 2 = somewhat relevant (minor FP), 3 = directly answers (major FP)

Return ONLY: {{"score": <0-3>, "rationale": "<one sentence>"}}"""
                result = _openrouter_call(prompt, "You are a retrieval quality judge. Return only valid JSON.", api_key)
                scored.append({"query_id": qid, "bucket": bucket, "score": int(result.get("score", 0)),
                               "rationale": result.get("rationale", ""), "is_negative": True,
                               "false_positive": int(result.get("score", 0)) >= 2,
                               "_top_fact_texts": [_fact_text(f) for f in facts[:1]]})
        else:
            if not facts:
                scored.append({"query_id": qid, "bucket": bucket, "score": 0,
                               "rationale": "No results.", "is_negative": False, "mrr_rank": None,
                               "_top_fact_texts": []})
            else:
                top1_text = _fact_text(facts[0])
                prompt = f"""Score the retrieval result for the query.

Query: {r['query']}
Expected (sketch): {r['expected_sketch']}
Retrieved: {top1_text}

3 = Directly relevant, specific, useful. 2 = Partially relevant. 1 = Tangential. 0 = Irrelevant.

Return ONLY: {{"score": <0-3>, "rationale": "<one sentence>"}}"""
                result = _openrouter_call(prompt, "You are a retrieval quality judge. Return only valid JSON.", api_key)
                top1_score = int(result.get("score", 0))
                time.sleep(0.5)

                mrr_rank = None
                if top1_score >= 2:
                    mrr_rank = 1
                else:
                    for rank, f in enumerate(facts[1:FINAL_TOPK], start=2):
                        fp = f"""Score: Query: {r['query']}
Expected: {r['expected_sketch']}
Retrieved: {_fact_text(f)}
3=Direct, 2=Partial, 1=Tangential, 0=Irrelevant. Return ONLY: {{"score": <0-3>, "rationale": "..."}}"""
                        fr = _openrouter_call(fp, "Retrieval quality judge. JSON only.", api_key)
                        if int(fr.get("score", 0)) >= 2:
                            mrr_rank = rank
                            break
                        time.sleep(0.5)

                scored.append({"query_id": qid, "bucket": bucket, "score": top1_score,
                               "rationale": result.get("rationale", ""), "is_negative": False,
                               "mrr_rank": mrr_rank,
                               "_top_fact_texts": [_fact_text(f) for f in facts[:1]]})

        logger.info(f"  Judged {qid}: score={scored[-1].get('score', 'N/A')}")
        time.sleep(0.3)

    return scored


# ─── Phase 4: Metrics computation ─────────────────────────────────────────────

def compute_metrics(scored: list[dict]) -> dict:
    positive = [s for s in scored if not s.get("is_negative", False)]
    negative = [s for s in scored if s.get("is_negative", False)]
    scores = [s["score"] for s in positive]

    metrics = {
        "total_positive": len(positive),
        "total_negative": len(negative),
        "top1_rate": round(sum(1 for s in scores if s == 3) / len(positive), 3) if positive else 0,
        "top1_count": sum(1 for s in scores if s == 3),
        "relevant_rate": round(sum(1 for s in scores if s >= 2) / len(positive), 3) if positive else 0,
        "relevant_count": sum(1 for s in scores if s >= 2),
        "mrr": round(sum(1.0 / s["mrr_rank"] if s.get("mrr_rank") else 0.0 for s in positive) / len(positive), 3) if positive else 0,
        "avg_score": round(sum(scores) / len(scores), 2) if scores else 0,
        "false_positives": sum(1 for s in negative if s.get("false_positive", False)),
    }

    # Per-bucket
    buckets: dict[str, list] = {}
    for s in scored:
        buckets.setdefault(s["bucket"], []).append(s)
    per_bucket = {}
    for b, items in sorted(buckets.items()):
        if b == "neg":
            fp = sum(1 for s in items if s.get("false_positive", False))
            per_bucket[b] = {"total": len(items), "false_positives": fp,
                             "fp_rate": round(fp / len(items), 3) if items else 0}
        else:
            b_scores = [s["score"] for s in items]
            per_bucket[b] = {
                "total": len(items),
                "relevant_rate": round(sum(1 for s in b_scores if s >= 2) / len(items), 3) if items else 0,
                "avg_score": round(sum(b_scores) / len(b_scores), 2) if b_scores else 0,
            }
    metrics["per_bucket"] = per_bucket

    # Per query class
    per_class = {}
    for cls in ["direct", "fragmented", "inferential"]:
        cls_items = [s for s in positive if QUERY_CLASS.get(s["query_id"]) == cls]
        if cls_items:
            cls_scores = [s["score"] for s in cls_items]
            per_class[cls] = {
                "total": len(cls_items),
                "relevant_rate": round(sum(1 for s in cls_scores if s >= 2) / len(cls_items), 3),
                "relevant_count": sum(1 for s in cls_scores if s >= 2),
                "avg_score": round(sum(cls_scores) / len(cls_scores), 2),
                "mrr": round(sum(1.0 / s["mrr_rank"] if s.get("mrr_rank") else 0.0 for s in cls_items) / len(cls_items), 3),
            }
    metrics["per_query_class"] = per_class

    return metrics


def print_comparison(all_metrics: dict[str, dict], baseline_synth_metrics: dict):
    """Print a comparison table of all strategies."""
    print("\n" + "=" * 80)
    print("LOCK-PUSH EXPERIMENT COMPARISON")
    print("=" * 80)

    synth_mrr = baseline_synth_metrics.get("mrr", 0.525)
    synth_rr = baseline_synth_metrics.get("relevant_rate", 0.442)

    print(f"\n{'Strategy':<30} {'MRR':>8} {'ΔMRR':>8} {'RR':>8} {'ΔRR':>8} {'FP':>4}")
    print("-" * 70)
    print(f"{'[synth baseline]':<30} {synth_mrr:>8.3f} {'':>8} {synth_rr:>8.3f} {'':>8} {'0':>4}")

    for name, m in sorted(all_metrics.items()):
        mrr = m.get("mrr", 0)
        rr = m.get("relevant_rate", 0)
        d_mrr = mrr - synth_mrr
        d_rr = rr - synth_rr
        fp = m.get("false_positives", 0)
        print(f"{name:<30} {mrr:>8.3f} {d_mrr:>+8.3f} {rr:>8.3f} {d_rr:>+8.3f} {fp:>4}")

    # Per-bucket detail for best strategy
    best_name = max(all_metrics, key=lambda n: all_metrics[n].get("mrr", 0))
    best = all_metrics[best_name]
    synth_buckets = baseline_synth_metrics.get("per_bucket", {})

    print(f"\nBest strategy: {best_name}")
    print(f"\n{'Bucket':<10} {'Synth RR':>10} {'Best RR':>10} {'Δ':>8}")
    print("-" * 40)
    for b in ["bio", "rel", "work", "pref", "heur"]:
        s_rr = synth_buckets.get(b, {}).get("relevant_rate", 0)
        b_rr = best.get("per_bucket", {}).get(b, {}).get("relevant_rate", 0)
        print(f"{b:<10} {s_rr:>10.3f} {b_rr:>10.3f} {b_rr - s_rr:>+8.3f}")

    neg = best.get("per_bucket", {}).get("neg", {})
    print(f"{'neg':<10} {'':>10} {'':>10} FP={neg.get('false_positives', 0)}")

    # Per query class for best
    synth_classes = baseline_synth_metrics.get("per_query_class", {})
    print(f"\n{'Class':<15} {'Synth RR':>10} {'Best RR':>10} {'Δ':>8} {'Best MRR':>10}")
    print("-" * 55)
    for cls in ["direct", "fragmented", "inferential"]:
        s_rr = synth_classes.get(cls, {}).get("relevant_rate", 0)
        b_cls = best.get("per_query_class", {}).get(cls, {})
        b_rr = b_cls.get("relevant_rate", 0)
        b_mrr = b_cls.get("mrr", 0)
        print(f"{cls:<15} {s_rr:>10.3f} {b_rr:>10.3f} {b_rr - s_rr:>+8.3f} {b_mrr:>10.3f}")

    print("=" * 80)


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-id", default="vnext_batch_chatgpt_constrained")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--skip-retrieve", action="store_true", help="Reuse existing retrieval pools")
    ap.add_argument("--strategies", nargs="*", help="Run only specific strategies")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    fh = logging.FileHandler(run_dir / "lock_push.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.root.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.root.addHandler(ch)
    logging.root.setLevel(logging.INFO)

    configure_env()
    api_key = os.environ["OPENROUTER_API_KEY"]

    # Phase 1: Retrieve
    pool_path = run_dir / "retrieval_pools.json"
    if args.skip_retrieve and pool_path.exists():
        logger.info("Reusing existing retrieval pools")
        pools = json.loads(pool_path.read_text())
    else:
        logger.info("Phase 1: Retrieving full pools...")
        pools = await phase1_retrieve(args.group_id, run_dir)

    # Load synth baseline metrics for comparison
    synth_baseline = json.loads(SYNTH_METRICS_PATH.read_text()) if SYNTH_METRICS_PATH.exists() else {}

    # Determine which strategies to run
    strategies_to_run = STRATEGIES
    if args.strategies:
        strategies_to_run = {k: v for k, v in STRATEGIES.items() if k in args.strategies}

    all_metrics = {}
    all_scored = {}

    for strat_name, strat_cfg in strategies_to_run.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Strategy: {strat_name}")
        logger.info(f"Config: {strat_cfg}")

        # Phase 2: Apply strategy
        strat_results = apply_strategy(pools, strat_name, strat_cfg)

        # Save strategy results
        strat_dir = run_dir / strat_name
        strat_dir.mkdir(exist_ok=True)
        (strat_dir / "query_results.json").write_text(json.dumps(strat_results, indent=2, default=str))

        # Phase 3: Judge (try to reuse judgments from baseline)
        existing = None
        if strat_name != "A_baseline" and "A_baseline" in all_scored:
            existing = {s["query_id"]: s for s in all_scored["A_baseline"]}

        logger.info(f"Phase 3: Judging {strat_name}...")
        scored = judge_results(strat_results, api_key, existing_judgments=existing)
        all_scored[strat_name] = scored

        (strat_dir / "scored_results.json").write_text(json.dumps(scored, indent=2))

        # Phase 4: Compute metrics
        metrics = compute_metrics(scored)
        all_metrics[strat_name] = metrics
        (strat_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        logger.info(f"  MRR: {metrics['mrr']:.3f}, RR: {metrics['relevant_rate']:.3f}, FP: {metrics['false_positives']}")

    # Save comparison
    comparison = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "synth_baseline": {
            "mrr": synth_baseline.get("mrr", 0.525),
            "relevant_rate": synth_baseline.get("relevant_rate", 0.442),
        },
        "strategies": {name: {
            "config": STRATEGIES.get(name, {}),
            "mrr": m.get("mrr", 0),
            "relevant_rate": m.get("relevant_rate", 0),
            "false_positives": m.get("false_positives", 0),
            "delta_mrr": round(m.get("mrr", 0) - synth_baseline.get("mrr", 0.525), 3),
            "delta_rr": round(m.get("relevant_rate", 0) - synth_baseline.get("relevant_rate", 0.442), 3),
            "per_bucket": m.get("per_bucket", {}),
            "per_query_class": m.get("per_query_class", {}),
        } for name, m in all_metrics.items()},
    }
    (run_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))

    print_comparison(all_metrics, synth_baseline)

    # Identify winner
    winner = max(all_metrics, key=lambda n: all_metrics[n].get("mrr", 0))
    winner_m = all_metrics[winner]
    print(f"\n🏆 Winner: {winner}")
    print(f"   MRR: {winner_m['mrr']:.3f} (Δ vs synth: {winner_m['mrr'] - synth_baseline.get('mrr', 0.525):+.3f})")
    print(f"   RR:  {winner_m['relevant_rate']:.3f}")
    print(f"   FP:  {winner_m['false_positives']}")


if __name__ == "__main__":
    asyncio.run(main())
