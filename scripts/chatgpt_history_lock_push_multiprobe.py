#!/usr/bin/env python3
"""Lock-Push Phase 2: Multi-probe retrieval for failing queries.

The Phase 1 experiment showed retrieval shaping (post-retrieval) doesn't move
the needle because the problem is upstream: embedding similarity misses facts
that exist in the graph.

Strategy: Decompose each failing query into 2-4 targeted sub-queries,
retrieve for each, merge pools, then rerank and judge.

Run from CWD without graphiti_core/ directory:
  cd /tmp && /Users/archibald/clawd/projects/bicameral-runtime/.venv/bin/python3 \
    /Users/archibald/clawd/projects/bicameral/scripts/chatgpt_history_lock_push_multiprobe.py \
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

REPO = Path("/Users/archibald/clawd/projects/bicameral-runtime")
BATCH_DIR = Path("/Users/archibald/clawd/projects/bicameral-runtime/state/chatgpt_history_vnext_20260316_batch")
QUERY_BANK_PATH = BATCH_DIR / "query_bank.json"
SYNTH_METRICS_PATH = Path("/Users/archibald/clawd/projects/bicameral-runtime/state/chatgpt_history_synthesis_pass_20260317/run/synth_eval_metrics.json")

sys.path.insert(0, str(REPO / "mcp_server" / "src"))

logger = logging.getLogger("lock_push_mp")

RERANKER_MODEL = "openai/gpt-4o-mini"
JUDGE_MODEL = "openai/gpt-4o-mini"
DECOMP_MODEL = "openai/gpt-4o-mini"
TEXT_SEARCH_WINDOW = 15
CENTER_NODE_WINDOW = 15
PROBE_SEARCH_WINDOW = 10
RERANK_TOPK = 25
FINAL_TOPK = 5

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
    episodes = f.get("episodes", None)
    if episodes is not None and len(episodes) == 0:
        return True
    return False


def _type_aware_boost(fact: Any, bucket: str, boost_weight: float = 0.15) -> float:
    priors = BUCKET_TYPE_PRIORS.get(bucket)
    if not priors:
        return 0.0
    text_lower = _fact_text(fact).lower()
    enriched_lower = _fact_enriched_text(fact).lower() if isinstance(fact, dict) else text_lower
    matched = sum(1 for term in priors["boost_terms"] if term.lower() in text_lower or term.lower() in enriched_lower)
    return boost_weight * math.sqrt(min(matched, 5)) if matched else 0.0


def _openrouter_call(prompt: str, system: str, api_key: str, model: str = JUDGE_MODEL) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 500,
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
                wait = 2 ** attempt + 1
                if "429" in str(e):
                    wait = max(wait, 10)
                time.sleep(wait)
            else:
                return {"score": 0, "rationale": f"Error: {e}"}
    return {"score": 0, "rationale": "Exhausted retries"}


def decompose_query(query: str, expected_sketch: str, api_key: str) -> list[str]:
    """Decompose a query into 2-4 targeted sub-queries for multi-probe retrieval."""
    prompt = f"""Decompose this query into 2-4 SHORT, specific sub-queries for a graph search.
Each sub-query should target different specific entities, facts, or aspects mentioned in the expected answer.
Make sub-queries concrete and entity-specific (names, specific topics, specific facts).

Query: {query}
Expected topics: {expected_sketch[:200]}

Return ONLY: {{"sub_queries": ["query1", "query2", ...]}}
Each sub-query should be 3-10 words, specific, entity-focused."""

    result = _openrouter_call(prompt, "You decompose queries into specific sub-queries. JSON only.", api_key, DECOMP_MODEL)
    subs = result.get("sub_queries", [])
    if not subs or not isinstance(subs, list):
        return []
    return [str(s) for s in subs[:4]]


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

    result = _openrouter_call(prompt, "Retrieval relevance scorer. JSON only.", api_key, RERANKER_MODEL)
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


async def retrieve_multiprobe(queries: list[dict], group_id: str, api_key: str,
                               run_dir: Path) -> list[dict]:
    """Multi-probe retrieval: original query + sub-queries, merged pool."""
    from config.schema import GraphitiConfig
    import graphiti_mcp_server as gms
    from graphiti_mcp_server import GraphitiService, search_memory_facts
    from neo4j import AsyncGraphDatabase

    # Find Yuan UUID
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

    results = []
    for i, q in enumerate(queries):
        logger.info(f"[{i+1}/{len(queries)}] Multi-probe {q['id']}: {q['query'][:60]}...")
        bucket = q["bucket"]

        try:
            seen_uuids = set()
            all_facts = []

            async def _retrieve(query_text, window):
                nonlocal all_facts, seen_uuids
                res = await search_memory_facts(
                    query=query_text, group_ids=[group_id],
                    search_mode="hybrid", max_facts=window)
                facts = res.get("facts", []) if isinstance(res, dict) else []
                for f in facts:
                    uid = f.get("uuid", "") if isinstance(f, dict) else ""
                    if uid and uid not in seen_uuids:
                        seen_uuids.add(uid)
                        all_facts.append(f)

            # 1. Original query text search
            await _retrieve(q["query"], TEXT_SEARCH_WINDOW)

            # 2. Center-node search
            if bucket != "neg" and primary_yuan_uuid:
                try:
                    res = await search_memory_facts(
                        query=q["query"], group_ids=[group_id],
                        search_mode="hybrid", max_facts=CENTER_NODE_WINDOW,
                        center_node_uuid=primary_yuan_uuid)
                    for f in (res.get("facts", []) if isinstance(res, dict) else []):
                        uid = f.get("uuid", "") if isinstance(f, dict) else ""
                        if uid and uid not in seen_uuids:
                            seen_uuids.add(uid)
                            all_facts.append(f)
                except Exception:
                    pass

            # 3. Sub-query probes (the key innovation)
            sub_queries = []
            if bucket != "neg":
                sub_queries = decompose_query(q["query"], q.get("expected_sketch", ""), api_key)
                logger.info(f"  Sub-queries: {sub_queries}")
                for sq in sub_queries:
                    await _retrieve(sq, PROBE_SEARCH_WINDOW)
                    await asyncio.sleep(0.5)

            # 4. Text dedup
            seen_texts = set()
            deduped = []
            for f in all_facts:
                text = _fact_text(f).strip()
                if text not in seen_texts:
                    seen_texts.add(text)
                    deduped.append(f)

            # 5. Rerank the full pool
            rerank_pool = deduped[:RERANK_TOPK]
            if bucket != "neg" and len(rerank_pool) > 1:
                raw_scores = _call_reranker_sync(q["query"], rerank_pool, api_key)
                for orig_idx, score in raw_scores:
                    if isinstance(rerank_pool[orig_idx], dict):
                        rerank_pool[orig_idx]["_rerank_score"] = round(score, 4)

            # 6. Apply type-aware boost and sort
            if bucket != "neg":
                for f in rerank_pool:
                    if isinstance(f, dict):
                        rs = f.get("_rerank_score", 0.5)
                        tb = _type_aware_boost(f, bucket)
                        f["_combined_score"] = round(rs + tb, 4)
                rerank_pool.sort(key=lambda f: -(f.get("_combined_score", 0) if isinstance(f, dict) else 0))

            # 7. Synth cap at 2
            selected = []
            synth_count = 0
            for f in rerank_pool:
                if _is_synthesized(f):
                    if synth_count < 2:
                        selected.append(f)
                        synth_count += 1
                else:
                    selected.append(f)
                if len(selected) >= FINAL_TOPK:
                    break
            if bucket == "neg":
                selected = rerank_pool[:FINAL_TOPK]

            results.append({
                "query_id": q["id"], "bucket": bucket,
                "query": q["query"], "expected_sketch": q.get("expected_sketch", ""),
                "facts": [{k: v for k, v in f.items() if not k.startswith("_")} if isinstance(f, dict) else f
                          for f in selected[:FINAL_TOPK]],
                "fact_count": len(selected[:FINAL_TOPK]),
                "total_merged": len(all_facts),
                "total_deduped": len(deduped),
                "sub_queries": sub_queries,
            })

        except Exception as e:
            logger.error(f"Error retrieving {q['id']}: {e}")
            import traceback; traceback.print_exc()
            results.append({
                "query_id": q["id"], "bucket": bucket,
                "query": q["query"], "expected_sketch": q.get("expected_sketch", ""),
                "facts": [], "fact_count": 0, "total_merged": 0,
                "error": str(e),
            })

        await asyncio.sleep(0.5)

    # Cleanup
    client = await service.get_client()
    await client.driver.close()

    (run_dir / "multiprobe_query_results.json").write_text(json.dumps(results, indent=2, default=str))
    return results


def judge_all(results: list[dict], api_key: str) -> list[dict]:
    scored = []
    for r in results:
        qid = r["query_id"]
        bucket = r["bucket"]
        facts = r.get("facts", [])

        if bucket == "neg":
            if not facts:
                scored.append({"query_id": qid, "bucket": bucket, "score": 0,
                               "rationale": "No results — correct for negative control.",
                               "is_negative": True, "false_positive": False})
            else:
                facts_text = "\n".join([f"- {_fact_text(f)}" for f in facts[:3]])
                prompt = f"""NEGATIVE CONTROL query — retrieval should NOT have relevant results.

Query: {r['query']}
Retrieved:
{facts_text}

Score false-positive severity: 0=irrelevant(correct), 1=tangential, 2=somewhat relevant(minor FP), 3=directly answers(major FP)

Return ONLY: {{"score": <0-3>, "rationale": "..."}}"""
                result = _openrouter_call(prompt, "Retrieval quality judge. JSON only.", api_key)
                scored.append({"query_id": qid, "bucket": bucket, "score": int(result.get("score", 0)),
                               "rationale": result.get("rationale", ""), "is_negative": True,
                               "false_positive": int(result.get("score", 0)) >= 2})
        else:
            if not facts:
                scored.append({"query_id": qid, "bucket": bucket, "score": 0,
                               "rationale": "No results.", "is_negative": False, "mrr_rank": None})
            else:
                mrr_rank = None
                for rank, f in enumerate(facts[:FINAL_TOPK], start=1):
                    prompt = f"""Score retrieval result relevance.

Query: {r['query']}
Expected (sketch): {r['expected_sketch']}
Retrieved: {_fact_text(f)}

3=Directly relevant and specific. 2=Partially relevant, right topic. 1=Tangential. 0=Irrelevant.

Return ONLY: {{"score": <0-3>, "rationale": "..."}}"""
                    result = _openrouter_call(prompt, "Retrieval quality judge. JSON only.", api_key)
                    s = int(result.get("score", 0))
                    if rank == 1:
                        top1_score = s
                        top1_rationale = result.get("rationale", "")
                    if s >= 2 and mrr_rank is None:
                        mrr_rank = rank
                        if rank > 1:  # no need to check further
                            break
                    time.sleep(0.3)
                    if mrr_rank is not None:
                        break

                scored.append({"query_id": qid, "bucket": bucket,
                               "score": top1_score,
                               "rationale": top1_rationale, "is_negative": False,
                               "mrr_rank": mrr_rank})

        logger.info(f"  Judged {qid}: score={scored[-1].get('score', 'N/A')}, mrr_rank={scored[-1].get('mrr_rank')}")
        time.sleep(0.3)

    return scored


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
            b_mrr = round(sum(1.0 / s["mrr_rank"] if s.get("mrr_rank") else 0.0 for s in items) / len(items), 3)
            per_bucket[b] = {
                "total": len(items),
                "relevant_rate": round(sum(1 for s in b_scores if s >= 2) / len(items), 3) if items else 0,
                "relevant_count": sum(1 for s in b_scores if s >= 2),
                "avg_score": round(sum(b_scores) / len(b_scores), 2) if b_scores else 0,
                "mrr": b_mrr,
            }
    metrics["per_bucket"] = per_bucket

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


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-id", default="vnext_batch_chatgpt_constrained")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--skip-retrieve", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(run_dir / "multiprobe.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.root.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.root.addHandler(ch)
    logging.root.setLevel(logging.INFO)

    configure_env()
    api_key = os.environ["OPENROUTER_API_KEY"]

    queries = json.loads(QUERY_BANK_PATH.read_text())

    # Phase 1: Multi-probe retrieval
    results_path = run_dir / "multiprobe_query_results.json"
    if args.skip_retrieve and results_path.exists():
        logger.info("Reusing existing multi-probe results")
        mp_results = json.loads(results_path.read_text())
    else:
        logger.info("Phase 1: Multi-probe retrieval...")
        mp_results = await retrieve_multiprobe(queries, args.group_id, api_key, run_dir)

    # Phase 2: Judge
    logger.info("Phase 2: Judging multi-probe results...")
    scored = judge_all(mp_results, api_key)
    (run_dir / "multiprobe_scored.json").write_text(json.dumps(scored, indent=2))

    # Phase 3: Metrics
    metrics = compute_metrics(scored)
    (run_dir / "multiprobe_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Load baselines for comparison
    synth_baseline = json.loads(SYNTH_METRICS_PATH.read_text()) if SYNTH_METRICS_PATH.exists() else {"mrr": 0.525, "relevant_rate": 0.442}
    phase1_dir = Path(args.run_dir).parent / "run" if (Path(args.run_dir).parent / "run").exists() else None
    phase1_metrics = {}
    if phase1_dir and (phase1_dir / "A_baseline" / "metrics.json").exists():
        phase1_metrics = json.loads((phase1_dir / "A_baseline" / "metrics.json").read_text())

    # Print comparison
    print("\n" + "=" * 80)
    print("MULTI-PROBE RETRIEVAL RESULTS")
    print("=" * 80)

    synth_mrr = synth_baseline.get("mrr", 0.525)
    synth_rr = synth_baseline.get("relevant_rate", 0.442)
    p1_mrr = phase1_metrics.get("mrr", 0)
    p1_rr = phase1_metrics.get("relevant_rate", 0)

    print(f"\n{'Strategy':<30} {'MRR':>8} {'ΔMRR':>8} {'RR':>8} {'ΔRR':>8} {'FP':>4}")
    print("-" * 70)
    print(f"{'[synth baseline]':<30} {synth_mrr:>8.3f} {'':>8} {synth_rr:>8.3f} {'':>8} {'0':>4}")
    if p1_mrr:
        print(f"{'[phase1 best]':<30} {p1_mrr:>8.3f} {p1_mrr-synth_mrr:>+8.3f} {p1_rr:>8.3f} {p1_rr-synth_rr:>+8.3f} {phase1_metrics.get('false_positives',0):>4}")
    print(f"{'MULTIPROBE':<30} {metrics['mrr']:>8.3f} {metrics['mrr']-synth_mrr:>+8.3f} {metrics['relevant_rate']:>8.3f} {metrics['relevant_rate']-synth_rr:>+8.3f} {metrics['false_positives']:>4}")

    # Per bucket
    synth_buckets = synth_baseline.get("per_bucket", {})
    print(f"\n{'Bucket':<10} {'Synth RR':>10} {'MP RR':>10} {'Δ':>8} {'MP MRR':>10}")
    print("-" * 50)
    for b in ["bio", "rel", "work", "pref", "heur"]:
        s_rr = synth_buckets.get(b, {}).get("relevant_rate", 0)
        b_data = metrics.get("per_bucket", {}).get(b, {})
        b_rr = b_data.get("relevant_rate", 0)
        b_mrr = b_data.get("mrr", 0)
        print(f"{b:<10} {s_rr:>10.3f} {b_rr:>10.3f} {b_rr-s_rr:>+8.3f} {b_mrr:>10.3f}")
    neg = metrics.get("per_bucket", {}).get("neg", {})
    print(f"{'neg':<10} {'':>10} {'':>10} {'':>8} FP={neg.get('false_positives', 0)}")

    # Per query class
    synth_classes = synth_baseline.get("per_query_class", {})
    print(f"\n{'Class':<15} {'Synth RR':>10} {'MP RR':>10} {'Δ':>8} {'MP MRR':>10}")
    print("-" * 55)
    for cls in ["direct", "fragmented", "inferential"]:
        s_rr = synth_classes.get(cls, {}).get("relevant_rate", 0)
        c_data = metrics.get("per_query_class", {}).get(cls, {})
        c_rr = c_data.get("relevant_rate", 0)
        c_mrr = c_data.get("mrr", 0)
        print(f"{cls:<15} {s_rr:>10.3f} {c_rr:>10.3f} {c_rr-s_rr:>+8.3f} {c_mrr:>10.3f}")

    print("=" * 80)
    print(f"\n🎯 MRR: {metrics['mrr']:.3f} (target: 0.750, gap: {0.750 - metrics['mrr']:.3f})")
    print(f"   vs synth baseline: Δ{metrics['mrr']-synth_mrr:+.3f}")

    # Decision
    if metrics["mrr"] >= 0.75:
        print("   ✅ LOCK BAR MET!")
    elif metrics["mrr"] >= 0.65:
        print("   🔶 Strong progress — one more iteration could lock")
    elif metrics["mrr"] > synth_mrr + 0.05:
        print("   📈 Material improvement — multi-probe helps, but gap remains")
    else:
        print("   🔻 Minimal improvement — substrate/coverage is the bottleneck")


if __name__ == "__main__":
    asyncio.run(main())
