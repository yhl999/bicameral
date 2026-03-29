#!/usr/bin/env python3
# NOTE: Must be run with the bicameral-runtime venv Python from a CWD that does
# NOT contain a graphiti_core/ directory (to avoid shadowing the venv package).
# Recommended invocation:
#   cd /tmp && /Users/archibald/clawd/projects/bicameral-runtime/.venv/bin/python3 \
#     /Users/archibald/clawd/projects/bicameral/scripts/evaluate_chatgpt_history_synthesis.py \
#     --run-dir /path/to/run
"""Evaluate ChatGPT History Synthesis Pass.

Re-runs the Phase 2b retrieval benchmark against the enriched graph (base +
synthesized nodes) and computes deltas by query class and bucket.

Usage:
  python3 scripts/evaluate_chatgpt_history_synthesis.py \
    --base-group-id vnext_batch_chatgpt_constrained \
    --synth-group-id vnext_batch_chatgpt_constrained \
    --run-dir /path/to/synthesis/run
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Reuse the query_bank from the existing benchmark
BATCH_DIR = Path("/Users/archibald/clawd/projects/bicameral-runtime/state/chatgpt_history_vnext_20260316_batch")
QUERY_BANK_PATH = BATCH_DIR / "query_bank.json"
BASELINE_METRICS_PATH = BATCH_DIR / "phase2b_metrics_constrained.json"

REPO = Path("/Users/archibald/clawd/projects/bicameral-runtime")
# Only add mcp_server/src to sys.path — do NOT add REPO root because
# bicameral-runtime/graphiti_core/ shadows the venv-installed package.
sys.path.insert(0, str(REPO / "mcp_server" / "src"))

logger = logging.getLogger("synth_eval")

# Retrieval settings (same as Phase 2b)
TEXT_SEARCH_WINDOW = 20
CENTER_NODE_WINDOW = 20
RERANK_TOPK = 20
FINAL_TOPK = 5
RERANKER_MODEL = "openai/gpt-4o-mini"
JUDGE_MODEL = "openai/gpt-4o-mini"

# Type-aware query classification (same as Phase 2b)
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
                             "GLP", "semaglutide", "health"], "boost_weight": 0.15},
    "heur": {"boost_terms": ["writing", "communication", "decision", "thinking",
                             "strategy", "pattern", "heuristic", "approach", "style",
                             "method", "process", "habit", "tendency", "belief",
                             "philosophy", "principle", "theory", "geopolitical",
                             "curate", "calibrate", "skepticism", "evaluate",
                             "navigate", "collaborate", "refine", "personalize"], "boost_weight": 0.15},
}

# Query class mapping from grounding audit
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


def _type_aware_boost(fact: Any, bucket: str) -> float:
    priors = BUCKET_TYPE_PRIORS.get(bucket)
    if not priors:
        return 0.0
    fact_text_lower = _fact_text(fact).lower()
    enriched_lower = _fact_enriched_text(fact).lower() if isinstance(fact, dict) else fact_text_lower
    matched = sum(1 for term in priors["boost_terms"] if term.lower() in fact_text_lower or term.lower() in enriched_lower)
    return priors["boost_weight"] * math.sqrt(min(matched, 5)) if matched else 0.0


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


async def run_eval(group_id: str, run_dir: Path, skip_query: bool = False, skip_judge: bool = False) -> dict:
    """Run the full evaluation pipeline."""
    from config.schema import GraphitiConfig
    import graphiti_mcp_server as gms
    from graphiti_mcp_server import GraphitiService, search_memory_facts

    queries = json.loads(QUERY_BANK_PATH.read_text())
    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    # Discover Yuan entity UUIDs
    from neo4j import AsyncGraphDatabase
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
                "RETURN e.uuid AS uuid ORDER BY size(e.name) DESC",
                gid=group_id,
            )
            yuan_uuids = [r.data()["uuid"] async for r in result]
    finally:
        await async_driver.close()

    primary_yuan_uuid = yuan_uuids[0] if yuan_uuids else None
    logger.info(f"Yuan UUID: {primary_yuan_uuid}")

    query_results_path = run_dir / "synth_eval_query_results.json"

    if not skip_query:
        # Initialize Graphiti service
        cfg = GraphitiConfig()
        gms.config = cfg
        service = GraphitiService(cfg, semaphore_limit=1)
        await service.initialize()
        gms.graphiti_service = service

        results = []
        for i, q in enumerate(queries):
            logger.info(f"Query {q['id']}: {q['query'][:60]}...")
            bucket = q["bucket"]

            try:
                # Pass 1: Text search
                text_res = await search_memory_facts(
                    query=q["query"], group_ids=[group_id],
                    search_mode="hybrid", max_facts=TEXT_SEARCH_WINDOW,
                )
                text_facts = text_res.get("facts", []) if isinstance(text_res, dict) else []

                # Pass 2: Center-node search
                center_facts = []
                if bucket != "neg" and primary_yuan_uuid:
                    try:
                        center_res = await search_memory_facts(
                            query=q["query"], group_ids=[group_id],
                            search_mode="hybrid", max_facts=CENTER_NODE_WINDOW,
                            center_node_uuid=primary_yuan_uuid,
                        )
                        center_facts = center_res.get("facts", []) if isinstance(center_res, dict) else []
                    except Exception as e:
                        logger.warning(f"Center-node search failed for {q['id']}: {e}")

                # Merge and deduplicate
                seen_uuids = set()
                merged_facts = []
                for f in text_facts:
                    uid = f.get("uuid", "") if isinstance(f, dict) else ""
                    if uid and uid not in seen_uuids:
                        seen_uuids.add(uid)
                        merged_facts.append(f)
                for f in center_facts:
                    uid = f.get("uuid", "") if isinstance(f, dict) else ""
                    if uid and uid not in seen_uuids:
                        seen_uuids.add(uid)
                        merged_facts.append(f)

                if not merged_facts:
                    results.append({"query_id": q["id"], "bucket": bucket, "query": q["query"],
                                    "expected_sketch": q["expected_sketch"], "fact_count": 0, "facts": [], "reranked": False})
                    continue

                # Type-aware boost
                if bucket != "neg":
                    for f in merged_facts:
                        if isinstance(f, dict):
                            f["_type_boost"] = _type_aware_boost(f, bucket)

                rerank_pool = merged_facts[:RERANK_TOPK]

                if bucket != "neg" and len(rerank_pool) > 1:
                    rerank_scores = _call_reranker_sync(q["query"], rerank_pool, api_key)
                    scored = []
                    for orig_idx, rerank_score in rerank_scores:
                        type_boost = rerank_pool[orig_idx].get("_type_boost", 0.0) if isinstance(rerank_pool[orig_idx], dict) else 0.0
                        scored.append((orig_idx, rerank_score + type_boost, rerank_score, rerank_pool[orig_idx]))
                    scored.sort(key=lambda x: -x[1])
                    reranked_facts = []
                    for _, combined, raw, fact in scored[:FINAL_TOPK]:
                        if isinstance(fact, dict):
                            fact = {**fact, "_rerank_score": round(raw, 4), "_combined_score": round(combined, 4)}
                        reranked_facts.append(fact)
                else:
                    reranked_facts = rerank_pool[:FINAL_TOPK]

                clean_facts = [{k: v for k, v in f.items() if not k.startswith("_")} if isinstance(f, dict) else f
                               for f in reranked_facts]

                results.append({
                    "query_id": q["id"], "bucket": bucket, "query": q["query"],
                    "expected_sketch": q["expected_sketch"],
                    "fact_count": len(clean_facts), "facts": clean_facts[:FINAL_TOPK],
                    "reranked": True, "total_merged": len(merged_facts),
                })
            except Exception as e:
                logger.error(f"Error querying {q['id']}: {e}")
                results.append({"query_id": q["id"], "bucket": bucket, "query": q["query"],
                                "expected_sketch": q["expected_sketch"], "fact_count": 0, "facts": [], "error": str(e)})

            if i < len(queries) - 1:
                await asyncio.sleep(1.0)

        query_results_path.write_text(json.dumps(results, indent=2, default=str))
        logger.info(f"Query results saved to {query_results_path}")

        client = await service.get_client()
        await client.driver.close()
    else:
        results = json.loads(query_results_path.read_text())

    # Judge results
    scored_results_path = run_dir / "synth_eval_scored_results.json"

    if not skip_judge:
        scored = []
        for r in results:
            query_id = r["query_id"]
            bucket = r["bucket"]
            facts = r.get("facts", [])

            if bucket == "neg":
                if not facts:
                    scored.append({"query_id": query_id, "bucket": bucket, "score": 0,
                                   "rationale": "No results — correct for negative control.",
                                   "is_negative": True, "false_positive": False})
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
                    scored.append({"query_id": query_id, "bucket": bucket, "score": int(result.get("score", 0)),
                                   "rationale": result.get("rationale", ""), "is_negative": True,
                                   "false_positive": int(result.get("score", 0)) >= 2})
            else:
                if not facts:
                    scored.append({"query_id": query_id, "bucket": bucket, "score": 0,
                                   "rationale": "No results.", "is_negative": False, "mrr_rank": None})
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
                    time.sleep(1)

                    mrr_rank = None
                    if top1_score >= 2:
                        mrr_rank = 1
                    else:
                        for rank, f in enumerate(facts[1:5], start=2):
                            fp = f"""Score: Query: {r['query']}
Expected: {r['expected_sketch']}
Retrieved: {_fact_text(f)}
3=Direct, 2=Partial, 1=Tangential, 0=Irrelevant. Return ONLY: {{"score": <0-3>, "rationale": "..."}}"""
                            fr = _openrouter_call(fp, "Retrieval quality judge. JSON only.", api_key)
                            if int(fr.get("score", 0)) >= 2:
                                mrr_rank = rank
                                break
                            time.sleep(1)

                    scored.append({"query_id": query_id, "bucket": bucket, "score": top1_score,
                                   "rationale": result.get("rationale", ""), "is_negative": False, "mrr_rank": mrr_rank})

            logger.info(f"Judged {query_id}: score={scored[-1].get('score', 'N/A')}")
            time.sleep(0.5)

        scored_results_path.write_text(json.dumps(scored, indent=2))
    else:
        scored = json.loads(scored_results_path.read_text())

    # Compute metrics
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

    # Per-bucket metrics
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
                "top1_rate": round(sum(1 for s in b_scores if s == 3) / len(items), 3) if items else 0,
                "relevant_rate": round(sum(1 for s in b_scores if s >= 2) / len(items), 3) if items else 0,
                "avg_score": round(sum(b_scores) / len(b_scores), 2) if b_scores else 0,
            }
    metrics["per_bucket"] = per_bucket

    # Per query class metrics
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
            }
    metrics["per_query_class"] = per_class

    # Compute deltas vs baseline
    baseline_metrics = {}
    if BASELINE_METRICS_PATH.exists():
        baseline_metrics = json.loads(BASELINE_METRICS_PATH.read_text())

    deltas = {}
    for key in ["top1_rate", "relevant_rate", "mrr", "avg_score"]:
        bv = baseline_metrics.get(key, 0)
        sv = metrics.get(key, 0)
        deltas[key] = {"baseline": bv, "synthesis": sv, "delta": round(sv - bv, 3)}
    metrics["deltas_vs_baseline"] = deltas

    # Per-bucket deltas
    bucket_deltas = {}
    for b in ["bio", "rel", "work", "pref", "heur"]:
        base_b = baseline_metrics.get("per_bucket", {}).get(b, {})
        synth_b = per_bucket.get(b, {})
        bucket_deltas[b] = {
            "baseline_relevant_rate": base_b.get("relevant_rate", 0),
            "synthesis_relevant_rate": synth_b.get("relevant_rate", 0),
            "delta": round(synth_b.get("relevant_rate", 0) - base_b.get("relevant_rate", 0), 3),
        }
    metrics["bucket_deltas_vs_baseline"] = bucket_deltas

    # Per-query-class deltas vs baseline (computed from baseline scored results)
    baseline_scored_path = BATCH_DIR / "phase2b_scored_results_constrained.json"
    class_deltas = {}
    if baseline_scored_path.exists():
        baseline_scored = json.loads(baseline_scored_path.read_text())
        baseline_positive = [s for s in baseline_scored if not s.get("is_negative", False)]
        for cls in ["direct", "fragmented", "inferential"]:
            base_cls = [s for s in baseline_positive if QUERY_CLASS.get(s["query_id"]) == cls]
            synth_cls = per_class.get(cls, {})
            if base_cls:
                base_scores = [s["score"] for s in base_cls]
                base_rr = round(sum(1 for s in base_scores if s >= 2) / len(base_cls), 3)
                base_avg = round(sum(base_scores) / len(base_scores), 2)
            else:
                base_rr, base_avg = 0, 0
            class_deltas[cls] = {
                "baseline_relevant_rate": base_rr,
                "synthesis_relevant_rate": synth_cls.get("relevant_rate", 0),
                "delta_relevant_rate": round(synth_cls.get("relevant_rate", 0) - base_rr, 3),
                "baseline_avg_score": base_avg,
                "synthesis_avg_score": synth_cls.get("avg_score", 0),
                "delta_avg_score": round(synth_cls.get("avg_score", 0) - base_avg, 2),
            }
    metrics["query_class_deltas_vs_baseline"] = class_deltas

    # Per-query detail
    per_query = []
    for s in scored:
        qid = s["query_id"]
        per_query.append({
            "query_id": qid,
            "bucket": s["bucket"],
            "query_class": QUERY_CLASS.get(qid, "unknown"),
            "score": s.get("score"),
            "mrr_rank": s.get("mrr_rank"),
            "is_negative": s.get("is_negative", False),
            "false_positive": s.get("false_positive", False),
            "rationale": s.get("rationale", ""),
        })
    metrics["per_query"] = per_query

    # Write metrics
    metrics_path = run_dir / "synth_eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Metrics saved to {metrics_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SYNTHESIS EVALUATION RESULTS")
    print("=" * 60)
    print(f"  relevant_rate: {metrics['relevant_rate']:.3f} (baseline: {deltas['relevant_rate']['baseline']:.3f}, Δ: {deltas['relevant_rate']['delta']:+.3f})")
    print(f"  top1_rate:     {metrics['top1_rate']:.3f} (baseline: {deltas['top1_rate']['baseline']:.3f}, Δ: {deltas['top1_rate']['delta']:+.3f})")
    print(f"  MRR:           {metrics['mrr']:.3f} (baseline: {deltas['mrr']['baseline']:.3f}, Δ: {deltas['mrr']['delta']:+.3f})")
    print(f"  avg_score:     {metrics['avg_score']:.2f} (baseline: {deltas['avg_score']['baseline']:.2f}, Δ: {deltas['avg_score']['delta']:+.2f})")
    print(f"  false_pos:     {metrics['false_positives']}")
    print("\nPer-bucket deltas:")
    for b, bd in bucket_deltas.items():
        print(f"  {b:6s}: {bd['baseline_relevant_rate']:.3f} → {bd['synthesis_relevant_rate']:.3f} (Δ: {bd['delta']:+.3f})")
    print("\nPer query class (with baseline deltas):")
    for cls, cd in per_class.items():
        delta_info = class_deltas.get(cls, {})
        base_rr = delta_info.get("baseline_relevant_rate", 0)
        delta_rr = delta_info.get("delta_relevant_rate", 0)
        print(f"  {cls:12s}: {cd['relevant_count']}/{cd['total']} relevant ({cd['relevant_rate']:.3f}) "
              f"[baseline: {base_rr:.3f}, Δ: {delta_rr:+.3f}]")
    print("=" * 60)

    return metrics


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-group-id", default="vnext_batch_chatgpt_constrained")
    ap.add_argument("--synth-group-id", default="vnext_batch_chatgpt_constrained")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--skip-query", action="store_true")
    ap.add_argument("--skip-judge", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "synth_eval.log"
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.root.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.root.addHandler(ch)
    logging.root.setLevel(logging.INFO)

    configure_env()

    metrics = await run_eval(args.synth_group_id, run_dir,
                             skip_query=args.skip_query, skip_judge=args.skip_judge)


if __name__ == "__main__":
    asyncio.run(main())
