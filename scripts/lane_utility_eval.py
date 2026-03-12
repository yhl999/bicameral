#!/usr/bin/env python3
"""Lane utility evaluation harness (Step 6).

Scores each target lane 0-10 using five 0-2 components:
1) episode coverage
2) timeline coverage
3) ontology conformance (r.name in lane ontology)
4) structural signal (rels/episode + relation diversity)
5) utility probe (pack outputs for content lanes; relation evidence for ops lanes)

Usage:
  uv run python ../bicameral-private/scripts/lane_utility_eval.py \
    --config ../bicameral-private/scripts/lane_utility_eval_queries.yaml \
    --runtime-repo . \
    --report ../bicameral-private/reports/canonical-truth/step6-lane-utility.md \
    --json ../bicameral-private/reports/canonical-truth/step6-lane-utility.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from neo4j import GraphDatabase

LOW_SIGNAL_PATTERNS = [
    re.compile(r"\bemploys rhetorical strategies?\b", re.IGNORECASE),
    re.compile(r"\bengage readers\b", re.IGNORECASE),
    re.compile(r"\bshows? (the )?author(?:'|’)s voice and style\b", re.IGNORECASE),
    re.compile(r"\bdemonstrates a mix of formal and casual tones\b", re.IGNORECASE),
]


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("- "):
            s = s[2:].strip()
        out.append(s)
    return out


def _low_signal_ratio(lines: list[str]) -> float:
    if not lines:
        return 1.0
    hits = sum(1 for line in lines if any(p.search(line) for p in LOW_SIGNAL_PATTERNS))
    return hits / len(lines)


def _token_set(lines: list[str]) -> set[str]:
    toks: set[str] = set()
    for line in lines:
        for t in re.findall(r"[a-z0-9]+", line.lower()):
            if len(t) >= 3:
                toks.add(t)
    return toks


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _score_two_level(v: float, good: float, okay: float) -> int:
    if v >= good:
        return 2
    if v >= okay:
        return 1
    return 0


def run_phase3_probe(private_root: Path, runtime_repo: Path, probe_cfg: dict[str, Any]) -> dict[str, Any]:
    out_dir = private_root / str(probe_cfg.get("output_dir", "_tmp/phase3-canary"))
    out_dir.mkdir(parents=True, exist_ok=True)

    long_task = str(probe_cfg.get("long_task", "draft an article about bicameral memory"))
    tweet_task = str(probe_cfg.get("tweet_task", "draft a tweet about our bicameral memory system"))

    long_json = out_dir / "content_long_form.eval.json"
    tweet_json = out_dir / "content_tweet.eval.json"
    python_executable = sys.executable or "python3"

    subprocess.run(
        [
            python_executable,
            str(runtime_repo / "scripts" / "runtime_pack_router.py"),
            "--consumer",
            "main_session_content_long_form",
            "--workflow-id",
            "content_long_form",
            "--step-id",
            "outline",
            "--task",
            long_task,
            "--scope",
            "private",
            "--materialize",
            "--out",
            str(long_json),
        ],
        check=True,
    )

    subprocess.run(
        [
            python_executable,
            str(runtime_repo / "scripts" / "runtime_pack_router.py"),
            "--consumer",
            "main_session_content_tweet",
            "--workflow-id",
            "content_tweet",
            "--step-id",
            "draft",
            "--task",
            tweet_task,
            "--scope",
            "private",
            "--materialize",
            "--out",
            str(tweet_json),
        ],
        check=True,
    )

    def extract_blocks(path: Path) -> tuple[list[str], list[str], list[str]]:
        voice: list[str] = []
        samples: list[str] = []
        artifacts: list[str] = []
        obj = json.loads(path.read_text(encoding="utf-8"))
        for pack in obj.get("packs", []):
            pid = pack.get("pack_id")
            content = str(pack.get("content", ""))
            lines = [
                (ln[2:].strip() if ln.strip().startswith("- ") else ln.strip())
                for ln in content.splitlines()
                if ln.strip()
            ]
            if pid == "content_voice_style":
                voice = lines
            elif pid == "content_writing_samples":
                samples = lines
            elif pid == "content_long_form_artifacts":
                artifacts = lines
        return voice, samples, artifacts

    voice, samples, artifacts = extract_blocks(long_json)
    voice_tweet, samples_tweet, _ = extract_blocks(tweet_json)

    return {
        "long_task": long_task,
        "tweet_task": tweet_task,
        "voice_lines": len(voice),
        "samples_lines": len(samples),
        "artifacts_lines": len(artifacts),
        "voice_low_signal": _low_signal_ratio(voice),
        "samples_low_signal": _low_signal_ratio(samples),
        "artifacts_low_signal": _low_signal_ratio(artifacts),
        "tweet_voice_lines": len(voice_tweet),
        "tweet_samples_lines": len(samples_tweet),
        "tweet_voice_low_signal": _low_signal_ratio(voice_tweet),
        "tweet_samples_low_signal": _low_signal_ratio(samples_tweet),
        "voice_samples_jaccard": _jaccard(_token_set(voice), _token_set(samples)),
        "tweet_voice_samples_jaccard": _jaccard(_token_set(voice_tweet), _token_set(samples_tweet)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 6 lane utility scorer")
    ap.add_argument("--config", required=True)
    ap.add_argument("--runtime-repo", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    private_root = cfg_path.parents[1]
    runtime_repo = Path(args.runtime_repo).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    lanes_cfg: dict[str, Any] = cfg["lanes"]
    score_target = float(cfg.get("score_target", 9.0))

    probe_cfg = cfg.get("pack_probe", {})
    probe = run_phase3_probe(private_root, runtime_repo, probe_cfg)

    ont_path = private_root / "mcp_server" / "config" / "extraction_ontologies.yaml"
    ont = yaml.safe_load(ont_path.read_text(encoding="utf-8"))

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    pw = os.environ.get("NEO4J_PASSWORD")
    db = os.environ.get("NEO4J_DATABASE", "neo4j")
    if not pw:
        raise SystemExit("NEO4J_PASSWORD is required")

    driver = GraphDatabase.driver(uri, auth=(user, pw))

    results: dict[str, Any] = {}
    with driver.session(database=db) as s:
        for lane, lane_cfg in lanes_cfg.items():
            exp = int(lane_cfg["expected_episodes"])
            min_rel_ep = float(lane_cfg["min_rel_per_episode"])
            min_distinct = int(lane_cfg["min_distinct_rel_names"])
            probe_kind = str(lane_cfg.get("probe", ""))

            episodes = s.run("MATCH (e:Episodic {group_id:$g}) RETURN count(e) AS c", g=lane).single()["c"]
            rels = s.run("MATCH ()-[r:RELATES_TO]->() WHERE r.group_id=$g RETURN count(r) AS c", g=lane).single()["c"]
            distinct = s.run("MATCH ()-[r:RELATES_TO]->() WHERE r.group_id=$g RETURN count(DISTINCT r.name) AS c", g=lane).single()["c"]
            next_links = s.run("MATCH (e:Episodic {group_id:$g})-[:NEXT_EPISODE]->() RETURN count(e) AS c", g=lane).single()["c"]

            allowed = [r["name"] for r in ont.get(lane, {}).get("relationship_types", []) if "name" in r]
            conf = 0
            if rels > 0 and allowed:
                conf = s.run(
                    "MATCH ()-[r:RELATES_TO]->() WHERE r.group_id=$g AND r.name IN $rels RETURN count(r) AS c",
                    g=lane,
                    rels=allowed,
                ).single()["c"]
            conformance = (conf / rels) if rels else 0.0

            coverage = (episodes / exp) if exp else 0.0
            timeline = (next_links / max(episodes - 1, 1)) if episodes else 0.0
            rel_per_episode = rels / max(episodes, 1)

            coverage_score = _score_two_level(coverage, 1.0, 0.9)
            timeline_score = _score_two_level(timeline, 1.0, 0.9)
            conformance_score = _score_two_level(conformance, 0.98, 0.95)

            struct_ok = (rel_per_episode >= min_rel_ep and distinct >= min_distinct)
            struct_score = 2 if struct_ok else (1 if (rel_per_episode >= 0.8 * min_rel_ep and distinct >= max(1, min_distinct - 1)) else 0)

            utility_probe_score = 0
            if probe_kind == "writing_samples":
                utility_probe_score = 2 if (probe["samples_lines"] >= 8 and probe["samples_low_signal"] <= 0.15 and probe["voice_samples_jaccard"] <= 0.60) else 1
            elif probe_kind == "content_strategy":
                utility_probe_score = 2 if (probe["artifacts_lines"] >= 8 and probe["artifacts_low_signal"] <= 0.15) else 1
            elif probe_kind == "inspiration_short_form":
                utility_probe_score = 2 if (probe["tweet_samples_lines"] >= 8 and probe["tweet_samples_low_signal"] <= 0.15 and probe["tweet_voice_samples_jaccard"] <= 0.60) else 1
            elif probe_kind == "inspiration_long_form":
                utility_probe_score = 2 if (probe["artifacts_lines"] >= 8 and probe["artifacts_low_signal"] <= 0.15) else 1
            elif probe_kind == "engineering":
                utility_probe_score = 2 if distinct >= 8 and rel_per_episode >= 2.0 else 1
            elif probe_kind == "self_audit":
                utility_probe_score = 2 if distinct >= 4 and rel_per_episode >= 1.2 else 1

            total = coverage_score + timeline_score + conformance_score + struct_score + utility_probe_score
            score10 = float(total)

            results[lane] = {
                "score": score10,
                "pass": score10 >= score_target,
                "components": {
                    "coverage": coverage_score,
                    "timeline": timeline_score,
                    "conformance": conformance_score,
                    "structure": struct_score,
                    "utility_probe": utility_probe_score,
                },
                "metrics": {
                    "episodes": episodes,
                    "expected_episodes": exp,
                    "coverage_ratio": round(coverage, 4),
                    "rels": rels,
                    "distinct_rel_names": distinct,
                    "rel_per_episode": round(rel_per_episode, 4),
                    "timeline_ratio": round(timeline, 4),
                    "conformance_ratio": round(conformance, 4),
                },
            }

    driver.close()

    avg = sum(v["score"] for v in results.values()) / max(len(results), 1)
    summary = {
        "score_target": score_target,
        "average_score": round(avg, 3),
        "all_pass": all(v["pass"] for v in results.values()),
        "probe_metrics": probe,
        "lanes": results,
    }

    json_path = Path(args.json).resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = ["# Step 6 Lane Utility Eval", "", f"- Score target: {score_target}", f"- Average score: {summary['average_score']}", f"- All lanes pass: {summary['all_pass']}", "", "| lane | score | pass | coverage | timeline | conformance | structure | probe |", "|---|---:|:---:|---:|---:|---:|---:|---:|"]
    for lane, row in results.items():
        c = row["components"]
        report_lines.append(f"| `{lane}` | {row['score']:.1f} | {'✅' if row['pass'] else '❌'} | {c['coverage']} | {c['timeline']} | {c['conformance']} | {c['structure']} | {c['utility_probe']} |")

    rep_path = Path(args.report).resolve()
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps({"report": str(rep_path), "json": str(json_path), "average_score": summary["average_score"], "all_pass": summary["all_pass"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
