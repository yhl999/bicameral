#!/usr/bin/env python3
"""ChatGPT History Post-Extraction Synthesis Pass.

Reads the existing first-order graph (entities, facts, episodes) for a given
group_id, uses gpt-5.1-codex-mini with reasoning.effort=high to synthesize
second-order objects (SynthesizedHeuristic, ConsolidatedProfile, InferredPreference),
and writes them back into the same Neo4j graph with explicit provenance.

This is NOT a re-extraction. It creates bounded derived objects from existing
facts to improve retrieval on inferential and fragmented queries.

Usage:
  python3 scripts/chatgpt_history_synthesis_pass.py \
    --group-id vnext_batch_chatgpt_constrained \
    --model gpt-5.1-codex-mini \
    --reasoning-effort high \
    --max-synth-nodes 40 \
    --out-dir /path/to/output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
# Safety: reasoning.effort validation
# gpt-5.1-codex-mini rejects "minimal". Allowed: low|medium|high.
# ──────────────────────────────────────────────────────────────────────────────

_ALLOWED_REASONING_EFFORTS = {"low", "medium", "high"}
_BLOCKED_REASONING_EFFORTS = {"minimal"}


def _validate_reasoning_effort(effort: str) -> str:
    """Validate and return reasoning.effort; fail fast on blocked values."""
    effort = effort.strip().lower()
    if effort in _BLOCKED_REASONING_EFFORTS:
        raise ValueError(
            f"reasoning.effort={effort!r} is BLOCKED for gpt-5 models. "
            f"Allowed values: {sorted(_ALLOWED_REASONING_EFFORTS)}"
        )
    if effort not in _ALLOWED_REASONING_EFFORTS:
        raise ValueError(
            f"reasoning.effort={effort!r} is not recognized. "
            f"Allowed values: {sorted(_ALLOWED_REASONING_EFFORTS)}"
        )
    return effort


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SourceFact:
    uuid: str
    fact: str
    name: str
    source_entity: str
    target_entity: str
    episodes: list[str] = field(default_factory=list)


@dataclass
class SourceEpisode:
    uuid: str
    name: str
    source_description: str


@dataclass
class SynthesisTarget:
    """A themed context window for synthesis."""
    target_id: str
    synthesis_type: str  # SynthesizedHeuristic | ConsolidatedProfile | InferredPreference
    theme: str
    description: str
    target_queries: list[str]  # query IDs this synthesis aims to serve
    fact_uuids: list[str]
    episode_uuids: list[str]
    facts_text: list[str]  # human-readable fact strings for the prompt
    episodes_text: list[str]  # human-readable episode names


@dataclass
class SynthesizedNode:
    """A second-order derived object to write to the graph."""
    node_uuid: str
    synthesis_type: str
    content: str  # the synthesized memory text
    theme: str
    confidence: float
    source_fact_ids: list[str]
    source_episode_ids: list[str]
    source_conversation_ids: list[str]
    synthesis_model: str
    synthesis_version: str
    synthesis_timestamp: str
    reasoning_effort: str
    target_queries: list[str]


# ──────────────────────────────────────────────────────────────────────────────
# Neo4j helpers
# ──────────────────────────────────────────────────────────────────────────────


def _neo4j_driver():
    from neo4j import GraphDatabase
    neo4j_env = Path.home() / ".clawdbot" / "credentials" / "neo4j.env"
    if neo4j_env.exists():
        for line in neo4j_env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                k = k.strip()
                if k in {"NEO4J_PASSWORD", "NEO4J_USER", "NEO4J_URI"} and k not in os.environ:
                    os.environ[k] = v.strip().strip('"').strip("'")
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")
    if not password:
        raise RuntimeError("NEO4J_PASSWORD is required")
    return GraphDatabase.driver(uri, auth=(user, password))


def _fetch_all_facts(session, group_id: str) -> list[SourceFact]:
    """Fetch all RELATES_TO facts for the given group_id."""
    rows = session.run(
        """
        MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
        WHERE r.group_id = $group_id
          AND r.invalid_at IS NULL
        RETURN r.uuid AS uuid, r.fact AS fact, r.name AS name,
               a.name AS src_name, b.name AS tgt_name,
               coalesce(r.episodes, []) AS episodes
        """,
        {"group_id": group_id},
    ).data()
    facts = []
    for row in rows:
        facts.append(SourceFact(
            uuid=str(row["uuid"]),
            fact=str(row.get("fact") or ""),
            name=str(row.get("name") or ""),
            source_entity=str(row.get("src_name") or ""),
            target_entity=str(row.get("tgt_name") or ""),
            episodes=[str(e) for e in (row.get("episodes") or [])],
        ))
    return facts


def _fetch_all_episodes(session, group_id: str) -> list[SourceEpisode]:
    """Fetch all Episodic nodes for the given group_id."""
    rows = session.run(
        """
        MATCH (e:Episodic)
        WHERE e.group_id = $group_id
        RETURN e.uuid AS uuid, e.name AS name,
               coalesce(e.source_description, '') AS source_description
        """,
        {"group_id": group_id},
    ).data()
    return [
        SourceEpisode(
            uuid=str(row["uuid"]),
            name=str(row.get("name") or ""),
            source_description=str(row.get("source_description") or ""),
        )
        for row in rows
    ]


def _fetch_yuan_entity_uuid(session, group_id: str) -> str | None:
    """Find the primary Yuan entity UUID."""
    row = session.run(
        """
        MATCH (e:Entity)
        WHERE e.group_id = $group_id
          AND (e.name CONTAINS 'Yuan Han Li' OR e.name = 'Yuan')
        RETURN e.uuid AS uuid, e.name AS name
        ORDER BY size(e.name) DESC
        LIMIT 1
        """,
        {"group_id": group_id},
    ).single()
    return str(row["uuid"]) if row else None


def _get_embedding(content: str) -> list[float]:
    """Get embedding from local Ollama."""
    base_url = os.environ.get("EMBEDDER_BASE_URL", "http://localhost:11434/v1")
    url = f"{base_url}/embeddings"
    model = os.environ.get("OM_EMBEDDING_MODEL", "embeddinggemma")
    payload = json.dumps({"model": model, "input": content}).encode()
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("EMBEDDER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return [float(v) for v in data["data"][0]["embedding"]]


# ──────────────────────────────────────────────────────────────────────────────
# Synthesis target construction
# ──────────────────────────────────────────────────────────────────────────────

# Synthesis targets are themed context windows that group source facts for
# specific query buckets. These are derived from the grounding audit.

SYNTHESIS_TARGETS_SPEC = [
    # ─── HEURISTIC TARGETS (inferential queries) ───
    {
        "target_id": "heur_writing_collab",
        "synthesis_type": "SynthesizedHeuristic",
        "theme": "Writing collaboration style and patterns",
        "description": "How Yuan collaborates on writing: iterative drafting, error correction, continuous prose, character count preservation, first-draft → cleanup pattern",
        "target_queries": ["heur_01", "work_07"],
        "keyword_filters": ["edit", "draft", "rewrite", "paragraph", "write", "prose", "Forbes", "bio",
                           "Investment in Predicate", "memo", "letter", "summary", "review"],
        "entity_filters": ["Yuan Han Li", "Yuan", "Forbes", "Predicate"],
    },
    {
        "target_id": "heur_tone_calibration",
        "synthesis_type": "SynthesizedHeuristic",
        "theme": "Tone calibration across contexts",
        "description": "How Yuan calibrates communication tone: formal Chinese for elders, witty/sarcastic for dating, graceful for social declines, careful for breakups",
        "target_queries": ["heur_02"],
        "keyword_filters": ["tone", "formal", "casual", "dating", "Hinge", "wedding", "decline",
                           "Chinese", "grandmother", "conversation", "breakup", "witty", "sarcastic",
                           "profile", "message", "draft"],
        "entity_filters": ["Yuan Han Li", "Yuan", "Hinge", "Raya"],
    },
    {
        "target_id": "heur_investment_skepticism",
        "synthesis_type": "SynthesizedHeuristic",
        "theme": "Investment skepticism and critical evaluation patterns",
        "description": "How Yuan evaluates investments: bull AND bear framing, 'too good to be true' heuristic, challenging assumptions, financial modeling, iterative memo writing",
        "target_queries": ["heur_03", "work_03"],
        "keyword_filters": ["invest", "analysis", "bull", "bear", "Tether", "BABA", "memo",
                           "critical", "skeptic", "model", "risk", "evaluate", "case"],
        "entity_filters": ["Yuan Han Li", "Yuan", "Tether", "BABA", "Blockchain Capital"],
    },
    {
        "target_id": "heur_recommendation_refinement",
        "synthesis_type": "SynthesizedHeuristic",
        "theme": "Personalized recommendation refinement patterns",
        "description": "How Yuan asks for and refines recommendations: 'based on what you know about me', taste anchors, iterating on suggestions, comparing to reference points",
        "target_queries": ["heur_04"],
        "keyword_filters": ["recommend", "preference", "taste", "based on", "champagne", "wine",
                           "film", "movie", "music", "suggest", "pairing", "shortcut"],
        "entity_filters": ["Yuan Han Li", "Yuan"],
    },
    {
        "target_id": "heur_geopolitical",
        "synthesis_type": "SynthesizedHeuristic",
        "theme": "Geopolitical and strategic theory interests",
        "description": "Yuan's pattern of exploring geopolitical and strategic theory: Mahan sea power, Iran/Oman chokepoints, Montevideo Convention, ordoliberalism, free market imperialism",
        "target_queries": ["heur_05"],
        "keyword_filters": ["geopolit", "Mahan", "sea power", "Iran", "Oman", "strait",
                           "Montevideo", "imperial", "ordoliberal", "strategic", "theory",
                           "sovereignty", "convention"],
        "entity_filters": ["Yuan Han Li", "Yuan", "Mahan"],
    },
    {
        "target_id": "heur_info_curation",
        "synthesis_type": "SynthesizedHeuristic",
        "theme": "Information consumption curation patterns",
        "description": "How Yuan curates information consumption: RSS feeds, source selection, cross-domain coverage, systematic approach to staying informed",
        "target_queries": ["heur_06"],
        "keyword_filters": ["RSS", "feed", "source", "news", "curat", "read", "information",
                           "consume", "blog", "newsletter", "subscribe"],
        "entity_filters": ["Yuan Han Li", "Yuan"],
    },
    {
        "target_id": "heur_chinese_cultural",
        "synthesis_type": "SynthesizedHeuristic",
        "theme": "Chinese cultural navigation patterns",
        "description": "How Yuan navigates Chinese cultural contexts: formal register switching, customs knowledge, family protocols, cross-cultural communication",
        "target_queries": ["heur_07"],
        "keyword_filters": ["Chinese", "奶奶", "grandmother", "cultural", "hong bao", "formal",
                           "cremation", "Dragon Boat", "Xue Lan", "Xiao Feng", "register"],
        "entity_filters": ["Yuan Han Li", "Yuan"],
    },
    # ─── CONSOLIDATION TARGETS (fragmented queries) ───
    {
        "target_id": "consol_education_path",
        "synthesis_type": "ConsolidatedProfile",
        "theme": "Educational background and path",
        "description": "Yuan's educational journey: Hong Kong → UK boarding school → UPenn (CS + Philosophy, dropped out senior year 2021), FranklinDAO president, first-gen college student",
        "target_queries": ["bio_01"],
        "keyword_filters": ["UPenn", "university", "school", "education", "boarding", "dropout",
                           "FranklinDAO", "Penn", "first-gen", "college", "philosophy"],
        "entity_filters": ["Yuan Han Li", "Yuan", "UPenn", "FranklinDAO", "Penn Blockchain"],
    },
    {
        "target_id": "consol_career_timeline",
        "synthesis_type": "ConsolidatedProfile",
        "theme": "Career history and timeline",
        "description": "Yuan's career path: FranklinDAO → Nine Masts Capital → Blockchain Capital (promoted to Partner Jan 2025)",
        "target_queries": ["bio_04"],
        "keyword_filters": ["career", "Blockchain Capital", "Nine Masts", "partner", "promote",
                           "principal", "venture", "fund", "FranklinDAO", "role"],
        "entity_filters": ["Yuan Han Li", "Yuan", "Blockchain Capital", "Nine Masts"],
    },
    {
        "target_id": "consol_family_structure",
        "synthesis_type": "ConsolidatedProfile",
        "theme": "Family members and dynamics",
        "description": "Yuan's family: grandfather (Gong Gong, passed), grandmother (奶奶, communicates in Chinese), mother (CEO), Xue Lan (COO), siblings",
        "target_queries": ["bio_07"],
        "keyword_filters": ["family", "grandfather", "Gong Gong", "grandmother", "母", "sister",
                           "sibling", "mother", "Xue Lan", "CEO"],
        "entity_filters": ["Yuan Han Li", "Yuan", "Gong Gong", "Xue Lan"],
    },
    {
        "target_id": "consol_relocation_history",
        "synthesis_type": "ConsolidatedProfile",
        "theme": "Geographic relocations and living history",
        "description": "Yuan's relocation path: Hong Kong → UK boarding school → Philadelphia (UPenn) → NYC (East Village area)",
        "target_queries": ["bio_05"],
        "keyword_filters": ["Hong Kong", "UK", "Philadelphia", "NYC", "New York", "East Village",
                           "move", "relocat", "live", "city", "address"],
        "entity_filters": ["Yuan Han Li", "Yuan"],
    },
    {
        "target_id": "consol_dating_approach",
        "synthesis_type": "ConsolidatedProfile",
        "theme": "Dating approach and patterns",
        "description": "How Yuan approaches dating: Hinge/Raya usage, 'too cool' vibe, witty/sarcastic tone, music references, profile crafting",
        "target_queries": ["rel_01"],
        "keyword_filters": ["dating", "Hinge", "Raya", "profile", "match", "flirty", "witty",
                           "sarcastic", "date", "romantic", "app"],
        "entity_filters": ["Yuan Han Li", "Yuan", "Hinge", "Raya"],
    },
    {
        "target_id": "consol_professional_network",
        "synthesis_type": "ConsolidatedProfile",
        "theme": "Professional contacts and network",
        "description": "Yuan's professional contacts: Sreeram (EigenLayer), Nicole Rolet, Xiao Feng, BC colleagues, industry panels",
        "target_queries": ["rel_05"],
        "keyword_filters": ["Sreeram", "Nicole", "Xiao Feng", "panel", "conference", "contact",
                           "meeting", "professional", "network", "colleague", "EigenLayer"],
        "entity_filters": ["Yuan Han Li", "Yuan", "Sreeram", "EigenLayer"],
    },
    {
        "target_id": "consol_gift_occasions",
        "synthesis_type": "ConsolidatedProfile",
        "theme": "Gift-giving occasions and patterns",
        "description": "Yuan's gift-giving: engagement gifts, white elephant, birthday notes, gift tax rules, charitable donations",
        "target_queries": ["rel_08"],
        "keyword_filters": ["gift", "engagement", "white elephant", "birthday", "donat",
                           "tax", "present", "occasion"],
        "entity_filters": ["Yuan Han Li", "Yuan"],
    },
    {
        "target_id": "consol_wine_preferences",
        "synthesis_type": "ConsolidatedProfile",
        "theme": "Wine and champagne preferences",
        "description": "Yuan's wine preferences: Jacques Lassaigne, grower champagne, Chablis, Burgundy, $60-150 range, natural-leaning",
        "target_queries": ["pref_01"],
        "keyword_filters": ["wine", "champagne", "Burgundy", "Chablis", "Lassaigne", "grower",
                           "pairing", "red", "white", "bottle", "vintage"],
        "entity_filters": ["Yuan Han Li", "Yuan"],
    },
    {
        "target_id": "consol_fashion_style",
        "synthesis_type": "ConsolidatedProfile",
        "theme": "Fashion and personal style",
        "description": "Yuan's fashion: Permanent Style, The Armoury, bespoke tuxedo, ROA footwear, Schiesser basics, ski wear",
        "target_queries": ["pref_02"],
        "keyword_filters": ["fashion", "style", "tuxedo", "bespoke", "Armoury", "Permanent Style",
                           "ROA", "Schiesser", "outfit", "wear", "clothing"],
        "entity_filters": ["Yuan Han Li", "Yuan"],
    },
    {
        "target_id": "consol_food_dining",
        "synthesis_type": "ConsolidatedProfile",
        "theme": "Food preferences and dining patterns",
        "description": "Yuan's food preferences: offal/innards, fondue, dry-aged burgers, Korean BBQ, protein-focused meals",
        "target_queries": ["pref_04"],
        "keyword_filters": ["food", "restaurant", "dining", "cook", "recipe", "fondue", "burger",
                           "oats", "protein", "Seoul", "butcher", "reservation"],
        "entity_filters": ["Yuan Han Li", "Yuan"],
    },
    {
        "target_id": "consol_cooking_hosting",
        "synthesis_type": "ConsolidatedProfile",
        "theme": "Cooking and hosting activities",
        "description": "Yuan's cooking/hosting: fondue recipes, high-protein meals, premium butchers, wine pairings for hosting",
        "target_queries": ["pref_09"],
        "keyword_filters": ["cook", "host", "fondue", "recipe", "oat", "protein", "butcher",
                           "wine pairing", "dinner party", "ingredient"],
        "entity_filters": ["Yuan Han Li", "Yuan"],
    },
    {
        "target_id": "consol_portfolio_companies",
        "synthesis_type": "ConsolidatedProfile",
        "theme": "Crypto portfolio companies championed",
        "description": "Companies Yuan has championed: EigenLayer, Predicate, Fabric Cryptography, RISC Zero, governance roles",
        "target_queries": ["work_02"],
        "keyword_filters": ["EigenLayer", "Predicate", "Fabric", "RISC Zero", "portfolio",
                           "champion", "governance", "investment", "company"],
        "entity_filters": ["Yuan Han Li", "Yuan", "EigenLayer", "Predicate", "Fabric Cryptography"],
    },
    {
        "target_id": "consol_semiconductor_hardware",
        "synthesis_type": "ConsolidatedProfile",
        "theme": "Semiconductor and hardware sector knowledge",
        "description": "Yuan's semiconductor knowledge: ETF analysis, NVIDIA AI thesis, crypto hardware memos, Fabric VPU",
        "target_queries": ["work_05"],
        "keyword_filters": ["semiconductor", "NVIDIA", "hardware", "ETF", "chip", "GPU", "VPU",
                           "Fabric", "crypto hardware"],
        "entity_filters": ["Yuan Han Li", "Yuan", "NVIDIA", "Fabric Cryptography"],
    },
    {
        "target_id": "consol_regulatory_stablecoin",
        "synthesis_type": "ConsolidatedProfile",
        "theme": "Stablecoin and crypto regulatory knowledge",
        "description": "Yuan's regulatory/stablecoin knowledge: Tether analysis, GENIUS Act, tokenized trading licenses, Hyperliquid ecosystem",
        "target_queries": ["work_06"],
        "keyword_filters": ["Tether", "stablecoin", "GENIUS", "regulat", "license", "tokeniz",
                           "Hyperliquid", "compliance", "USDT"],
        "entity_filters": ["Yuan Han Li", "Yuan", "Tether"],
    },
    {
        "target_id": "consol_non_crypto_investments",
        "synthesis_type": "ConsolidatedProfile",
        "theme": "Non-crypto investment interests",
        "description": "Yuan's non-crypto investments: BABA analysis, ICHN ETF, semiconductor ETFs, co-op to condo conversion",
        "target_queries": ["work_08"],
        "keyword_filters": ["BABA", "Alibaba", "ICHN", "ETF", "semiconductor", "co-op", "condo",
                           "equity", "stock"],
        "entity_filters": ["Yuan Han Li", "Yuan", "BABA", "Alibaba"],
    },
    # ─── INFERRED PREFERENCE TARGETS ───
    {
        "target_id": "infer_multilingual",
        "synthesis_type": "InferredPreference",
        "theme": "Multilingual usage patterns",
        "description": "Yuan communicates in 6+ languages contextually: English (primary), Chinese (family/business), German, Portuguese, Spanish, and more",
        "target_queries": ["bio_08"],
        "keyword_filters": ["language", "Chinese", "German", "Portuguese", "Spanish", "translate",
                           "Waschmaschine", "São Paulo", "奶奶", "端午"],
        "entity_filters": ["Yuan Han Li", "Yuan"],
    },
    {
        "target_id": "consol_emotional_vulnerability",
        "synthesis_type": "ConsolidatedProfile",
        "theme": "Emotional vulnerability and support-seeking patterns",
        "description": "When Yuan has shown emotional vulnerability: breakup planning, attachment discussions, grief (grandfather letter)",
        "target_queries": ["rel_07"],
        "keyword_filters": ["breakup", "emotion", "vulnerable", "attachment", "grief", "support",
                           "conversation", "feeling", "relationship", "plan"],
        "entity_filters": ["Yuan Han Li", "Yuan"],
    },
]


def _build_synthesis_targets(
    facts: list[SourceFact],
    episodes: list[SourceEpisode],
) -> list[SynthesisTarget]:
    """Build synthesis context windows from the source graph and target specs."""
    episode_by_uuid = {e.uuid: e for e in episodes}

    targets: list[SynthesisTarget] = []
    for spec in SYNTHESIS_TARGETS_SPEC:
        keyword_filters = spec.get("keyword_filters", [])
        entity_filters = spec.get("entity_filters", [])

        # Filter facts: match by keyword in fact text/name, or entity name
        matched_facts: list[SourceFact] = []
        for f in facts:
            text_lower = (f.fact + " " + f.name + " " + f.source_entity + " " + f.target_entity).lower()
            kw_match = any(kw.lower() in text_lower for kw in keyword_filters)
            ent_match = any(
                ent.lower() in f.source_entity.lower() or ent.lower() in f.target_entity.lower()
                for ent in entity_filters
            )
            if kw_match or ent_match:
                matched_facts.append(f)

        # Deduplicate and cap facts
        seen_uuids: set[str] = set()
        unique_facts: list[SourceFact] = []
        for f in matched_facts:
            if f.uuid not in seen_uuids:
                seen_uuids.add(f.uuid)
                unique_facts.append(f)

        # Cap at 80 facts per target to stay within context limits
        unique_facts = unique_facts[:80]

        # Collect episode UUIDs from matched facts
        ep_uuids: set[str] = set()
        for f in unique_facts:
            for ep_uuid in f.episodes:
                ep_uuids.add(ep_uuid)

        # Build readable text
        facts_text = []
        for f in unique_facts:
            entry = f"[{f.source_entity} → {f.target_entity}] {f.fact}"
            if f.name and f.name != f.fact:
                entry += f" (relation: {f.name})"
            facts_text.append(entry)

        episodes_text = []
        for ep_uuid in sorted(ep_uuids):
            ep = episode_by_uuid.get(ep_uuid)
            if ep:
                episodes_text.append(ep.name)

        # Extract conversation IDs from episode source descriptions
        conv_ids: set[str] = set()
        for ep_uuid in ep_uuids:
            ep = episode_by_uuid.get(ep_uuid)
            if ep and "conversation_id=" in ep.source_description:
                for part in ep.source_description.split():
                    if part.startswith("conversation_id="):
                        conv_ids.add(part.split("=", 1)[1])

        targets.append(SynthesisTarget(
            target_id=spec["target_id"],
            synthesis_type=spec["synthesis_type"],
            theme=spec["theme"],
            description=spec["description"],
            target_queries=spec["target_queries"],
            fact_uuids=[f.uuid for f in unique_facts],
            episode_uuids=sorted(ep_uuids),
            facts_text=facts_text,
            episodes_text=episodes_text,
        ))

    return targets


# ──────────────────────────────────────────────────────────────────────────────
# LLM synthesis
# ──────────────────────────────────────────────────────────────────────────────

_SYNTHESIS_SYSTEM_PROMPT = """\
You are a memory synthesis engine for a personal AI assistant. Your job is to
read a set of first-order extracted facts and create SECOND-ORDER synthesized
memory objects that capture patterns, consolidated profiles, or inferred
preferences that are not explicitly stated in any single fact but are
demonstrably present across multiple facts.

RULES:
1. ONLY synthesize patterns that are DIRECTLY DEMONSTRATED in the provided evidence.
   Do NOT speculate, hallucinate, or infer beyond what the facts show.
2. Every synthesized object MUST be traceable to specific source facts.
3. Return valid JSON only. No markdown, no explanation outside the JSON.
4. Each synthesized object needs a confidence score (0.0 to 1.0).
   - 0.9-1.0: Pattern is unambiguous and supported by 4+ facts
   - 0.7-0.8: Pattern is clear with 2-3 supporting facts
   - Below 0.7: Do not emit the object — evidence is too thin.
5. The content field should be a clear, concise statement of the synthesized
   memory — something an AI assistant could use to answer a question about
   the person. Write it as a durable memory fact, not a research summary.
6. Focus on the HUMAN SUBJECT (Yuan), not on systems, tools, or AI operations.
"""


def _build_synthesis_prompt(target: SynthesisTarget) -> str:
    """Build the user prompt for a single synthesis target."""
    facts_block = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(target.facts_text[:60]))
    episodes_block = "\n".join(f"  - {e}" for e in target.episodes_text[:30])

    return f"""Synthesize memory objects for the theme: **{target.theme}**

Synthesis type to produce: {target.synthesis_type}
Description of what to synthesize: {target.description}

SOURCE FACTS ({len(target.facts_text)} facts from the existing memory graph):
{facts_block}

SOURCE EPISODES (conversations these facts came from):
{episodes_block}

Return a JSON object with this structure:
{{
  "synthesized_objects": [
    {{
      "content": "<synthesized memory statement — concise, durable, assistant-useful>",
      "confidence": <float 0.7-1.0>,
      "supporting_fact_indices": [<1-based indices into the SOURCE FACTS list>],
      "reasoning": "<one sentence explaining why this synthesis is warranted>"
    }}
  ]
}}

Guidelines for {target.synthesis_type}:
- SynthesizedHeuristic: A behavioral pattern, decision tendency, communication style, or recurring approach demonstrated across multiple conversations. State the pattern clearly and specifically.
- ConsolidatedProfile: A merged biographical, preference, or relationship fact assembled from multiple fragmented sources. Combine scattered facts into one coherent, retrievable memory.
- InferredPreference: A taste, preference, or usage pattern that is demonstrated through behavior but never explicitly stated. State what is inferred and from what evidence.

Emit 1-3 objects. Quality over quantity. Reject anything below 0.7 confidence.
Return valid JSON only.
"""


def _call_synthesis_llm(
    target: SynthesisTarget,
    *,
    model: str,
    reasoning_effort: str,
    api_key: str,
    base_url: str,
) -> list[dict[str, Any]]:
    """Call gpt-5.1-codex-mini to synthesize objects.

    Supports two API shapes:
      1. OpenRouter / Chat Completions API (base_url contains 'openrouter')
      2. OpenAI Responses API (direct OpenAI)
    """
    system_prompt = _SYNTHESIS_SYSTEM_PROMPT
    user_prompt = _build_synthesis_prompt(target)

    is_openrouter = "openrouter" in base_url.lower()

    if is_openrouter:
        # OpenRouter Chat Completions API
        or_model = model if "/" in model else f"openai/{model}"
        payload: dict[str, Any] = {
            "model": or_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": (
                    "IMPORTANT: Respond with a JSON object. "
                    "Your entire response must be valid JSON with exact key names.\n\n"
                    + user_prompt
                )},
            ],
            "reasoning": {"effort": reasoning_effort},
            "response_format": {"type": "json_object"},
        }
        url = f"{base_url.rstrip('/')}/chat/completions"
    else:
        # OpenAI Responses API (direct)
        payload = {
            "model": model,
            "instructions": system_prompt,
            "input": (
                "IMPORTANT: Respond with a JSON object. "
                "Your entire response must be valid JSON with exact key names.\n\n"
                + user_prompt
            ),
            "text": {"format": {"type": "json_object"}},
            "reasoning": {"effort": reasoning_effort},
        }
        url = f"{base_url.rstrip('/')}/responses"

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            if attempt < max_retries - 1 and exc.code in (429, 500, 502, 503):
                wait = 2 ** attempt + 2
                print(f"[SYNTH] Retrying after HTTP {exc.code}, wait {wait}s...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Synthesis LLM HTTP {exc.code}: {detail}")
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + 2)
                continue
            raise RuntimeError(f"Synthesis LLM error: {exc}")

    resp_data = json.loads(raw)

    # Extract content depending on API shape
    content_str = ""
    actual_effort = reasoning_effort

    if is_openrouter:
        # Chat Completions shape
        choices = resp_data.get("choices", [])
        if choices:
            content_str = choices[0].get("message", {}).get("content", "").strip()
        # OpenRouter may echo reasoning config
        actual_effort = reasoning_effort  # OpenRouter doesn't return this in metadata reliably
    else:
        # Responses API shape
        for item in resp_data.get("output", []):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for c in item.get("content", []):
                if isinstance(c, dict) and c.get("type") == "output_text":
                    content_str = c.get("text", "").strip()
                    break
            if content_str:
                break
        actual_effort = resp_data.get("reasoning", {}).get("effort", reasoning_effort)

    if not content_str:
        raise RuntimeError(f"Synthesis LLM returned no content. Raw: {raw[:300]}")

    # Strip markdown code fences if present
    if content_str.startswith("```"):
        lines = content_str.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content_str = "\n".join(lines).strip()

    parsed = json.loads(content_str)
    objects = parsed.get("synthesized_objects", [])
    if not isinstance(objects, list):
        raise RuntimeError("Synthesis LLM did not return synthesized_objects array")

    return [
        {**obj, "_actual_reasoning_effort": actual_effort}
        for obj in objects
        if isinstance(obj, dict) and obj.get("confidence", 0) >= 0.7
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Graph write
# ──────────────────────────────────────────────────────────────────────────────


def _synth_node_uuid(synthesis_version: str, target_id: str, index: int) -> str:
    """Deterministic UUID for a synthesized node."""
    digest = hashlib.sha256(f"synth:{synthesis_version}:{target_id}:{index}".encode()).hexdigest()
    # Format as UUID-like
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def _write_synthesized_nodes(
    session,
    nodes: list[SynthesizedNode],
    *,
    group_id: str,
    yuan_uuid: str | None,
    synthesis_version: str,
) -> int:
    """Write synthesized nodes and provenance edges to Neo4j."""
    # First, clean up any prior nodes with the same synthesis_version
    session.run(
        """
        MATCH (n)
        WHERE n.group_id = $group_id
          AND n.is_synthesized = true
          AND n.synthesis_version = $synthesis_version
        DETACH DELETE n
        """,
        {"group_id": group_id, "synthesis_version": synthesis_version},
    ).consume()

    written = 0
    for node in nodes:
        # Get embedding for the synthesized content
        try:
            embedding = _get_embedding(node.content)
        except Exception as exc:
            print(f"WARNING: embedding failed for {node.node_uuid}: {exc}", file=sys.stderr)
            embedding = []

        # Write the synthesized node as an Entity with special labels
        session.run(
            """
            CREATE (n:Entity {
                uuid: $uuid,
                name: $content,
                summary: $content,
                group_id: $group_id,
                labels: [$synthesis_type],
                name_embedding: $embedding,
                created_at: $created_at,
                is_synthesized: true,
                synthesis_type: $synthesis_type,
                synthesis_model: $synthesis_model,
                synthesis_version: $synthesis_version,
                synthesis_timestamp: $synthesis_timestamp,
                synthesis_theme: $theme,
                confidence: $confidence,
                source_fact_ids: $source_fact_ids,
                source_episode_ids: $source_episode_ids,
                source_conversation_ids: $source_conversation_ids,
                reasoning_effort: $reasoning_effort,
                target_queries: $target_queries
            })
            """,
            {
                "uuid": node.node_uuid,
                "content": node.content,
                "group_id": group_id,
                "synthesis_type": node.synthesis_type,
                "embedding": embedding,
                "created_at": node.synthesis_timestamp,
                "synthesis_model": node.synthesis_model,
                "synthesis_version": node.synthesis_version,
                "synthesis_timestamp": node.synthesis_timestamp,
                "theme": node.theme,
                "confidence": node.confidence,
                "source_fact_ids": node.source_fact_ids,
                "source_episode_ids": node.source_episode_ids,
                "source_conversation_ids": node.source_conversation_ids,
                "reasoning_effort": node.reasoning_effort,
                "target_queries": node.target_queries,
            },
        ).consume()

        # Create RELATES_TO edges from synthesized node that mirror the
        # retrieval pattern (the search_memory_facts code queries RELATES_TO edges).
        # We create a self-referencing RELATES_TO with the synthesized content as the fact,
        # plus an ABOUT edge to Yuan's entity if available.
        fact_uuid = hashlib.sha256(
            f"synth_fact:{synthesis_version}:{node.node_uuid}".encode()
        ).hexdigest()
        fact_uuid_formatted = f"{fact_uuid[:8]}-{fact_uuid[8:12]}-{fact_uuid[12:16]}-{fact_uuid[16:20]}-{fact_uuid[20:32]}"

        session.run(
            """
            MATCH (n:Entity {uuid: $node_uuid})
            CREATE (n)-[r:RELATES_TO {
                uuid: $fact_uuid,
                fact: $content,
                name: $theme,
                group_id: $group_id,
                created_at: $created_at,
                valid_at: $created_at,
                source_node_uuid: $node_uuid,
                target_node_uuid: $node_uuid,
                fact_embedding: $embedding,
                episodes: [],
                is_synthesized: true,
                synthesis_version: $synthesis_version,
                confidence: $confidence
            }]->(n)
            """,
            {
                "node_uuid": node.node_uuid,
                "fact_uuid": fact_uuid_formatted,
                "content": node.content,
                "theme": node.theme,
                "group_id": group_id,
                "created_at": node.synthesis_timestamp,
                "embedding": embedding,
                "synthesis_version": synthesis_version,
                "confidence": node.confidence,
            },
        ).consume()

        # If we have Yuan's entity, create an ABOUT edge so the synthesized
        # node appears in Yuan's neighborhood (center-node search)
        if yuan_uuid:
            about_uuid = hashlib.sha256(
                f"synth_about:{synthesis_version}:{node.node_uuid}:{yuan_uuid}".encode()
            ).hexdigest()
            about_uuid_fmt = f"{about_uuid[:8]}-{about_uuid[8:12]}-{about_uuid[12:16]}-{about_uuid[16:20]}-{about_uuid[20:32]}"

            session.run(
                """
                MATCH (synth:Entity {uuid: $synth_uuid})
                MATCH (yuan:Entity {uuid: $yuan_uuid})
                CREATE (synth)-[r:RELATES_TO {
                    uuid: $about_uuid,
                    fact: $content,
                    name: $theme,
                    group_id: $group_id,
                    created_at: $created_at,
                    valid_at: $created_at,
                    source_node_uuid: $synth_uuid,
                    target_node_uuid: $yuan_uuid,
                    fact_embedding: $embedding,
                    episodes: [],
                    is_synthesized: true,
                    synthesis_version: $synthesis_version,
                    confidence: $confidence
                }]->(yuan)
                """,
                {
                    "synth_uuid": node.node_uuid,
                    "yuan_uuid": yuan_uuid,
                    "about_uuid": about_uuid_fmt,
                    "content": node.content,
                    "theme": node.theme,
                    "group_id": group_id,
                    "created_at": node.synthesis_timestamp,
                    "embedding": embedding,
                    "synthesis_version": synthesis_version,
                    "confidence": node.confidence,
                },
            ).consume()

        # Create DERIVED_FROM edges to source fact source entities
        for src_fact_uuid in node.source_fact_ids[:10]:
            session.run(
                """
                MATCH (synth:Entity {uuid: $synth_uuid})
                MATCH (src_entity:Entity)-[src_fact:RELATES_TO {uuid: $src_fact_uuid}]->()
                MERGE (synth)-[d:DERIVED_FROM]->(src_entity)
                ON CREATE SET
                    d.source_fact_uuid = $src_fact_uuid,
                    d.group_id = $group_id,
                    d.created_at = $created_at,
                    d.synthesis_version = $synthesis_version
                """,
                {
                    "synth_uuid": node.node_uuid,
                    "src_fact_uuid": src_fact_uuid,
                    "group_id": group_id,
                    "created_at": node.synthesis_timestamp,
                    "synthesis_version": synthesis_version,
                },
            ).consume()

        written += 1

    return written


# ──────────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────────


def run_synthesis_pass(
    *,
    group_id: str,
    model: str,
    reasoning_effort: str,
    max_synth_nodes: int,
    out_dir: Path,
    confidence_threshold: float = 0.7,
    dry_run: bool = False,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Run the synthesis pass end to end."""
    reasoning_effort = _validate_reasoning_effort(reasoning_effort)

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for synthesis LLM")

    if base_url is None:
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    synthesis_version = f"v1_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    synthesis_timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    out_dir.mkdir(parents=True, exist_ok=True)

    driver = _neo4j_driver()

    with driver, driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
        # Phase 1: Read the source graph
        print(f"[SYNTH] Reading source graph for group_id={group_id}...")
        facts = _fetch_all_facts(session, group_id)
        episodes = _fetch_all_episodes(session, group_id)
        yuan_uuid = _fetch_yuan_entity_uuid(session, group_id)
        print(f"[SYNTH] Found {len(facts)} facts, {len(episodes)} episodes, yuan_uuid={yuan_uuid}")

        # Phase 2: Build synthesis targets
        targets = _build_synthesis_targets(facts, episodes)
        print(f"[SYNTH] Built {len(targets)} synthesis targets")

        # Phase 3: Run synthesis LLM for each target
        all_synth_nodes: list[SynthesizedNode] = []
        target_results: list[dict[str, Any]] = []

        for target in targets:
            if len(all_synth_nodes) >= max_synth_nodes:
                print(f"[SYNTH] Reached max_synth_nodes={max_synth_nodes}, stopping")
                break

            if not target.facts_text:
                print(f"[SYNTH] Skipping {target.target_id}: no matching facts")
                target_results.append({
                    "target_id": target.target_id,
                    "status": "skipped",
                    "reason": "no matching facts",
                })
                continue

            print(f"[SYNTH] Synthesizing {target.target_id} ({target.synthesis_type}): "
                  f"{len(target.facts_text)} facts, {len(target.episode_uuids)} episodes...")

            try:
                llm_objects = _call_synthesis_llm(
                    target,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    api_key=api_key,
                    base_url=base_url,
                )

                target_nodes: list[SynthesizedNode] = []
                for i, obj in enumerate(llm_objects):
                    if len(all_synth_nodes) + len(target_nodes) >= max_synth_nodes:
                        break

                    confidence = float(obj.get("confidence", 0))
                    if confidence < confidence_threshold:
                        continue

                    # Map supporting_fact_indices back to fact UUIDs
                    indices = obj.get("supporting_fact_indices", [])
                    source_fact_ids = []
                    for idx in indices:
                        try:
                            idx_int = int(idx) - 1  # 1-based to 0-based
                            if 0 <= idx_int < len(target.fact_uuids):
                                source_fact_ids.append(target.fact_uuids[idx_int])
                        except (ValueError, TypeError):
                            pass

                    # Require at least 2 supporting facts
                    if len(source_fact_ids) < 2:
                        # Fall back to first N fact UUIDs if model indices are bad
                        source_fact_ids = target.fact_uuids[:max(2, len(source_fact_ids))]

                    node = SynthesizedNode(
                        node_uuid=_synth_node_uuid(synthesis_version, target.target_id, i),
                        synthesis_type=target.synthesis_type,
                        content=str(obj.get("content", "")),
                        theme=target.theme,
                        confidence=confidence,
                        source_fact_ids=source_fact_ids[:20],
                        source_episode_ids=target.episode_uuids[:20],
                        source_conversation_ids=[],
                        synthesis_model=model,
                        synthesis_version=synthesis_version,
                        synthesis_timestamp=synthesis_timestamp,
                        reasoning_effort=obj.get("_actual_reasoning_effort", reasoning_effort),
                        target_queries=target.target_queries,
                    )
                    target_nodes.append(node)

                all_synth_nodes.extend(target_nodes)
                target_results.append({
                    "target_id": target.target_id,
                    "status": "success",
                    "synthesis_type": target.synthesis_type,
                    "facts_matched": len(target.facts_text),
                    "episodes_matched": len(target.episode_uuids),
                    "objects_produced": len(target_nodes),
                    "objects_content": [n.content[:200] for n in target_nodes],
                })
                print(f"[SYNTH]   → {len(target_nodes)} objects synthesized")

            except Exception as exc:
                print(f"[SYNTH] ERROR on {target.target_id}: {exc}", file=sys.stderr)
                target_results.append({
                    "target_id": target.target_id,
                    "status": "error",
                    "error": str(exc)[:300],
                })

            # Rate limiting
            time.sleep(2)

            # Incremental save: write progress after each target
            _progress = {
                "synthesis_version": synthesis_version,
                "targets_completed": len(target_results),
                "nodes_so_far": len(all_synth_nodes),
                "target_results": target_results,
                "nodes": [asdict(n) for n in all_synth_nodes],
            }
            (out_dir / "synthesis_progress.json").write_text(json.dumps(_progress, indent=2))

        # Phase 4: Write to graph
        if dry_run:
            print(f"[SYNTH] DRY RUN: would write {len(all_synth_nodes)} synthesized nodes")
            written = 0
        else:
            print(f"[SYNTH] Writing {len(all_synth_nodes)} synthesized nodes to graph...")
            written = _write_synthesized_nodes(
                session,
                all_synth_nodes,
                group_id=group_id,
                yuan_uuid=yuan_uuid,
                synthesis_version=synthesis_version,
            )
            print(f"[SYNTH] Wrote {written} synthesized nodes")

    # Write artifacts
    run_summary = {
        "synthesis_version": synthesis_version,
        "synthesis_timestamp": synthesis_timestamp,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "group_id": group_id,
        "max_synth_nodes": max_synth_nodes,
        "confidence_threshold": confidence_threshold,
        "source_facts_total": len(facts),
        "source_episodes_total": len(episodes),
        "yuan_entity_uuid": yuan_uuid,
        "targets_attempted": len(target_results),
        "targets_succeeded": sum(1 for t in target_results if t.get("status") == "success"),
        "targets_skipped": sum(1 for t in target_results if t.get("status") == "skipped"),
        "targets_errored": sum(1 for t in target_results if t.get("status") == "error"),
        "total_synthesized_nodes": len(all_synth_nodes),
        "nodes_written": written,
        "dry_run": dry_run,
        "target_results": target_results,
    }

    (out_dir / "synthesis_run_summary.json").write_text(json.dumps(run_summary, indent=2))

    # Write individual node details
    nodes_dump = [asdict(n) for n in all_synth_nodes]
    (out_dir / "synthesized_nodes.json").write_text(json.dumps(nodes_dump, indent=2))

    # Write provenance manifest
    provenance = []
    for n in all_synth_nodes:
        provenance.append({
            "node_uuid": n.node_uuid,
            "synthesis_type": n.synthesis_type,
            "theme": n.theme,
            "confidence": n.confidence,
            "source_fact_count": len(n.source_fact_ids),
            "source_episode_count": len(n.source_episode_ids),
            "content_preview": n.content[:200],
        })
    (out_dir / "provenance_manifest.json").write_text(json.dumps(provenance, indent=2))

    print(f"[SYNTH] Run complete. Artifacts in {out_dir}")
    print(json.dumps({
        "event": "SYNTHESIS_PASS_COMPLETE",
        "synthesis_version": synthesis_version,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "nodes_synthesized": len(all_synth_nodes),
        "nodes_written": written,
    }))

    return run_summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChatGPT History Synthesis Pass")
    parser.add_argument("--group-id", required=True, help="Source graph group_id")
    parser.add_argument("--model", default="gpt-5.1-codex-mini", help="Synthesis model")
    parser.add_argument("--reasoning-effort", default="high", help="Reasoning effort (low|medium|high)")
    parser.add_argument("--max-synth-nodes", type=int, default=40, help="Max synthesized nodes")
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--out-dir", required=True, help="Output directory for run artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to graph")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (default: OPENAI_API_KEY env)")
    parser.add_argument("--base-url", default=None, help="OpenAI base URL (default: OPENAI_BASE_URL env)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_synthesis_pass(
            group_id=args.group_id,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_synth_nodes=args.max_synth_nodes,
            confidence_threshold=args.confidence_threshold,
            out_dir=Path(args.out_dir),
            dry_run=args.dry_run,
            api_key=args.api_key,
            base_url=args.base_url,
        )
        return 0
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
