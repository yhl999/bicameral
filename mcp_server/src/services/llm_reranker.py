"""
LLMRerankerService — production LLM reranker for the hybrid default surface.

Takes the RRF-merged candidate pool from hybrid retrieval and applies
LLM-based relevance scoring to produce a semantically reranked final list.

Design goals:
- Async-first (production MCP surface is async).
- Fail-soft: if LLM reranking fails, return the RRF-ordered pool unchanged
  with diagnostics attached.
- Deterministic fallback: same inputs → same fallback output.
- Bounded LLM calls: one batched call per query (not per candidate).
- Clean abstraction: no benchmark-script imports, no experiment harness deps.

Configuration is via environment variables:
- BICAMERAL_RERANK_MODEL: model id (default: gpt-5.4-nano)
- BICAMERAL_RERANK_API_KEY: API key (falls back to OPENAI_API_KEY)
- BICAMERAL_RERANK_API_BASE: API base URL (auto-detected from key prefix)
- BICAMERAL_RERANK_ENABLED: '1' to enable, '0' to disable (default: '1')
- BICAMERAL_RERANK_TIMEOUT: request timeout seconds (default: 30)

When no API key is available or reranking is disabled, the service returns
the RRF-ordered pool unchanged (passthrough mode), not an error.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Query-type classification ─────────────────────────────────────────────────

QUERY_TYPES = frozenset({
    "person", "project", "event", "decision",
    "technical", "financial", "preference", "generic",
})

_CLASSIFY_SYSTEM_PROMPT = (
    "Classify the user's query into exactly one of these types: "
    "person, project, event, decision, technical, financial, preference, generic.\n\n"
    "Definitions:\n"
    "- person: about a specific person (who is X, what do I know about X, has X contacted me)\n"
    "- project: about a project's status, timeline, stakeholders, or deliverables\n"
    "- event: about a specific event — when, who attended, what happened\n"
    "- decision: about why a decision was made, tradeoffs, alternatives considered\n"
    "- technical: how something works, architecture, specs, debugging\n"
    "- financial: prices, budgets, valuations, P&L, financial figures\n"
    "- preference: personal opinions, tastes, likes/dislikes\n"
    "- generic: mixed, unclear, or doesn't fit other types\n\n"
    "Respond with ONLY the single type word, nothing else."
)

# ── TTL cache for query-type classification ───────────────────────────────────

_CLASSIFY_CACHE: dict[str, tuple[str, float]] = {}
_CLASSIFY_CACHE_TTL = 3600.0  # 1 hour


def _cache_get(query_text: str) -> str | None:
    """Return cached query type if still valid, else None."""
    key = hashlib.sha256(query_text.encode()).hexdigest()
    entry = _CLASSIFY_CACHE.get(key)
    if entry is None:
        return None
    value, ts = entry
    if time.monotonic() - ts > _CLASSIFY_CACHE_TTL:
        _CLASSIFY_CACHE.pop(key, None)
        return None
    return value


def _cache_set(query_text: str, query_type: str) -> None:
    """Store a classification result with monotonic timestamp."""
    key = hashlib.sha256(query_text.encode()).hexdigest()
    _CLASSIFY_CACHE[key] = (query_type, time.monotonic())


# ── Type-specific scoring rules ───────────────────────────────────────────────

_TYPE_SCORING_RULES: dict[str, str] = {
    "person": (
        "Reward facts about the person's background, role, interactions, opinions, "
        "or personal details. Penalize generic org overviews or project descriptions "
        "that merely mention the person's name."
    ),
    "project": (
        "Reward facts about project status, timeline, stakeholders, milestones, "
        "and decisions. Penalize general company descriptions or unrelated "
        "personal context."
    ),
    "event": (
        "Reward facts with specific dates, attendees, outcomes, agendas, "
        "and what happened. Penalize organizational history or generic context "
        "that doesn't reference the event."
    ),
    "decision": (
        "Reward facts containing rationale, tradeoffs, alternatives considered, "
        "and who decided. Penalize implementation details or procedural steps "
        "unrelated to the decision context."
    ),
    "technical": (
        "Reward explanations, architecture details, specs, debugging info, "
        "and how things work. Penalize unrelated organizational or personal facts."
    ),
    "financial": (
        "Reward specific numbers, budgets, valuations, pricing, revenue, "
        "and financial metrics. Penalize narrative context without quantitative data."
    ),
    "preference": (
        "Reward stated opinions, tastes, aversions, and personal preferences. "
        "Penalize purely factual or procedural information that doesn't reflect "
        "a personal stance."
    ),
    "generic": (
        "No type-specific penalty. Reward specificity, recency, and direct "
        "relevance to the query intent."
    ),
}

# ── Configuration defaults ────────────────────────────────────────────────────

_DEFAULT_MODEL = "gpt-5.4-nano"
_DEFAULT_TIMEOUT = 30
_DEFAULT_BATCH_SIZE = 20
_MAX_RETRIES = 2
_RETRY_BASE_DELAY = 1.0

# ── System prompt (v2 — type-aware) ───────────────────────────────────────────

_RERANK_SYSTEM_PROMPT_BASE = (
    "You are a relevance judge for a personal AI memory retrieval system. "
    "Given a query and a numbered list of memory candidates, "
    "score EACH candidate's relevance to the query.\n\n"
    "Respond with ONLY a JSON array. Each element must have:\n"
    '  {"index": <int>, "score": <float 0.0-1.0>, "rationale": "<brief reason>"}\n\n'
    "Scoring rules:\n"
    "- 0.8-1.0: candidate directly and substantially answers the query\n"
    "- 0.5-0.7: candidate contains partially relevant information\n"
    "- 0.2-0.4: candidate is tangentially related\n"
    "- 0.0-0.1: candidate is irrelevant to the query\n"
    "- Return one object per candidate, in order of index\n"
    "- Be strict: high scores only for candidates that genuinely help answer the query\n\n"
    "General penalties (apply always):\n"
    "- Penalize generic/organizational facts (e.g., company overview, public knowledge) "
    "unless the query specifically asks for that\n"
    "- Penalize facts that are lexically similar but off-type "
    "(e.g., project description when query is about a person)\n"
    "- Penalize vague/abstract facts without specific details\n"
    "- Penalize metadata-only facts (e.g., 'directory of' without content)\n\n"
    "General rewards (apply always):\n"
    "- Reward facts that directly answer the query\n"
    "- Reward specific, recent, and actionable context\n"
    "- Reward facts that match the query type"
)


def _build_type_aware_system_prompt(query_type: str) -> str:
    """Build the full system prompt with type-specific scoring rules injected.

    Args:
        query_type: One of the 8 recognized query types.

    Returns:
        Complete system prompt string with type-specific guidance.
    """
    type_rule = _TYPE_SCORING_RULES.get(query_type, _TYPE_SCORING_RULES["generic"])
    return (
        f"{_RERANK_SYSTEM_PROMPT_BASE}\n\n"
        f"Query type: {query_type}\n"
        f"Type-specific scoring guidance:\n"
        f"- {type_rule}"
    )


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class RerankedCandidate:
    """A hybrid candidate with LLM rerank scores attached."""

    original: dict[str, Any]
    llm_score: float
    rrf_score: float
    blended_score: float
    final_rank: int
    rationale: str = ""

    def to_annotated_dict(self) -> dict[str, Any]:
        """Return the original candidate dict with rerank annotations added."""
        result = dict(self.original)
        result["_rerank_score"] = round(self.llm_score, 4)
        result["_blended_score"] = round(self.blended_score, 4)
        result["_final_rank"] = self.final_rank
        if self.rationale:
            result["_rerank_rationale"] = self.rationale
        return result


@dataclass
class RerankResult:
    """Output of the LLM rerank pass for a single query."""

    candidates: list[dict[str, Any]]  # Annotated candidate dicts
    total_scored: int
    method: str  # 'llm' | 'passthrough' | 'fallback'
    model: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_degraded(self) -> bool:
        return self.method in ("passthrough", "fallback")


# ── RRF blend weight ─────────────────────────────────────────────────────────
# Weight for blending RRF score with LLM score.
# 0.2 means: 80% LLM semantic score + 20% RRF positional score.
# Matches the calibrated value from the benchmark harness.
_RRF_BLEND_WEIGHT = 0.2
# RRF scores are tiny fractions (1/k+rank); scale factor aligns them to [0,1].
_RRF_SCALE_FACTOR = 60.0


# ── Service class ─────────────────────────────────────────────────────────────

class LLMRerankerService:
    """Production LLM reranker for the hybrid retrieval surface.

    Async, fail-soft, bounded. Uses the OpenAI-compatible chat completions
    API to score candidates in a single batched call per query.

    When LLM reranking is unavailable (no key, disabled, or API failure),
    returns the input pool in its existing RRF order with ``method='passthrough'``
    or ``method='fallback'`` diagnostics.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        enabled: bool | None = None,
        timeout: int | None = None,
        batch_size: int | None = None,
        rrf_blend_weight: float = _RRF_BLEND_WEIGHT,
    ):
        self._model = model or os.environ.get("BICAMERAL_RERANK_MODEL", _DEFAULT_MODEL)
        self._api_key = (
            api_key
            if api_key is not None
            else os.environ.get(
                "BICAMERAL_RERANK_API_KEY",
                os.environ.get("OPENAI_API_KEY", ""),
            )
        )
        self._api_base = (
            api_base
            or os.environ.get("BICAMERAL_RERANK_API_BASE")
            or self._detect_api_base(self._api_key)
        )
        self._enabled = (
            enabled
            if enabled is not None
            else os.environ.get("BICAMERAL_RERANK_ENABLED", "1") == "1"
        )
        self._timeout = timeout or int(
            os.environ.get("BICAMERAL_RERANK_TIMEOUT", str(_DEFAULT_TIMEOUT))
        )
        self._batch_size = batch_size or _DEFAULT_BATCH_SIZE
        self._rrf_blend_weight = rrf_blend_weight

    @staticmethod
    def _detect_api_base(api_key: str) -> str:
        """Auto-detect API base from key prefix."""
        if api_key and api_key.startswith("sk-or-"):
            return "https://openrouter.ai/api/v1"
        return "https://api.openai.com/v1"

    @property
    def is_available(self) -> bool:
        """Whether LLM reranking is available (enabled + has API key)."""
        return self._enabled and bool(self._api_key)

    async def rerank(
        self,
        *,
        query: str,
        candidates: list[dict[str, Any]],
        max_results: int | None = None,
    ) -> RerankResult:
        """Rerank candidates by LLM relevance scoring.

        Args:
            query: The user's search query.
            candidates: RRF-merged candidate pool (from hybrid merge).
            max_results: Cap on returned results (default: len(candidates)).

        Returns:
            RerankResult with reranked candidates and metadata.
            On failure, returns candidates in original RRF order with
            method='fallback' and diagnostics.
        """
        cap = max_results if max_results is not None else len(candidates)

        if not candidates:
            return RerankResult(
                candidates=[],
                total_scored=0,
                method="passthrough",
                diagnostics={"reason": "empty_input"},
            )

        if not self.is_available:
            reason = "disabled" if not self._enabled else "no_api_key"
            logger.debug("LLM reranker passthrough: %s", reason)
            return RerankResult(
                candidates=candidates[:cap],
                total_scored=len(candidates),
                method="passthrough",
                diagnostics={"reason": reason},
            )

        try:
            # ── Phase 1A: classify query type for type-aware reranking ────────
            query_type = await self.classify_query_type(query)

            scored = await self._score_candidates(query, candidates, query_type=query_type)
            if not scored:
                # LLM returned empty/unparseable — fall back to RRF order
                logger.warning("LLM reranker returned empty scores — falling back to RRF order")
                return RerankResult(
                    candidates=candidates[:cap],
                    total_scored=len(candidates),
                    method="fallback",
                    model=self._model,
                    diagnostics={"reason": "empty_llm_response", "query_type": query_type},
                )

            # Blend LLM scores with RRF scores
            reranked = self._blend_and_sort(candidates, scored)
            annotated = [rc.to_annotated_dict() for rc in reranked[:cap]]

            return RerankResult(
                candidates=annotated,
                total_scored=len(reranked),
                method="llm",
                model=self._model,
                diagnostics={"query_type": query_type},
            )

        except Exception as e:
            logger.warning(
                "LLM reranker failed (falling back to RRF order): %s",
                e,
                exc_info=True,
            )
            return RerankResult(
                candidates=candidates[:cap],
                total_scored=len(candidates),
                method="fallback",
                model=self._model,
                diagnostics={
                    "reason": "llm_error",
                    "error": str(e),
                },
            )

    async def classify_query_type(self, query_text: str) -> str:
        """Classify a query into one of 8 types using a lightweight LLM call.

        Uses a 1-hour in-memory cache keyed by SHA-256 of query_text.
        Falls back to 'generic' on any error.

        Args:
            query_text: The user's search query.

        Returns:
            One of: person, project, event, decision, technical, financial,
            preference, generic.
        """
        # Check cache first
        cached = _cache_get(query_text)
        if cached is not None:
            logger.debug("Query type cache hit: %s -> %s", query_text[:60], cached)
            return cached

        try:
            payload = {
                "model": self._resolve_model(),
                "messages": [
                    {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
                    {"role": "user", "content": query_text},
                ],
                "temperature": 0.0,
                "max_tokens": 10,
            }
            result = await asyncio.to_thread(
                self._http_post,
                f"{self._api_base}/chat/completions",
                payload,
            )
            raw = result["choices"][0]["message"]["content"].strip().lower()
            # Extract just the type word (handle edge cases like "type: person")
            for qt in QUERY_TYPES:
                if qt in raw:
                    _cache_set(query_text, qt)
                    logger.debug("Query type classified: %s -> %s", query_text[:60], qt)
                    return qt

            # LLM returned something unexpected — default to generic
            logger.warning(
                "Query type classifier returned unrecognized type '%s' for query '%s' — defaulting to generic",
                raw[:50], query_text[:60],
            )
            _cache_set(query_text, "generic")
            return "generic"

        except Exception as e:
            logger.warning(
                "Query type classification failed (defaulting to generic): %s", e
            )
            return "generic"

    async def _score_candidates(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        query_type: str = "generic",
    ) -> list[dict[str, Any]]:
        """Call the LLM to score candidates in batches.

        Returns a list of {index, score, rationale} dicts.
        """
        all_scores: list[dict[str, Any]] = []

        for batch_start in range(0, len(candidates), self._batch_size):
            batch = candidates[batch_start: batch_start + self._batch_size]
            batch_scores = await self._call_llm_batch(query, batch, query_type=query_type)

            for item in batch_scores:
                item["index"] = batch_start + item["index"]
            all_scores.extend(batch_scores)

        return all_scores

    async def _call_llm_batch(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        query_type: str = "generic",
    ) -> list[dict[str, Any]]:
        """Make a single batched LLM call for a set of candidates.

        Returns list of {index, score, rationale} dicts (indices relative
        to the batch, not the full candidate list).
        """
        user_prompt = self._build_user_prompt(query, candidates, query_type=query_type)
        system_prompt = _build_type_aware_system_prompt(query_type)
        max_tokens = max(100, min(4000, len(candidates) * 80 + 100))

        payload = {
            "model": self._resolve_model(),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }

        # Use asyncio.to_thread to avoid blocking the event loop with urllib
        for attempt in range(_MAX_RETRIES + 1):
            try:
                result = await asyncio.to_thread(
                    self._http_post,
                    f"{self._api_base}/chat/completions",
                    payload,
                )
                content = result["choices"][0]["message"]["content"].strip()
                return self._parse_response(content, len(candidates))

            except Exception as e:
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_BASE_DELAY * (attempt + 1)
                    # Check for 429 retry-after
                    if hasattr(e, "retry_after"):
                        delay = max(delay, getattr(e, "retry_after"))
                    logger.debug(
                        "LLM rerank attempt %d/%d failed, retrying in %.1fs: %s",
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        return []  # unreachable but satisfies type checker

    def _http_post(self, url: str, payload: dict) -> dict:
        """Synchronous HTTP POST (runs in thread pool)."""
        import urllib.error
        import urllib.request

        data = json.dumps(payload).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        # Add OpenRouter headers if applicable
        if "openrouter.ai" in self._api_base:
            headers["HTTP-Referer"] = "https://github.com/yhl999/bicameral"
            headers["X-Title"] = "Bicameral MCP Reranker"

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            error = RuntimeError(f"LLM API error {e.code}: {detail}")
            if e.code == 429:
                # Try to parse Retry-After
                retry_after = e.headers.get("Retry-After") if e.headers else None
                if retry_after:
                    try:
                        error.retry_after = float(retry_after)  # type: ignore[attr-defined]
                    except (TypeError, ValueError):
                        pass
            raise error from e

    def _resolve_model(self) -> str:
        """Resolve model id for the target API provider."""
        model = self._model
        if "openrouter.ai" in self._api_base and "/" not in model:
            return f"openai/{model}"
        if "api.openai.com" in self._api_base and model.startswith("openai/"):
            return model.split("/", 1)[1]
        return model

    def _build_user_prompt(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        query_type: str = "generic",
    ) -> str:
        """Build the user prompt for the batched rerank call.

        Includes query type label so the model can apply type-specific scoring.
        """
        parts = [
            f"Query (type: {query_type}): {query}",
            "",
            f"Score each of these {len(candidates)} memory candidates:",
            "",
        ]
        for i, cand in enumerate(candidates):
            # Extract readable text from the candidate
            text = self._extract_candidate_text(cand)
            source = cand.get("_source", "unknown")
            parts.append(f"[{i}] ({source}) {text[:400]}")

        return "\n".join(parts)

    @staticmethod
    def _extract_candidate_text(cand: dict[str, Any]) -> str:
        """Extract human-readable text from a hybrid candidate dict."""
        # Graph facts have 'fact' key
        fact = cand.get("fact", "")
        if fact:
            return str(fact)
        # Typed items may have subject/predicate/value
        parts = [
            str(cand.get("subject", "") or ""),
            str(cand.get("predicate", "") or ""),
            str(cand.get("value", "") or ""),
        ]
        text = " ".join(p for p in parts if p)
        if text:
            return text
        # Fallback: name or uuid
        return str(cand.get("name", "") or cand.get("uuid", "") or "")

    def _parse_response(
        self,
        content: str,
        expected_count: int,
    ) -> list[dict[str, Any]]:
        """Parse the LLM's JSON array response into normalized scores."""
        content = content.strip()

        # Strip markdown code fences
        if content.startswith("```"):
            lines = content.split("\n")
            start = 1
            end = len(lines)
            for i in range(1, len(lines)):
                if lines[i].strip().startswith("```"):
                    end = i
                    break
            content = "\n".join(lines[start:end]).strip()

        # Try direct JSON parse
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return self._normalize_scores(parsed, expected_count)
            if isinstance(parsed, dict):
                for key in ("scores", "results", "items", "candidates"):
                    maybe = parsed.get(key)
                    if isinstance(maybe, list):
                        return self._normalize_scores(maybe, expected_count)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON array from mixed content
        match = re.search(r"\[[\s\S]*\]", content)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    return self._normalize_scores(parsed, expected_count)
            except json.JSONDecodeError:
                pass

        # Try line-by-line JSON objects
        results = []
        for line in content.split("\n"):
            line = line.strip().rstrip(",")
            if line.startswith("{"):
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if results:
            return self._normalize_scores(results, expected_count)

        return []

    @staticmethod
    def _normalize_scores(
        items: list[dict],
        expected_count: int,
    ) -> list[dict[str, Any]]:
        """Normalize raw LLM scores into a consistent shape."""
        normalized = []
        for item in items:
            if not isinstance(item, dict):
                continue

            # Flexible index extraction
            idx = item.get("index", item.get("candidate_index", item.get("idx", -1)))
            score = item.get("score", item.get("relevance", 0.0))
            rationale = item.get("rationale", item.get("reason", ""))

            try:
                idx = int(idx)
            except (TypeError, ValueError):
                continue
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0

            score = max(0.0, min(1.0, score))

            if 0 <= idx < expected_count:
                normalized.append({
                    "index": idx,
                    "score": round(score, 4),
                    "rationale": str(rationale)[:200],
                })

        return normalized

    def _blend_and_sort(
        self,
        candidates: list[dict[str, Any]],
        scores: list[dict[str, Any]],
    ) -> list[RerankedCandidate]:
        """Blend LLM scores with RRF scores and sort."""
        score_map = {item["index"]: item for item in scores}

        reranked: list[RerankedCandidate] = []
        for i, cand in enumerate(candidates):
            rrf_score = float(cand.get("_hybrid_score", 0.0))
            score_info = score_map.get(i)

            if score_info is not None:
                llm_score = score_info["score"]
                rationale = score_info.get("rationale", "")
            else:
                # Candidate wasn't scored (beyond batch window) — use RRF only
                llm_score = 0.0
                rationale = ""

            blended = (
                (1 - self._rrf_blend_weight) * llm_score
                + self._rrf_blend_weight * rrf_score * _RRF_SCALE_FACTOR
            )

            reranked.append(RerankedCandidate(
                original=cand,
                llm_score=llm_score,
                rrf_score=rrf_score,
                blended_score=blended,
                final_rank=0,
                rationale=rationale,
            ))

        reranked.sort(key=lambda rc: -rc.blended_score)
        for i, rc in enumerate(reranked, start=1):
            rc.final_rank = i

        return reranked
