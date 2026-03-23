"""
Tests for LLMRerankerService — covers query-type classifier, type-aware
prompt generation, caching, blending, passthrough/fallback paths, and
response parsing.

Phase 1A: query-type classifier + type-aware reranker prompting.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Module imports ────────────────────────────────────────────────────────────

import sys
import os

# Ensure the MCP server source is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.llm_reranker import (
    QUERY_TYPES,
    _QUERY_TYPES_SORTED,
    LLMRerankerService,
    RerankedCandidate,
    RerankResult,
    _CLASSIFY_CACHE,
    _CLASSIFY_CACHE_TTL,
    _RERANK_SYSTEM_PROMPT_BASE,
    _TYPE_SCORING_RULES,
    _build_type_aware_system_prompt,
    _cache_get,
    _cache_set,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_candidates(n: int = 5) -> list[dict[str, Any]]:
    """Generate n fake hybrid candidates for testing."""
    return [
        {
            "fact": f"Fact number {i} about topic {i}",
            "_source": "graph" if i % 2 == 0 else "typed",
            "_hybrid_score": round(0.1 / (i + 1), 6),
            "uuid": f"uuid-{i:04d}",
        }
        for i in range(n)
    ]


def _mock_llm_response(scores: list[dict]) -> dict:
    """Build a mock OpenAI chat completion response."""
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(scores),
                }
            }
        ]
    }


def _mock_classify_response(query_type: str) -> dict:
    """Build a mock chat completion for classification."""
    return {
        "choices": [
            {
                "message": {
                    "content": query_type,
                }
            }
        ]
    }


def _service(**kwargs) -> LLMRerankerService:
    """Create a test service with a fake API key so it's 'available'."""
    defaults = {"api_key": "sk-or-test-key-1234", "enabled": True}
    defaults.update(kwargs)
    return LLMRerankerService(**defaults)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_classify_cache():
    """Clear the classification cache before each test."""
    _CLASSIFY_CACHE.clear()
    yield
    _CLASSIFY_CACHE.clear()


# ══════════════════════════════════════════════════════════════════════════════
# Section 1: Query-Type Constants & Prompt Generation
# ══════════════════════════════════════════════════════════════════════════════

class TestQueryTypeConstants:
    """Verify the 8 query types are properly defined."""

    def test_query_types_count(self):
        assert len(QUERY_TYPES) == 8

    def test_query_types_values(self):
        expected = {"person", "project", "event", "decision",
                    "technical", "financial", "preference", "generic"}
        assert QUERY_TYPES == expected

    def test_all_types_have_scoring_rules(self):
        for qt in QUERY_TYPES:
            assert qt in _TYPE_SCORING_RULES, f"Missing scoring rule for type: {qt}"
            assert len(_TYPE_SCORING_RULES[qt]) > 10, f"Scoring rule too short for type: {qt}"

    def test_sorted_query_types_is_deterministic(self):
        """_QUERY_TYPES_SORTED must be a plain sorted list for deterministic iteration."""
        assert isinstance(_QUERY_TYPES_SORTED, list)
        assert _QUERY_TYPES_SORTED == sorted(QUERY_TYPES)
        # Verify it's actually sorted (alphabetical)
        assert _QUERY_TYPES_SORTED == sorted(_QUERY_TYPES_SORTED)


class TestTypeAwareSystemPrompt:
    """Test _build_type_aware_system_prompt generates correct prompts."""

    def test_includes_base_prompt(self):
        prompt = _build_type_aware_system_prompt("person")
        assert "relevance judge" in prompt
        assert "JSON array" in prompt

    def test_includes_query_type_label(self):
        for qt in QUERY_TYPES:
            prompt = _build_type_aware_system_prompt(qt)
            assert f"Query type: {qt}" in prompt

    def test_includes_type_specific_rule(self):
        prompt = _build_type_aware_system_prompt("person")
        assert "background" in prompt and "role" in prompt

        prompt = _build_type_aware_system_prompt("financial")
        assert "numbers" in prompt or "budgets" in prompt

        prompt = _build_type_aware_system_prompt("event")
        assert "dates" in prompt or "attendees" in prompt

    def test_generic_type_no_penalty(self):
        prompt = _build_type_aware_system_prompt("generic")
        assert "No type-specific penalty" in prompt

    def test_unknown_type_falls_back_to_generic(self):
        prompt = _build_type_aware_system_prompt("nonexistent_type")
        assert "No type-specific penalty" in prompt

    def test_includes_general_penalties(self):
        prompt = _build_type_aware_system_prompt("person")
        assert "Penalize generic/organizational facts" in prompt
        assert "Penalize facts that are lexically similar but off-type" in prompt

    def test_includes_general_rewards(self):
        prompt = _build_type_aware_system_prompt("project")
        assert "Reward facts that directly answer the query" in prompt
        assert "Reward specific, recent, and actionable context" in prompt

    @pytest.mark.parametrize("query_type", list(QUERY_TYPES))
    def test_all_types_generate_valid_prompt(self, query_type: str):
        prompt = _build_type_aware_system_prompt(query_type)
        assert isinstance(prompt, str)
        assert len(prompt) > 200  # Non-trivial prompt


# ══════════════════════════════════════════════════════════════════════════════
# Section 2: Classification Cache
# ══════════════════════════════════════════════════════════════════════════════

class TestClassificationCache:
    """Test the TTL-based classification cache."""

    def test_cache_set_and_get(self):
        _cache_set("who is alice?", "person")
        assert _cache_get("who is alice?") == "person"

    def test_cache_miss(self):
        assert _cache_get("never seen this query") is None

    def test_cache_expiry(self):
        _cache_set("test query", "project")
        # Manually expire the entry
        import hashlib
        key = hashlib.sha256("test query".encode()).hexdigest()
        # Set timestamp far in the past
        _CLASSIFY_CACHE[key] = ("project", time.monotonic() - _CLASSIFY_CACHE_TTL - 1)
        assert _cache_get("test query") is None

    def test_cache_different_queries(self):
        _cache_set("who is alice?", "person")
        _cache_set("project status?", "project")
        assert _cache_get("who is alice?") == "person"
        assert _cache_get("project status?") == "project"

    def test_cache_overwrite(self):
        _cache_set("ambiguous query", "generic")
        _cache_set("ambiguous query", "person")
        assert _cache_get("ambiguous query") == "person"


# ══════════════════════════════════════════════════════════════════════════════
# Section 3: classify_query_type
# ══════════════════════════════════════════════════════════════════════════════

class TestClassifyQueryType:
    """Test the classify_query_type method."""

    @pytest.mark.asyncio
    async def test_classify_person_query(self):
        svc = _service()
        with patch.object(svc, "_http_post", return_value=_mock_classify_response("person")):
            result = await svc.classify_query_type("Who is Alice Chen?")
        assert result == "person"

    @pytest.mark.asyncio
    async def test_classify_project_query(self):
        svc = _service()
        with patch.object(svc, "_http_post", return_value=_mock_classify_response("project")):
            result = await svc.classify_query_type("What's the status of Project Alpha?")
        assert result == "project"

    @pytest.mark.asyncio
    async def test_classify_event_query(self):
        svc = _service()
        with patch.object(svc, "_http_post", return_value=_mock_classify_response("event")):
            result = await svc.classify_query_type("What happened at the offsite last week?")
        assert result == "event"

    @pytest.mark.asyncio
    async def test_classify_decision_query(self):
        svc = _service()
        with patch.object(svc, "_http_post", return_value=_mock_classify_response("decision")):
            result = await svc.classify_query_type("Why did we decide to use Rust?")
        assert result == "decision"

    @pytest.mark.asyncio
    async def test_classify_technical_query(self):
        svc = _service()
        with patch.object(svc, "_http_post", return_value=_mock_classify_response("technical")):
            result = await svc.classify_query_type("How does the RRF merge algorithm work?")
        assert result == "technical"

    @pytest.mark.asyncio
    async def test_classify_financial_query(self):
        svc = _service()
        with patch.object(svc, "_http_post", return_value=_mock_classify_response("financial")):
            result = await svc.classify_query_type("What's the valuation of Acme Corp?")
        assert result == "financial"

    @pytest.mark.asyncio
    async def test_classify_preference_query(self):
        svc = _service()
        with patch.object(svc, "_http_post", return_value=_mock_classify_response("preference")):
            result = await svc.classify_query_type("Do I like sushi?")
        assert result == "preference"

    @pytest.mark.asyncio
    async def test_classify_generic_query(self):
        svc = _service()
        with patch.object(svc, "_http_post", return_value=_mock_classify_response("generic")):
            result = await svc.classify_query_type("Tell me something interesting")
        assert result == "generic"

    @pytest.mark.asyncio
    async def test_classify_uses_cache(self):
        svc = _service()
        mock_post = MagicMock(return_value=_mock_classify_response("person"))
        with patch.object(svc, "_http_post", mock_post):
            result1 = await svc.classify_query_type("Who is Bob?")
            result2 = await svc.classify_query_type("Who is Bob?")
        assert result1 == "person"
        assert result2 == "person"
        # Should only call LLM once — second hit served from cache
        assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_classify_error_falls_back_to_generic(self):
        svc = _service()
        with patch.object(svc, "_http_post", side_effect=RuntimeError("API error")):
            result = await svc.classify_query_type("Who is Alice?")
        assert result == "generic"

    @pytest.mark.asyncio
    async def test_classify_unrecognized_response_falls_back(self):
        svc = _service()
        with patch.object(svc, "_http_post", return_value=_mock_classify_response("banana")):
            result = await svc.classify_query_type("Something weird")
        assert result == "generic"

    @pytest.mark.asyncio
    async def test_classify_extracts_type_from_verbose_response(self):
        """LLM might return 'The query type is: person' instead of just 'person'."""
        svc = _service()
        resp = _mock_classify_response("The query type is: person")
        with patch.object(svc, "_http_post", return_value=resp):
            result = await svc.classify_query_type("Who is Alice?")
        assert result == "person"

    @pytest.mark.asyncio
    async def test_classify_handles_uppercase_response(self):
        svc = _service()
        resp = _mock_classify_response("FINANCIAL")
        with patch.object(svc, "_http_post", return_value=resp):
            result = await svc.classify_query_type("Budget for Q1?")
        assert result == "financial"

    # ── Substring false-positive regression tests ─────────────────────────────

    @pytest.mark.asyncio
    async def test_classify_no_false_positive_event_in_prevent(self):
        """'event' must NOT match in 'prevent' — substring false positive."""
        svc = _service()
        resp = _mock_classify_response("prevent")
        with patch.object(svc, "_http_post", return_value=resp):
            result = await svc.classify_query_type("How to prevent issues?")
        assert result == "generic"  # not "event"

    @pytest.mark.asyncio
    async def test_classify_no_false_positive_person_in_personal(self):
        """'person' must NOT match in 'personal' — substring false positive."""
        svc = _service()
        resp = _mock_classify_response("personal")
        with patch.object(svc, "_http_post", return_value=resp):
            result = await svc.classify_query_type("What are my personal notes?")
        assert result == "generic"  # not "person"

    @pytest.mark.asyncio
    async def test_classify_no_false_positive_event_in_eventually(self):
        """'event' must NOT match in 'eventually'."""
        svc = _service()
        resp = _mock_classify_response("eventually generic")
        with patch.object(svc, "_http_post", return_value=resp):
            result = await svc.classify_query_type("What will eventually happen?")
        assert result == "generic"  # not "event"

    @pytest.mark.asyncio
    async def test_classify_no_false_positive_decision_in_indecision(self):
        """'decision' must NOT match in 'indecision'."""
        svc = _service()
        resp = _mock_classify_response("indecision")
        with patch.object(svc, "_http_post", return_value=resp):
            result = await svc.classify_query_type("test query")
        assert result == "generic"  # not "decision"

    @pytest.mark.asyncio
    async def test_classify_word_boundary_matches_in_verbose_response(self):
        """Word-boundary regex should still match 'event' as a standalone word."""
        svc = _service()
        resp = _mock_classify_response("I believe the type is event.")
        with patch.object(svc, "_http_post", return_value=resp):
            result = await svc.classify_query_type("What happened at the conference?")
        assert result == "event"

    @pytest.mark.asyncio
    async def test_classify_exact_match_preferred_over_regex(self):
        """Clean single-word response should use exact match, not regex."""
        svc = _service()
        resp = _mock_classify_response("person")
        with patch.object(svc, "_http_post", return_value=resp):
            result = await svc.classify_query_type("Who is Alice?")
        assert result == "person"

    @pytest.mark.asyncio
    async def test_classify_deterministic_across_runs(self):
        """The same ambiguous response must always produce the same type
        regardless of frozenset iteration order (hash seed)."""
        svc = _service()
        # Response containing both "decision" and "event" as whole words
        resp = _mock_classify_response("this is about a decision at the event")
        results = set()
        for _ in range(50):  # run many times to detect non-determinism
            _CLASSIFY_CACHE.clear()
            with patch.object(svc, "_http_post", return_value=resp):
                result = await svc.classify_query_type("some query")
            results.add(result)
        assert len(results) == 1, f"Non-deterministic results: {results}"


# ══════════════════════════════════════════════════════════════════════════════
# Section 4: Type-Aware User Prompt
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildUserPrompt:
    """Test that _build_user_prompt includes query type."""

    def test_user_prompt_includes_query_type(self):
        svc = _service()
        candidates = _make_candidates(3)
        prompt = svc._build_user_prompt("Who is Alice?", candidates, query_type="person")
        assert "Query (type: person):" in prompt
        assert "Who is Alice?" in prompt

    def test_user_prompt_default_type_is_generic(self):
        svc = _service()
        candidates = _make_candidates(2)
        prompt = svc._build_user_prompt("test query", candidates)
        assert "Query (type: generic):" in prompt

    def test_user_prompt_includes_all_candidates(self):
        svc = _service()
        candidates = _make_candidates(5)
        prompt = svc._build_user_prompt("test", candidates, query_type="technical")
        for i in range(5):
            assert f"[{i}]" in prompt


# ══════════════════════════════════════════════════════════════════════════════
# Section 5: Full Rerank Flow (integration with mocks)
# ══════════════════════════════════════════════════════════════════════════════

class TestRerankIntegration:
    """Test the full rerank flow with mocked LLM calls."""

    @pytest.mark.asyncio
    async def test_rerank_empty_candidates(self):
        svc = _service()
        result = await svc.rerank(query="test", candidates=[])
        assert result.method == "passthrough"
        assert result.candidates == []

    @pytest.mark.asyncio
    async def test_rerank_disabled(self):
        svc = _service(enabled=False)
        candidates = _make_candidates(3)
        result = await svc.rerank(query="test", candidates=candidates)
        assert result.method == "passthrough"
        assert result.diagnostics["reason"] == "disabled"

    @pytest.mark.asyncio
    async def test_rerank_no_api_key(self):
        svc = LLMRerankerService(api_key="", enabled=True)
        candidates = _make_candidates(3)
        result = await svc.rerank(query="test", candidates=candidates)
        assert result.method == "passthrough"
        assert result.diagnostics["reason"] == "no_api_key"

    @pytest.mark.asyncio
    async def test_rerank_llm_success_includes_query_type(self):
        """Full flow: classify → score → blend. Verify query_type in diagnostics."""
        svc = _service()
        candidates = _make_candidates(3)

        call_count = 0

        def mock_post(url, payload):
            nonlocal call_count
            call_count += 1
            messages = payload["messages"]
            user_msg = messages[-1]["content"]

            if call_count == 1:
                # First call is classification
                return _mock_classify_response("person")
            else:
                # Second call is reranking
                scores = [
                    {"index": 0, "score": 0.9, "rationale": "directly about the person"},
                    {"index": 1, "score": 0.3, "rationale": "tangential"},
                    {"index": 2, "score": 0.7, "rationale": "partially relevant"},
                ]
                return _mock_llm_response(scores)

        with patch.object(svc, "_http_post", side_effect=mock_post):
            result = await svc.rerank(query="Who is Alice?", candidates=candidates)

        assert result.method == "llm"
        assert result.diagnostics.get("query_type") == "person"
        assert len(result.candidates) == 3
        # First result should be the highest scored (index 0, score 0.9)
        assert result.candidates[0].get("_rerank_score") == 0.9

    @pytest.mark.asyncio
    async def test_rerank_llm_failure_falls_back(self):
        svc = _service()
        candidates = _make_candidates(3)

        call_count = 0

        def mock_post(url, payload):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_classify_response("generic")
            raise RuntimeError("API failure")

        with patch.object(svc, "_http_post", side_effect=mock_post):
            result = await svc.rerank(query="test", candidates=candidates)

        assert result.method == "fallback"
        assert "llm_error" in result.diagnostics.get("reason", "")

    @pytest.mark.asyncio
    async def test_rerank_max_results_caps_output(self):
        svc = _service()
        candidates = _make_candidates(5)

        call_count = 0

        def mock_post(url, payload):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_classify_response("generic")
            scores = [{"index": i, "score": 0.5, "rationale": "ok"} for i in range(5)]
            return _mock_llm_response(scores)

        with patch.object(svc, "_http_post", side_effect=mock_post):
            result = await svc.rerank(query="test", candidates=candidates, max_results=2)

        assert len(result.candidates) <= 2

    @pytest.mark.asyncio
    async def test_rerank_classification_failure_still_reranks(self):
        """If classifier fails, reranking should still proceed with 'generic' type."""
        svc = _service()
        candidates = _make_candidates(3)

        call_count = 0

        def mock_post(url, payload):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Classification fails
                raise RuntimeError("classify failed")
            # Reranking succeeds
            scores = [{"index": i, "score": 0.5 + i * 0.1, "rationale": "ok"} for i in range(3)]
            return _mock_llm_response(scores)

        with patch.object(svc, "_http_post", side_effect=mock_post):
            result = await svc.rerank(query="test", candidates=candidates)

        assert result.method == "llm"
        assert result.diagnostics.get("query_type") == "generic"


# ══════════════════════════════════════════════════════════════════════════════
# Section 6: Response Parsing
# ══════════════════════════════════════════════════════════════════════════════

class TestParseResponse:
    """Test _parse_response handles various LLM output formats."""

    def _parse(self, content: str, expected_count: int = 3) -> list:
        svc = _service()
        return svc._parse_response(content, expected_count)

    def test_clean_json_array(self):
        scores = [
            {"index": 0, "score": 0.8, "rationale": "good"},
            {"index": 1, "score": 0.3, "rationale": "meh"},
        ]
        result = self._parse(json.dumps(scores), 3)
        assert len(result) == 2
        assert result[0]["score"] == 0.8

    def test_markdown_fenced_json(self):
        content = '```json\n[{"index": 0, "score": 0.9, "rationale": "yes"}]\n```'
        result = self._parse(content, 1)
        assert len(result) == 1
        assert result[0]["score"] == 0.9

    def test_json_in_wrapper_object(self):
        content = json.dumps({"scores": [{"index": 0, "score": 0.5, "rationale": "ok"}]})
        result = self._parse(content, 1)
        assert len(result) == 1

    def test_out_of_range_index_filtered(self):
        scores = [
            {"index": 0, "score": 0.8, "rationale": "ok"},
            {"index": 99, "score": 0.9, "rationale": "bad index"},
        ]
        result = self._parse(json.dumps(scores), 3)
        assert len(result) == 1

    def test_score_clamped_to_0_1(self):
        scores = [{"index": 0, "score": 1.5, "rationale": "too high"}]
        result = self._parse(json.dumps(scores), 1)
        assert result[0]["score"] == 1.0

    def test_empty_response(self):
        result = self._parse("", 3)
        assert result == []

    def test_line_by_line_json(self):
        content = '{"index": 0, "score": 0.8, "rationale": "a"}\n{"index": 1, "score": 0.6, "rationale": "b"}'
        result = self._parse(content, 3)
        assert len(result) == 2


# ══════════════════════════════════════════════════════════════════════════════
# Section 7: Blend and Sort
# ══════════════════════════════════════════════════════════════════════════════

class TestBlendAndSort:
    """Test _blend_and_sort correctly blends LLM + RRF scores."""

    def test_llm_dominates_blended_score(self):
        svc = _service()
        candidates = [
            {"fact": "A", "_hybrid_score": 0.01, "_source": "graph"},
            {"fact": "B", "_hybrid_score": 0.02, "_source": "graph"},
        ]
        scores = [
            {"index": 0, "score": 0.9, "rationale": "high llm"},
            {"index": 1, "score": 0.1, "rationale": "low llm"},
        ]
        reranked = svc._blend_and_sort(candidates, scores)
        # Candidate 0 has high LLM score → should be first
        assert reranked[0].original["fact"] == "A"
        assert reranked[0].final_rank == 1

    def test_unscored_candidate_gets_zero_llm(self):
        svc = _service()
        candidates = [
            {"fact": "A", "_hybrid_score": 0.01, "_source": "graph"},
            {"fact": "B", "_hybrid_score": 0.02, "_source": "graph"},
        ]
        scores = [{"index": 0, "score": 0.5, "rationale": "ok"}]
        reranked = svc._blend_and_sort(candidates, scores)
        assert reranked[1].llm_score == 0.0  # Candidate 1 unscored


# ══════════════════════════════════════════════════════════════════════════════
# Section 8: RerankedCandidate & RerankResult data classes
# ══════════════════════════════════════════════════════════════════════════════

class TestDataClasses:
    """Test RerankedCandidate and RerankResult behavior."""

    def test_annotated_dict_includes_scores(self):
        rc = RerankedCandidate(
            original={"fact": "test", "_source": "graph"},
            llm_score=0.85,
            rrf_score=0.01,
            blended_score=0.7,
            final_rank=1,
            rationale="very relevant",
        )
        d = rc.to_annotated_dict()
        assert d["_rerank_score"] == 0.85
        assert d["_blended_score"] == 0.7
        assert d["_final_rank"] == 1
        assert d["_rerank_rationale"] == "very relevant"

    def test_annotated_dict_no_rationale(self):
        rc = RerankedCandidate(
            original={"fact": "test"},
            llm_score=0.5,
            rrf_score=0.01,
            blended_score=0.4,
            final_rank=2,
        )
        d = rc.to_annotated_dict()
        assert "_rerank_rationale" not in d

    def test_rerank_result_is_degraded(self):
        assert RerankResult(candidates=[], total_scored=0, method="passthrough").is_degraded
        assert RerankResult(candidates=[], total_scored=0, method="fallback").is_degraded
        assert not RerankResult(candidates=[], total_scored=0, method="llm").is_degraded


# ══════════════════════════════════════════════════════════════════════════════
# Section 9: Service Configuration
# ══════════════════════════════════════════════════════════════════════════════

class TestServiceConfiguration:
    """Test LLMRerankerService init and config detection."""

    def test_default_model_is_nano(self):
        svc = _service()
        # Model should remain gpt-5.4-nano for cost efficiency.
        # Semantic work comes from type-aware prompt + query-type classifier.
        assert "nano" in svc._model

    def test_openrouter_api_base_detection(self):
        svc = _service(api_key="sk-or-v1-abc123")
        assert "openrouter.ai" in svc._api_base

    def test_openai_api_base_detection(self):
        svc = _service(api_key="sk-abc123")
        assert "api.openai.com" in svc._api_base

    def test_is_available_requires_key_and_enabled(self):
        assert _service(api_key="sk-test", enabled=True).is_available
        assert not _service(api_key="", enabled=True).is_available
        assert not _service(api_key="sk-test", enabled=False).is_available

    def test_custom_model_override(self):
        svc = _service(model="custom-model-v2")
        assert svc._model == "custom-model-v2"


# ══════════════════════════════════════════════════════════════════════════════
# Section 10: Extract Candidate Text
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractCandidateText:
    """Test _extract_candidate_text handles various candidate shapes."""

    def test_fact_key(self):
        text = LLMRerankerService._extract_candidate_text({"fact": "Alice is an engineer"})
        assert text == "Alice is an engineer"

    def test_subject_predicate_value(self):
        text = LLMRerankerService._extract_candidate_text({
            "subject": "Alice",
            "predicate": "works at",
            "value": "Acme Corp",
        })
        assert "Alice" in text and "works at" in text and "Acme Corp" in text

    def test_fallback_to_name(self):
        text = LLMRerankerService._extract_candidate_text({"name": "Some Entity"})
        assert text == "Some Entity"

    def test_fallback_to_uuid(self):
        text = LLMRerankerService._extract_candidate_text({"uuid": "abc-123"})
        assert text == "abc-123"

    def test_empty_candidate(self):
        text = LLMRerankerService._extract_candidate_text({})
        assert text == ""


# ══════════════════════════════════════════════════════════════════════════════
# Section 11: Model Resolution
# ══════════════════════════════════════════════════════════════════════════════

class TestModelResolution:
    """Test _resolve_model for different API bases."""

    def test_openrouter_adds_prefix(self):
        svc = _service(api_base="https://openrouter.ai/api/v1", model="gpt-5.4-nano")
        assert svc._resolve_model() == "openai/gpt-5.4-nano"

    def test_openai_strips_prefix(self):
        svc = _service(api_base="https://api.openai.com/v1", model="openai/gpt-5.4-nano")
        assert svc._resolve_model() == "gpt-5.4-nano"

    def test_model_with_slash_kept_for_openrouter(self):
        svc = _service(api_base="https://openrouter.ai/api/v1", model="openai/gpt-5.4-nano")
        assert svc._resolve_model() == "openai/gpt-5.4-nano"
