"""
Tests for LLMRerankerService — covers unified system prompt, blending,
passthrough/fallback paths, and response parsing.

Phase 1A simplified: single unified prompt, no query-type classifier.
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
    LLMRerankerService,
    RerankedCandidate,
    RerankResult,
    _RERANK_SYSTEM_PROMPT,
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


def _service(**kwargs) -> LLMRerankerService:
    """Create a test service with a fake API key so it's 'available'."""
    defaults = {"api_key": "sk-or-test-key-1234", "enabled": True}
    defaults.update(kwargs)
    return LLMRerankerService(**defaults)


# ══════════════════════════════════════════════════════════════════════════════
# Section 1: Unified System Prompt
# ══════════════════════════════════════════════════════════════════════════════

class TestUnifiedSystemPrompt:
    """Verify the unified system prompt contains all required heuristics."""

    def test_prompt_is_string(self):
        assert isinstance(_RERANK_SYSTEM_PROMPT, str)

    def test_prompt_includes_scoring_scale(self):
        assert "0.8-1.0" in _RERANK_SYSTEM_PROMPT
        assert "0.0-0.1" in _RERANK_SYSTEM_PROMPT

    def test_prompt_includes_json_format(self):
        assert "JSON array" in _RERANK_SYSTEM_PROMPT

    def test_prompt_includes_core_principle(self):
        assert "Core Principle" in _RERANK_SYSTEM_PROMPT
        assert "query's intent" in _RERANK_SYSTEM_PROMPT

    def test_prompt_includes_person_heuristic(self):
        assert "Asking about a person" in _RERANK_SYSTEM_PROMPT
        assert "background" in _RERANK_SYSTEM_PROMPT

    def test_prompt_includes_project_heuristic(self):
        assert "Asking about a project" in _RERANK_SYSTEM_PROMPT
        assert "milestones" in _RERANK_SYSTEM_PROMPT

    def test_prompt_includes_event_heuristic(self):
        assert "Asking about an event" in _RERANK_SYSTEM_PROMPT
        assert "attendees" in _RERANK_SYSTEM_PROMPT

    def test_prompt_includes_decision_heuristic(self):
        assert "Asking about a decision" in _RERANK_SYSTEM_PROMPT
        assert "tradeoffs" in _RERANK_SYSTEM_PROMPT

    def test_prompt_includes_technical_heuristic(self):
        assert "Asking how something works" in _RERANK_SYSTEM_PROMPT
        assert "architecture" in _RERANK_SYSTEM_PROMPT

    def test_prompt_includes_financial_heuristic(self):
        assert "Asking about numbers or finances" in _RERANK_SYSTEM_PROMPT
        assert "valuations" in _RERANK_SYSTEM_PROMPT

    def test_prompt_includes_preference_heuristic(self):
        assert "Asking about preferences or opinions" in _RERANK_SYSTEM_PROMPT
        assert "tastes" in _RERANK_SYSTEM_PROMPT

    def test_prompt_includes_penalty_section(self):
        assert "What to Penalize" in _RERANK_SYSTEM_PROMPT
        assert "Keyword matches" in _RERANK_SYSTEM_PROMPT
        assert "Metadata-only facts" in _RERANK_SYSTEM_PROMPT

    def test_prompt_includes_reward_section(self):
        assert "What to Reward" in _RERANK_SYSTEM_PROMPT
        assert "Specificity and concrete details" in _RERANK_SYSTEM_PROMPT
        assert "Recency" in _RERANK_SYSTEM_PROMPT

    def test_prompt_nontrivial_length(self):
        assert len(_RERANK_SYSTEM_PROMPT) > 500


# ══════════════════════════════════════════════════════════════════════════════
# Section 2: User Prompt
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildUserPrompt:
    """Test that _build_user_prompt formats correctly."""

    def test_user_prompt_includes_query(self):
        svc = _service()
        candidates = _make_candidates(3)
        prompt = svc._build_user_prompt("Who is Alice?", candidates)
        assert "Query: Who is Alice?" in prompt

    def test_user_prompt_no_type_label(self):
        """Unified prompt should NOT include a query type label."""
        svc = _service()
        candidates = _make_candidates(2)
        prompt = svc._build_user_prompt("test query", candidates)
        assert "(type:" not in prompt

    def test_user_prompt_includes_all_candidates(self):
        svc = _service()
        candidates = _make_candidates(5)
        prompt = svc._build_user_prompt("test", candidates)
        for i in range(5):
            assert f"[{i}]" in prompt


# ══════════════════════════════════════════════════════════════════════════════
# Section 3: Full Rerank Flow (integration with mocks)
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
    async def test_rerank_llm_success(self):
        """Full flow: single LLM call for scoring, no classifier."""
        svc = _service()
        candidates = _make_candidates(3)
        scores = [
            {"index": 0, "score": 0.9, "rationale": "directly about the person"},
            {"index": 1, "score": 0.3, "rationale": "tangential"},
            {"index": 2, "score": 0.7, "rationale": "partially relevant"},
        ]

        with patch.object(svc, "_http_post", return_value=_mock_llm_response(scores)):
            result = await svc.rerank(query="Who is Alice?", candidates=candidates)

        assert result.method == "llm"
        assert "query_type" not in result.diagnostics
        assert len(result.candidates) == 3
        # First result should be the highest scored (index 0, score 0.9)
        assert result.candidates[0].get("_rerank_score") == 0.9

    @pytest.mark.asyncio
    async def test_rerank_only_one_llm_call(self):
        """Unified prompt means only one LLM call (scoring), not two (classify + score)."""
        svc = _service()
        candidates = _make_candidates(3)
        scores = [{"index": i, "score": 0.5, "rationale": "ok"} for i in range(3)]
        mock_post = MagicMock(return_value=_mock_llm_response(scores))

        with patch.object(svc, "_http_post", mock_post):
            result = await svc.rerank(query="test", candidates=candidates)

        assert result.method == "llm"
        # Exactly 1 call (scoring batch), not 2 (no classifier)
        assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_rerank_llm_failure_falls_back(self):
        svc = _service()
        candidates = _make_candidates(3)

        with patch.object(svc, "_http_post", side_effect=RuntimeError("API failure")):
            result = await svc.rerank(query="test", candidates=candidates)

        assert result.method == "fallback"
        assert "llm_error" in result.diagnostics.get("reason", "")

    @pytest.mark.asyncio
    async def test_rerank_max_results_caps_output(self):
        svc = _service()
        candidates = _make_candidates(5)
        scores = [{"index": i, "score": 0.5, "rationale": "ok"} for i in range(5)]

        with patch.object(svc, "_http_post", return_value=_mock_llm_response(scores)):
            result = await svc.rerank(query="test", candidates=candidates, max_results=2)

        assert len(result.candidates) <= 2

    @pytest.mark.asyncio
    async def test_rerank_empty_llm_response_falls_back(self):
        """If the LLM returns unparseable output, fall back to RRF order."""
        svc = _service()
        candidates = _make_candidates(3)
        empty_resp = {"choices": [{"message": {"content": "I can't do that"}}]}

        with patch.object(svc, "_http_post", return_value=empty_resp):
            result = await svc.rerank(query="test", candidates=candidates)

        assert result.method == "fallback"
        assert result.diagnostics.get("reason") == "empty_llm_response"


# ══════════════════════════════════════════════════════════════════════════════
# Section 4: Response Parsing
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
# Section 5: Blend and Sort
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
# Section 6: RerankedCandidate & RerankResult data classes
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
# Section 7: Service Configuration
# ══════════════════════════════════════════════════════════════════════════════

class TestServiceConfiguration:
    """Test LLMRerankerService init and config detection."""

    def test_default_model_is_nano(self):
        svc = _service()
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
# Section 8: Extract Candidate Text
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
# Section 9: Model Resolution
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


# ══════════════════════════════════════════════════════════════════════════════
# Section 10: Classifier Removal Regression
# ══════════════════════════════════════════════════════════════════════════════

class TestClassifierRemoved:
    """Verify the query-type classifier is fully removed."""

    def test_no_classify_method(self):
        svc = _service()
        assert not hasattr(svc, "classify_query_type")

    def test_no_query_types_constant(self):
        import services.llm_reranker as mod
        assert not hasattr(mod, "QUERY_TYPES")

    def test_no_classify_cache(self):
        import services.llm_reranker as mod
        assert not hasattr(mod, "_CLASSIFY_CACHE")

    def test_no_type_scoring_rules(self):
        import services.llm_reranker as mod
        assert not hasattr(mod, "_TYPE_SCORING_RULES")

    def test_no_build_type_aware_prompt(self):
        import services.llm_reranker as mod
        assert not hasattr(mod, "_build_type_aware_system_prompt")

    def test_unified_prompt_exists(self):
        import services.llm_reranker as mod
        assert hasattr(mod, "_RERANK_SYSTEM_PROMPT")
