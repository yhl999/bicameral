"""Tests for the LLM Reranker Service (llm_reranker.py).

Covers:
- Passthrough mode when disabled or no API key
- Score parsing from various LLM response formats
- Blend-and-sort logic (LLM score + RRF score blending)
- Graceful fallback when LLM call fails
- Empty input handling
- Candidate text extraction from graph and typed items
- Model resolution for OpenRouter vs OpenAI
- Response normalization edge cases
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# Make the service importable.
_svc_path = str(
    pathlib.Path(__file__).parent.parent / "mcp_server" / "src" / "services"
)
if _svc_path not in sys.path:
    sys.path.insert(0, _svc_path)

from mcp_server.src.services.llm_reranker import (
    LLMRerankerService,
    RerankedCandidate,
    RerankResult,
    _RRF_BLEND_WEIGHT,
    _RRF_SCALE_FACTOR,
)


def _run(coro):
    return asyncio.run(coro)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _graph_candidates(n: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "uuid": f"g{i}",
            "fact": f"graph fact {i}",
            "_source": "graph",
            "_hybrid_score": round(1.0 / (60 + i), 6),
        }
        for i in range(1, n + 1)
    ]


def _typed_candidates(n: int = 2) -> list[dict[str, Any]]:
    return [
        {
            "uuid": f"sf_{i}",
            "fact": f"typed state fact {i}",
            "_source": "typed_state",
            "_object_type": "state_fact",
            "_hybrid_score": round(0.85 / (60 + i), 6),
            "_original": {
                "object_id": f"sf_{i}",
                "subject": "user",
                "predicate": "likes",
                "value": f"thing {i}",
            },
        }
        for i in range(1, n + 1)
    ]


def _mixed_candidates() -> list[dict[str, Any]]:
    return _graph_candidates(3) + _typed_candidates(2)


# ─────────────────────────────────────────────────────────────────────────────
# §1  Passthrough mode (no API key or disabled)
# ─────────────────────────────────────────────────────────────────────────────


def test_passthrough_when_disabled():
    """Disabled reranker returns candidates in original order with method='passthrough'."""
    svc = LLMRerankerService(enabled=False, api_key="sk-test")
    candidates = _mixed_candidates()
    result = _run(svc.rerank(query="test", candidates=candidates))

    assert isinstance(result, RerankResult)
    assert result.method == "passthrough"
    assert result.diagnostics.get("reason") == "disabled"
    assert len(result.candidates) == len(candidates)
    # Order preserved
    for orig, returned in zip(candidates, result.candidates):
        assert orig["uuid"] == returned["uuid"]


def test_passthrough_when_no_api_key():
    """No API key → passthrough mode."""
    svc = LLMRerankerService(enabled=True, api_key="")
    result = _run(svc.rerank(query="test", candidates=_graph_candidates()))

    assert result.method == "passthrough"
    assert result.diagnostics.get("reason") == "no_api_key"


def test_passthrough_empty_input():
    """Empty candidates → empty result with passthrough."""
    svc = LLMRerankerService(enabled=True, api_key="sk-test")
    result = _run(svc.rerank(query="test", candidates=[]))

    assert result.method == "passthrough"
    assert result.candidates == []
    assert result.total_scored == 0


def test_is_available_property():
    """is_available reflects enabled + API key presence."""
    assert LLMRerankerService(enabled=True, api_key="sk-test").is_available
    assert not LLMRerankerService(enabled=False, api_key="sk-test").is_available
    assert not LLMRerankerService(enabled=True, api_key="").is_available


# ─────────────────────────────────────────────────────────────────────────────
# §2  Score parsing
# ─────────────────────────────────────────────────────────────────────────────


def test_parse_response_json_array():
    """Standard JSON array response is parsed correctly."""
    svc = LLMRerankerService(api_key="sk-test")
    content = json.dumps([
        {"index": 0, "score": 0.9, "rationale": "directly relevant"},
        {"index": 1, "score": 0.3, "rationale": "tangential"},
    ])
    result = svc._parse_response(content, expected_count=2)
    assert len(result) == 2
    assert result[0]["score"] == 0.9
    assert result[1]["score"] == 0.3


def test_parse_response_with_markdown_fences():
    """JSON wrapped in markdown code fences is parsed."""
    svc = LLMRerankerService(api_key="sk-test")
    content = '```json\n[{"index": 0, "score": 0.8, "rationale": "good"}]\n```'
    result = svc._parse_response(content, expected_count=1)
    assert len(result) == 1
    assert result[0]["score"] == 0.8


def test_parse_response_dict_with_results_key():
    """Response as dict with 'results' key is parsed."""
    svc = LLMRerankerService(api_key="sk-test")
    content = json.dumps({
        "results": [{"index": 0, "score": 0.5, "rationale": "ok"}]
    })
    result = svc._parse_response(content, expected_count=1)
    assert len(result) == 1


def test_parse_response_alternative_field_names():
    """Alternative field names (relevance, candidate_index, reason) are accepted."""
    svc = LLMRerankerService(api_key="sk-test")
    content = json.dumps([
        {"candidate_index": 0, "relevance": 0.7, "reason": "somewhat relevant"},
    ])
    result = svc._parse_response(content, expected_count=1)
    assert len(result) == 1
    assert result[0]["score"] == 0.7


def test_parse_response_out_of_range_index_dropped():
    """Scores with index >= expected_count are dropped."""
    svc = LLMRerankerService(api_key="sk-test")
    content = json.dumps([
        {"index": 0, "score": 0.9},
        {"index": 99, "score": 0.5},  # out of range
    ])
    result = svc._parse_response(content, expected_count=2)
    assert len(result) == 1
    assert result[0]["index"] == 0


def test_parse_response_score_clamped():
    """Scores outside [0, 1] are clamped."""
    svc = LLMRerankerService(api_key="sk-test")
    content = json.dumps([
        {"index": 0, "score": 1.5},
        {"index": 1, "score": -0.3},
    ])
    result = svc._parse_response(content, expected_count=2)
    assert result[0]["score"] == 1.0
    assert result[1]["score"] == 0.0


def test_parse_response_empty_content():
    """Empty or unparseable content returns empty list."""
    svc = LLMRerankerService(api_key="sk-test")
    assert svc._parse_response("", expected_count=2) == []
    assert svc._parse_response("not json at all", expected_count=2) == []


def test_normalize_scores_handles_non_dict_items():
    """Non-dict items in the score list are skipped."""
    result = LLMRerankerService._normalize_scores(
        [{"index": 0, "score": 0.8}, "garbage", 42, {"index": 1, "score": 0.3}],
        expected_count=2,
    )
    assert len(result) == 2


# ─────────────────────────────────────────────────────────────────────────────
# §3  Blend-and-sort logic
# ─────────────────────────────────────────────────────────────────────────────


def test_blend_and_sort_llm_scores_dominate():
    """With default 80/20 blend, a high LLM score dominates a low RRF score."""
    svc = LLMRerankerService(api_key="sk-test")
    candidates = [
        {"uuid": "a", "_hybrid_score": 0.016, "_source": "graph", "fact": "a"},
        {"uuid": "b", "_hybrid_score": 0.014, "_source": "graph", "fact": "b"},
    ]
    scores = [
        {"index": 0, "score": 0.2, "rationale": "low relevance"},
        {"index": 1, "score": 0.9, "rationale": "highly relevant"},
    ]
    result = svc._blend_and_sort(candidates, scores)
    assert len(result) == 2
    # b (LLM=0.9) should rank above a (LLM=0.2) despite a having higher RRF
    assert result[0].original["uuid"] == "b"
    assert result[1].original["uuid"] == "a"
    # Verify ranks
    assert result[0].final_rank == 1
    assert result[1].final_rank == 2


def test_blend_and_sort_unscored_candidates_ranked_last():
    """Candidates not in the score map get LLM=0.0 and rank lower."""
    svc = LLMRerankerService(api_key="sk-test")
    candidates = [
        {"uuid": "a", "_hybrid_score": 0.01, "_source": "graph", "fact": "a"},
        {"uuid": "b", "_hybrid_score": 0.01, "_source": "graph", "fact": "b"},
    ]
    # Only score index 0
    scores = [{"index": 0, "score": 0.8, "rationale": "good"}]
    result = svc._blend_and_sort(candidates, scores)
    assert result[0].original["uuid"] == "a"  # scored
    assert result[1].original["uuid"] == "b"  # unscored → rank 2


def test_blend_weight_formula():
    """Verify the blended score formula: (1-w)*llm + w*rrf*scale."""
    svc = LLMRerankerService(api_key="sk-test", rrf_blend_weight=0.2)
    candidates = [{"uuid": "a", "_hybrid_score": 0.01, "_source": "graph", "fact": "a"}]
    scores = [{"index": 0, "score": 0.5}]
    result = svc._blend_and_sort(candidates, scores)

    expected = (1 - 0.2) * 0.5 + 0.2 * 0.01 * 60.0
    assert abs(result[0].blended_score - expected) < 0.001


# ─────────────────────────────────────────────────────────────────────────────
# §4  Fallback on LLM failure
# ─────────────────────────────────────────────────────────────────────────────


def test_fallback_on_llm_exception():
    """LLM call raising an exception → fallback with RRF order preserved."""
    svc = LLMRerankerService(api_key="sk-test", enabled=True)

    candidates = _graph_candidates(3)
    with patch.object(svc, "_score_candidates", side_effect=RuntimeError("API down")):
        result = _run(svc.rerank(query="test", candidates=candidates))

    assert result.method == "fallback"
    assert "llm_error" in str(result.diagnostics.get("reason", ""))
    assert len(result.candidates) == 3
    # Order preserved (fallback = original RRF order)
    for orig, returned in zip(candidates, result.candidates):
        assert orig["uuid"] == returned["uuid"]


def test_fallback_on_empty_llm_response():
    """LLM returning empty scores → fallback."""
    svc = LLMRerankerService(api_key="sk-test", enabled=True)

    async def empty_scores(*args, **kwargs):
        return []

    candidates = _graph_candidates(2)
    with patch.object(svc, "_score_candidates", side_effect=empty_scores):
        result = _run(svc.rerank(query="test", candidates=candidates))

    assert result.method == "fallback"
    assert result.diagnostics.get("reason") == "empty_llm_response"


# ─────────────────────────────────────────────────────────────────────────────
# §5  Candidate text extraction
# ─────────────────────────────────────────────────────────────────────────────


def test_extract_text_from_graph_fact():
    """Graph facts use 'fact' field."""
    text = LLMRerankerService._extract_candidate_text(
        {"fact": "user prefers dark roast", "_source": "graph"}
    )
    assert text == "user prefers dark roast"


def test_extract_text_from_typed_spv():
    """Typed items use subject/predicate/value."""
    text = LLMRerankerService._extract_candidate_text(
        {"subject": "user", "predicate": "likes", "value": "tea", "_source": "typed_state"}
    )
    assert "user" in text
    assert "likes" in text
    assert "tea" in text


def test_extract_text_fallback_to_name():
    """Falls back to name when no fact/spv."""
    text = LLMRerankerService._extract_candidate_text({"name": "reset password"})
    assert text == "reset password"


def test_extract_text_fallback_to_uuid():
    """Falls back to uuid when nothing else."""
    text = LLMRerankerService._extract_candidate_text({"uuid": "abc-123"})
    assert text == "abc-123"


# ─────────────────────────────────────────────────────────────────────────────
# §6  Model resolution
# ─────────────────────────────────────────────────────────────────────────────


def test_resolve_model_openrouter_adds_prefix():
    """OpenRouter API base → model gets openai/ prefix if missing."""
    svc = LLMRerankerService(
        api_key="sk-or-test",
        api_base="https://openrouter.ai/api/v1",
        model="gpt-5.4-nano",
    )
    assert svc._resolve_model() == "openai/gpt-5.4-nano"


def test_resolve_model_openrouter_preserves_existing_prefix():
    """OpenRouter with existing prefix is preserved."""
    svc = LLMRerankerService(
        api_key="sk-or-test",
        api_base="https://openrouter.ai/api/v1",
        model="anthropic/claude-3.5-sonnet",
    )
    assert svc._resolve_model() == "anthropic/claude-3.5-sonnet"


def test_resolve_model_openai_strips_prefix():
    """OpenAI API base → openai/ prefix is stripped."""
    svc = LLMRerankerService(
        api_key="sk-test",
        api_base="https://api.openai.com/v1",
        model="openai/gpt-5.4-nano",
    )
    assert svc._resolve_model() == "gpt-5.4-nano"


def test_detect_api_base_from_openrouter_key():
    """sk-or- prefix → OpenRouter base."""
    base = LLMRerankerService._detect_api_base("sk-or-abc123")
    assert "openrouter" in base


def test_detect_api_base_from_openai_key():
    """Normal key → OpenAI base."""
    base = LLMRerankerService._detect_api_base("sk-abc123")
    assert "openai" in base


# ─────────────────────────────────────────────────────────────────────────────
# §7  RerankedCandidate.to_annotated_dict
# ─────────────────────────────────────────────────────────────────────────────


def test_annotated_dict_includes_rerank_fields():
    """to_annotated_dict adds _rerank_score, _blended_score, _final_rank."""
    rc = RerankedCandidate(
        original={"uuid": "a", "_source": "graph", "fact": "test"},
        llm_score=0.85,
        rrf_score=0.01,
        blended_score=0.72,
        final_rank=1,
        rationale="directly relevant",
    )
    d = rc.to_annotated_dict()
    assert d["_rerank_score"] == 0.85
    assert d["_blended_score"] == 0.72
    assert d["_final_rank"] == 1
    assert d["_rerank_rationale"] == "directly relevant"
    # Original fields preserved
    assert d["uuid"] == "a"
    assert d["_source"] == "graph"


def test_annotated_dict_omits_empty_rationale():
    """Empty rationale is not added to the dict."""
    rc = RerankedCandidate(
        original={"uuid": "a"},
        llm_score=0.5,
        rrf_score=0.01,
        blended_score=0.4,
        final_rank=2,
        rationale="",
    )
    d = rc.to_annotated_dict()
    assert "_rerank_rationale" not in d


# ─────────────────────────────────────────────────────────────────────────────
# §8  max_results cap
# ─────────────────────────────────────────────────────────────────────────────


def test_max_results_caps_output():
    """max_results limits the number of returned candidates."""
    svc = LLMRerankerService(enabled=False)
    candidates = _graph_candidates(10)
    result = _run(svc.rerank(query="test", candidates=candidates, max_results=3))
    assert len(result.candidates) == 3


def test_max_results_none_returns_all():
    """max_results=None returns all candidates."""
    svc = LLMRerankerService(enabled=False)
    candidates = _graph_candidates(5)
    result = _run(svc.rerank(query="test", candidates=candidates, max_results=None))
    assert len(result.candidates) == 5


# ─────────────────────────────────────────────────────────────────────────────
# §9  RerankResult properties
# ─────────────────────────────────────────────────────────────────────────────


def test_rerank_result_is_degraded():
    """is_degraded is True for passthrough/fallback, False for llm."""
    assert RerankResult(candidates=[], total_scored=0, method="passthrough").is_degraded
    assert RerankResult(candidates=[], total_scored=0, method="fallback").is_degraded
    assert not RerankResult(candidates=[], total_scored=0, method="llm").is_degraded


# ─────────────────────────────────────────────────────────────────────────────
# §10  User prompt construction
# ─────────────────────────────────────────────────────────────────────────────


def test_build_user_prompt_includes_query():
    """User prompt includes the query text."""
    svc = LLMRerankerService(api_key="sk-test")
    prompt = svc._build_user_prompt(
        "user preferences",
        [{"uuid": "g1", "fact": "likes coffee", "_source": "graph"}],
    )
    assert "user preferences" in prompt
    assert "likes coffee" in prompt
    assert "(graph)" in prompt


def test_build_user_prompt_truncates_long_text():
    """Candidate text is truncated to 400 chars."""
    svc = LLMRerankerService(api_key="sk-test")
    long_fact = "x" * 1000
    prompt = svc._build_user_prompt(
        "test",
        [{"uuid": "g1", "fact": long_fact, "_source": "graph"}],
    )
    # The prompt should contain at most 400 chars of the fact
    lines = prompt.split("\n")
    candidate_line = [l for l in lines if l.startswith("[0]")][0]
    # 400 chars + prefix overhead
    assert len(candidate_line) < 500
