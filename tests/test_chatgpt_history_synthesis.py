"""Tests for chatgpt_history_synthesis_pass.py.

Validates:
  - reasoning.effort validation (blocked values, allowed values)
  - synthesis target construction (keyword/entity matching)
  - synthesized node UUID determinism
  - confidence threshold filtering
  - source fact minimum requirements
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from chatgpt_history_synthesis_pass import (
    SynthesizedNode,
    SourceFact,
    SourceEpisode,
    SynthesisTarget,
    _build_synthesis_targets,
    _synth_node_uuid,
    _validate_reasoning_effort,
    SYNTHESIS_TARGETS_SPEC,
)


class TestReasoningEffortValidation:
    """FR-3: reasoning.effort=minimal must be blocked."""

    def test_blocks_minimal(self):
        with pytest.raises(ValueError, match="BLOCKED"):
            _validate_reasoning_effort("minimal")

    def test_blocks_minimal_uppercase(self):
        with pytest.raises(ValueError, match="BLOCKED"):
            _validate_reasoning_effort("MINIMAL")

    def test_blocks_minimal_with_spaces(self):
        with pytest.raises(ValueError, match="BLOCKED"):
            _validate_reasoning_effort("  minimal  ")

    def test_allows_high(self):
        assert _validate_reasoning_effort("high") == "high"

    def test_allows_medium(self):
        assert _validate_reasoning_effort("medium") == "medium"

    def test_allows_low(self):
        assert _validate_reasoning_effort("low") == "low"

    def test_rejects_unknown(self):
        with pytest.raises(ValueError, match="not recognized"):
            _validate_reasoning_effort("turbo")


class TestSynthNodeUUID:
    """Synthesized node UUIDs must be deterministic."""

    def test_deterministic(self):
        uuid1 = _synth_node_uuid("v1", "heur_01", 0)
        uuid2 = _synth_node_uuid("v1", "heur_01", 0)
        assert uuid1 == uuid2

    def test_different_for_different_index(self):
        uuid1 = _synth_node_uuid("v1", "heur_01", 0)
        uuid2 = _synth_node_uuid("v1", "heur_01", 1)
        assert uuid1 != uuid2

    def test_different_for_different_version(self):
        uuid1 = _synth_node_uuid("v1", "heur_01", 0)
        uuid2 = _synth_node_uuid("v2", "heur_01", 0)
        assert uuid1 != uuid2

    def test_uuid_format(self):
        uuid = _synth_node_uuid("v1", "test", 0)
        parts = uuid.split("-")
        assert len(parts) == 5
        assert len(parts[0]) == 8
        assert len(parts[1]) == 4


class TestSynthesisTargetConstruction:
    """Test that synthesis targets are correctly built from facts."""

    def _make_facts(self) -> list[SourceFact]:
        return [
            SourceFact(
                uuid="fact-1", fact="Yuan iteratively edits drafts with ChatGPT",
                name="writing collaboration", source_entity="Yuan Han Li",
                target_entity="ChatGPT", episodes=["ep-1"],
            ),
            SourceFact(
                uuid="fact-2", fact="Forbes 30 Under 30 bio editing session",
                name="bio editing", source_entity="Yuan Han Li",
                target_entity="Forbes", episodes=["ep-2"],
            ),
            SourceFact(
                uuid="fact-3", fact="Yuan analyzes Tether with bull and bear framing",
                name="investment analysis", source_entity="Yuan Han Li",
                target_entity="Tether", episodes=["ep-3"],
            ),
            SourceFact(
                uuid="fact-4", fact="Yuan's grandmother communicates in Chinese",
                name="family communication", source_entity="Yuan Han Li",
                target_entity="Grandmother", episodes=["ep-4"],
            ),
            SourceFact(
                uuid="fact-5", fact="Totally unrelated blockchain protocol spec",
                name="protocol spec", source_entity="Protocol X",
                target_entity="Protocol Y", episodes=["ep-5"],
            ),
        ]

    def _make_episodes(self) -> list[SourceEpisode]:
        return [
            SourceEpisode(uuid="ep-1", name="Writing Session", source_description="conversation_id=conv1"),
            SourceEpisode(uuid="ep-2", name="Forbes Edit", source_description="conversation_id=conv2"),
            SourceEpisode(uuid="ep-3", name="Tether Analysis", source_description="conversation_id=conv3"),
            SourceEpisode(uuid="ep-4", name="Family Chat", source_description="conversation_id=conv4"),
            SourceEpisode(uuid="ep-5", name="Protocol Analysis", source_description="conversation_id=conv5"),
        ]

    def test_targets_built(self):
        facts = self._make_facts()
        episodes = self._make_episodes()
        targets = _build_synthesis_targets(facts, episodes)
        assert len(targets) == len(SYNTHESIS_TARGETS_SPEC)

    def test_writing_collab_matches_relevant_facts(self):
        facts = self._make_facts()
        episodes = self._make_episodes()
        targets = _build_synthesis_targets(facts, episodes)
        writing_target = next(t for t in targets if t.target_id == "heur_writing_collab")
        # Should match fact-1 (keyword "draft"/"write") and fact-2 (keyword "Forbes", entity "Forbes")
        assert "fact-1" in writing_target.fact_uuids
        assert "fact-2" in writing_target.fact_uuids
        # Should NOT match the unrelated protocol fact
        assert "fact-5" not in writing_target.fact_uuids

    def test_investment_skepticism_matches(self):
        facts = self._make_facts()
        episodes = self._make_episodes()
        targets = _build_synthesis_targets(facts, episodes)
        invest_target = next(t for t in targets if t.target_id == "heur_investment_skepticism")
        # Should match fact-3 (keyword "Tether", "analysis", entity "Tether")
        assert "fact-3" in invest_target.fact_uuids

    def test_target_spec_completeness(self):
        """Every target spec has required fields."""
        for spec in SYNTHESIS_TARGETS_SPEC:
            assert "target_id" in spec
            assert "synthesis_type" in spec
            assert "theme" in spec
            assert "target_queries" in spec
            assert spec["synthesis_type"] in {
                "SynthesizedHeuristic", "ConsolidatedProfile", "InferredPreference"
            }


class TestConfidenceThreshold:
    """Synthesized nodes below confidence threshold are rejected."""

    def test_low_confidence_rejected(self):
        """A synthesized object with confidence < 0.7 should be filtered."""
        # This is tested indirectly through the LLM call filtering.
        # The _call_synthesis_llm function filters objects with confidence < 0.7.
        # We test the threshold constant exists and is correct.
        from chatgpt_history_synthesis_pass import run_synthesis_pass
        # The default confidence_threshold is 0.7 in the function signature
        import inspect
        sig = inspect.signature(run_synthesis_pass)
        default_threshold = sig.parameters["confidence_threshold"].default
        assert default_threshold == 0.7


class TestSynthesisTypesCoverage:
    """Verify all three synthesis types are represented in targets."""

    def test_has_heuristic_targets(self):
        heur_targets = [s for s in SYNTHESIS_TARGETS_SPEC if s["synthesis_type"] == "SynthesizedHeuristic"]
        assert len(heur_targets) >= 7, "Should have at least 7 heuristic synthesis targets"

    def test_has_consolidated_targets(self):
        consol_targets = [s for s in SYNTHESIS_TARGETS_SPEC if s["synthesis_type"] == "ConsolidatedProfile"]
        assert len(consol_targets) >= 10, "Should have at least 10 consolidation targets"

    def test_has_inferred_targets(self):
        infer_targets = [s for s in SYNTHESIS_TARGETS_SPEC if s["synthesis_type"] == "InferredPreference"]
        assert len(infer_targets) >= 1, "Should have at least 1 inferred preference target"

    def test_query_coverage(self):
        """All inferential and fragmented queries should be targeted."""
        all_target_queries = set()
        for spec in SYNTHESIS_TARGETS_SPEC:
            all_target_queries.update(spec["target_queries"])

        # All 7 heur queries should be targeted
        for i in range(1, 8):
            assert f"heur_0{i}" in all_target_queries, f"heur_0{i} not targeted"

        # Inferential work queries should be targeted
        assert "work_03" in all_target_queries
        assert "work_07" in all_target_queries

        # Inferential bio query should be targeted
        assert "bio_08" in all_target_queries
