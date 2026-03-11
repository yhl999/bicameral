"""Tests for truth.entity_type_routing."""
from __future__ import annotations

import pytest

from truth.entity_type_routing import route_entity_edge


def test_procedure_label_on_source() -> None:
    result = route_entity_edge(
        source_name="SendWeeklyDigest",
        target_name="Email",
        rel_name="DELIVERS_TO",
        fact="SendWeeklyDigest delivers content to Email subscribers every Monday.",
        a_labels=["Procedure", "Entity"],
        b_labels=["Entity"],
    )
    assert result is not None
    predicate, assertion_type, value, confidence = result
    assert assertion_type == "procedure"
    assert predicate == "procedure.steps"
    assert value["name"] == "SendWeeklyDigest"
    assert value["trigger"] == "DELIVERS_TO"
    assert confidence == pytest.approx(0.72)


def test_procedure_label_on_target() -> None:
    result = route_entity_edge(
        source_name="Yuan",
        target_name="MorningWorkout",
        rel_name="FOLLOWS",
        fact="Yuan follows MorningWorkout before 10:30am.",
        a_labels=["Entity"],
        b_labels=["Procedure"],
    )
    assert result is not None
    _, assertion_type, _, _ = result
    assert assertion_type == "procedure"


def test_episode_label_routing() -> None:
    result = route_entity_edge(
        source_name="CodexQuotaExhaustion",
        target_name="PRBlockedEvent",
        rel_name="CAUSED",
        fact="Codex quota exhaustion caused 6 PRs to be blocked.",
        a_labels=["EngineeringIncident"],
        b_labels=["Entity"],
    )
    assert result is not None
    _, assertion_type, value, _ = result
    assert assertion_type == "episode"
    assert "CodexQuotaExhaustion" in value["participants"]


def test_no_special_labels_returns_none() -> None:
    result = route_entity_edge(
        source_name="Yuan",
        target_name="BlockchainCapital",
        rel_name="WORKS_AT",
        fact="Yuan works at Blockchain Capital as a Partner.",
        a_labels=["Person"],
        b_labels=["Organization"],
    )
    assert result is None


def test_none_labels_returns_none() -> None:
    result = route_entity_edge(
        source_name="Yuan",
        target_name="BlockchainCapital",
        rel_name="WORKS_AT",
        fact="Yuan works at Blockchain Capital.",
        a_labels=None,
        b_labels=None,
    )
    assert result is None


def test_empty_labels_returns_none() -> None:
    result = route_entity_edge(
        source_name="Yuan",
        target_name="BlockchainCapital",
        rel_name="WORKS_AT",
        fact="Yuan works at Blockchain Capital.",
        a_labels=[],
        b_labels=[],
    )
    assert result is None


def test_rel_name_fallback_when_empty() -> None:
    result = route_entity_edge(
        source_name="SendDigest",
        target_name="Subscriber",
        rel_name="",
        fact="SendDigest sends to Subscriber.",
        a_labels=["Procedure"],
        b_labels=["Entity"],
    )
    assert result is not None
    _, _, value, _ = result
    assert value["trigger"] == "relates_to"
