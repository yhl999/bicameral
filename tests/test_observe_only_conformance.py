"""Tests for FR-11 observe-only ontology conformance gate.

Covers:
  1. _build_episode_body strips untrusted metadata (sanitization fix)
  2. compute_conformance_metrics basic calculations on fixture/mock data
  3. Ontology config loads for s1_sessions_main and s1_pilot_fr11_20260227 lanes
  4. Conformance evaluator exit-code-0 guarantee (observe-only contract)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ── 1. _build_episode_body sanitization ───────────────────────────────────────


def test_build_episode_body_strips_untrusted_metadata():
    """_build_episode_body must sanitize message content before building the body."""
    from scripts.mcp_ingest_sessions import _build_episode_body

    msg_with_metadata = (
        "Hello, this is my real message.\n\n"
        "Conversation info (untrusted metadata):\n"
        "```json\n"
        '{"message_id": "abc123", "platform": "telegram"}\n'
        "```\n\n"
        "See you later."
    )
    messages_by_id = {
        "msg1": {
            "created_at": "2026-02-27T10:00:00Z",
            "role": "user",
            "content": msg_with_metadata,
        }
    }
    body = _build_episode_body(["msg1"], messages_by_id)

    # The untrusted metadata block must be stripped
    assert "Conversation info (untrusted metadata):" not in body
    assert '```json' not in body
    assert "message_id" not in body
    # Real content must be preserved
    assert "Hello, this is my real message." in body
    assert "See you later." in body


def test_build_episode_body_clean_content_unchanged():
    """Content without untrusted metadata blocks should be returned as-is."""
    from scripts.mcp_ingest_sessions import _build_episode_body

    clean_content = "This is a normal message without any metadata."
    messages_by_id = {
        "m1": {"created_at": "2026-02-27T09:00:00", "role": "assistant", "content": clean_content}
    }
    body = _build_episode_body(["m1"], messages_by_id)
    assert clean_content in body


def test_build_episode_body_multiple_messages_sanitized():
    """Multiple messages should each be sanitized independently."""
    from scripts.mcp_ingest_sessions import _build_episode_body

    messages_by_id = {
        "a": {
            "created_at": "2026-02-27T08:00:00",
            "role": "user",
            "content": (
                "First real content.\n\n"
                "Sender (untrusted metadata):\n"
                "```json\n{\"user_id\": \"u1\"}\n```\n\n"
            ),
        },
        "b": {
            "created_at": "2026-02-27T08:01:00",
            "role": "assistant",
            "content": "Second clean content.",
        },
    }
    body = _build_episode_body(["a", "b"], messages_by_id)
    assert "user_id" not in body
    assert "First real content." in body
    assert "Second clean content." in body


def test_build_episode_body_empty_content():
    """Empty content should produce an empty result (not crash)."""
    from scripts.mcp_ingest_sessions import _build_episode_body

    messages_by_id = {
        "x": {"created_at": "2026-02-27T12:00:00", "role": "user", "content": ""},
    }
    # Should not raise
    body = _build_episode_body(["x"], messages_by_id)
    assert isinstance(body, str)


# ── 2. compute_conformance_metrics ───────────────────────────────────────────


def test_conformance_metrics_all_in_schema():
    """When all entities and relations are in-schema, rates should be 1.0."""
    from scripts.evaluate_ontology_conformance import compute_conformance_metrics

    entities = [
        {"entity_type": "Preference", "name": "e1"},
        {"entity_type": "Requirement", "name": "e2"},
        {"entity_type": "Organization", "name": "e3"},
    ]
    relations = [
        {"relation_type": "RELATES_TO"},
        {"relation_type": "PREFERS"},
        {"relation_type": "REQUIRES"},
    ]
    allowed_entities = {"Preference", "Requirement", "Organization"}
    allowed_relations = {"RELATES_TO", "PREFERS", "REQUIRES"}

    m = compute_conformance_metrics(entities, relations, allowed_entities, allowed_relations)

    assert m["typed_entity_rate"] == 1.0
    assert m["allowed_relation_rate"] == 1.0
    assert m["out_of_schema_count"] == 0
    assert m["out_of_schema_types"] == []
    assert m["off_schema_entity_types"] == []


def test_conformance_metrics_partial_schema():
    """Partial off-schema data should yield rates < 1.0."""
    from scripts.evaluate_ontology_conformance import compute_conformance_metrics

    entities = [
        {"entity_type": "Preference", "name": "e1"},
        {"entity_type": "UNKNOWN_TYPE", "name": "e2"},
    ]
    relations = [
        {"relation_type": "RELATES_TO"},
        {"relation_type": "OFF_SCHEMA_REL"},
    ]
    allowed_entities = {"Preference", "Requirement"}
    allowed_relations = {"RELATES_TO", "PREFERS"}

    m = compute_conformance_metrics(entities, relations, allowed_entities, allowed_relations)

    assert m["total_entities"] == 2
    assert m["typed_entities"] == 1
    assert m["typed_entity_rate"] == pytest.approx(0.5)
    assert m["total_relations"] == 2
    assert m["allowed_relations"] == 1
    assert m["allowed_relation_rate"] == pytest.approx(0.5)
    assert m["out_of_schema_count"] == 1
    assert "OFF_SCHEMA_REL" in m["out_of_schema_types"]
    assert "UNKNOWN_TYPE" in m["off_schema_entity_types"]


def test_conformance_metrics_empty_inputs():
    """Empty inputs should return 1.0 rates (vacuously true) and zero counts."""
    from scripts.evaluate_ontology_conformance import compute_conformance_metrics

    m = compute_conformance_metrics([], [], {"Preference"}, {"RELATES_TO"})

    assert m["typed_entity_rate"] == 1.0
    assert m["allowed_relation_rate"] == 1.0
    assert m["total_entities"] == 0
    assert m["total_relations"] == 0
    assert m["out_of_schema_count"] == 0


def test_conformance_metrics_all_off_schema():
    """All-off-schema data should yield 0.0 rates."""
    from scripts.evaluate_ontology_conformance import compute_conformance_metrics

    entities = [{"entity_type": "NOPE", "name": "x"}]
    relations = [{"relation_type": "UNKNOWN_REL"}]

    m = compute_conformance_metrics(entities, relations, {"Preference"}, {"RELATES_TO"})

    assert m["typed_entity_rate"] == 0.0
    assert m["allowed_relation_rate"] == 0.0
    assert m["out_of_schema_count"] == 1


# ── 3. Ontology config loads for new lanes ────────────────────────────────────


def _get_ontology_path(filename: str = "extraction_ontologies.yaml") -> Path:
    """Find the extraction_ontologies.yaml path relative to the repo root."""
    repo_root = Path(__file__).resolve().parents[1]
    # Prefer mcp_server config as the canonical runtime copy
    candidates = [
        repo_root / "mcp_server" / "config" / filename,
        repo_root / "config" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    pytest.skip(f"{filename} not found in expected locations")


@pytest.mark.parametrize("group_id", ["s1_sessions_main", "s1_pilot_fr11_20260227"])
def test_sessions_lane_ontology_loads(group_id: str):
    """s1_sessions_main and s1_pilot_fr11_20260227 must load from the registry."""
    import importlib

    ontology_registry = importlib.import_module("mcp_server.src.services.ontology_registry")
    OntologyRegistry = ontology_registry.OntologyRegistry

    config_path = _get_ontology_path()
    registry = OntologyRegistry.load(config_path)

    profile = registry.get(group_id)
    assert profile is not None, f"No profile found for {group_id}"


@pytest.mark.parametrize("group_id", ["s1_sessions_main", "s1_pilot_fr11_20260227"])
def test_sessions_lane_extraction_mode_constrained_soft(group_id: str):
    """Sessions lanes must have extraction_mode=constrained_soft."""
    import importlib

    ontology_registry = importlib.import_module("mcp_server.src.services.ontology_registry")
    OntologyRegistry = ontology_registry.OntologyRegistry

    config_path = _get_ontology_path()
    registry = OntologyRegistry.load(config_path)

    profile = registry.get(group_id)
    assert profile is not None
    assert profile.extraction_mode == "constrained_soft", (
        f"{group_id} should use constrained_soft, got {profile.extraction_mode!r}"
    )


@pytest.mark.parametrize("group_id", ["s1_sessions_main", "s1_pilot_fr11_20260227"])
def test_sessions_lane_has_relates_to(group_id: str):
    """Sessions lanes must include RELATES_TO for backwards compatibility."""
    import importlib

    ontology_registry = importlib.import_module("mcp_server.src.services.ontology_registry")
    OntologyRegistry = ontology_registry.OntologyRegistry

    config_path = _get_ontology_path()
    registry = OntologyRegistry.load(config_path)

    profile = registry.get(group_id)
    assert profile is not None
    assert "RELATES_TO" in profile.edge_types, (
        f"{group_id} must include RELATES_TO in relationship_types for backwards compatibility"
    )


@pytest.mark.parametrize("group_id", ["s1_sessions_main", "s1_pilot_fr11_20260227"])
def test_sessions_lane_entity_types_include_graphiti_defaults(group_id: str):
    """Sessions lanes must include the standard graphiti entity types."""
    import importlib

    ontology_registry = importlib.import_module("mcp_server.src.services.ontology_registry")
    OntologyRegistry = ontology_registry.OntologyRegistry

    config_path = _get_ontology_path()
    registry = OntologyRegistry.load(config_path)

    profile = registry.get(group_id)
    assert profile is not None

    expected_types = {"Preference", "Requirement", "Procedure", "Location", "Event", "Organization"}
    actual_types = set(profile.entity_types.keys())
    missing = expected_types - actual_types
    assert not missing, f"{group_id} is missing entity types: {missing}"


# ── 4. Observe-only contract: exit code 0 guarantee ──────────────────────────


def test_conformance_evaluator_dry_run_exit_zero():
    """Dry-run invocation must always exit 0 (observe-only contract)."""
    from scripts.evaluate_ontology_conformance import main

    exit_code = main(["--group-id", "s1_sessions_main", "--dry-run"])
    assert exit_code == 0, "Conformance evaluator must exit 0 in dry-run mode"


def test_conformance_evaluator_dry_run_below_threshold_still_zero():
    """Even when metrics are below threshold, exit code must be 0 (observe-only)."""
    from scripts.evaluate_ontology_conformance import main

    # Use very high thresholds to trigger warnings, but still expect exit 0
    exit_code = main([
        "--group-id", "s1_sessions_main",
        "--dry-run",
        "--typed-entity-threshold", "0.99",
        "--allowed-relation-threshold", "0.99",
    ])
    assert exit_code == 0, "Observe-only: below-threshold must NOT produce non-zero exit"


def test_conformance_evaluator_dry_run_produces_valid_json(capsys):
    """Dry-run output must be valid JSON with required report fields."""
    from scripts.evaluate_ontology_conformance import main

    main(["--group-id", "s1_pilot_fr11_20260227", "--dry-run", "--output", "json"])
    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert report["group_id"] == "s1_pilot_fr11_20260227"
    assert report["observe_only"] is True
    assert report["dry_run"] is True
    assert "metrics" in report
    assert "conformance_passed" in report
    assert "note" in report
    # The note must mention observe-only
    assert "observe" in report["note"].lower()


def test_conformance_evaluator_observe_only_note_in_report(capsys):
    """The JSON report must explicitly document observe-only behavior."""
    from scripts.evaluate_ontology_conformance import main

    main(["--group-id", "s1_sessions_main", "--dry-run", "--output", "json"])
    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert report["observe_only"] is True
    # Must not say anything about dropping episodes
    note = report["note"].lower()
    assert "no episodes were dropped" in note or "observe-only" in note


def test_conformance_evaluator_dry_run_fixture_metrics(capsys):
    """Dry-run with fixture data should produce plausible metric values."""
    from scripts.evaluate_ontology_conformance import main

    main([
        "--group-id", "s1_sessions_main",
        "--dry-run",
        "--output", "json",
    ])
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    m = report["metrics"]

    assert m["total_entities"] > 0
    assert m["total_relations"] > 0
    assert 0.0 <= m["typed_entity_rate"] <= 1.0
    assert 0.0 <= m["allowed_relation_rate"] <= 1.0
    assert m["out_of_schema_count"] >= 0
