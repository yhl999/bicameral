from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

om_compressor = importlib.import_module("scripts.om_compressor")


def test_default_ontology_config_resolution_is_repo_relative(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OM_ONTOLOGY_CONFIG_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    resolved = om_compressor._resolve_ontology_config_path(None)

    assert resolved == om_compressor.REPO_ROOT / om_compressor.DEFAULT_ONTOLOGY_CONFIG_REL


def test_ontology_config_env_override_resolves_relative_to_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OM_ONTOLOGY_CONFIG_PATH", "config/extraction_ontologies.yaml")

    resolved = om_compressor._resolve_ontology_config_path(None)

    assert resolved == om_compressor.REPO_ROOT / "config/extraction_ontologies.yaml"


def test_lock_path_default_not_legacy_tmp_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OM_COMPRESSOR_LOCK_PATH", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)

    lock_path = om_compressor._resolve_lock_path()

    assert lock_path.name == om_compressor.DEFAULT_LOCK_FILENAME
    assert str(lock_path) != "/tmp/om_graph_write.lock"


def test_lock_path_env_override_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OM_COMPRESSOR_LOCK_PATH", "state/custom/om.lock")

    lock_path = om_compressor._resolve_lock_path()

    assert lock_path == om_compressor.REPO_ROOT / "state/custom/om.lock"


# ---------------------------------------------------------------------------
# OM-1: metadata stripping in om_compressor path
# ---------------------------------------------------------------------------


def test_strip_untrusted_metadata_removes_contamination_block() -> None:
    """Contaminated 'Conversation info (untrusted metadata)' JSON block is stripped."""
    content = (
        "Useful message content.\n\n"
        "Conversation info (untrusted metadata):\n"
        "```json\n"
        '{"message_id": "12345", "session_id": "abc", "source": "telegram"}\n'
        "```\n"
        "More content after block."
    )
    result = om_compressor.strip_untrusted_metadata(content)
    assert "Conversation info" not in result
    assert "message_id" not in result
    assert "session_id" not in result
    assert "Useful message content." in result
    assert "More content after block." in result


def test_strip_untrusted_metadata_clean_passthrough() -> None:
    """Content without metadata blocks passes through unchanged."""
    content = "Clean message with no metadata blocks. Just plain text."
    assert om_compressor.strip_untrusted_metadata(content) == content


def test_strip_untrusted_metadata_empty_string_safe() -> None:
    """Empty string returns empty string (no exception)."""
    assert om_compressor.strip_untrusted_metadata("") == ""


def test_strip_untrusted_metadata_none_passthrough() -> None:
    """None-equivalent (falsy) content returns the original value without crash."""
    # The function guards `if not content: return content`
    # so passing None would need the caller to coerce; test empty str only.
    assert om_compressor.strip_untrusted_metadata("") == ""


def test_strip_untrusted_metadata_all_four_prefixes() -> None:
    """All four untrusted-metadata prefixes are stripped."""
    prefixes = [
        "Conversation info:",
        "Sender (untrusted metadata):",
        "Replied message (untrusted, for context):",
        "Conversation info (untrusted metadata):",
    ]
    for prefix in prefixes:
        content = f"{prefix}\n```json\n{{\"key\": \"value\"}}\n```\nKeep this."
        result = om_compressor.strip_untrusted_metadata(content)
        assert prefix not in result, f"Prefix {prefix!r} was not stripped"
        assert "Keep this." in result, f"Non-metadata content was lost for prefix {prefix!r}"


def test_strip_untrusted_metadata_multiblock() -> None:
    """Multiple metadata blocks in one message are all stripped."""
    content = (
        "Lead text.\n\n"
        "Conversation info:\n```json\n{\"a\": 1}\n```\n"
        "Middle text.\n\n"
        "Sender (untrusted metadata):\n```json\n{\"b\": 2}\n```\n"
        "Tail text."
    )
    result = om_compressor.strip_untrusted_metadata(content)
    assert "Conversation info" not in result
    assert "Sender (untrusted metadata)" not in result
    assert "Lead text." in result
    assert "Middle text." in result
    assert "Tail text." in result


def test_strip_untrusted_metadata_strips_function_exists_on_module() -> None:
    """strip_untrusted_metadata is exported from om_compressor module."""
    assert callable(om_compressor.strip_untrusted_metadata)


# ---------------------------------------------------------------------------
# Extractor-path verification: OM_EXTRACTOR_PATH event
# ---------------------------------------------------------------------------


def _make_messages(n: int = 1) -> list:
    return [
        om_compressor.MessageRow(
            message_id=f"msg{i}",
            source_session_id="sess1",
            content=f"test content number {i}",
            created_at="2026-02-28T00:00:00Z",
            content_embedding=[],
            om_extract_attempts=0,
        )
        for i in range(n)
    ]


def _make_cfg(model_id: str = "gpt-5.1-codex-mini") -> om_compressor.ExtractorConfig:
    return om_compressor.ExtractorConfig(
        schema_version="v1",
        prompt_template="OM_PROMPT_TEMPLATE_V1",
        model_id=model_id,
        extractor_version="abc123",
    )


def test_extractor_path_event_emitted(capsys: pytest.CaptureFixture) -> None:
    """OM_EXTRACTOR_PATH event is emitted by _extract_items with required fields."""
    om_compressor._extract_items(_make_messages(), _make_cfg())

    captured = capsys.readouterr()
    events = [json.loads(ln) for ln in captured.out.splitlines() if ln.strip() and '"event"' in ln]
    path_events = [e for e in events if e.get("event") == "OM_EXTRACTOR_PATH"]

    assert len(path_events) >= 1, "Expected at least one OM_EXTRACTOR_PATH event"
    evt = path_events[0]
    assert "extractor_mode" in evt, "extractor_mode field missing"
    assert evt["extractor_mode"] in ("model", "fallback"), f"Invalid extractor_mode: {evt['extractor_mode']!r}"
    assert "model_id" in evt, "model_id field missing"
    assert evt["model_id"] == "gpt-5.1-codex-mini"


def test_extractor_path_event_fallback_has_reason(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Fallback path emits OM_EXTRACTOR_PATH with a non-empty reason field."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OM_EXTRACTOR_API_KEY", raising=False)

    om_compressor._extract_items(_make_messages(), _make_cfg())

    captured = capsys.readouterr()
    events = [json.loads(ln) for ln in captured.out.splitlines() if ln.strip() and '"event"' in ln]
    path_events = [e for e in events if e.get("event") == "OM_EXTRACTOR_PATH"]

    assert len(path_events) >= 1
    evt = path_events[0]
    assert evt["extractor_mode"] == "fallback"
    assert "reason" in evt, "reason field missing from fallback event"
    assert evt["reason"], "reason must be non-empty"


def test_extractor_path_event_model_id_propagated(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """model_id in OM_EXTRACTOR_PATH matches the ExtractorConfig.model_id."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OM_EXTRACTOR_API_KEY", raising=False)

    custom_model = "my-custom-model-xyz"
    om_compressor._extract_items(_make_messages(), _make_cfg(model_id=custom_model))

    captured = capsys.readouterr()
    events = [json.loads(ln) for ln in captured.out.splitlines() if ln.strip() and '"event"' in ln]
    path_events = [e for e in events if e.get("event") == "OM_EXTRACTOR_PATH"]

    assert len(path_events) >= 1
    assert path_events[0]["model_id"] == custom_model


def test_is_model_client_available_false_when_no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """_is_model_client_available returns False when no API keys are set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OM_EXTRACTOR_API_KEY", raising=False)
    assert om_compressor._is_model_client_available() is False


def test_is_model_client_available_true_with_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """_is_model_client_available returns True when OPENAI_API_KEY is set."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    assert om_compressor._is_model_client_available() is True


def test_is_model_client_available_true_with_om_extractor_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """_is_model_client_available returns True when OM_EXTRACTOR_API_KEY is set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OM_EXTRACTOR_API_KEY", "sk-om-test-key")
    assert om_compressor._is_model_client_available() is True


# ---------------------------------------------------------------------------


def test_relation_type_validation_and_interpolation_guard() -> None:
    assert om_compressor._validated_relation_type_for_cypher("motivates") == "MOTIVATES"

    edge = om_compressor.ExtractionEdge(
        source_node_id="n1",
        target_node_id="n2",
        relation_type="MOTIVATES",
    )
    assert om_compressor._assert_relation_type_safe_for_interpolation("MOTIVATES", edge) == "MOTIVATES"

    with pytest.raises(om_compressor.OMCompressorError):
        om_compressor._validated_relation_type_for_cypher("MOTIVATES]->(x) DETACH DELETE x //")

    with pytest.raises(om_compressor.OMCompressorError):
        om_compressor._assert_relation_type_safe_for_interpolation("BAD-TOKEN", edge)
