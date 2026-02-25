from __future__ import annotations

import importlib
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
