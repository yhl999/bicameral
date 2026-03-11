"""Tests for scripts/import_graphiti_candidates generic extraction."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Mock external runtime dependencies before importing the script under test.
# graph_driver and truth.candidates require a live Neo4j / SQLite runtime;
# we only need to exercise the pure Python logic here.
sys.modules.setdefault("graph_driver", MagicMock())
_mock_candidates = MagicMock()
sys.modules.setdefault("truth.candidates", _mock_candidates)
sys.modules.setdefault("truth.entity_type_routing", MagicMock())

# Ensure repo root is on sys.path for the script's own import resolution.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.import_graphiti_candidates import load_anchors_from_config, Anchor  # noqa: E402


def test_load_anchors_from_config_basic(tmp_path: Path) -> None:
    cfg = tmp_path / "anchors.json"
    cfg.write_text(json.dumps({
        "anchors": [
            {"canonical_name": "Alice", "subject_id": "user:alice", "aliases": ["Alice Smith"]},
            {"canonical_name": "Bot", "subject_id": "agent:bot", "aliases": []},
        ]
    }))
    anchors = load_anchors_from_config(str(cfg))
    assert len(anchors) == 2
    assert anchors[0].canonical_name == "Alice"
    assert anchors[0].subject_id == "user:alice"
    assert "Alice Smith" in anchors[0].aliases


def test_load_anchors_no_config_returns_empty() -> None:
    anchors = load_anchors_from_config(None)
    assert anchors == []


def test_load_anchors_empty_anchors_key(tmp_path: Path) -> None:
    cfg = tmp_path / "anchors.json"
    cfg.write_text(json.dumps({"anchors": []}))
    anchors = load_anchors_from_config(str(cfg))
    assert anchors == []
