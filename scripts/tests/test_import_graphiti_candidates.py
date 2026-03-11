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

from scripts.import_graphiti_candidates import (  # noqa: E402
    Anchor,
    iter_anchor_aliases,
    load_anchors_from_config,
)


# ---------------------------------------------------------------------------
# iter_anchor_aliases — Issue #1 fix: canonical_name must be included
# ---------------------------------------------------------------------------


def test_iter_anchor_aliases_includes_canonical_name() -> None:
    """canonical_name must be yielded so graph nodes using that name are matched."""
    anchors = [Anchor(canonical_name="Yuan", subject_id="user:yuan", aliases=())]
    pairs = list(iter_anchor_aliases(anchors))
    names = [n for n, _ in pairs]
    assert "Yuan" in names, "canonical_name must appear in iter_anchor_aliases output"


def test_iter_anchor_aliases_includes_aliases_alongside_canonical() -> None:
    anchors = [
        Anchor(
            canonical_name="Yuan",
            subject_id="user:yuan",
            aliases=("Yuan Han", "Yuan Han Li"),
        )
    ]
    pairs = list(iter_anchor_aliases(anchors))
    names = [n for n, _ in pairs]
    assert set(names) == {"Yuan", "Yuan Han", "Yuan Han Li"}
    # All pairs must map to the same subject_id
    assert all(sid == "user:yuan" for _, sid in pairs)


def test_iter_anchor_aliases_canonical_only_no_aliases() -> None:
    """An anchor with no aliases must still yield one pair (canonical_name)."""
    anchors = [Anchor(canonical_name="Bot", subject_id="agent:bot", aliases=())]
    pairs = list(iter_anchor_aliases(anchors))
    assert pairs == [("Bot", "agent:bot")]


def test_iter_anchor_aliases_multiple_anchors() -> None:
    anchors = [
        Anchor(canonical_name="Alice", subject_id="user:alice", aliases=("Al",)),
        Anchor(canonical_name="Bob", subject_id="user:bob", aliases=()),
    ]
    pairs = list(iter_anchor_aliases(anchors))
    assert ("Alice", "user:alice") in pairs
    assert ("Al", "user:alice") in pairs
    assert ("Bob", "user:bob") in pairs


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
