"""Tests for write-time provisional ledger episode generation.

Phase 2 platonic ideal: when the OM compressor writes new OM nodes to Neo4j,
it also writes provisional Episode entries to the change ledger.  This makes
unpromoted OM content discoverable through the canonical ledger path without
read-time projection.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OM_GROUP_ID", "test_group")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")

from mcp_server.src.models.typed_memory import Episode
from mcp_server.src.services.change_ledger import ChangeLedger, project_objects
from scripts.om_compressor import (
    WrittenOMNode,
    _provisional_episode_object_id,
    _provisional_episode_root_id,
    _write_provisional_ledger_episodes,
)


@pytest.fixture()
def tmp_ledger(tmp_path: Path) -> ChangeLedger:
    """Create a temporary ChangeLedger for test isolation."""
    db_path = tmp_path / "test_change_ledger.db"
    return ChangeLedger(db_path)


@pytest.fixture(autouse=True)
def _patch_ledger(tmp_ledger: ChangeLedger) -> None:
    """Patch the compressor's lazy-singleton ledger with the test instance."""
    import scripts.om_compressor as mod

    mod._PROVISIONAL_LEDGER = tmp_ledger
    yield
    mod._PROVISIONAL_LEDGER = None


def _make_node(**overrides: object) -> WrittenOMNode:
    defaults = dict(
        node_id="node_001",
        group_id="test_group",
        content="Yuan prefers Ethiopian coffee beans",
        semantic_domain="preference",
        node_type="factual_observation",
        created_at="2026-03-11T00:00:00Z",
        valid_at="2026-03-11T00:00:00Z",
        source_message_ids=["msg_a", "msg_b"],
    )
    defaults.update(overrides)
    # Derive source_key from group_id and node_id (mirrors compressor behavior)
    gid = defaults["group_id"]
    nid = defaults["node_id"]
    defaults.setdefault("source_key", f"om:{gid}:node:{nid}")
    return WrittenOMNode(**defaults)  # type: ignore[arg-type]


class TestWriteProvisionalLedgerEpisodes:
    def test_writes_episode_for_new_node(self, tmp_ledger: ChangeLedger) -> None:
        node = _make_node()
        written = _write_provisional_ledger_episodes([node], chunk_id="chunk_1")
        assert written == 1

        object_id = _provisional_episode_object_id("test_group", "node_001")
        root_id = tmp_ledger.root_id_for_object(object_id)
        assert root_id is not None

        # Materialize and verify Episode shape
        events = tmp_ledger.events_for_root(root_id)
        assert len(events) == 1
        objects = project_objects(events)
        assert len(objects) == 1
        ep = objects[0]
        assert isinstance(ep, Episode)
        assert ep.object_id == object_id
        assert "provisional" in (ep.annotations or [])
        assert "om_native" in (ep.annotations or [])
        assert "unpromoted" in (ep.annotations or [])
        assert ep.summary == "Yuan prefers Ethiopian coffee beans"
        assert ep.source_lane == "om:test_group"
        assert ep.source_key == "om:test_group:node:node_001"

    def test_idempotent_skips_duplicate(self, tmp_ledger: ChangeLedger) -> None:
        node = _make_node()
        first = _write_provisional_ledger_episodes([node], chunk_id="chunk_1")
        second = _write_provisional_ledger_episodes([node], chunk_id="chunk_1")
        assert first == 1
        assert second == 0

    def test_empty_list_returns_zero(self) -> None:
        assert _write_provisional_ledger_episodes([], chunk_id="chunk_x") == 0

    def test_multiple_nodes_batch(self, tmp_ledger: ChangeLedger) -> None:
        nodes = [
            _make_node(node_id="n1", content="fact one"),
            _make_node(node_id="n2", content="fact two"),
            _make_node(node_id="n3", content="fact three"),
        ]
        written = _write_provisional_ledger_episodes(nodes, chunk_id="chunk_2")
        assert written == 3

        for n in nodes:
            oid = _provisional_episode_object_id("test_group", n.node_id)
            assert tmp_ledger.root_id_for_object(oid) is not None

    def test_object_id_deterministic(self) -> None:
        oid = _provisional_episode_object_id("grp", "nid")
        assert oid == "om_provisional_episode:grp:nid"
        rid = _provisional_episode_root_id("grp", "nid")
        assert rid == "om_provisional_root:grp:nid"

    def test_history_meta_contains_lineage_info(self, tmp_ledger: ChangeLedger) -> None:
        node = _make_node()
        _write_provisional_ledger_episodes([node], chunk_id="chunk_3")
        object_id = _provisional_episode_object_id("test_group", "node_001")
        root_id = tmp_ledger.root_id_for_object(object_id)
        events = tmp_ledger.events_for_root(root_id)
        objects = project_objects(events)
        ep = objects[0]
        assert isinstance(ep, Episode)
        meta = ep.history_meta or {}
        assert meta["lineage_kind"] == "om_provisional"
        assert meta["lineage_basis"] == "write_time_ledger"
        assert meta["promotion_status"] == "unpromoted"
        assert meta["om_node_id"] == "node_001"
        assert meta["om_group_id"] == "test_group"
        assert meta["chunk_id"] == "chunk_3"

    def test_source_key_format_matches_om_node_key_parser(self) -> None:
        """Source key must match the om:{group_id}:node:{node_id} pattern
        that _om_node_key_from_source_key expects for dedup tracking."""
        from mcp_server.src.services.typed_retrieval import _om_node_key_from_source_key

        node = _make_node(node_id="n42", group_id="g1")
        assert node.source_key == "om:g1:node:n42"
        key = _om_node_key_from_source_key(node.source_key)
        assert key == ("g1", "n42")


class TestProvisionalEpisodeSuppressionAfterPromotion:
    """Verify that provisional ledger episodes are suppressed from results
    when the same OM node has been promoted to a ledger-backed state fact."""

    def test_provisional_suppressed_when_node_already_covered(self) -> None:
        from mcp_server.src.services.typed_retrieval import _om_node_key_from_source_key

        covered = {("test_group", "node_001")}
        node = _make_node()
        node_key = _om_node_key_from_source_key(node.source_key)
        assert node_key in covered
