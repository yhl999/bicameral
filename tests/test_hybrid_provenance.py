"""Tests for hybrid provenance materialization (build_provenance).

Covers:
- Graph-only results produce correct refs with source='graph'
- Typed-state results produce refs with source_id from _original
- Typed-procedure results produce refs with source='typed_procedure'
- Mixed results produce one ref per merged item
- Evidence from typed_results is resolved into resolved_evidence
- Inline evidence_refs are resolved
- Empty inputs produce empty provenance
- Scalar evidence_refs (string URIs) are handled
- Missing _original gracefully falls back
"""
from __future__ import annotations

import pathlib
import sys
from typing import Any

_svc_path = str(
    pathlib.Path(__file__).parent.parent / "mcp_server" / "src" / "services"
)
if _svc_path not in sys.path:
    sys.path.insert(0, _svc_path)

from mcp_server.src.services.typed_retrieval_service import build_provenance


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _graph_item(uuid: str = "g1", group_id: str = "s1_sessions_main") -> dict[str, Any]:
    return {
        "uuid": uuid,
        "fact": "user likes coffee",
        "_source": "graph",
        "_hybrid_score": 0.016,
        "group_id": group_id,
    }


def _typed_state_item(
    object_id: str = "sf_01",
    source_lane: str = "s1_sessions_main",
) -> dict[str, Any]:
    return {
        "uuid": f"typed:{object_id}",
        "fact": "user likes coffee",
        "_source": "typed_state",
        "_hybrid_score": 0.014,
        "_object_type": "state_fact",
        "_original": {
            "object_id": object_id,
            "object_type": "state_fact",
            "subject": "user",
            "predicate": "likes",
            "value": "coffee",
            "source_lane": source_lane,
            "evidence_refs": [],
        },
    }


def _typed_procedure_item(object_id: str = "pr_01") -> dict[str, Any]:
    return {
        "uuid": f"typed:{object_id}",
        "fact": "reset password procedure",
        "_source": "typed_procedure",
        "_hybrid_score": 0.012,
        "_object_type": "procedure",
        "_original": {
            "object_id": object_id,
            "object_type": "procedure",
            "name": "reset password",
            "trigger": "user forgets password",
            "source_lane": "s1_sessions_main",
        },
    }


def _typed_results_with_evidence(
    parent_object_id: str = "sf_01",
) -> dict[str, Any]:
    return {
        "state": [{"object_id": parent_object_id, "subject": "user", "predicate": "likes", "value": "coffee"}],
        "procedures": [],
        "evidence": [
            {
                "parent_object_id": parent_object_id,
                "kind": "session_extract",
                "source_system": "discord",
                "canonical_uri": "session://2026-03-21/msg-42",
                "snippet": "Yuan said he likes coffee",
                "observed_at": "2026-03-21T12:00:00Z",
            },
            {
                "parent_object_id": parent_object_id,
                "kind": "chatgpt_import",
                "source_system": "chatgpt",
                "canonical_uri": "chatgpt://conv-abc/msg-7",
                "snippet": "user preference: coffee",
                "observed_at": "2026-02-15T08:00:00Z",
            },
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# §1  Empty inputs
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_merged_results():
    """Empty merged_results → empty refs and resolved_evidence."""
    prov = build_provenance([], None)
    assert prov == {"refs": [], "resolved_evidence": {}}


def test_empty_merged_with_typed_results():
    """Empty merged_results with non-empty typed_results → still empty."""
    typed = {"state": [{"object_id": "sf_01"}], "evidence": []}
    prov = build_provenance([], typed)
    assert prov["refs"] == []
    assert prov["resolved_evidence"] == {}


# ─────────────────────────────────────────────────────────────────────────────
# §2  Graph-only results
# ─────────────────────────────────────────────────────────────────────────────


def test_graph_item_produces_correct_ref():
    """Graph item → ref with source='graph', source_id=uuid, object_type='entity_edge'."""
    prov = build_provenance([_graph_item("g1", "s1_sessions_main")])
    assert len(prov["refs"]) == 1
    ref = prov["refs"][0]
    assert ref["source"] == "graph"
    assert ref["source_id"] == "g1"
    assert ref["object_type"] == "entity_edge"
    assert ref["lane"] == "s1_sessions_main"
    assert ref["evidence_keys"] == []


def test_multiple_graph_items():
    """Multiple graph items → one ref per item."""
    items = [_graph_item(f"g{i}") for i in range(5)]
    prov = build_provenance(items)
    assert len(prov["refs"]) == 5
    assert all(r["source"] == "graph" for r in prov["refs"])
    assert prov["resolved_evidence"] == {}


# ─────────────────────────────────────────────────────────────────────────────
# §3  Typed-state results
# ─────────────────────────────────────────────────────────────────────────────


def test_typed_state_item_produces_correct_ref():
    """Typed state item → ref with source='typed_state', source_id from _original."""
    prov = build_provenance([_typed_state_item("sf_42")])
    assert len(prov["refs"]) == 1
    ref = prov["refs"][0]
    assert ref["source"] == "typed_state"
    assert ref["source_id"] == "sf_42"
    assert ref["object_type"] == "state_fact"
    assert ref["lane"] == "s1_sessions_main"


def test_typed_state_with_evidence_resolves():
    """Typed state item with evidence in typed_results → evidence resolved."""
    item = _typed_state_item("sf_01")
    typed_results = _typed_results_with_evidence("sf_01")

    prov = build_provenance([item], typed_results)
    ref = prov["refs"][0]
    assert len(ref["evidence_keys"]) == 2

    # Check resolved evidence
    for key in ref["evidence_keys"]:
        assert key in prov["resolved_evidence"]
        ev = prov["resolved_evidence"][key]
        assert ev["status"] == "resolved"
        assert ev["kind"] in ("session_extract", "chatgpt_import")


def test_typed_state_evidence_has_correct_structure():
    """Resolved evidence entries have all expected fields."""
    item = _typed_state_item("sf_01")
    typed_results = _typed_results_with_evidence("sf_01")

    prov = build_provenance([item], typed_results)
    ev_key = prov["refs"][0]["evidence_keys"][0]
    ev = prov["resolved_evidence"][ev_key]

    assert "kind" in ev
    assert "source_system" in ev
    assert "canonical_uri" in ev
    assert "snippet" in ev
    assert "observed_at" in ev
    assert ev["status"] == "resolved"


# ─────────────────────────────────────────────────────────────────────────────
# §4  Typed-procedure results
# ─────────────────────────────────────────────────────────────────────────────


def test_typed_procedure_produces_correct_ref():
    """Typed procedure item → ref with source='typed_procedure'."""
    prov = build_provenance([_typed_procedure_item("pr_01")])
    ref = prov["refs"][0]
    assert ref["source"] == "typed_procedure"
    assert ref["source_id"] == "pr_01"
    assert ref["object_type"] == "procedure"
    assert ref["lane"] == "s1_sessions_main"


# ─────────────────────────────────────────────────────────────────────────────
# §5  Mixed results
# ─────────────────────────────────────────────────────────────────────────────


def test_mixed_results_one_ref_per_item():
    """Mixed graph + typed → one ref per merged item, in order."""
    items = [
        _graph_item("g1"),
        _typed_state_item("sf_01"),
        _graph_item("g2"),
        _typed_procedure_item("pr_01"),
    ]
    prov = build_provenance(items)
    assert len(prov["refs"]) == 4
    assert prov["refs"][0]["source"] == "graph"
    assert prov["refs"][1]["source"] == "typed_state"
    assert prov["refs"][2]["source"] == "graph"
    assert prov["refs"][3]["source"] == "typed_procedure"


# ─────────────────────────────────────────────────────────────────────────────
# §6  Inline evidence_refs
# ─────────────────────────────────────────────────────────────────────────────


def test_inline_evidence_refs_resolved():
    """Inline evidence_refs in _original are resolved into provenance."""
    item = _typed_state_item("sf_01")
    item["_original"]["evidence_refs"] = [
        {
            "kind": "manual_annotation",
            "source_system": "human",
            "canonical_uri": "manual://note-1",
            "title": "Yuan mentioned this in a call",
        },
    ]
    prov = build_provenance([item])
    ref = prov["refs"][0]
    assert len(ref["evidence_keys"]) == 1
    ev_key = ref["evidence_keys"][0]
    ev = prov["resolved_evidence"][ev_key]
    assert ev["kind"] == "manual_annotation"
    assert ev["status"] == "resolved"


def test_scalar_evidence_ref():
    """String evidence_ref (URI) is resolved into a minimal provenance entry."""
    item = _typed_state_item("sf_01")
    item["_original"]["evidence_refs"] = ["session://2026-03-21/msg-99"]

    prov = build_provenance([item])
    ref = prov["refs"][0]
    assert len(ref["evidence_keys"]) == 1
    ev = prov["resolved_evidence"][ref["evidence_keys"][0]]
    assert ev["kind"] == "ref"
    assert ev["canonical_uri"] == "session://2026-03-21/msg-99"
    assert ev["status"] == "resolved"


# ─────────────────────────────────────────────────────────────────────────────
# §7  Missing _original fallback
# ─────────────────────────────────────────────────────────────────────────────


def test_typed_item_without_original_falls_back():
    """Typed item missing _original still produces a ref with available data."""
    item = {
        "uuid": "sf_01",
        "_source": "typed_state",
        "_object_type": "state_fact",
        "_hybrid_score": 0.01,
    }
    prov = build_provenance([item])
    ref = prov["refs"][0]
    assert ref["source"] == "typed_state"
    assert ref["source_id"] == "sf_01"  # falls back to uuid
    assert ref["object_type"] == "state_fact"  # from _object_type


def test_typed_item_with_empty_original():
    """Typed item with empty _original dict still works."""
    item = {
        "uuid": "x",
        "_source": "typed_state",
        "_hybrid_score": 0.01,
        "_original": {},
    }
    prov = build_provenance([item])
    ref = prov["refs"][0]
    assert ref["source"] == "typed_state"
    assert ref["source_id"] == ""  # no object_id, no uuid in original
    assert ref["evidence_keys"] == []  # no object_id → no evidence lookup


# ─────────────────────────────────────────────────────────────────────────────
# §8  Evidence key uniqueness
# ─────────────────────────────────────────────────────────────────────────────


def test_evidence_keys_are_unique_across_items():
    """Evidence keys for different items use distinct prefixes."""
    item_a = _typed_state_item("sf_01")
    item_b = _typed_state_item("sf_02")
    typed_results = {
        "state": [],
        "procedures": [],
        "evidence": [
            {"parent_object_id": "sf_01", "kind": "ev_a"},
            {"parent_object_id": "sf_02", "kind": "ev_b"},
        ],
    }
    prov = build_provenance([item_a, item_b], typed_results)
    all_keys = []
    for ref in prov["refs"]:
        all_keys.extend(ref["evidence_keys"])
    assert len(all_keys) == len(set(all_keys)), "Evidence keys must be unique"


# ─────────────────────────────────────────────────────────────────────────────
# §9  Snippet truncation
# ─────────────────────────────────────────────────────────────────────────────


def test_evidence_snippet_truncated():
    """Long evidence snippets are truncated to 500 chars."""
    item = _typed_state_item("sf_01")
    typed_results = {
        "state": [],
        "procedures": [],
        "evidence": [
            {
                "parent_object_id": "sf_01",
                "kind": "long",
                "snippet": "x" * 1000,
            },
        ],
    }
    prov = build_provenance([item], typed_results)
    ev_key = prov["refs"][0]["evidence_keys"][0]
    assert len(prov["resolved_evidence"][ev_key]["snippet"]) <= 500


# ─────────────────────────────────────────────────────────────────────────────
# §10  No typed_results parameter
# ─────────────────────────────────────────────────────────────────────────────


def test_graph_items_no_typed_results():
    """Graph items without typed_results → no evidence, clean provenance."""
    prov = build_provenance([_graph_item()], None)
    assert len(prov["refs"]) == 1
    assert prov["resolved_evidence"] == {}


def test_typed_items_no_typed_results():
    """Typed items without typed_results → refs produced, no evidence resolved."""
    prov = build_provenance([_typed_state_item()], None)
    ref = prov["refs"][0]
    assert ref["source"] == "typed_state"
    assert ref["evidence_keys"] == []  # no evidence to resolve
