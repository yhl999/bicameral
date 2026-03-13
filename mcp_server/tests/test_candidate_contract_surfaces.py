from __future__ import annotations

import json
from pathlib import Path

from mcp_server.src.routers import candidates as candidates_router


def _schema_path() -> Path:
    return Path(__file__).parent.parent / 'schemas' / 'Candidate.json'


def _graphiti_source_path() -> Path:
    return Path(__file__).parent.parent / 'src' / 'graphiti_mcp_server.py'


def _load_candidate_schema() -> dict:
    return json.loads(_schema_path().read_text(encoding='utf-8'))


def _load_graphiti_source() -> str:
    return _graphiti_source_path().read_text(encoding='utf-8')


def test_candidate_schema_tracks_runtime_status_and_shape():
    schema = _load_candidate_schema()

    assert schema['required'][:5] == ['uuid', 'type', 'subject', 'predicate', 'value']
    assert 'candidate_id' not in schema['properties']
    assert 'fact_type' not in schema['properties']
    assert set(schema['properties']['status']['enum']) == set(candidates_router.VALID_STATUSES)
    assert schema['properties']['status']['default'] == 'quarantine'


def test_candidate_tool_metadata_matches_runtime_contract():
    source = _load_graphiti_source()

    assert "name='list_candidates'" in source
    assert '"quarantine" | "promoted" | "rejected" | null' in source
    assert 'pending' not in source[source.index("name='list_candidates'"):source.index("name='promote_candidate'")]
    assert "examples=[{'status': 'quarantine'" in source
    assert "output='list[Candidate]'" in source

    promote_block = source[source.index("name='promote_candidate'"):source.index("name='reject_candidate'")]
    assert '"supersede" | "parallel" | "cancel"' in promote_block
    assert "examples=[{'candidate_id': 'cand-001', 'resolution': 'supersede'}]" in promote_block

    reject_block = source[source.index("name='reject_candidate'"):source.index("name='list_packs'")]
    assert 'pending queue' not in reject_block
    assert "examples=[{'candidate_id': 'cand-002'}]" in reject_block
