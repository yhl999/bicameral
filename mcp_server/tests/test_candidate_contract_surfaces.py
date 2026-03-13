from __future__ import annotations

import json
from pathlib import Path

from mcp_server.src.routers import candidates as candidates_router


def _schema_path() -> Path:
    return Path(__file__).parent.parent / 'schemas' / 'Candidate.json'


def _graphiti_source_path() -> Path:
    # In integration, candidate tool contracts live in the candidates router
    # (graphiti_mcp_server.py uses _ROUTER_TOOL_CONTRACTS delegation pattern)
    return Path(__file__).parent.parent / 'src' / 'routers' / 'candidates.py'


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

    # Integration uses dict-form TOOL_CONTRACTS in candidates.py router
    # (not keyword-arg _tool_schema_entry() form from graphiti_mcp_server.py)
    assert "'name': 'list_candidates'" in source
    assert 'quarantine' in source
    assert 'pending' not in source[source.index("'name': 'list_candidates'"):source.index("'name': 'promote_candidate'")]
    # examples are in dict form; check via list_block below
    # list_candidates output now includes the full {'status': 'ok', 'candidates': list[Candidate], ...} envelope.
    assert '"candidates": list[Candidate]' in source

    # Verify examples are present (dict form used in router TOOL_CONTRACTS)
    list_block = source[source.index("'name': 'list_candidates'"):source.index("'name': 'promote_candidate'")]
    assert "'status': 'quarantine'" in list_block

    promote_block = source[source.index("'name': 'promote_candidate'"):source.index("'name': 'reject_candidate'")]
    assert 'supersede' in promote_block
    assert "'resolution': 'supersede'" in promote_block

    reject_block = source[source.index("'name': 'reject_candidate'"):source.index("'name': 'reject_candidate'") + 500]
    assert 'pending queue' not in reject_block
    assert "'candidate_id': 'cand-002'" in reject_block
