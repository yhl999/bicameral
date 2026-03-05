from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_SRC = ROOT / 'mcp_server' / 'src' / 'graphiti_mcp_server.py'


def test_om_adapter_is_neo4j_gated_for_nodes_and_facts():
    src = SERVER_SRC.read_text(encoding='utf-8')

    # Both search_nodes and search_memory_facts must guard OM adapter usage.
    expected_guard = "provider_name == 'neo4j'"
    assert src.count(expected_guard) >= 2

    assert 'use_observational_adapter = (' in src
    assert 'and search_service.includes_observational_memory(effective_group_ids)' in src
