from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_SRC = ROOT / 'mcp_server' / 'src' / 'graphiti_mcp_server.py'
SEARCH_SERVICE_SRC = ROOT / 'mcp_server' / 'src' / 'services' / 'search_service.py'


def test_mixed_lane_path_fuses_graphiti_and_om_results():
    src = SERVER_SRC.read_text(encoding='utf-8')
    assert 'merged_nodes = _fuse_node_like_results(' in src
    assert 'supplemental=om_nodes' in src
    assert 'merged_facts = _fuse_node_like_results(' in src
    assert 'supplemental=om_facts' in src


def test_om_payload_includes_lane_provenance_fields():
    src = SEARCH_SERVICE_SRC.read_text(encoding='utf-8')
    assert "'group_id': str(row.get('group_id') or group_id)" in src
    assert "'attributes': {" in src
    assert "'source': 'om_primitive'" in src
