from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_SRC = ROOT / 'mcp_server' / 'src' / 'graphiti_mcp_server.py'
CONFIG_YAML = ROOT / 'mcp_server' / 'config' / 'config.yaml'


def test_lane_alias_resolution_function_is_explicit_and_ordered():
    src = SERVER_SRC.read_text(encoding='utf-8')

    assert 'def _resolve_effective_group_ids(' in src
    assert 'if group_ids is not None:' in src
    assert 'elif lane_alias is not None:' in src
    assert 'alias_map = config.graphiti.lane_aliases or {}' in src
    assert "elif config.graphiti.group_id:" in src


def test_lane_alias_config_maps_observational_memory_group_id():
    cfg = CONFIG_YAML.read_text(encoding='utf-8')
    assert 'lane_aliases:' in cfg
    assert 'observational_memory: ["s1_observational_memory"]' in cfg
