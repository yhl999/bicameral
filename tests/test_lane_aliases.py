import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MCP_SERVER = ROOT / 'mcp_server' / 'src' / 'graphiti_mcp_server.py'
SCHEMA = ROOT / 'mcp_server' / 'src' / 'config' / 'schema.py'
CONFIG_YAML = ROOT / 'mcp_server' / 'config' / 'config.yaml'


class LaneAliasContractSourceTests(unittest.TestCase):
    def test_search_tools_accept_lane_alias_and_search_mode(self):
        src = MCP_SERVER.read_text()
        self.assertIn('async def search_nodes(', src)
        self.assertIn('lane_alias: list[str] | None = None', src)
        self.assertIn("search_mode: str = 'hybrid'", src)
        self.assertIn('async def search_memory_facts(', src)

    def test_group_resolution_precedence_is_present(self):
        src = MCP_SERVER.read_text()
        self.assertIn('def _resolve_effective_group_ids(', src)
        self.assertIn('if group_ids:', src)
        self.assertIn('if lane_alias is not None:', src)
        self.assertIn('if config.graphiti.group_id:', src)

    def test_schema_and_config_declare_lane_aliases(self):
        schema = SCHEMA.read_text()
        cfg = CONFIG_YAML.read_text()
        self.assertIn('lane_aliases: dict[str, list[str]]', schema)
        self.assertIn('lane_aliases:', cfg)
        self.assertIn('sessions_main:', cfg)


if __name__ == '__main__':
    unittest.main()
