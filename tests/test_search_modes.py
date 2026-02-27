import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MCP_SERVER = ROOT / 'mcp_server' / 'src' / 'graphiti_mcp_server.py'


class SearchModeContractSourceTests(unittest.TestCase):
    def test_valid_search_modes_are_declared(self):
        src = MCP_SERVER.read_text()
        self.assertIn("VALID_SEARCH_MODES = {'hybrid', 'semantic', 'keyword'}", src)

    def test_node_mode_builder_has_semantic_and_keyword_paths(self):
        src = MCP_SERVER.read_text()
        self.assertIn('def _build_node_search_config(search_mode: str, max_nodes: int):', src)
        self.assertIn("if search_mode == 'hybrid':", src)
        self.assertIn("elif search_mode == 'semantic':", src)
        self.assertIn("elif search_mode == 'keyword':", src)
        self.assertIn('NodeSearchMethod.cosine_similarity', src)
        self.assertIn('NodeSearchMethod.bm25', src)

    def test_edge_mode_builder_has_semantic_and_keyword_paths(self):
        src = MCP_SERVER.read_text()
        self.assertIn('def _build_edge_search_config(search_mode: str, max_facts: int, center_node_uuid: str | None):', src)
        self.assertIn('EdgeSearchMethod.cosine_similarity', src)
        self.assertIn('EdgeSearchMethod.bm25', src)
        self.assertIn("'trust_weight': TRUST_WEIGHT", src)


if __name__ == '__main__':
    unittest.main()
