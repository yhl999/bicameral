import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MCP_SERVER = ROOT / 'mcp_server' / 'src' / 'graphiti_mcp_server.py'


class AddMemoryContractSourceTests(unittest.TestCase):
    def test_add_memory_uses_fallback_entity_types_kwarg(self):
        src = MCP_SERVER.read_text()
        self.assertIn('fallback_entity_types=graphiti_service.entity_types', src)

    def test_ontology_registry_is_initialized(self):
        src = MCP_SERVER.read_text()
        self.assertIn('self.ontology_registry = None', src)
        self.assertIn('OntologyRegistry.load(', src)
        self.assertIn('overlay_paths=overlay_paths', src)

    def test_configured_overlay_failures_raise_in_initialize(self):
        src = MCP_SERVER.read_text()
        self.assertIn('Configured ontology overlay load failed', src)
        self.assertIn('require base ontology file', src)
        self.assertIn('logger.exception(', src)


if __name__ == '__main__':
    unittest.main()
