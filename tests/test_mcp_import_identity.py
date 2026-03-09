from __future__ import annotations

import importlib
import sys

from tests.helpers_mcp_import import load_graphiti_mcp_server


def test_load_graphiti_mcp_server_uses_canonical_package_modules() -> None:
    load_graphiti_mcp_server()

    ontology_registry = importlib.import_module('mcp_server.src.services.ontology_registry')
    typed_memory = importlib.import_module('mcp_server.src.models.typed_memory')
    typed_retrieval = importlib.import_module('mcp_server.src.services.typed_retrieval')

    assert sys.modules.get('services.ontology_registry') in (None, ontology_registry)
    assert sys.modules.get('models.typed_memory') in (None, typed_memory)
    assert typed_retrieval.StateFact is typed_memory.StateFact
    assert typed_retrieval.EvidenceRef is typed_memory.EvidenceRef
