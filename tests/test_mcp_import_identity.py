from __future__ import annotations

import importlib
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from tests.helpers_mcp_import import load_graphiti_mcp_server

_MISSING = object()


def test_load_graphiti_mcp_server_uses_canonical_package_modules() -> None:
    load_graphiti_mcp_server()

    ontology_registry = importlib.import_module('mcp_server.src.services.ontology_registry')
    typed_memory = importlib.import_module('mcp_server.src.models.typed_memory')
    typed_retrieval = importlib.import_module('mcp_server.src.services.typed_retrieval')

    assert sys.modules.get('services.ontology_registry') in (None, ontology_registry)
    assert sys.modules.get('models.typed_memory') in (None, typed_memory)
    assert typed_retrieval.StateFact is typed_memory.StateFact
    assert typed_retrieval.EvidenceRef is typed_memory.EvidenceRef


def test_episodes_procedures_test_module_import_restores_stub_modules() -> None:
    module_path = Path(__file__).resolve().parents[1] / 'mcp_server' / 'tests' / 'test_episodes_procedures.py'
    watched_modules = [
        'mcp',
        'mcp.server',
        'mcp.server.fastmcp',
        'mcp.server.auth.middleware.auth_context',
        'services.factories',
    ]
    originals = {name: sys.modules.get(name, _MISSING) for name in watched_modules}

    module_name = 'tests_import_hygiene_test_episodes_procedures'
    spec = spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)

    for name, original in originals.items():
        if original is _MISSING:
            assert name not in sys.modules
        else:
            assert sys.modules.get(name) is original
