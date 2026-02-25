from __future__ import annotations

import importlib

ontology_registry = importlib.import_module("mcp_server.src.services.ontology_registry")


def test_dynamic_entity_model_has_explicit_module() -> None:
    built = ontology_registry._build_entity_types(
        [{"name": "OperationalRule", "description": "Rule description"}]
    )

    model = built["OperationalRule"]
    assert model.__module__ == ontology_registry.__name__
    assert model.__doc__ == "Rule description"
