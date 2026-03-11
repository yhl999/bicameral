"""Schema validation layer for Bicameral MCP typed objects."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Path to schema directory
_SCHEMAS_DIR = Path(__file__).parent.parent.parent / 'schemas'

# In-memory schema registry — loaded at import time
SCHEMA_REGISTRY: dict[str, dict[str, Any]] = {}


def _load_schemas() -> None:
    """Load all JSON schemas from the schemas directory into the registry."""
    if not _SCHEMAS_DIR.exists():
        logger.warning(f'Schemas directory not found: {_SCHEMAS_DIR}')
        return
    for schema_file in _SCHEMAS_DIR.glob('*.json'):
        try:
            schema = json.loads(schema_file.read_text(encoding='utf-8'))
            schema_id = schema_file.stem  # filename without .json
            SCHEMA_REGISTRY[schema_id] = schema
            logger.debug(f'Loaded schema: {schema_id}')
        except Exception as exc:
            logger.error(f'Failed to load schema {schema_file}: {exc}')


# Load schemas at module import time
_load_schemas()


def _validate_typed_object(
    obj: dict[str, Any],
    obj_type: str,
    strict: bool = True,
) -> tuple[bool, str | None]:
    """Validate a typed object against its JSON schema.

    Args:
        obj: The object dict to validate.
        obj_type: Schema name (e.g. "Preference", "Commitment", "TypedFact").
        strict: If True, fail on unknown fields that aren't in additionalProperties.

    Returns:
        (True, None) on success, (False, error_message) on failure.
    """
    schema = SCHEMA_REGISTRY.get(obj_type)
    if schema is None:
        return False, f'Unknown schema type: {obj_type!r}'

    if not isinstance(obj, dict):
        return False, f'Expected a dict, got {type(obj).__name__}'

    required_fields: list[str] = schema.get('required', [])
    properties: dict[str, Any] = schema.get('properties', {})

    # Check required fields
    for field in required_fields:
        if field not in obj or obj[field] is None or obj[field] == '':
            return False, f'Missing required field: {field!r}'

    # Type-check known properties
    for field, field_schema in properties.items():
        if field not in obj:
            continue
        value = obj[field]
        expected_type = field_schema.get('type')
        if expected_type and value is not None:
            ok, err = _check_type(field, value, field_schema)
            if not ok:
                return False, err

    # Enum validation for top-level fields
    for field, field_schema in properties.items():
        if field not in obj:
            continue
        if 'enum' in field_schema and obj[field] not in field_schema['enum']:
            return False, (
                f'Invalid value for {field!r}: {obj[field]!r}. '
                f'Must be one of: {field_schema["enum"]}'
            )
        if 'const' in field_schema and obj[field] != field_schema['const']:
            return False, (
                f'Field {field!r} must equal {field_schema["const"]!r}, '
                f'got {obj[field]!r}'
            )

    # String constraint validation
    for field, field_schema in properties.items():
        if field not in obj or not isinstance(obj[field], str):
            continue
        min_len = field_schema.get('minLength')
        max_len = field_schema.get('maxLength')
        if min_len is not None and len(obj[field]) < min_len:
            return False, f'Field {field!r} too short (min {min_len})'
        if max_len is not None and len(obj[field]) > max_len:
            return False, f'Field {field!r} too long (max {max_len})'

    # Array minItems validation
    for field, field_schema in properties.items():
        if field not in obj or not isinstance(obj[field], list):
            continue
        min_items = field_schema.get('minItems')
        if min_items is not None and len(obj[field]) < min_items:
            return False, f'Field {field!r} requires at least {min_items} item(s)'

    return True, None


def _check_type(field: str, value: Any, field_schema: dict[str, Any]) -> tuple[bool, str | None]:
    """Check that value matches the expected JSON Schema type."""
    expected = field_schema.get('type')
    if expected == 'string' and not isinstance(value, str):
        return False, f'Field {field!r} must be a string, got {type(value).__name__}'
    if expected == 'integer' and not isinstance(value, int):
        return False, f'Field {field!r} must be an integer, got {type(value).__name__}'
    if expected == 'number' and not isinstance(value, (int, float)):
        return False, f'Field {field!r} must be a number, got {type(value).__name__}'
    if expected == 'boolean' and not isinstance(value, bool):
        return False, f'Field {field!r} must be a boolean, got {type(value).__name__}'
    if expected == 'array' and not isinstance(value, list):
        return False, f'Field {field!r} must be an array, got {type(value).__name__}'
    if expected == 'object' and not isinstance(value, dict):
        return False, f'Field {field!r} must be an object, got {type(value).__name__}'
    return True, None


def detect_conflict(
    new_fact: dict[str, Any],
    existing_facts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return a conflict descriptor if new_fact contradicts existing_facts.

    This is a stub. Exec 1 implements full conflict detection logic.
    """
    return None
