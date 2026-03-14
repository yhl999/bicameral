"""Schema validation layer for Bicameral MCP typed objects."""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator, FormatChecker, ValidationError

logger = logging.getLogger(__name__)

# Path to schema directory
_SCHEMAS_DIR = Path(__file__).parent.parent.parent / 'schemas'
_FORMAT_CHECKER = FormatChecker()

# In-memory schema registry — loaded at import time
SCHEMA_REGISTRY: dict[str, dict[str, Any]] = {}
_VALIDATORS: dict[str, Draft7Validator] = {}


def _build_validator(schema: dict[str, Any], *, strict: bool) -> Draft7Validator:
    schema_to_validate = copy.deepcopy(schema)
    if not strict:
        schema_to_validate['additionalProperties'] = True
    return Draft7Validator(schema_to_validate, format_checker=_FORMAT_CHECKER)


def _load_schemas() -> None:
    """Load all JSON schemas from the schemas directory into the registry."""
    if not _SCHEMAS_DIR.exists():
        raise RuntimeError(f'Schemas directory not found: {_SCHEMAS_DIR}')

    loaded_schemas: dict[str, dict[str, Any]] = {}
    loaded_validators: dict[str, Draft7Validator] = {}

    for schema_file in sorted(_SCHEMAS_DIR.glob('*.json')):
        try:
            schema = json.loads(schema_file.read_text(encoding='utf-8'))
            Draft7Validator.check_schema(schema)
            schema_id = schema_file.stem  # filename without .json
            loaded_schemas[schema_id] = schema
            loaded_validators[schema_id] = _build_validator(schema, strict=True)
            logger.debug('Loaded schema: %s', schema_id)
        except Exception as exc:
            raise RuntimeError(f'Failed to load schema {schema_file}: {exc}') from exc

    SCHEMA_REGISTRY.clear()
    SCHEMA_REGISTRY.update(loaded_schemas)
    _VALIDATORS.clear()
    _VALIDATORS.update(loaded_validators)


# Load schemas at module import time
_load_schemas()


def _format_validation_error(obj_type: str, error: ValidationError) -> str:
    path = '.'.join(str(part) for part in error.absolute_path)
    location = path or '<root>'

    if error.validator == 'additionalProperties':
        return f'{obj_type} has unknown field(s): {error.message}'
    if error.validator == 'required':
        return f'{obj_type} missing required field: {error.message}'
    if error.validator == 'type':
        return f'{obj_type}.{location}: {error.message}'
    if error.validator in {
        'enum',
        'const',
        'format',
        'pattern',
        'minLength',
        'maxLength',
        'minItems',
        'maxItems',
        'maxProperties',
    }:
        return f'{obj_type}.{location}: {error.message}'
    return f'{obj_type}.{location}: {error.message}'


def parse_date_time_string(value: Any, *, field_name: str) -> tuple[datetime | None, str | None]:
    if not isinstance(value, str):
        return None, f'{field_name} must be a string'

    candidate = value[:-1] + '+00:00' if value.endswith('Z') else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None, f'{field_name}: {value!r} is not a valid date-time'

    if parsed.tzinfo is None:
        return None, f'{field_name}: date-time values must include a timezone offset'

    return parsed, None


def _validate_declared_formats(
    obj: dict[str, Any],
    obj_type: str,
    schema: dict[str, Any],
) -> tuple[bool, str | None]:
    properties = schema.get('properties', {})
    for field_name, field_schema in properties.items():
        if field_name not in obj:
            continue

        if field_schema.get('format') != 'date-time':
            continue

        value = obj[field_name]
        if not isinstance(value, str):
            continue

        _, error = parse_date_time_string(value, field_name=f'{obj_type}.{field_name}')
        if error is not None:
            return False, error

    return True, None


def _validate_typed_object(
    obj: dict[str, Any],
    obj_type: str,
    strict: bool = True,
) -> tuple[bool, str | None]:
    """Validate a typed object against its JSON schema.

    Args:
        obj: The object dict to validate.
        obj_type: Schema name (e.g. "Preference", "Commitment", "TypedFact").
        strict: If True, fail on unknown fields. If False, allow unknown top-level fields.

    Returns:
        (True, None) on success, (False, error_message) on failure.
    """
    schema = SCHEMA_REGISTRY.get(obj_type)
    if schema is None:
        return False, f'Unknown schema type: {obj_type!r}'

    if not isinstance(obj, dict):
        return False, f'Expected a dict, got {type(obj).__name__}'

    validator = _VALIDATORS.get(obj_type)
    if validator is None or not strict:
        validator = _build_validator(schema, strict=strict)

    errors = sorted(validator.iter_errors(obj), key=lambda err: (list(err.absolute_path), err.message))
    if errors:
        return False, _format_validation_error(obj_type, errors[0])

    formats_ok, format_error = _validate_declared_formats(obj, obj_type, schema)
    if not formats_ok:
        return False, format_error

    return True, None


def detect_conflict(
    new_fact: dict[str, Any],
    existing_facts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return a lightweight conflict descriptor if new_fact contradicts existing_facts.

    Conflict is detected when there is an active fact already present for the
    same `subject` + `predicate` + `scope` where the `value` differs.
    """

    if not isinstance(new_fact, dict):
        return {
            'conflict': False,
            'reason': 'new_fact_must_be_object',
        }

    new_subject = str(new_fact.get('subject') or '').strip().lower()
    new_predicate = str(new_fact.get('predicate') or '').strip().lower()
    new_value = new_fact.get('value')
    new_scope = str(new_fact.get('scope') or 'private').strip().lower()

    if not new_subject or not new_predicate:
        return {
            'conflict': False,
            'reason': 'missing_subject_or_predicate',
        }

    for existing in existing_facts or []:
        if not isinstance(existing, dict):
            continue

        existing_subject = str(existing.get('subject') or '').strip().lower()
        existing_predicate = str(existing.get('predicate') or '').strip().lower()
        if not existing_subject or not existing_predicate:
            continue

        existing_scope = str(existing.get('scope') or existing.get('policy_scope') or 'private').strip().lower()
        if (
            existing_subject != new_subject
            or existing_predicate != new_predicate
            or existing_scope != new_scope
        ):
            continue

        existing_value = existing.get('value')
        if _values_equivalent(existing_value, new_value):
            return {
                'conflict': False,
                'existing_fact': dict(existing),
                'reason': 'duplicate',
            }

        return {
            'conflict': True,
            'reason': 'contradictory_value',
            'existing_fact': dict(existing),
        }

    return None


def _values_equivalent(a: Any, b: Any) -> bool:
    """Compare two values for equality with normalized JSON repr fallback."""

    if a is b:
        return True

    if a is None or b is None:
        return a is b

    if type(a) is type(b):
        return a == b

    try:
        return _normalize_for_compare(a) == _normalize_for_compare(b)
    except Exception:
        return False


def _normalize_for_compare(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)):
        return str(value).strip().lower()
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), sort_keys=True, ensure_ascii=False)
    return str(value).strip().lower()
