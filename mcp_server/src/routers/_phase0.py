"""Shared helpers for Bicameral MCP Surface Phase 0 router stubs."""

from __future__ import annotations

import re
from typing import Any

try:
    from ..models.response_types import ErrorResponse, SuccessResponse
    from ..services.schema_validation import _validate_typed_object, parse_date_time_string
except ImportError:  # pragma: no cover - script/top-level import fallback
    from models.response_types import ErrorResponse, SuccessResponse
    from services.schema_validation import _validate_typed_object, parse_date_time_string


_PHASE0_EMPTY_RESULT_MESSAGE = (
    '{method_name} is a Phase 0 stub; returning an empty result until the implementation lands.'
)
_PHASE0_NOT_IMPLEMENTED_MESSAGE = (
    '{method_name} is a Phase 0 stub and is not implemented yet.'
)
_PACK_ID_RE = re.compile(r'^[a-z0-9_-]{1,128}$')
_PHASE0_IDENTIFIER_RE = re.compile(r'^[a-z0-9][a-z0-9:_-]{0,255}$')


def error_response(
    error: str,
    *,
    message: str | None = None,
    details: Any | None = None,
) -> ErrorResponse:
    response: ErrorResponse = {'error': error}
    if message is not None:
        response['message'] = message
    if details is not None:
        response['details'] = details
    return response


def success_response(message: str, *, details: Any | None = None) -> SuccessResponse:
    response: SuccessResponse = {'message': message}
    if details is not None:
        response['details'] = details
    return response


def phase0_empty_list_response(method_name: str, list_key: str) -> dict[str, Any]:
    return {
        'message': _PHASE0_EMPTY_RESULT_MESSAGE.format(method_name=method_name),
        list_key: [],
    }


def phase0_paginated_list_response(
    method_name: str,
    list_key: str,
    *,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    return {
        'message': _PHASE0_EMPTY_RESULT_MESSAGE.format(method_name=method_name),
        list_key: [],
        'limit': limit,
        'offset': offset,
        'total': 0,
        'has_more': False,
        'next_offset': None,
    }


def phase0_not_implemented(method_name: str) -> ErrorResponse:
    return error_response(
        'not_implemented',
        message=_PHASE0_NOT_IMPLEMENTED_MESSAGE.format(method_name=method_name),
    )


def require_string(field_name: str, value: Any) -> ErrorResponse | None:
    if not isinstance(value, str):
        return error_response(
            'validation_error',
            message=f'{field_name} must be a string',
            details={'field': field_name, 'expected_type': 'string', 'actual_type': type(value).__name__},
        )
    return None


def require_non_empty_string(field_name: str, value: Any) -> ErrorResponse | None:
    string_error = require_string(field_name, value)
    if string_error is not None:
        return string_error
    if value.strip() == '':
        return error_response(
            'validation_error',
            message=f'{field_name} must be a non-empty string',
            details={'field': field_name},
        )
    return None


def require_patterned_string(
    field_name: str,
    value: Any,
    *,
    pattern: re.Pattern[str],
    pattern_description: str,
) -> ErrorResponse | None:
    string_error = require_non_empty_string(field_name, value)
    if string_error is not None:
        return string_error
    if not pattern.fullmatch(value.strip()):
        return error_response(
            'validation_error',
            message=f'{field_name} must match {pattern_description}',
            details={
                'field': field_name,
                'pattern': pattern.pattern,
                'actual': value,
            },
        )
    return None


def require_pack_id(field_name: str, value: Any) -> ErrorResponse | None:
    return require_patterned_string(
        field_name,
        value,
        pattern=_PACK_ID_RE,
        pattern_description='1-128 lowercase letters, numbers, underscores, or hyphens',
    )


def require_identifier(field_name: str, value: Any) -> ErrorResponse | None:
    return require_patterned_string(
        field_name,
        value,
        pattern=_PHASE0_IDENTIFIER_RE,
        pattern_description='lowercase letters, numbers, colons, underscores, or hyphens',
    )


def require_optional_non_empty_string(field_name: str, value: Any) -> ErrorResponse | None:
    if value is None:
        return None
    return require_non_empty_string(field_name, value)


def require_boolean(field_name: str, value: Any) -> ErrorResponse | None:
    if not isinstance(value, bool):
        return error_response(
            'validation_error',
            message=f'{field_name} must be a boolean',
            details={'field': field_name, 'expected_type': 'boolean', 'actual_type': type(value).__name__},
        )
    return None


def require_optional_string_list(field_name: str, value: Any) -> ErrorResponse | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return error_response(
            'validation_error',
            message=f'{field_name} must be a list of strings when provided',
            details={'field': field_name, 'expected_type': 'list[string]', 'actual_type': type(value).__name__},
        )
    for index, item in enumerate(value):
        item_error = require_non_empty_string(f'{field_name}[{index}]', item)
        if item_error is not None:
            return item_error
    return None


def validate_pagination(
    *,
    limit: Any,
    offset: Any,
    default_limit: int = 10,
    default_offset: int = 0,
) -> tuple[int, int, ErrorResponse | None]:
    if limit is None:
        limit_value = default_limit
    elif isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        return 0, 0, error_response(
            'validation_error',
            message='limit must be a positive integer when provided',
            details={'field': 'limit', 'actual': limit},
        )
    else:
        limit_value = limit

    if offset is None:
        offset_value = default_offset
    elif isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        return 0, 0, error_response(
            'validation_error',
            message='offset must be a non-negative integer when provided',
            details={'field': 'offset', 'actual': offset},
        )
    else:
        offset_value = offset

    return limit_value, offset_value, None


def require_optional_dict(field_name: str, value: Any) -> ErrorResponse | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        return error_response(
            'validation_error',
            message=f'{field_name} must be an object/dict when provided',
            details={'field': field_name, 'expected_type': 'object', 'actual_type': type(value).__name__},
        )
    return None


def require_dict(field_name: str, value: Any) -> ErrorResponse | None:
    if not isinstance(value, dict):
        return error_response(
            'validation_error',
            message=f'{field_name} must be an object/dict',
            details={'field': field_name, 'expected_type': 'object', 'actual_type': type(value).__name__},
        )
    return None


def require_enum(field_name: str, value: Any, allowed: set[str]) -> ErrorResponse | None:
    string_error = require_optional_non_empty_string(field_name, value)
    if string_error is not None:
        return string_error
    if value is None:
        return None
    normalized = value.strip()
    if normalized not in allowed:
        return error_response(
            'validation_error',
            message=f'{field_name} must be one of: {sorted(allowed)}',
            details={'field': field_name, 'allowed': sorted(allowed), 'actual': value},
        )
    return None


def validate_time_range(time_range: Any) -> ErrorResponse | None:
    dict_error = require_optional_dict('time_range', time_range)
    if dict_error is not None or time_range is None:
        return dict_error

    unexpected_keys = sorted(set(time_range) - {'start', 'end'})
    if unexpected_keys:
        return error_response(
            'validation_error',
            message='time_range only supports start/end keys',
            details={'field': 'time_range', 'unexpected_keys': unexpected_keys},
        )

    parsed_bounds: dict[str, Any] = {}
    for key in ('start', 'end'):
        if key in time_range:
            value_error = require_non_empty_string(f'time_range.{key}', time_range[key])
            if value_error is not None:
                return value_error

            parsed, parse_error = parse_date_time_string(
                time_range[key],
                field_name=f'time_range.{key}',
            )
            if parse_error is not None:
                return error_response(
                    'validation_error',
                    message=parse_error,
                    details={'field': f'time_range.{key}'},
                )
            parsed_bounds[key] = parsed

    if 'start' in parsed_bounds and 'end' in parsed_bounds and parsed_bounds['start'] > parsed_bounds['end']:
        return error_response(
            'validation_error',
            message='time_range.start must be before or equal to time_range.end',
            details={'field': 'time_range'},
        )
    return None


def validate_schema_object(field_name: str, value: Any, schema_type: str) -> ErrorResponse | None:
    dict_error = require_dict(field_name, value)
    if dict_error is not None:
        return dict_error

    is_valid, error_message = _validate_typed_object(value, schema_type)
    if is_valid:
        return None
    return error_response(
        'validation_error',
        message=error_message or f'{field_name} failed {schema_type} validation',
        details={'field': field_name, 'schema_type': schema_type},
    )
