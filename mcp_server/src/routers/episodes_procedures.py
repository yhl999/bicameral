"""Episodes and procedures router — episodic/procedural memory stubs.

Owned by: Exec 5 (episode/procedure retrieval).
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from ._phase0 import (
        phase0_not_implemented,
        phase0_paginated_list_response,
        require_boolean,
        require_identifier,
        require_non_empty_string,
        require_optional_string_list,
        require_string,
        validate_pagination,
        validate_time_range,
    )
except ImportError:  # pragma: no cover - script/top-level import fallback
    from _phase0 import (  # type: ignore[no-redef]
        phase0_not_implemented,
        phase0_paginated_list_response,
        require_boolean,
        require_identifier,
        require_non_empty_string,
        require_optional_string_list,
        require_string,
        validate_pagination,
        validate_time_range,
    )

logger = logging.getLogger(__name__)

TOOL_CONTRACTS: list[dict[str, Any]] = [
    {
        'name': 'search_episodes',
        'description': 'Search episodic memory by semantic query with optional time range, scope, and pagination controls',
        'mode_hint': 'typed',
        'schema': {
            'inputs': {
                'query': 'string',
                'time_range': 'object with optional start/end ISO timestamps | null',
                'include_history': 'boolean (default false)',
                'group_ids': 'list[string] | null',
                'lane_alias': 'list[string] | null',
                'limit': 'integer | null (default 10)',
                'offset': 'integer | null (default 0)',
            },
            'output': 'EpisodeSearchResponse | ErrorResponse',
        },
        'examples': [{
            'query': 'last deployment',
            'time_range': None,
            'include_history': False,
            'group_ids': None,
            'lane_alias': None,
            'limit': 10,
            'offset': 0,
        }],
        'phase0_behavior': 'Validates filters/scope/pagination and returns an empty paginated episode list.',
    },
    {
        'name': 'get_episode',
        'description': 'Validate lookup input for a specific episode within optional lane scope',
        'mode_hint': 'typed',
        'schema': {
            'inputs': {
                'episode_id': 'string',
                'group_ids': 'list[string] | null',
                'lane_alias': 'list[string] | null',
            },
            'output': 'ErrorResponse(error="not_implemented") in Phase 0 after validation; future: Episode | ErrorResponse',
        },
        'examples': [{'episode_id': 'ep-001', 'group_ids': None, 'lane_alias': None}],
        'phase0_behavior': 'Validates episode_id/scope and then returns not_implemented.',
    },
    {
        'name': 'search_procedures',
        'description': 'Search procedural memory with optional scope and pagination controls',
        'mode_hint': 'typed',
        'schema': {
            'inputs': {
                'query': 'string',
                'include_all': 'boolean (default false)',
                'group_ids': 'list[string] | null',
                'lane_alias': 'list[string] | null',
                'limit': 'integer | null (default 10)',
                'offset': 'integer | null (default 0)',
            },
            'output': 'ProcedureSearchResponse | ErrorResponse',
        },
        'examples': [{
            'query': 'how to run tests',
            'include_all': False,
            'group_ids': None,
            'lane_alias': None,
            'limit': 10,
            'offset': 0,
        }],
        'phase0_behavior': 'Validates filters/scope/pagination and returns an empty paginated procedure list.',
    },
    {
        'name': 'get_procedure',
        'description': 'Validate lookup input for a procedure by trigger or ID within optional lane scope',
        'mode_hint': 'typed',
        'schema': {
            'inputs': {
                'trigger_or_id': 'string',
                'group_ids': 'list[string] | null',
                'lane_alias': 'list[string] | null',
            },
            'output': 'ErrorResponse(error="not_implemented") in Phase 0 after validation; future: Procedure | ErrorResponse',
        },
        'examples': [{'trigger_or_id': 'deploy to production', 'group_ids': None, 'lane_alias': None}],
        'phase0_behavior': 'Validates trigger_or_id/scope and then returns not_implemented.',
    },
]


def register_tools(mcp: Any) -> dict[str, Any]:
    """Register all episode and procedure router tools with the MCP server instance."""

    @mcp.tool()
    async def search_episodes(
        query: str,
        time_range: dict[str, Any] | None = None,
        include_history: bool = False,
        group_ids: list[str] | None = None,
        lane_alias: list[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """Search episodic memory by semantic query and optional time range.

        Episodes are discrete narrative chunks representing past events,
        conversations, or experiences. Use this to find relevant context
        from the agent's experience.

        Args:
            query: Natural-language query to search episodes.
            time_range: Optional time range filter dict with 'start' and/or
                        'end' ISO 8601 timestamps.
            include_history: Whether to include historical versions when supported.
            group_ids: Optional explicit list of lane/group IDs.
            lane_alias: Optional lane aliases resolved by runtime config.
            limit: Optional page size (defaults to 10 when omitted).
            offset: Optional page offset (defaults to 0 when omitted).

        Returns:
            Dict with 'episodes' plus pagination metadata.
        """
        query_error = require_string('query', query)
        if query_error is not None:
            return query_error

        time_range_error = validate_time_range(time_range)
        if time_range_error is not None:
            return time_range_error

        include_history_error = require_boolean('include_history', include_history)
        if include_history_error is not None:
            return include_history_error

        group_ids_error = require_optional_string_list('group_ids', group_ids)
        if group_ids_error is not None:
            return group_ids_error

        lane_alias_error = require_optional_string_list('lane_alias', lane_alias)
        if lane_alias_error is not None:
            return lane_alias_error

        limit_value, offset_value, pagination_error = validate_pagination(limit=limit, offset=offset)
        if pagination_error is not None:
            return pagination_error

        logger.debug(
            'search_episodes Phase 0 stub called include_history=%r group_ids=%r lane_alias=%r limit=%r offset=%r',
            include_history,
            group_ids,
            lane_alias,
            limit_value,
            offset_value,
        )
        return phase0_paginated_list_response(
            'search_episodes',
            'episodes',
            limit=limit_value,
            offset=offset_value,
        )

    @mcp.tool()
    async def get_episode(
        episode_id: str,
        group_ids: list[str] | None = None,
        lane_alias: list[str] | None = None,
    ) -> dict[str, Any]:
        """Retrieve a specific episode by ID.

        Returns the full episode record including content, timestamps,
        evidence refs, and annotations.

        Args:
            episode_id: Unique ID of the episode to retrieve.
            group_ids: Optional explicit list of lane/group IDs.
            lane_alias: Optional lane aliases resolved by runtime config.

        Returns:
            Episode dict, or ErrorResponse dict if not found.
        """
        episode_id_error = require_identifier('episode_id', episode_id)
        if episode_id_error is not None:
            return episode_id_error

        group_ids_error = require_optional_string_list('group_ids', group_ids)
        if group_ids_error is not None:
            return group_ids_error

        lane_alias_error = require_optional_string_list('lane_alias', lane_alias)
        if lane_alias_error is not None:
            return lane_alias_error

        logger.debug(
            'get_episode Phase 0 stub validated episode_id=%r group_ids=%r lane_alias=%r',
            episode_id,
            group_ids,
            lane_alias,
        )
        return phase0_not_implemented('get_episode')

    @mcp.tool()
    async def search_procedures(
        query: str,
        include_all: bool = False,
        group_ids: list[str] | None = None,
        lane_alias: list[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """Search procedural memory for relevant procedures.

        Procedures are step-by-step instructions for accomplishing tasks,
        derived from past experience or explicitly defined. Use this to
        find how the agent has done things before.

        Args:
            query: Natural-language query describing the task or procedure.
            include_all: Whether to include broader matches when supported.
            group_ids: Optional explicit list of lane/group IDs.
            lane_alias: Optional lane aliases resolved by runtime config.
            limit: Optional page size (defaults to 10 when omitted).
            offset: Optional page offset (defaults to 0 when omitted).

        Returns:
            Dict with 'procedures' plus pagination metadata.
        """
        query_error = require_string('query', query)
        if query_error is not None:
            return query_error

        include_all_error = require_boolean('include_all', include_all)
        if include_all_error is not None:
            return include_all_error

        group_ids_error = require_optional_string_list('group_ids', group_ids)
        if group_ids_error is not None:
            return group_ids_error

        lane_alias_error = require_optional_string_list('lane_alias', lane_alias)
        if lane_alias_error is not None:
            return lane_alias_error

        limit_value, offset_value, pagination_error = validate_pagination(limit=limit, offset=offset)
        if pagination_error is not None:
            return pagination_error

        logger.debug(
            'search_procedures Phase 0 stub called include_all=%r group_ids=%r lane_alias=%r limit=%r offset=%r',
            include_all,
            group_ids,
            lane_alias,
            limit_value,
            offset_value,
        )
        return phase0_paginated_list_response(
            'search_procedures',
            'procedures',
            limit=limit_value,
            offset=offset_value,
        )

    @mcp.tool()
    async def get_procedure(
        trigger_or_id: str,
        group_ids: list[str] | None = None,
        lane_alias: list[str] | None = None,
    ) -> dict[str, Any]:
        """Retrieve a procedure by trigger phrase or ID.

        Looks up a procedure by its natural-language trigger (e.g. "how to
        deploy to production") or by its unique object ID.

        Args:
            trigger_or_id: Trigger phrase or unique procedure ID.
            group_ids: Optional explicit list of lane/group IDs.
            lane_alias: Optional lane aliases resolved by runtime config.

        Returns:
            Procedure dict, or ErrorResponse dict if not found.
        """
        trigger_error = require_non_empty_string('trigger_or_id', trigger_or_id)
        if trigger_error is not None:
            return trigger_error

        group_ids_error = require_optional_string_list('group_ids', group_ids)
        if group_ids_error is not None:
            return group_ids_error

        lane_alias_error = require_optional_string_list('lane_alias', lane_alias)
        if lane_alias_error is not None:
            return lane_alias_error

        logger.debug(
            'get_procedure Phase 0 stub validated trigger_or_id=%r group_ids=%r lane_alias=%r',
            trigger_or_id,
            group_ids,
            lane_alias,
        )
        return phase0_not_implemented('get_procedure')

    return {
        'search_episodes': search_episodes,
        'get_episode': get_episode,
        'search_procedures': search_procedures,
        'get_procedure': get_procedure,
    }
