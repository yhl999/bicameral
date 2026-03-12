"""Episodes and procedures router — episodic/procedural memory stubs.

Owned by: Exec 5 (episode/procedure retrieval).
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from ._phase0 import (
        phase0_empty_list_response,
        phase0_not_implemented,
        require_identifier,
        require_non_empty_string,
        validate_time_range,
    )
except ImportError:  # pragma: no cover - script/top-level import fallback
    from _phase0 import (  # type: ignore[no-redef]
        phase0_empty_list_response,
        phase0_not_implemented,
        require_identifier,
        require_non_empty_string,
        validate_time_range,
    )

logger = logging.getLogger(__name__)


def register_tools(mcp: Any) -> dict[str, Any]:
    """Register all episode and procedure router tools with the MCP server instance."""

    @mcp.tool()
    async def search_episodes(
        query: str,
        time_range: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Search episodic memory by semantic query and optional time range.

        Episodes are discrete narrative chunks representing past events,
        conversations, or experiences. Use this to find relevant context
        from the agent's experience.

        Args:
            query: Natural-language query to search episodes.
            time_range: Optional time range filter dict with 'start' and/or
                        'end' ISO 8601 timestamps.

        Returns:
            Dict with 'episodes' key containing list of Episode dicts.
        """
        query_error = require_non_empty_string('query', query)
        if query_error is not None:
            return query_error

        time_range_error = validate_time_range(time_range)
        if time_range_error is not None:
            return time_range_error

        logger.debug('search_episodes Phase 0 stub called')
        return phase0_empty_list_response('search_episodes', 'episodes')

    @mcp.tool()
    async def get_episode(
        episode_id: str,
    ) -> dict[str, Any]:
        """Retrieve a specific episode by ID.

        Returns the full episode record including content, timestamps,
        evidence refs, and annotations.

        Args:
            episode_id: Unique ID of the episode to retrieve.

        Returns:
            Episode dict, or ErrorResponse dict if not found.
        """
        episode_id_error = require_identifier('episode_id', episode_id)
        if episode_id_error is not None:
            return episode_id_error

        logger.debug('get_episode Phase 0 stub validated episode_id=%r', episode_id)
        return phase0_not_implemented('get_episode')

    @mcp.tool()
    async def search_procedures(
        query: str,
    ) -> dict[str, Any]:
        """Search procedural memory for relevant procedures.

        Procedures are step-by-step instructions for accomplishing tasks,
        derived from past experience or explicitly defined. Use this to
        find how the agent has done things before.

        Args:
            query: Natural-language query describing the task or procedure.

        Returns:
            Dict with 'procedures' key containing list of Procedure dicts.
        """
        query_error = require_non_empty_string('query', query)
        if query_error is not None:
            return query_error

        logger.debug('search_procedures Phase 0 stub called')
        return phase0_empty_list_response('search_procedures', 'procedures')

    @mcp.tool()
    async def get_procedure(
        trigger_or_id: str,
    ) -> dict[str, Any]:
        """Retrieve a procedure by trigger phrase or ID.

        Looks up a procedure by its natural-language trigger (e.g. "how to
        deploy to production") or by its unique object ID.

        Args:
            trigger_or_id: Trigger phrase or unique procedure ID.

        Returns:
            Procedure dict, or ErrorResponse dict if not found.
        """
        trigger_error = require_non_empty_string('trigger_or_id', trigger_or_id)
        if trigger_error is not None:
            return trigger_error

        logger.debug('get_procedure Phase 0 stub validated trigger_or_id=%r', trigger_or_id)
        return phase0_not_implemented('get_procedure')

    return {
        'search_episodes': search_episodes,
        'get_episode': get_episode,
        'search_procedures': search_procedures,
        'get_procedure': get_procedure,
    }
