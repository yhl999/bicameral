"""Episodes and procedures router — episodic/procedural memory stubs.

Owned by: Exec 5 (episode/procedure retrieval).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def register_tools(mcp: Any) -> None:
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
        # Stub: Exec 5 implements full episode search
        logger.debug('search_episodes stub called with query=%r', query[:80] if query else '')
        return {
            'status': 'stub',
            'message': 'search_episodes not yet implemented (Phase 0 stub)',
            'query': query,
            'episodes': [],
        }

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
        # Stub: Exec 5 implements full episode retrieval
        logger.debug('get_episode stub called for episode_id=%r', episode_id)
        return {
            'status': 'stub',
            'message': 'get_episode not yet implemented (Phase 0 stub)',
            'episode_id': episode_id,
        }

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
        # Stub: Exec 5 implements full procedure search
        logger.debug('search_procedures stub called with query=%r', query[:80] if query else '')
        return {
            'status': 'stub',
            'message': 'search_procedures not yet implemented (Phase 0 stub)',
            'query': query,
            'procedures': [],
        }

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
        # Stub: Exec 5 implements full procedure retrieval
        logger.debug('get_procedure stub called for trigger_or_id=%r', trigger_or_id)
        return {
            'status': 'stub',
            'message': 'get_procedure not yet implemented (Phase 0 stub)',
            'trigger_or_id': trigger_or_id,
        }
