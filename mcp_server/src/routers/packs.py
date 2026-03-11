"""Packs router — pack CRUD + registry stubs.

Owned by: Exec 3 (list/get/describe/create packs).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def register_tools(mcp: Any) -> None:
    """Register all pack router tools with the MCP server instance."""

    @mcp.tool()
    async def list_packs(
        filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """List available context and workflow packs.

        Packs bundle curated knowledge (context packs) or step-by-step
        procedures (workflow packs) for injection into agent prompts.

        Args:
            filter: Optional filter dict (e.g. {"scope": "private", "intent": "coding"}).

        Returns:
            Dict with 'packs' key containing list of PackRegistry dicts.
        """
        # Stub: Exec 3 implements full pack registry retrieval
        logger.debug('list_packs stub called with filter=%r', filter)
        return {
            'status': 'stub',
            'message': 'list_packs not yet implemented (Phase 0 stub)',
            'packs': [],
        }

    @mcp.tool()
    async def get_context_pack(
        pack_id: str,
        task: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve a materialized context pack for prompt injection.

        Returns the pack's curated facts and knowledge, optionally filtered
        for relevance to a specific task description.

        Args:
            pack_id: ID of the context pack to retrieve.
            task: Optional task description to filter pack content for relevance.

        Returns:
            PackMaterialized dict with 'items' key containing relevant facts,
            or ErrorResponse dict on failure.
        """
        # Stub: Exec 3 implements full pack materialization
        logger.debug('get_context_pack stub called for pack_id=%r', pack_id)
        return {
            'status': 'stub',
            'message': 'get_context_pack not yet implemented (Phase 0 stub)',
            'pack_id': pack_id,
            'items': [],
        }

    @mcp.tool()
    async def get_workflow_pack(
        pack_id: str,
        task: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve a materialized workflow pack with step-by-step procedures.

        Returns the pack's procedures and instructions, optionally filtered
        for relevance to a specific task.

        Args:
            pack_id: ID of the workflow pack to retrieve.
            task: Optional task description to filter pack content for relevance.

        Returns:
            PackMaterialized dict with 'steps' key containing procedure steps,
            or ErrorResponse dict on failure.
        """
        # Stub: Exec 3 implements full workflow pack materialization
        logger.debug('get_workflow_pack stub called for pack_id=%r', pack_id)
        return {
            'status': 'stub',
            'message': 'get_workflow_pack not yet implemented (Phase 0 stub)',
            'pack_id': pack_id,
            'steps': [],
        }

    @mcp.tool()
    async def describe_pack(
        pack_id: str,
    ) -> dict[str, Any]:
        """Get the full definition of a pack including schema and metadata.

        Returns the pack's full PackDefinition including scope, intent, consumer,
        version, and the rules/content that make up the pack.

        Args:
            pack_id: ID of the pack to describe.

        Returns:
            PackDefinition dict, or ErrorResponse dict if not found.
        """
        # Stub: Exec 3 implements full pack description retrieval
        logger.debug('describe_pack stub called for pack_id=%r', pack_id)
        return {
            'status': 'stub',
            'message': 'describe_pack not yet implemented (Phase 0 stub)',
            'pack_id': pack_id,
        }

    @mcp.tool()
    async def create_workflow_pack(
        definition: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a new workflow pack from a definition.

        Registers a new workflow pack in the pack registry. The definition
        must include scope, intent, consumer, version, and steps.

        Args:
            definition: PackDefinition dict describing the new workflow pack.

        Returns:
            PackRegistry dict with the newly created pack's metadata,
            or ErrorResponse dict on validation failure.
        """
        # Stub: Exec 3 implements full pack creation logic
        logger.debug('create_workflow_pack stub called')
        return {
            'status': 'stub',
            'message': 'create_workflow_pack not yet implemented (Phase 0 stub)',
        }
