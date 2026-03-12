"""Packs router — pack CRUD + registry stubs.

Owned by: Exec 3 (list/get/describe/create packs).
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from ._phase0 import (
        error_response,
        phase0_empty_list_response,
        phase0_not_implemented,
        require_optional_dict,
        require_optional_non_empty_string,
        require_pack_id,
        validate_schema_object,
    )
except ImportError:  # pragma: no cover - script/top-level import fallback
    from _phase0 import (  # type: ignore[no-redef]
        error_response,
        phase0_empty_list_response,
        phase0_not_implemented,
        require_optional_dict,
        require_optional_non_empty_string,
        require_pack_id,
        validate_schema_object,
    )

logger = logging.getLogger(__name__)


def register_tools(mcp: Any) -> dict[str, Any]:
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
        filter_error = require_optional_dict('filter', filter)
        if filter_error is not None:
            return filter_error

        logger.debug('list_packs Phase 0 stub called with filter=%r', filter)
        return phase0_empty_list_response('list_packs', 'packs')

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
        pack_id_error = require_pack_id('pack_id', pack_id)
        if pack_id_error is not None:
            return pack_id_error

        task_error = require_optional_non_empty_string('task', task)
        if task_error is not None:
            return task_error

        logger.debug('get_context_pack Phase 0 stub validated pack_id=%r', pack_id)
        return phase0_not_implemented('get_context_pack')

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
        pack_id_error = require_pack_id('pack_id', pack_id)
        if pack_id_error is not None:
            return pack_id_error

        task_error = require_optional_non_empty_string('task', task)
        if task_error is not None:
            return task_error

        logger.debug('get_workflow_pack Phase 0 stub validated pack_id=%r', pack_id)
        return phase0_not_implemented('get_workflow_pack')

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
        pack_id_error = require_pack_id('pack_id', pack_id)
        if pack_id_error is not None:
            return pack_id_error

        logger.debug('describe_pack Phase 0 stub validated pack_id=%r', pack_id)
        return phase0_not_implemented('describe_pack')

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
        definition_error = validate_schema_object('definition', definition, 'PackDefinition')
        if definition_error is not None:
            return definition_error

        if definition.get('scope') not in {'workflow', 'both'}:
            return error_response(
                'validation_error',
                message='definition.scope must be workflow or both for create_workflow_pack',
                details={'field': 'definition.scope', 'actual': definition.get('scope')},
            )

        logger.debug('create_workflow_pack Phase 0 stub validated definition successfully')
        return phase0_not_implemented('create_workflow_pack')

    return {
        'list_packs': list_packs,
        'get_context_pack': get_context_pack,
        'get_workflow_pack': get_workflow_pack,
        'describe_pack': describe_pack,
        'create_workflow_pack': create_workflow_pack,
    }
