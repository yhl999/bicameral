"""Candidates router — candidate lifecycle stubs.

Owned by: Exec 4 (list/promote/reject candidates).
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from ._phase0 import (
        phase0_empty_list_response,
        phase0_not_implemented,
        require_enum,
        require_identifier,
        require_non_empty_string,
    )
except ImportError:  # pragma: no cover - script/top-level import fallback
    from _phase0 import (  # type: ignore[no-redef]
        phase0_empty_list_response,
        phase0_not_implemented,
        require_enum,
        require_identifier,
        require_non_empty_string,
    )

logger = logging.getLogger(__name__)

_ALLOWED_CANDIDATE_STATUS = {'pending', 'promoted', 'rejected'}

TOOL_CONTRACTS: list[dict[str, Any]] = [
    {
        'name': 'list_candidates',
        'description': 'List quarantined fact candidates awaiting promotion review',
        'mode_hint': 'typed',
        'schema': {
            'inputs': {'status': '"pending" | "promoted" | "rejected" | null'},
            'output': '{"message": string, "candidates": list[Candidate]} | ErrorResponse',
        },
        'examples': [{'status': 'pending'}],
        'phase0_behavior': 'Returns an empty candidate list after input validation.',
    },
    {
        'name': 'promote_candidate',
        'description': 'Validate promotion input for a candidate fact; Exec 4 wires the ledger integration',
        'mode_hint': 'typed',
        'schema': {
            'inputs': {
                'candidate_id': 'string',
                'resolution': 'string',
            },
            'output': 'ErrorResponse(error="not_implemented") in Phase 0 after validation; future: SuccessResponse | ErrorResponse',
        },
        'examples': [{'candidate_id': 'cand-001', 'resolution': 'Verified correct'}],
        'phase0_behavior': 'Validates candidate_id/resolution and then returns not_implemented.',
    },
    {
        'name': 'reject_candidate',
        'description': 'Validate rejection input for a candidate fact; Exec 4 wires the ledger integration',
        'mode_hint': 'typed',
        'schema': {
            'inputs': {'candidate_id': 'string'},
            'output': 'ErrorResponse(error="not_implemented") in Phase 0 after validation; future: SuccessResponse | ErrorResponse',
        },
        'examples': [{'candidate_id': 'cand-002'}],
        'phase0_behavior': 'Validates candidate_id and then returns not_implemented.',
    },
]


def register_tools(mcp: Any) -> dict[str, Any]:
    """Register all candidate router tools with the MCP server instance."""

    @mcp.tool()
    async def list_candidates(
        status: str | None = None,
    ) -> dict[str, Any]:
        """List quarantined fact candidates awaiting promotion review.

        Candidates are facts that have been extracted but not yet confirmed
        as ground truth. They may conflict with existing facts or require
        human review before being promoted to the main ledger.

        Args:
            status: Optional filter by candidate status.
                    One of: "pending", "promoted", "rejected".
                    If omitted, returns all candidates.

        Returns:
            Dict with 'candidates' key containing list of Candidate dicts.
        """
        status_error = require_enum('status', status, _ALLOWED_CANDIDATE_STATUS)
        if status_error is not None:
            return status_error

        logger.debug('list_candidates Phase 0 stub called with status=%r', status)
        return phase0_empty_list_response('list_candidates', 'candidates')

    @mcp.tool()
    async def promote_candidate(
        candidate_id: str,
        resolution: str,
    ) -> dict[str, Any]:
        """Promote a candidate fact to the main memory ledger.

        Moves a candidate from quarantine to the active ledger after review.
        Records the resolution reason for audit trail.

        Args:
            candidate_id: ID of the candidate to promote.
            resolution: Human-readable reason for promotion.

        Returns:
            SuccessResponse dict on success, ErrorResponse dict on failure.
        """
        candidate_id_error = require_identifier('candidate_id', candidate_id)
        if candidate_id_error is not None:
            return candidate_id_error

        resolution_error = require_non_empty_string('resolution', resolution)
        if resolution_error is not None:
            return resolution_error

        logger.debug('promote_candidate Phase 0 stub validated candidate_id=%r', candidate_id)
        return phase0_not_implemented('promote_candidate')

    @mcp.tool()
    async def reject_candidate(
        candidate_id: str,
    ) -> dict[str, Any]:
        """Reject a candidate fact, removing it from quarantine.

        Marks the candidate as rejected and removes it from the pending queue.

        Args:
            candidate_id: ID of the candidate to reject.

        Returns:
            SuccessResponse dict on success, ErrorResponse dict on failure.
        """
        candidate_id_error = require_identifier('candidate_id', candidate_id)
        if candidate_id_error is not None:
            return candidate_id_error

        logger.debug('reject_candidate Phase 0 stub validated candidate_id=%r', candidate_id)
        return phase0_not_implemented('reject_candidate')

    return {
        'list_candidates': list_candidates,
        'promote_candidate': promote_candidate,
        'reject_candidate': reject_candidate,
    }
