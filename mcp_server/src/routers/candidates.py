"""Candidates router — candidate lifecycle stubs.

Owned by: Exec 4 (list/promote/reject candidates).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def register_tools(mcp: Any) -> None:
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
        # Stub: Exec 4 implements full candidate retrieval
        logger.debug('list_candidates stub called with status=%r', status)
        return {
            'status': 'stub',
            'message': 'list_candidates not yet implemented (Phase 0 stub)',
            'candidates': [],
        }

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
        # Stub: Exec 4 implements full promotion logic
        logger.debug('promote_candidate stub called for candidate_id=%r', candidate_id)
        return {
            'status': 'stub',
            'message': 'promote_candidate not yet implemented (Phase 0 stub)',
            'candidate_id': candidate_id,
        }

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
        # Stub: Exec 4 implements full rejection logic
        logger.debug('reject_candidate stub called for candidate_id=%r', candidate_id)
        return {
            'status': 'stub',
            'message': 'reject_candidate not yet implemented (Phase 0 stub)',
            'candidate_id': candidate_id,
        }
