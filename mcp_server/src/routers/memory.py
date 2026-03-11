"""Memory router — remember_fact, get_current_state, get_history stubs.

Owned by: Exec 1 (remember_fact) and Exec 2 (get_current_state, get_history).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def register_tools(mcp: Any) -> None:
    """Register all memory router tools with the MCP server instance."""

    @mcp.tool()
    async def remember_fact(
        text: str,
        hint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write a typed fact to the memory ledger.

        Parses natural-language or structured text into a typed StateFact,
        checks for conflicts with existing current facts, and persists to
        the ledger. Returns the created TypedFact or a ConflictDialog if
        a conflict is detected.

        Args:
            text: Natural-language or structured fact text to record.
            hint: Optional structured hint dict to guide fact extraction
                  (e.g. {"fact_type": "preference", "subject": "user"}).

        Returns:
            TypedFact dict on success, ConflictDialog dict if conflict detected,
            or ErrorResponse dict on validation failure.
        """
        # Stub: Exec 1 implements full ledger-first write logic
        logger.debug('remember_fact stub called with text=%r', text[:80] if text else '')
        return {
            'status': 'stub',
            'message': 'remember_fact not yet implemented (Phase 0 stub)',
            'text': text,
        }

    @mcp.tool()
    async def get_current_state(
        subject: str,
        predicate: str | None = None,
    ) -> dict[str, Any]:
        """Query the ledger for the current non-superseded fact(s).

        Returns the most recent, non-invalidated fact for the given subject
        and optional predicate. Uses the typed fact ledger for precise
        point-in-time queries.

        Args:
            subject: The entity to query (e.g. "user", "project-alpha").
            predicate: Optional relationship/attribute to filter by.
                       If omitted, returns all current facts for subject.

        Returns:
            Dict with 'facts' key containing list of current TypedFact dicts,
            or ErrorResponse dict on failure.
        """
        # Stub: Exec 2 implements full ledger query logic
        logger.debug('get_current_state stub called for subject=%r', subject)
        return {
            'status': 'stub',
            'message': 'get_current_state not yet implemented (Phase 0 stub)',
            'subject': subject,
            'predicate': predicate,
            'facts': [],
        }

    @mcp.tool()
    async def get_history(
        subject: str,
        predicate: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve the full change history for a subject/predicate.

        Returns all versioned facts for the given subject, ordered by
        valid_at timestamp (newest first). Includes superseded and
        invalidated facts to show the full temporal evolution.

        Args:
            subject: The entity to query history for.
            predicate: Optional relationship/attribute to filter by.

        Returns:
            Dict with 'history' key containing list of TypedFact dicts
            (all versions), or ErrorResponse dict on failure.
        """
        # Stub: Exec 2 implements full history retrieval logic
        logger.debug('get_history stub called for subject=%r', subject)
        return {
            'status': 'stub',
            'message': 'get_history not yet implemented (Phase 0 stub)',
            'subject': subject,
            'predicate': predicate,
            'history': [],
        }
