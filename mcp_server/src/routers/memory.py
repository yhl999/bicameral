"""Memory router — remember_fact, get_current_state, get_history stubs.

Owned by: Exec 1 (remember_fact) and Exec 2 (get_current_state, get_history).
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from ._phase0 import (
        phase0_empty_list_response,
        phase0_not_implemented,
        require_non_empty_string,
        require_optional_dict,
        require_optional_non_empty_string,
    )
except ImportError:  # pragma: no cover - script/top-level import fallback
    from _phase0 import (  # type: ignore[no-redef]
        phase0_empty_list_response,
        phase0_not_implemented,
        require_non_empty_string,
        require_optional_dict,
        require_optional_non_empty_string,
    )

logger = logging.getLogger(__name__)


def register_tools(mcp: Any) -> dict[str, Any]:
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
            Phase 0 validates the stub inputs and returns ErrorResponse(
            error="not_implemented") until Exec 1 lands.
        """
        text_error = require_non_empty_string('text', text)
        if text_error is not None:
            return text_error

        hint_error = require_optional_dict('hint', hint)
        if hint_error is not None:
            return hint_error

        logger.debug('remember_fact Phase 0 stub validated input successfully')
        return phase0_not_implemented('remember_fact')

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
        subject_error = require_non_empty_string('subject', subject)
        if subject_error is not None:
            return subject_error

        predicate_error = require_optional_non_empty_string('predicate', predicate)
        if predicate_error is not None:
            return predicate_error

        logger.debug('get_current_state Phase 0 stub called for subject=%r', subject)
        return phase0_empty_list_response('get_current_state', 'facts')

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
        subject_error = require_non_empty_string('subject', subject)
        if subject_error is not None:
            return subject_error

        predicate_error = require_optional_non_empty_string('predicate', predicate)
        if predicate_error is not None:
            return predicate_error

        logger.debug('get_history Phase 0 stub called for subject=%r', subject)
        return phase0_empty_list_response('get_history', 'history')

    return {
        'remember_fact': remember_fact,
        'get_current_state': get_current_state,
        'get_history': get_history,
    }
