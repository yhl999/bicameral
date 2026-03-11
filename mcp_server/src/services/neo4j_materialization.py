"""Neo4j materialization callback for typed state-fact writes.

Exec 1 intentionally keeps this fail-open and side-effect light:
- If materialization succeeds, return metadata payload (e.g. edge_uuid)
- If unavailable/unconfigured, return None
- Callers should never fail the ledger write due to Neo4j callback failure
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

try:
    from ..models.typed_memory import StateFact
except ImportError:  # pragma: no cover - top-level import fallback
    from models.typed_memory import StateFact  # type: ignore[no-redef]


MaterializationWriter = Callable[[StateFact, str | None], Awaitable[dict[str, Any] | None]]


async def _default_writer(_fact: StateFact, _supersedes_object_id: str | None) -> dict[str, Any] | None:
    """Default no-op writer.

    Real Neo4j dual-write wiring is optional and can be injected by runtime.
    """

    return None


class Neo4jMaterializationService:
    """Thin injectable callback wrapper for typed-fact -> Neo4j dual-write."""

    def __init__(self, writer: MaterializationWriter | None = None):
        self._writer = writer or _default_writer

    async def materialize_state_fact(
        self,
        fact: StateFact,
        *,
        supersedes_object_id: str | None = None,
    ) -> dict[str, Any] | None:
        return await self._writer(fact, supersedes_object_id)
