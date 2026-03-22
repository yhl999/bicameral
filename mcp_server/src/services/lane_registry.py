"""Canonical lane identity registry.

Cleanly separates three orthogonal concerns that were previously conflated:

1. **Canonical semantic lane identity** (``source_lane`` on typed objects):
   ``s1_sessions_main``, ``s1_observational_memory``, ``learning_self_audit``, etc.

2. **Graph / corpus group_ids**: may be identical to canonical lane IDs for
   primary lanes, or may be versioned replay IDs, benchmark IDs, etc.

3. **Visibility / policy scope**: ``private``, ``public``, ``owner`` — these are
   access-control labels, NOT lane identity, and must never appear as
   ``source_lane``.

The registry derives its canonical lane set from the existing ``lane_aliases``
configuration (``GraphitiAppConfig.lane_aliases``), which already maps
human-readable aliases to canonical group_ids.  This module adds:

- Scope-not-lane validation (prevents ``'private'`` leaking into ``source_lane``)
- Resolution of ``effective_group_ids`` → canonical ``source_lane`` values for
  typed retrieval filtering

Design rationale: rather than inventing a separate registry file, we extend the
existing config surface so there is exactly one shared authority for lane
identity.  New lanes are added by adding a ``lane_aliases`` entry; everything
else follows automatically.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── Scope values that must NEVER be used as source_lane ──────────────────────
# These are visibility/policy scope labels, not lane identity.  Any write path
# that attempts to set source_lane to one of these values has a scope-as-lane
# leakage bug and must be corrected.
SCOPE_NOT_LANE: frozenset[str] = frozenset({
    'private',
    'public',
    'owner',
    'global',
    'all',
})


class LaneRegistry:
    """Shared authority for canonical lane identity.

    Constructed from the ``lane_aliases`` config dict (``alias → [group_id, ...]``).
    The union of all group_ids across alias values forms the canonical lane set.

    Usage::

        registry = LaneRegistry(config.graphiti.lane_aliases)
        # Validate at write time:
        safe_lane = registry.validate_source_lane(raw_value)
        # Resolve at read time:
        source_lanes = registry.resolve_typed_source_lanes(effective_group_ids)
    """

    __slots__ = ('_canonical_lanes', '_alias_to_lanes')

    def __init__(
        self,
        lane_aliases: dict[str, list[str]] | None = None,
        *,
        extra_canonical_lanes: frozenset[str] | None = None,
    ) -> None:
        self._alias_to_lanes: dict[str, list[str]] = dict(lane_aliases or {})

        # Derive canonical set from all group_ids referenced in lane_aliases.
        # The 'all' alias conventionally maps to [] (meaning "every lane"),
        # so its value contributes nothing to the canonical set — which is
        # correct: "all" is a meta-alias, not a lane identity.
        canonical: set[str] = set()
        for group_ids in self._alias_to_lanes.values():
            for gid in group_ids:
                clean = gid.strip()
                if clean and clean.lower() not in SCOPE_NOT_LANE:
                    canonical.add(clean)

        if extra_canonical_lanes:
            canonical.update(
                lane for lane in extra_canonical_lanes
                if lane.strip() and lane.strip().lower() not in SCOPE_NOT_LANE
            )

        self._canonical_lanes: frozenset[str] = frozenset(canonical)

    # ── Read-only properties ──────────────────────────────────────────────

    @property
    def canonical_lanes(self) -> frozenset[str]:
        """The set of all known canonical lane identifiers."""
        return self._canonical_lanes

    # ── Validation ────────────────────────────────────────────────────────

    @staticmethod
    def is_scope_not_lane(value: str) -> bool:
        """Return True if *value* is a scope label that must not be used as source_lane."""
        return value.strip().lower() in SCOPE_NOT_LANE

    def validate_source_lane(self, value: str | None) -> str | None:
        """Validate and return *value* as a source_lane, or ``None`` if invalid.

        Returns ``None`` (and logs a warning) when *value* is:
        - Empty/whitespace-only
        - A scope label (``private``, ``public``, ``owner``, etc.)

        Unknown-but-valid values are passed through for forward-compatibility
        (new lanes can be added to the config without restarting all writers
        first).
        """
        if value is None:
            return None
        clean = value.strip()
        if not clean:
            return None
        if clean.lower() in SCOPE_NOT_LANE:
            logger.warning(
                'lane_registry: rejected scope value %r as source_lane '
                '(scope is not lane identity)',
                clean,
            )
            return None
        return clean

    # ── Resolution ────────────────────────────────────────────────────────

    def resolve_typed_source_lanes(
        self,
        effective_group_ids: list[str],
    ) -> list[str]:
        """Resolve ``effective_group_ids`` to canonical ``source_lane`` values.

        For typed retrieval, ``source_lane`` on ``typed_roots`` stores canonical
        lane identifiers.  Callers that pass versioned/experimental group_ids
        through ``_resolve_effective_group_ids`` may receive IDs that have no
        corresponding ``typed_roots`` rows.

        This method:
        1. Filters out scope values (``private``, ``public``, etc.)
        2. Passes through canonical lanes and unknown-but-valid IDs
           (forward-compat for lanes not yet in the alias config)
        3. Preserves input order and deduplicates

        An empty input returns an empty list (caller should treat this as
        "no lane filter" or "deny" depending on context — same as before).
        """
        if not effective_group_ids:
            return []

        seen: set[str] = set()
        result: list[str] = []
        for gid in effective_group_ids:
            clean = gid.strip()
            if not clean or clean in seen:
                continue
            if clean.lower() in SCOPE_NOT_LANE:
                logger.debug(
                    'lane_registry: dropping scope value %r from typed source_lane filter',
                    clean,
                )
                continue
            seen.add(clean)
            result.append(clean)
        return result

    # ── Introspection ─────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f'LaneRegistry(canonical_lanes={sorted(self._canonical_lanes)}, '
            f'aliases={sorted(self._alias_to_lanes)})'
        )


# ── Module-level singleton ────────────────────────────────────────────────────
# Initialized lazily from the server config.  Tests can replace this via
# ``set_lane_registry()`` or by monkeypatching ``_lane_registry`` directly.

_lane_registry: LaneRegistry | None = None


def get_lane_registry() -> LaneRegistry:
    """Return the module-level LaneRegistry singleton.

    If not yet initialized (server startup hasn't called
    ``init_lane_registry``), returns a permissive empty registry that
    passes all values through.  This is the safe backward-compatible
    default for tests and standalone scripts that don't load server config.
    """
    if _lane_registry is None:
        return LaneRegistry()
    return _lane_registry


def init_lane_registry(
    lane_aliases: dict[str, list[str]] | None = None,
    *,
    extra_canonical_lanes: frozenset[str] | None = None,
) -> LaneRegistry:
    """Initialize the module-level registry from server config.

    Called once during server startup.  Returns the created registry.
    """
    global _lane_registry
    _lane_registry = LaneRegistry(
        lane_aliases,
        extra_canonical_lanes=extra_canonical_lanes,
    )
    logger.info(
        'lane_registry: initialized with %d canonical lanes: %s',
        len(_lane_registry.canonical_lanes),
        sorted(_lane_registry.canonical_lanes),
    )
    return _lane_registry


def set_lane_registry(registry: LaneRegistry | None) -> None:
    """Replace the module-level registry.  Primarily for test fixtures."""
    global _lane_registry
    _lane_registry = registry
