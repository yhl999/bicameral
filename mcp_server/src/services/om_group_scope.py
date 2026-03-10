"""Helpers for recognizing OM-native group scopes.

Patch 1 keeps the retrieval adapter architecture but expands activation from the
canonical OM lane to explicit experimental groups that are still OM-native.

Important distinction:
- canonical `s1_observational_memory` remains a strict OM-only scope
- explicit experimental `_om_` groups may probe the OM adapter first, but they
  can safely fall back to Graphiti retrieval when no OM primitives are found
"""

from __future__ import annotations

DEFAULT_OM_GROUP_ID = 's1_observational_memory'


def _normalize_group_id(group_id: str | None) -> str:
    return str(group_id or '').strip()


def _looks_like_experimental_om_native_group(group_id: str) -> bool:
    """Return True for explicit experimental groups that encode an `_om_` token.

    This keeps recognition intentionally narrow:
    - canonical `s1_observational_memory` is handled separately
    - `s1_*` lanes other than the canonical OM lane are not auto-promoted here
    - experimental bakeoff groups like `ontbk15batch_20260310_om_f` are treated
      as OM-native because `om` appears as its own underscore-delimited token
    """

    normalized = _normalize_group_id(group_id).lower()
    if not normalized or normalized == DEFAULT_OM_GROUP_ID:
        return False
    if normalized.startswith('s1_'):
        return False

    tokens = [token for token in normalized.split('_') if token]
    if len(tokens) < 3:
        return False

    return any(index > 0 and index < len(tokens) - 1 and token == 'om' for index, token in enumerate(tokens))


def is_canonical_om_group_id(group_id: str | None, *, default_group_id: str = DEFAULT_OM_GROUP_ID) -> bool:
    normalized = _normalize_group_id(group_id)
    if not normalized:
        return False
    return normalized == _normalize_group_id(default_group_id)


def is_om_native_group_id(group_id: str | None, *, default_group_id: str = DEFAULT_OM_GROUP_ID) -> bool:
    normalized = _normalize_group_id(group_id)
    if not normalized:
        return False
    if is_canonical_om_group_id(normalized, default_group_id=default_group_id):
        return True
    return _looks_like_experimental_om_native_group(normalized)


def includes_om_native_group(group_ids: list[str], *, default_group_id: str = DEFAULT_OM_GROUP_ID) -> bool:
    """Whether OM-native retrieval should activate for the requested scope.

    Empty group scope preserves the MCP all-lanes contract and keeps routing to
    the canonical OM lane only.
    """

    if len(group_ids) == 0:
        return True
    return any(is_om_native_group_id(group_id, default_group_id=default_group_id) for group_id in group_ids)


def om_native_groups_in_scope(group_ids: list[str], *, default_group_id: str = DEFAULT_OM_GROUP_ID) -> list[str]:
    """Return the explicit OM-native groups the adapter should search.

    For all-lanes scope (`[]`), keep historical behavior and search only the
    canonical OM lane. Experimental OM-native groups are queried only when the
    caller explicitly targets them.
    """

    if len(group_ids) == 0:
        return [default_group_id]

    seen: set[str] = set()
    scoped: list[str] = []
    for group_id in group_ids:
        normalized = _normalize_group_id(group_id)
        if not is_om_native_group_id(normalized, default_group_id=default_group_id):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        scoped.append(normalized)
    return scoped


def is_om_native_only_scope(group_ids: list[str], *, default_group_id: str = DEFAULT_OM_GROUP_ID) -> bool:
    """Whether the request targets a single OM-native scope.

    This includes both the canonical OM lane and explicit experimental OM-native
    groups. Use ``requires_strict_om_native_only_scope`` when the caller needs
    to know whether OM misses should fail closed instead of falling back.
    """

    return len(group_ids) == 1 and is_om_native_group_id(group_ids[0], default_group_id=default_group_id)


def requires_strict_om_native_only_scope(
    group_ids: list[str], *, default_group_id: str = DEFAULT_OM_GROUP_ID
) -> bool:
    """Return True only for the canonical OM lane fail-closed scope.

    Experimental OM-native groups are adapter-eligible, but they do not fail
    closed when the OM path returns zero rows; those scopes may continue to the
    normal Graphiti retrieval path.
    """

    return len(group_ids) == 1 and is_canonical_om_group_id(
        group_ids[0],
        default_group_id=default_group_id,
    )
