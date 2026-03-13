"""Episodes and procedures router — episodic/procedural memory retrieval.

Owned by: Exec 5 (episode/procedure retrieval).
Integrated from Phase-0 stubs: real ledger-backed retrieval is now wired.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from ._phase0 import (
        phase0_not_implemented,
        phase0_paginated_list_response,
        require_boolean,
        require_identifier,
        require_non_empty_string,
        require_optional_string_list,
        require_string,
        validate_pagination,
        validate_time_range,
    )
except ImportError:  # pragma: no cover - script/top-level import fallback
    from _phase0 import (  # type: ignore[no-redef]
        phase0_not_implemented,
        phase0_paginated_list_response,
        require_boolean,
        require_identifier,
        require_non_empty_string,
        require_optional_string_list,
        require_string,
        validate_pagination,
        validate_time_range,
    )

logger = logging.getLogger(__name__)

TOOL_CONTRACTS: list[dict[str, Any]] = [
    {
        'name': 'search_episodes',
        'description': 'Search episodic memory by semantic query with optional time range, scope, and pagination controls',
        'mode_hint': 'typed',
        'schema': {
            'inputs': {
                'query': 'string',
                'time_range': 'object with optional start/end ISO timestamps | null',
                'include_history': 'boolean (default false)',
                'group_ids': 'list[string] | null',
                'lane_alias': 'list[string] | null',
                'limit': 'integer | null (default 10)',
                'offset': 'integer | null (default 0)',
            },
            'output': 'EpisodeSearchResponse | ErrorResponse',
        },
        'examples': [{
            'query': 'last deployment',
            'time_range': None,
            'include_history': False,
            'group_ids': None,
            'lane_alias': None,
            'limit': 10,
            'offset': 0,
        }],
    },
    {
        'name': 'get_episode',
        'description': 'Retrieve a specific episode by ID within optional lane scope',
        'mode_hint': 'typed',
        'schema': {
            'inputs': {
                'episode_id': 'string',
                'group_ids': 'list[string] | null',
                'lane_alias': 'list[string] | null',
            },
            'output': 'Episode dict | ErrorResponse',
            'security_note': (
                'Cross-lane access returns {"error": "not_found"} (not "access_denied") to '
                'prevent existence leaks — callers cannot distinguish a forbidden episode '
                'from a truly absent one.'
            ),
        },
        'examples': [{'episode_id': 'ep-001', 'group_ids': None, 'lane_alias': None}],
    },
    {
        'name': 'search_procedures',
        'description': 'Search procedural memory with optional scope and pagination controls',
        'mode_hint': 'typed',
        'schema': {
            'inputs': {
                'query': 'string',
                'include_all': 'boolean (default false)',
                'group_ids': 'list[string] | null',
                'lane_alias': 'list[string] | null',
                'limit': 'integer | null (default 10)',
                'offset': 'integer | null (default 0)',
            },
            'output': 'ProcedureSearchResponse | ErrorResponse',
        },
        'examples': [{
            'query': 'how to run tests',
            'include_all': False,
            'group_ids': None,
            'lane_alias': None,
            'limit': 10,
            'offset': 0,
        }],
    },
    {
        'name': 'get_procedure',
        'description': 'Retrieve a procedure by trigger phrase or ID within optional lane scope',
        'mode_hint': 'typed',
        'schema': {
            'inputs': {
                'trigger_or_id': 'string',
                'group_ids': 'list[string] | null',
                'lane_alias': 'list[string] | null',
            },
            'output': 'Procedure dict | ErrorResponse',
        },
        'examples': [{'trigger_or_id': 'deploy to production', 'group_ids': None, 'lane_alias': None}],
    },
]


def _load_services() -> tuple[Any, Any, Any, Any]:
    """Lazily load service dependencies.

    Returns:
        (TypedRetrievalService, ChangeLedger, ProcedureService, resolve_ledger_path)
    """
    try:
        from ..services.change_ledger import ChangeLedger, resolve_ledger_path
        from ..services.procedure_service import ProcedureService
        from ..services.typed_retrieval import TypedRetrievalService
    except ImportError:  # pragma: no cover - top-level import fallback
        from services.change_ledger import ChangeLedger, resolve_ledger_path  # type: ignore[no-redef]
        from services.procedure_service import ProcedureService  # type: ignore[no-redef]
        from services.typed_retrieval import TypedRetrievalService  # type: ignore[no-redef]
    return TypedRetrievalService, ChangeLedger, ProcedureService, resolve_ledger_path


def _load_typed_models() -> tuple[Any, Any]:
    """Lazily load typed memory models; returns (Episode, Procedure)."""
    try:
        from ..models.typed_memory import Episode, Procedure
    except ImportError:
        try:
            from models.typed_memory import Episode, Procedure  # type: ignore[no-redef]
        except ImportError:
            return None, None  # type: ignore[return-value]
    return Episode, Procedure


def _resolve_effective_group_ids(
    group_ids: list[str] | None,
    lane_alias: list[str] | None,
) -> list[str] | None:
    """Resolve group_ids + lane_alias into an effective lane scope.

    Delegates to the main server resolver (which also intersects with
    authorized_group_ids).

    Fail-closed semantics:
    - Returns ``None`` only when no lane scoping is active AND the server
      has no ``authorized_group_ids`` restriction.
    - Returns ``[]`` when the resolved scope is empty (deny all).
    - When scope parameters are omitted but the server has
      ``authorized_group_ids`` configured, returns that list so omitted
      params do not widen access beyond the server-authorized scope.
    """
    try:
        from ..graphiti_mcp_server import _resolve_effective_group_ids as _server_resolve
        effective, invalid = _server_resolve(
            group_ids=group_ids,
            lane_alias=lane_alias,
        )
        if invalid:
            return []  # fail closed on invalid aliases

        # If caller explicitly requested scope, respect the result
        if group_ids is not None or lane_alias is not None:
            return effective

        # Caller omitted scope params — check if server restricts access
        try:
            from ..graphiti_mcp_server import config as _server_config
            authorized = _server_config.graphiti.authorized_group_ids
            if authorized:
                # Server has authorized restrictions; scope to those lanes
                # so omitting params doesn't widen beyond authorized scope.
                return list(authorized)
        except (ImportError, AttributeError, Exception):
            pass

        # No server restrictions and no explicit scope → all lanes visible
        if effective:
            return effective
        return None

    except (ImportError, Exception):
        # Fallback: server resolver unavailable
        if group_ids is not None:
            return list(group_ids) if group_ids else []
        if lane_alias is not None:
            return []  # can't resolve aliases without server → fail closed
        return None


def _build_metadata_filters(
    group_ids: list[str] | None,
    time_range: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata_filters dict for typed retrieval from lane scope and time range."""
    filters: dict[str, Any] = {}
    if group_ids:
        filters['source_lane'] = {'in': sorted(set(group_ids))}
    if time_range:
        if 'start' in time_range:
            filters.setdefault('time_range', {})['start'] = time_range['start']
        if 'end' in time_range:
            filters.setdefault('time_range', {})['end'] = time_range['end']
    return filters


def _episode_to_dict(ep: Any) -> dict[str, Any]:
    """Convert an Episode object to a serializable dict."""
    try:
        return ep.model_dump(mode='json')
    except Exception:
        return {
            'object_id': getattr(ep, 'object_id', None),
            'root_id': getattr(ep, 'root_id', None),
            'title': getattr(ep, 'title', ''),
            'content': getattr(ep, 'content', ''),
            'source_lane': getattr(ep, 'source_lane', None),
            'recorded_at': str(getattr(ep, 'recorded_at', '') or ''),
        }


def _procedure_to_dict(proc: Any) -> dict[str, Any]:
    """Convert a Procedure object to a serializable dict."""
    try:
        return proc.model_dump(mode='json')
    except Exception:
        return {
            'object_id': getattr(proc, 'object_id', None),
            'name': getattr(proc, 'name', ''),
            'trigger': getattr(proc, 'trigger', ''),
            'steps': getattr(proc, 'steps', []),
            'expected_outcome': getattr(proc, 'expected_outcome', ''),
            'promotion_status': getattr(proc, 'promotion_status', 'proposed'),
        }


def _passes_lane_filter(obj: Any, group_ids: list[str] | None) -> bool:
    """Return True if the object's source_lane is in the allowed group_ids (or no filter set).

    Fail-closed semantics:
    - ``None`` means no lane filter is active → allow all.
    - ``[]`` (empty list) means the resolved scope is empty → deny all.
      This covers: invalid aliases, disallowed lanes, explicit empty scope,
      and disjoint authorized-scope intersections.
    """
    if group_ids is None:
        return True
    if not group_ids:
        # Empty scope = deny (fail closed).  Never treat [] as "all lanes".
        return False
    source_lane = getattr(obj, 'source_lane', None)
    return source_lane in group_ids


def register_tools(mcp: Any) -> dict[str, Any]:
    """Register all episode and procedure router tools with the MCP server instance."""

    @mcp.tool()
    async def search_episodes(
        query: str,
        time_range: dict[str, Any] | None = None,
        include_history: bool = False,
        group_ids: list[str] | None = None,
        lane_alias: list[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """Search episodic memory by semantic query and optional time range.

        Episodes are discrete narrative chunks representing past events,
        conversations, or experiences. Use this to find relevant context
        from the agent's experience.

        Args:
            query: Natural-language query to search episodes.
            time_range: Optional time range filter dict with 'start' and/or
                        'end' ISO 8601 timestamps.
            include_history: Whether to include non-current (historical) versions.
            group_ids: Optional explicit list of lane/group IDs to scope search.
            lane_alias: Optional lane aliases resolved by runtime config.
            limit: Optional page size (defaults to 10 when omitted).
            offset: Optional page offset (defaults to 0 when omitted).

        Returns:
            Dict with 'episodes' plus pagination metadata.
        """
        query_error = require_string('query', query)
        if query_error is not None:
            return query_error

        time_range_error = validate_time_range(time_range)
        if time_range_error is not None:
            return time_range_error

        include_history_error = require_boolean('include_history', include_history)
        if include_history_error is not None:
            return include_history_error

        group_ids_error = require_optional_string_list('group_ids', group_ids)
        if group_ids_error is not None:
            return group_ids_error

        lane_alias_error = require_optional_string_list('lane_alias', lane_alias)
        if lane_alias_error is not None:
            return lane_alias_error

        limit_value, offset_value, pagination_error = validate_pagination(limit=limit, offset=offset)
        if pagination_error is not None:
            return pagination_error

        try:
            TypedRetrievalService, ChangeLedger, ProcedureService, _resolve_ledger = _load_services()
        except Exception as e:
            logger.error('search_episodes: service import failed: %s', e)
            return phase0_paginated_list_response(
                'search_episodes', 'episodes', limit=limit_value, offset=offset_value
            )

        # Resolve lane_alias into effective group IDs (includes server-authorized
        # scope intersection when the server resolver is available).
        effective_groups = _resolve_effective_group_ids(group_ids, lane_alias)

        history_mode = 'all' if include_history else 'current'
        metadata_filters = _build_metadata_filters(effective_groups, time_range)

        try:
            service = TypedRetrievalService(ledger_path=_resolve_ledger())
            result = await service.search(
                query=query,
                object_types=['episode'],
                metadata_filters=metadata_filters,
                history_mode=history_mode,
                current_only=not include_history,
                max_results=limit_value + offset_value + 1,
                max_evidence=1,
                effective_group_ids=effective_groups,
            )
        except Exception as e:
            logger.error('search_episodes: retrieval error: %s', e)
            return {
                'error': 'retrieval_error',
                'message': f'Episode search failed: {e}',
            }

        raw_episodes = result.get('episodes', [])
        # Apply lane filter (defence-in-depth in case service returned cross-lane data).
        # Use ``is not None`` so that an empty list (fail-closed scope) correctly
        # filters out ALL episodes, rather than being skipped as falsy.
        if effective_groups is not None:
            raw_episodes = [ep for ep in raw_episodes if _passes_lane_filter(ep, effective_groups)]

        total = len(raw_episodes)
        page = raw_episodes[offset_value:offset_value + limit_value]
        has_more = (offset_value + len(page)) < total
        next_offset = offset_value + len(page) if has_more else None

        episodes_out: list[dict[str, Any]] = []
        for ep in page:
            if isinstance(ep, dict):
                episodes_out.append(ep)
            else:
                episodes_out.append(_episode_to_dict(ep))

        return {
            'episodes': episodes_out,
            'limit': limit_value,
            'offset': offset_value,
            'total': total,
            'has_more': has_more,
            'next_offset': next_offset,
        }

    @mcp.tool()
    async def get_episode(
        episode_id: str,
        group_ids: list[str] | None = None,
        lane_alias: list[str] | None = None,
    ) -> dict[str, Any]:
        """Retrieve a specific episode by ID.

        Returns the full episode record including content, timestamps,
        evidence refs, and annotations.

        Args:
            episode_id: Unique ID of the episode to retrieve.
            group_ids: Optional explicit list of lane/group IDs for access scoping.
            lane_alias: Optional lane aliases resolved by runtime config.

        Returns:
            Episode dict, or ErrorResponse dict if not found or access denied.
        """
        episode_id_error = require_identifier('episode_id', episode_id)
        if episode_id_error is not None:
            return episode_id_error

        group_ids_error = require_optional_string_list('group_ids', group_ids)
        if group_ids_error is not None:
            return group_ids_error

        lane_alias_error = require_optional_string_list('lane_alias', lane_alias)
        if lane_alias_error is not None:
            return lane_alias_error

        try:
            TypedRetrievalService, ChangeLedger, ProcedureService, _resolve_ledger = _load_services()
        except Exception as e:
            logger.error('get_episode: service import failed: %s', e)
            return phase0_not_implemented('get_episode')

        try:
            with ChangeLedger(_resolve_ledger()) as ledger:
                obj = ledger.materialize_object(episode_id)
        except Exception as e:
            logger.error('get_episode: ledger error for %r: %s', episode_id, e)
            return {'error': 'retrieval_error', 'message': f'Ledger access failed: {e}'}

        if obj is None:
            return {'error': 'not_found', 'message': f'Episode not found: {episode_id}'}

        EpisodeModel, _ProcModel = _load_typed_models()

        if EpisodeModel is not None and not isinstance(obj, EpisodeModel):
            return {
                'error': 'not_found',
                'message': f'Object {episode_id!r} is not an episode (type: {type(obj).__name__})',
            }

        # Resolve lane_alias into effective group IDs for access check
        effective_groups = _resolve_effective_group_ids(group_ids, lane_alias)
        # Lane access check: uniform not_found so caller cannot distinguish
        # "exists but forbidden" from "truly absent" (no existence leak).
        if not _passes_lane_filter(obj, effective_groups):
            return {
                'error': 'not_found',
                'message': f'Episode not found: {episode_id}',
            }

        return _episode_to_dict(obj)

    @mcp.tool()
    async def search_procedures(
        query: str,
        include_all: bool = False,
        group_ids: list[str] | None = None,
        lane_alias: list[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """Search procedural memory for relevant procedures.

        Procedures are step-by-step instructions for accomplishing tasks,
        derived from past experience or explicitly defined. Use this to
        find how the agent has done things before.

        Args:
            query: Natural-language query describing the task or procedure.
            include_all: Whether to include proposed (non-promoted) procedures.
            group_ids: Optional explicit list of lane/group IDs to scope search.
            lane_alias: Optional lane aliases resolved by runtime config.
            limit: Optional page size (defaults to 10 when omitted).
            offset: Optional page offset (defaults to 0 when omitted).

        Returns:
            Dict with 'procedures' plus pagination metadata.
        """
        query_error = require_string('query', query)
        if query_error is not None:
            return query_error

        include_all_error = require_boolean('include_all', include_all)
        if include_all_error is not None:
            return include_all_error

        group_ids_error = require_optional_string_list('group_ids', group_ids)
        if group_ids_error is not None:
            return group_ids_error

        lane_alias_error = require_optional_string_list('lane_alias', lane_alias)
        if lane_alias_error is not None:
            return lane_alias_error

        limit_value, offset_value, pagination_error = validate_pagination(limit=limit, offset=offset)
        if pagination_error is not None:
            return pagination_error

        try:
            TypedRetrievalService, ChangeLedger, ProcedureService, _resolve_ledger = _load_services()
        except Exception as e:
            logger.error('search_procedures: service import failed: %s', e)
            return phase0_paginated_list_response(
                'search_procedures', 'procedures', limit=limit_value, offset=offset_value
            )

        # Resolve lane scope BEFORE retrieval so lane filtering is applied
        # within the candidate pool (prevents cross-lane top-K shadowing).
        effective_groups = _resolve_effective_group_ids(group_ids, lane_alias)

        try:
            with ChangeLedger(_resolve_ledger()) as ledger:
                svc = ProcedureService(ledger)
                # Fetch enough to satisfy pagination; retrieve_procedures returns top-k by score.
                # Lane filtering happens inside retrieve_procedures via effective_group_ids.
                fetch_limit = max(limit_value + offset_value + 1, 50)
                matches = svc.retrieve_procedures(
                    query,
                    limit=fetch_limit,
                    include_proposed=bool(include_all),
                    effective_group_ids=effective_groups,
                )
        except Exception as e:
            logger.error('search_procedures: retrieval error: %s', e)
            return {
                'error': 'retrieval_error',
                'message': f'Procedure search failed: {e}',
            }

        # Defence-in-depth: post-hoc lane filter in case service missed any.
        if effective_groups is not None:
            matches = [m for m in matches if _passes_lane_filter(m.procedure, effective_groups)]

        total = len(matches)
        page = matches[offset_value:offset_value + limit_value]
        has_more = (offset_value + len(page)) < total
        next_offset = offset_value + len(page) if has_more else None

        procedures_out: list[dict[str, Any]] = []
        for match in page:
            proc_dict = _procedure_to_dict(match.procedure)
            proc_dict['_score'] = match.score
            proc_dict['_matched_terms'] = match.matched_terms
            procedures_out.append(proc_dict)

        return {
            'procedures': procedures_out,
            'limit': limit_value,
            'offset': offset_value,
            'total': total,
            'has_more': has_more,
            'next_offset': next_offset,
        }

    @mcp.tool()
    async def get_procedure(
        trigger_or_id: str,
        group_ids: list[str] | None = None,
        lane_alias: list[str] | None = None,
    ) -> dict[str, Any]:
        """Retrieve a procedure by trigger phrase or ID.

        Looks up a procedure by its natural-language trigger (e.g. "how to
        deploy to production") or by its unique object ID.

        Args:
            trigger_or_id: Trigger phrase or unique procedure ID.
            group_ids: Optional explicit list of lane/group IDs for access scoping.
            lane_alias: Optional lane aliases resolved by runtime config.

        Returns:
            Procedure dict, or ErrorResponse dict if not found.
        """
        trigger_error = require_non_empty_string('trigger_or_id', trigger_or_id)
        if trigger_error is not None:
            return trigger_error

        group_ids_error = require_optional_string_list('group_ids', group_ids)
        if group_ids_error is not None:
            return group_ids_error

        lane_alias_error = require_optional_string_list('lane_alias', lane_alias)
        if lane_alias_error is not None:
            return lane_alias_error

        try:
            TypedRetrievalService, ChangeLedger, ProcedureService, _resolve_ledger = _load_services()
        except Exception as e:
            logger.error('get_procedure: service import failed: %s', e)
            return phase0_not_implemented('get_procedure')

        _EpisodeModel, ProcedureModel = _load_typed_models()

        try:
            ledger = ChangeLedger(_resolve_ledger())
        except Exception as e:
            logger.error('get_procedure: ledger open failed: %s', e)
            return {'error': 'service_unavailable', 'message': f'Ledger unavailable: {e}'}

        # Resolve lane scope FIRST so lane filtering is applied before
        # selecting a match.  This prevents cross-lane existence leaks
        # (returning access_denied for a forbidden match reveals its
        # existence) and shadowing (a forbidden higher-ranked match
        # hiding a permitted lower-ranked same-lane match).
        effective_groups = _resolve_effective_group_ids(group_ids, lane_alias)

        proc: Any = None

        # Use ledger as context manager to guarantee the SQLite connection is
        # closed on exit regardless of which return path is taken.
        with ledger:
            # First: try to resolve as a direct object ID (with lane check)
            try:
                candidate = ledger.materialize_object(trigger_or_id)
                if ProcedureModel is not None and isinstance(candidate, ProcedureModel):
                    if _passes_lane_filter(candidate, effective_groups):
                        proc = candidate
            except Exception:
                pass  # Not an object ID — try trigger search

            # Second: search by trigger phrase with lane-scoped retrieval.
            # Lane filtering happens inside retrieve_procedures so forbidden-lane
            # matches never consume top-K slots (no shadowing, no magic-number hack).
            if proc is None:
                try:
                    svc = ProcedureService(ledger)
                    matches = svc.retrieve_procedures(
                        trigger_or_id,
                        limit=5,
                        include_proposed=True,
                        effective_group_ids=effective_groups,
                    )
                    if matches:
                        proc = matches[0].procedure
                except Exception as e:
                    logger.error('get_procedure: search error for %r: %s', trigger_or_id, e)
                    return {'error': 'retrieval_error', 'message': f'Procedure lookup failed: {e}'}

            if proc is None:
                # Uniform not_found: caller cannot distinguish "exists but
                # forbidden" from "truly absent" (no existence leak).
                return {
                    'error': 'not_found',
                    'message': f'No procedure found for trigger or ID: {trigger_or_id!r}',
                }

            return _procedure_to_dict(proc)

    return {
        'search_episodes': search_episodes,
        'get_episode': get_episode,
        'search_procedures': search_procedures,
        'get_procedure': get_procedure,
    }
