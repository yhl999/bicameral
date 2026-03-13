"""Regression tests for authorized-scope empty-intersection fail-closed behaviour.

Proves:
  1. _resolve_effective_group_ids fails closed (returns []) when a caller
     requests lanes that are ALL outside the authorized_group_ids allowlist.
  2. _resolve_effective_group_ids scopes all-lanes requests to the authorized
     set when authorized_group_ids is configured (no longer returns [] which
     would be misread as all-lanes by the OM adapter and Graphiti client).
  3. search_nodes and search_memory_facts (both facts and typed paths) return
     an empty/denied response — not a wider search — when effective_group_ids
     is [] after authorized intersection AND authorized_group_ids is configured.
  4. No regression: when authorized_group_ids is empty/absent (the default),
     the legacy all-lanes [] sentinel is preserved unchanged.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import path setup — match the pattern used by test_graphiti_mcp_server.py
# ---------------------------------------------------------------------------

_MCP_SRC = Path(__file__).parent.parent / 'src'
if str(_MCP_SRC) not in sys.path:
    sys.path.insert(0, str(_MCP_SRC.parent))
    sys.path.insert(0, str(_MCP_SRC))

_REPO_TESTS = Path(__file__).resolve().parents[2] / 'tests'
if str(_REPO_TESTS) not in sys.path:
    sys.path.insert(0, str(_REPO_TESTS))


def _load_server_module():
    return importlib.import_module('helpers_mcp_import').load_graphiti_mcp_server()


# ---------------------------------------------------------------------------
# Config mock factory
# ---------------------------------------------------------------------------


def _make_graphiti_cfg(*, authorized_group_ids=None, group_id=None, lane_aliases=None):
    cfg = MagicMock()
    cfg.authorized_group_ids = authorized_group_ids if authorized_group_ids is not None else []
    cfg.group_id = group_id or ''
    cfg.lane_aliases = lane_aliases or {}
    return cfg


def _make_db_cfg(*, provider='neo4j'):
    cfg = MagicMock()
    cfg.provider = provider
    return cfg


def _make_config(*, authorized=None, group_id=None, provider='neo4j'):
    cfg = MagicMock()
    cfg.graphiti = _make_graphiti_cfg(authorized_group_ids=authorized or [], group_id=group_id)
    cfg.database = _make_db_cfg(provider=provider)
    return cfg


# ---------------------------------------------------------------------------
# Unit tests: _resolve_effective_group_ids (no I/O)
# ---------------------------------------------------------------------------


class TestResolveEffectiveGroupIds:
    """Unit tests for the resolver itself — pure logic, no backend calls."""

    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _load_server_module()

    def _call(self, *, group_ids, lane_alias, authorized=None, default_group=None):
        fake_config = _make_config(authorized=authorized, group_id=default_group)
        with patch.object(self.mod, 'config', fake_config):
            return self.mod._resolve_effective_group_ids(
                group_ids=group_ids,
                lane_alias=lane_alias,
            )

    # --- fail-closed: caller requests unauthorized lanes ---

    def test_all_requested_lanes_denied_returns_empty(self):
        """When caller requests only unauthorized lanes, intersection yields []."""
        effective, invalid = self._call(
            group_ids=['lane_private', 'lane_secret'],
            lane_alias=None,
            authorized=['lane_public'],
        )
        assert effective == [], (
            f"Expected [] (deny), got {effective!r}"
        )
        assert invalid == []

    def test_partial_denial_returns_only_authorized_lanes(self):
        """When some requested lanes pass and some don't, only the passing ones are returned."""
        effective, invalid = self._call(
            group_ids=['lane_a', 'lane_forbidden'],
            lane_alias=None,
            authorized=['lane_a', 'lane_b'],
        )
        assert effective == ['lane_a'], (
            f"Expected ['lane_a'], got {effective!r}"
        )

    def test_single_unauthorized_lane_yields_empty_not_all_lanes(self):
        """Single disallowed lane must yield [] (deny), not all-lanes sentinel."""
        effective, invalid = self._call(
            group_ids=['disallowed'],
            lane_alias=None,
            authorized=['lane_allowed'],
        )
        assert effective == []
        assert invalid == []

    # --- all-lanes scoping: no explicit scope + authorized set ---

    def test_no_scope_with_authorized_scopes_to_authorized_lanes(self):
        """All-lanes request (group_ids=None) with authorized_group_ids scopes to authorized."""
        effective, invalid = self._call(
            group_ids=None,
            lane_alias=None,
            authorized=['lane_a', 'lane_b'],
        )
        assert set(effective) == {'lane_a', 'lane_b'}, (
            f"All-lanes + authorized must scope to authorized set, got {effective!r}"
        )
        assert invalid == []

    def test_no_scope_with_authorized_does_not_return_all_lanes_sentinel(self):
        """All-lanes request with authorized must NOT return [] (would mean all-lanes)."""
        effective, invalid = self._call(
            group_ids=None,
            lane_alias=None,
            authorized=['lane_a'],
        )
        assert effective != [], (
            "All-lanes request with authorized_group_ids must NOT return [] — "
            "that would be mis-interpreted as all-lanes by OM adapter and Graphiti client"
        )
        assert effective == ['lane_a']

    # --- no regression: no authorized_group_ids preserves legacy all-lanes ---

    def test_no_authorized_restriction_all_lanes_returns_empty_list(self):
        """When authorized_group_ids is unset/empty, [] all-lanes sentinel is preserved."""
        effective, invalid = self._call(
            group_ids=None,
            lane_alias=None,
            authorized=[],  # no restriction
        )
        assert effective == [], (
            f"No auth restriction: legacy all-lanes [] sentinel must be preserved, got {effective!r}"
        )

    def test_explicit_lanes_without_authorized_passes_through(self):
        """Explicit lanes without any authorization restriction pass through unchanged."""
        effective, invalid = self._call(
            group_ids=['lane_x', 'lane_y'],
            lane_alias=None,
            authorized=[],
        )
        assert effective == ['lane_x', 'lane_y']

    def test_authorized_subset_of_requested_returns_intersection(self):
        """When authorized is a strict subset of requested, only authorized lanes pass."""
        effective, invalid = self._call(
            group_ids=['lane_a', 'lane_b', 'lane_c'],
            lane_alias=None,
            authorized=['lane_b'],
        )
        assert effective == ['lane_b']

    def test_authorized_superset_of_requested_returns_requested(self):
        """When all requested lanes are authorized, caller gets exactly what they asked."""
        effective, invalid = self._call(
            group_ids=['lane_a'],
            lane_alias=None,
            authorized=['lane_a', 'lane_b', 'lane_c'],
        )
        assert effective == ['lane_a']


# ---------------------------------------------------------------------------
# Integration-level: search_nodes fails closed on empty intersection
# ---------------------------------------------------------------------------


class TestSearchNodesDeniesEmptyAuthorizedIntersection:
    """search_nodes must return empty/denied when authorized intersection is empty."""

    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _load_server_module()

    @pytest.mark.anyio
    async def test_search_nodes_empty_not_wider_when_intersection_empty(self):
        """search_nodes fails closed (empty nodes) when authorized intersection yields []."""
        fake_config = _make_config(authorized=['lane_allowed'])
        fake_graphiti = MagicMock()

        with (
            patch.object(self.mod, 'config', fake_config),
            patch.object(self.mod, 'graphiti_service', fake_graphiti),
            patch.object(self.mod, '_SEARCH_RATE_LIMIT_ENABLED', False),
        ):
            result = await self.mod.search_nodes(
                query='test query',
                group_ids=['lane_disallowed'],  # not in authorized → intersection = []
            )

        # Backend must NOT have been called — fail closed before OM adapter or Graphiti search
        fake_graphiti.get_client_for_group.assert_not_called()
        # Result must be empty nodes, not an error from a backend failure
        if hasattr(result, 'nodes'):
            assert result.nodes == [], (
                f"search_nodes must return empty when authorized intersection yields [], got: {result}"
            )
        else:
            assert result.get('nodes', []) == []

    @pytest.mark.anyio
    async def test_search_nodes_no_restriction_still_reaches_backend(self):
        """search_nodes without authorized restriction still calls the backend."""
        fake_config = _make_config(authorized=[])  # no restriction
        fake_graphiti = MagicMock()
        fake_client = AsyncMock()
        fake_results = MagicMock()
        fake_results.nodes = []
        fake_client.search_ = AsyncMock(return_value=fake_results)
        fake_graphiti.get_client_for_group = AsyncMock(return_value=fake_client)
        fake_search_service = MagicMock()
        fake_search_service.includes_observational_memory = lambda _: False

        with (
            patch.object(self.mod, 'config', fake_config),
            patch.object(self.mod, 'graphiti_service', fake_graphiti),
            patch.object(self.mod, '_SEARCH_RATE_LIMIT_ENABLED', False),
            patch.object(self.mod, 'search_service', fake_search_service),
        ):
            await self.mod.search_nodes(
                query='test query',
                group_ids=['lane_a'],
            )

        # Backend must have been reached
        fake_graphiti.get_client_for_group.assert_called_once()

    @pytest.mark.anyio
    async def test_search_nodes_all_lanes_with_authorized_scopes_to_authorized(self):
        """search_nodes without explicit scope but with authorized_group_ids passes authorized scope
        to the backend — it must NOT be intercepted by the fail-closed guard (which would only
        fire if authorized intersection yielded empty).

        Proves: all-lanes request + authorized=['lane_a'] resolves to effective=['lane_a'],
        which is non-empty, so the fail-closed guard is NOT triggered and the backend IS reached.
        """
        fake_config = _make_config(authorized=['lane_a'])
        fake_graphiti = MagicMock()
        fake_client = AsyncMock()
        fake_graphiti.get_client_for_group = AsyncMock(return_value=fake_client)
        fake_search_service = MagicMock()
        fake_search_service.includes_observational_memory = lambda _: False

        with (
            patch.object(self.mod, 'config', fake_config),
            patch.object(self.mod, 'graphiti_service', fake_graphiti),
            patch.object(self.mod, '_SEARCH_RATE_LIMIT_ENABLED', False),
            patch.object(self.mod, 'search_service', fake_search_service),
        ):
            await self.mod.search_nodes(
                query='test query',
                group_ids=None,  # all-lanes request — must be scoped to ['lane_a']
            )

        # Backend must be reached (all-lanes + authorized=['lane_a'] → effective=['lane_a'],
        # non-empty, so fail-closed guard does NOT fire).
        # get_client_for_group is called exactly once with 'lane_a' (not the empty-list fallback).
        fake_graphiti.get_client_for_group.assert_called_once()
        called_with_group = fake_graphiti.get_client_for_group.call_args.args[0]
        assert called_with_group == 'lane_a', (
            f"All-lanes + authorized=['lane_a'] must route to lane_a client, "
            f"got: {called_with_group!r}"
        )


# ---------------------------------------------------------------------------
# Integration-level: search_memory_facts fails closed on empty intersection
# ---------------------------------------------------------------------------


class TestSearchMemoryFactsDeniesEmptyAuthorizedIntersection:
    """search_memory_facts must return empty when authorized intersection is empty."""

    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _load_server_module()

    @pytest.mark.anyio
    async def test_facts_path_empty_not_wider_on_denied_intersection(self):
        """search_memory_facts (facts) fails closed when authorized intersection is []."""
        fake_config = _make_config(authorized=['lane_allowed'])
        fake_graphiti = MagicMock()

        with (
            patch.object(self.mod, 'config', fake_config),
            patch.object(self.mod, 'graphiti_service', fake_graphiti),
            patch.object(self.mod, '_SEARCH_RATE_LIMIT_ENABLED', False),
        ):
            result = await self.mod.search_memory_facts(
                query='test query',
                group_ids=['lane_disallowed'],
                result_format='facts',
            )

        # Backend must NOT have been called
        fake_graphiti.get_client_for_group.assert_not_called()
        if hasattr(result, 'facts'):
            assert result.facts == [], (
                f"search_memory_facts must return empty when denied, got: {result}"
            )
        else:
            assert result.get('facts', []) == []

    @pytest.mark.anyio
    async def test_typed_path_empty_not_wider_on_denied_intersection(self):
        """search_memory_facts (typed) fails closed when authorized intersection is []."""
        fake_config = _make_config(authorized=['lane_allowed'])
        fake_graphiti = MagicMock()

        with (
            patch.object(self.mod, 'config', fake_config),
            patch.object(self.mod, 'graphiti_service', fake_graphiti),
            patch.object(self.mod, '_SEARCH_RATE_LIMIT_ENABLED', False),
        ):
            result = await self.mod.search_memory_facts(
                query='test query',
                group_ids=['lane_disallowed'],
                result_format='typed',
            )

        # TypedRetrievalService must NOT have been invoked
        fake_graphiti.get_client_for_group.assert_not_called()
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert result.get('state_facts', []) == []
        assert result.get('episodes', []) == []
        assert result.get('procedures', []) == []

    @pytest.mark.anyio
    async def test_facts_no_restriction_still_reaches_backend(self):
        """search_memory_facts without authorized restriction calls the backend."""
        fake_config = _make_config(authorized=[])
        fake_graphiti = MagicMock()
        fake_client = AsyncMock()
        fake_results = MagicMock()
        fake_results.edges = []
        fake_client.search_ = AsyncMock(return_value=fake_results)
        fake_graphiti.get_client_for_group = AsyncMock(return_value=fake_client)
        fake_search_service = MagicMock()
        fake_search_service.includes_observational_memory = lambda _: False

        with (
            patch.object(self.mod, 'config', fake_config),
            patch.object(self.mod, 'graphiti_service', fake_graphiti),
            patch.object(self.mod, '_SEARCH_RATE_LIMIT_ENABLED', False),
            patch.object(self.mod, 'search_service', fake_search_service),
        ):
            await self.mod.search_memory_facts(
                query='test query',
                group_ids=['lane_a'],
                result_format='facts',
            )

        fake_graphiti.get_client_for_group.assert_called_once()
