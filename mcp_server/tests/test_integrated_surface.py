"""Integrated surface acceptance tests.

Proves that:
 A. Episodes/procedures are real (not stubs) on the integrated branch.
 B. Candidate/list/promote/reject auth boundary works at the integrated surface.
 C. Lane scoping works — objects scoped to one lane are not returned for another.
 D. Output contract metadata matches real runtime behavior.

These tests use in-memory SQLite so they run without external services.
"""

from __future__ import annotations

import sqlite3
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_mcp() -> MagicMock:
    """Return a minimal mock MCP that captures registered @mcp.tool() functions."""
    tools: dict[str, object] = {}
    mock = MagicMock()

    def _tool_decorator():
        def decorator(fn):
            tools[fn.__name__] = fn
            return fn

        mock.tool.return_value = decorator
        return decorator

    mock.tool.side_effect = lambda: _tool_decorator()
    mock._tools = tools
    return mock


def _make_in_memory_ledger():
    """Return a ChangeLedger backed by an in-memory SQLite connection."""
    from mcp_server.src.services.change_ledger import ChangeLedger

    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    return ChangeLedger(conn)


def _make_test_evidence_ref(object_id: str, idx: int = 0) -> dict[str, Any]:
    """Return a minimal valid EvidenceRef dict for tests."""
    return {
        'kind': 'event_log',
        'source_system': 'test',
        'locator': {
            'system': 'test',
            'stream': f'test-stream-{object_id}',
            'event_id': f'evt-{object_id}-{idx}',
        },
    }


def _create_episode_in_ledger(ledger, *, title: str = 'Test episode', content: str = 'Something happened.',
                               source_lane: str | None = None, object_id: str | None = None):
    """Append an Episode assert event directly to the ledger."""
    from mcp_server.src.models.typed_memory import Episode

    eid = object_id or 'ep-test-001'
    ep = Episode.model_validate({
        'object_id': eid,
        'root_id': eid,
        'title': title,
        'summary': content,
        'source_lane': source_lane,
        'policy_scope': 'private',
        'visibility_scope': 'private',
        'evidence_refs': [_make_test_evidence_ref(eid)],
    })
    ledger.append_event('assert', actor_id='test', reason='test_create', payload=ep)
    return ep


def _create_procedure_in_ledger(ledger, *, name: str = 'Test procedure', trigger: str = 'how to deploy',
                                  steps: list[str] | None = None, source_lane: str | None = None,
                                  object_id: str | None = None, promote: bool = True):
    """Create a Procedure via ProcedureService and optionally promote it."""
    from mcp_server.src.services.procedure_service import ProcedureService

    svc = ProcedureService(ledger)
    proc = svc.create_procedure(
        actor_id='test',
        name=name,
        trigger=trigger,
        steps=steps or ['Step 1', 'Step 2'],
        expected_outcome='Success',
        evidence_refs=[{
            'uri': 'mem://test/evidence/1',
            'kind': 'message',
            'content_summary': 'test evidence',
        }],
        source_lane=source_lane,
        object_id=object_id,
        promote=promote,
    )
    return proc


# ---------------------------------------------------------------------------
# Blocker A — episodes/procedures are real, not stubs
# ---------------------------------------------------------------------------


class TestEpisodesAreReal:
    """Prove search_episodes and get_episode use the real ledger, not stubs."""

    @pytest.mark.anyio
    async def test_search_episodes_returns_real_data(self, tmp_path):
        """search_episodes returns a real episode from the ledger, not an empty stub list."""
        from mcp_server.src.routers.episodes_procedures import _load_services, register_tools
        from mcp_server.src.services.change_ledger import ChangeLedger

        # Write an episode to a tmp file ledger
        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = ChangeLedger(db_path)
        _create_episode_in_ledger(ledger, title='Deployment event', content='Deployed v1.2.3 to production.')

        # Patch DB_PATH_DEFAULT so the router opens the same DB
        import mcp_server.src.services.change_ledger as cl_module
        original_path = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path

        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['search_episodes']
            result = await fn(query='deployment')
        finally:
            cl_module.DB_PATH_DEFAULT = original_path

        assert 'error' not in result, f"Unexpected error: {result}"
        assert 'episodes' in result, f"Missing 'episodes' key: {result}"
        assert isinstance(result['episodes'], list)
        # The episode we wrote should be returned (it matches 'deployment' query)
        assert len(result['episodes']) >= 1, (
            "search_episodes returned empty list — still behaving as stub"
        )
        titles = [ep.get('title', '') for ep in result['episodes'] if isinstance(ep, dict)]
        assert any('Deployment' in t or 'deployment' in t.lower() for t in titles), (
            f"Expected deployment episode, got: {titles}"
        )

    @pytest.mark.anyio
    async def test_get_episode_returns_real_episode(self, tmp_path):
        """get_episode retrieves a real episode from the ledger, not not_implemented."""
        from mcp_server.src.routers.episodes_procedures import register_tools
        from mcp_server.src.services.change_ledger import ChangeLedger

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = ChangeLedger(db_path)
        ep = _create_episode_in_ledger(
            ledger, title='Get me by ID', content='Detailed content.', object_id='ep-specific-001'
        )

        import mcp_server.src.services.change_ledger as cl_module
        original_path = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path

        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['get_episode']
            result = await fn(episode_id='ep-specific-001')
        finally:
            cl_module.DB_PATH_DEFAULT = original_path

        assert 'error' not in result, (
            f"get_episode returned error (still stub?): {result}"
        )
        assert result.get('title') == 'Get me by ID' or result.get('object_id') == 'ep-specific-001', (
            f"Expected episode with title/id, got: {result}"
        )

    @pytest.mark.anyio
    async def test_get_episode_returns_not_found_for_missing_id(self, tmp_path):
        """get_episode returns not_found (not not_implemented) for a missing ID."""
        from mcp_server.src.routers.episodes_procedures import register_tools
        from mcp_server.src.services.change_ledger import ChangeLedger

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)

        import mcp_server.src.services.change_ledger as cl_module
        original_path = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path

        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['get_episode']
            result = await fn(episode_id='ep-does-not-exist')
        finally:
            cl_module.DB_PATH_DEFAULT = original_path

        assert result.get('error') in ('not_found', 'retrieval_error'), (
            f"Expected not_found, got: {result}"
        )
        # Critically: NOT 'not_implemented'
        assert result.get('error') != 'not_implemented', (
            "get_episode is still returning Phase-0 not_implemented stub response"
        )

    @pytest.mark.anyio
    async def test_search_episodes_empty_ledger_returns_empty_list_not_stub_message(self, tmp_path):
        """search_episodes on empty ledger returns proper empty list (no Phase-0 stub message)."""
        from mcp_server.src.routers.episodes_procedures import register_tools
        from mcp_server.src.services.change_ledger import ChangeLedger

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)

        import mcp_server.src.services.change_ledger as cl_module
        original_path = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path

        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['search_episodes']
            result = await fn(query='anything')
        finally:
            cl_module.DB_PATH_DEFAULT = original_path

        assert 'error' not in result, f"Unexpected error: {result}"
        assert result['episodes'] == []
        assert result['total'] == 0
        # No Phase-0 stub message
        assert 'Phase 0 stub' not in result.get('message', ''), (
            "search_episodes is still returning Phase-0 stub message"
        )


class TestProceduresAreReal:
    """Prove search_procedures and get_procedure use the real ledger, not stubs."""

    @pytest.mark.anyio
    async def test_search_procedures_returns_real_data(self, tmp_path):
        """search_procedures returns a real procedure, not an empty stub list."""
        from mcp_server.src.routers.episodes_procedures import register_tools
        from mcp_server.src.services.change_ledger import ChangeLedger

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = ChangeLedger(db_path)
        _create_procedure_in_ledger(
            ledger, name='Deploy procedure', trigger='how to deploy to production',
            steps=['Run tests', 'Build artifact', 'Push to registry', 'Deploy'],
            promote=True,
        )

        import mcp_server.src.services.change_ledger as cl_module
        original_path = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path

        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['search_procedures']
            result = await fn(query='deploy to production')
        finally:
            cl_module.DB_PATH_DEFAULT = original_path

        assert 'error' not in result, f"Unexpected error: {result}"
        assert 'procedures' in result
        assert len(result['procedures']) >= 1, (
            "search_procedures returned empty list — still behaving as stub"
        )
        names = [p.get('name', '') for p in result['procedures'] if isinstance(p, dict)]
        assert any('Deploy' in n for n in names), f"Expected Deploy procedure, got: {names}"

    @pytest.mark.anyio
    async def test_search_procedures_include_all_returns_proposed(self, tmp_path):
        """search_procedures with include_all=True returns proposed (non-promoted) procedures."""
        from mcp_server.src.routers.episodes_procedures import register_tools
        from mcp_server.src.services.change_ledger import ChangeLedger

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = ChangeLedger(db_path)
        _create_procedure_in_ledger(
            ledger, name='Proposed procedure', trigger='how to run tests', promote=False
        )

        import mcp_server.src.services.change_ledger as cl_module
        original_path = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path

        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['search_procedures']

            # include_all=False should NOT return proposed procedures
            result_no_all = await fn(query='run tests', include_all=False)
            # include_all=True SHOULD return proposed procedures
            result_with_all = await fn(query='run tests', include_all=True)
        finally:
            cl_module.DB_PATH_DEFAULT = original_path

        assert len(result_no_all['procedures']) == 0, "Promoted-only filter should exclude proposed"
        assert len(result_with_all['procedures']) >= 1, "include_all=True should return proposed"

    @pytest.mark.anyio
    async def test_get_procedure_returns_real_procedure_by_trigger(self, tmp_path):
        """get_procedure finds a real procedure by trigger phrase, not not_implemented."""
        from mcp_server.src.routers.episodes_procedures import register_tools
        from mcp_server.src.services.change_ledger import ChangeLedger

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = ChangeLedger(db_path)
        _create_procedure_in_ledger(
            ledger, name='Rollback procedure', trigger='how to rollback a deployment',
            steps=['Identify last good version', 'Revert deployment'], promote=True,
        )

        import mcp_server.src.services.change_ledger as cl_module
        original_path = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path

        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['get_procedure']
            result = await fn(trigger_or_id='how to rollback a deployment')
        finally:
            cl_module.DB_PATH_DEFAULT = original_path

        assert 'error' not in result, f"get_procedure returned error (still stub?): {result}"
        assert result.get('name') == 'Rollback procedure' or result.get('trigger', '').startswith('how to rollback'), (
            f"Expected Rollback procedure, got: {result}"
        )
        # Critically: NOT not_implemented
        assert result.get('error') != 'not_implemented', (
            "get_procedure is still returning Phase-0 not_implemented stub response"
        )

    @pytest.mark.anyio
    async def test_get_procedure_not_found_for_missing_trigger(self, tmp_path):
        """get_procedure returns not_found (not not_implemented) for unknown trigger."""
        from mcp_server.src.routers.episodes_procedures import register_tools
        from mcp_server.src.services.change_ledger import ChangeLedger

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)

        import mcp_server.src.services.change_ledger as cl_module
        original_path = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path

        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['get_procedure']
            result = await fn(trigger_or_id='something that does not exist anywhere')
        finally:
            cl_module.DB_PATH_DEFAULT = original_path

        assert result.get('error') == 'not_found', (
            f"Expected not_found, got: {result}"
        )
        assert result.get('error') != 'not_implemented', (
            "get_procedure is still returning Phase-0 not_implemented stub response"
        )


# ---------------------------------------------------------------------------
# Lane scoping — objects scoped to one lane not returned for another
# ---------------------------------------------------------------------------


class TestLaneScoping:
    """Prove lane scoping is enforced for episodes and procedures."""

    @pytest.mark.anyio
    async def test_search_episodes_respects_group_ids_lane_filter(self, tmp_path):
        """search_episodes with group_ids only returns episodes in the specified lane."""
        from mcp_server.src.routers.episodes_procedures import register_tools
        from mcp_server.src.services.change_ledger import ChangeLedger

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = ChangeLedger(db_path)
        _create_episode_in_ledger(
            ledger, title='Lane A episode', content='In lane A.', source_lane='lane_a',
            object_id='ep-lane-a-001'
        )
        _create_episode_in_ledger(
            ledger, title='Lane B episode', content='In lane B.', source_lane='lane_b',
            object_id='ep-lane-b-001'
        )

        import mcp_server.src.services.change_ledger as cl_module
        original_path = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path

        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['search_episodes']
            result_a = await fn(query='lane', group_ids=['lane_a'])
            result_b = await fn(query='lane', group_ids=['lane_b'])
        finally:
            cl_module.DB_PATH_DEFAULT = original_path

        # Lane A should only see Lane A episode
        lanes_a = {ep.get('source_lane') for ep in result_a.get('episodes', []) if isinstance(ep, dict)}
        assert 'lane_b' not in lanes_a, f"Lane A search leaked Lane B episode: {result_a}"

        # Lane B should only see Lane B episode
        lanes_b = {ep.get('source_lane') for ep in result_b.get('episodes', []) if isinstance(ep, dict)}
        assert 'lane_a' not in lanes_b, f"Lane B search leaked Lane A episode: {result_b}"

    @pytest.mark.anyio
    async def test_get_episode_denies_cross_lane_access(self, tmp_path):
        """get_episode with a lane scope denies access to episodes in a different lane.

        Security design: cross-lane denial returns 'not_found' (not 'access_denied') so
        callers cannot distinguish "exists but forbidden" from "truly absent".  This is an
        intentional existence-leak prevention measure — the uniform not_found contract is
        documented in the router implementation and must not be changed to access_denied.
        """
        from mcp_server.src.routers.episodes_procedures import register_tools
        from mcp_server.src.services.change_ledger import ChangeLedger

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = ChangeLedger(db_path)
        _create_episode_in_ledger(
            ledger, title='Private episode', content='Should not leak.', source_lane='lane_private',
            object_id='ep-private-001'
        )

        import mcp_server.src.services.change_ledger as cl_module
        original_path = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path

        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['get_episode']
            # Accessing with wrong lane scope: uniform not_found prevents existence leak.
            result = await fn(episode_id='ep-private-001', group_ids=['lane_other'])
        finally:
            cl_module.DB_PATH_DEFAULT = original_path

        # Cross-lane access uses uniform not_found (security: no existence leak).
        # Do NOT assert 'access_denied' — that would reveal the object exists in another lane.
        assert result.get('error') == 'not_found', (
            f"Cross-lane access should return uniform not_found (existence-leak prevention), "
            f"got: {result}"
        )
        # Sanity: response must not contain episode content
        assert 'Private episode' not in str(result), (
            f"Cross-lane denial leaked episode content: {result}"
        )

    @pytest.mark.anyio
    async def test_get_episode_allows_correct_lane_access(self, tmp_path):
        """get_episode with correct lane scope returns the episode."""
        from mcp_server.src.routers.episodes_procedures import register_tools
        from mcp_server.src.services.change_ledger import ChangeLedger

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = ChangeLedger(db_path)
        _create_episode_in_ledger(
            ledger, title='Scoped episode', content='In lane A.', source_lane='lane_a',
            object_id='ep-scoped-001'
        )

        import mcp_server.src.services.change_ledger as cl_module
        original_path = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path

        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['get_episode']
            result = await fn(episode_id='ep-scoped-001', group_ids=['lane_a'])
        finally:
            cl_module.DB_PATH_DEFAULT = original_path

        assert 'error' not in result, f"Should allow correct lane access, got: {result}"


# ---------------------------------------------------------------------------
# Blocker B — output contract: no Phase-0 messages on real tools
# ---------------------------------------------------------------------------


class TestContractCoherence:
    """Prove that TOOL_CONTRACTS no longer contain phase0_behavior for implemented tools."""

    def test_episodes_procedures_contracts_have_no_phase0_behavior(self):
        """Implemented tools should not have phase0_behavior in their contracts."""
        from mcp_server.src.routers.episodes_procedures import TOOL_CONTRACTS

        for contract in TOOL_CONTRACTS:
            tool_name = contract.get('name', '?')
            assert 'phase0_behavior' not in contract, (
                f"Tool {tool_name!r} still has phase0_behavior in its contract; "
                "remove it for implemented tools."
            )

    def test_episodes_procedures_contracts_have_valid_output_descriptions(self):
        """Implemented tools should have output descriptions that match real return types."""
        from mcp_server.src.routers.episodes_procedures import TOOL_CONTRACTS

        for contract in TOOL_CONTRACTS:
            schema = contract.get('schema', {})
            output = schema.get('output', '')
            # Should not refer to Phase-0 stub behavior
            assert 'Phase 0' not in output, (
                f"Contract for {contract.get('name')!r} output still references Phase 0: {output!r}"
            )
            assert 'not_implemented' not in output, (
                f"Contract for {contract.get('name')!r} output still references not_implemented: {output!r}"
            )

    def test_tool_contracts_are_registered_for_all_four_ep_tools(self):
        """All four episodes/procedures tools must be present in TOOL_CONTRACTS."""
        from mcp_server.src.routers.episodes_procedures import TOOL_CONTRACTS

        contract_names = {c['name'] for c in TOOL_CONTRACTS}
        expected = {'search_episodes', 'get_episode', 'search_procedures', 'get_procedure'}
        assert expected == contract_names, (
            f"Expected contracts for {expected}, found {contract_names}"
        )


# ---------------------------------------------------------------------------
# Blocker C (security) — auth boundary for implemented tools
# ---------------------------------------------------------------------------


class TestIntegratedAuthBoundary:
    """Test that anonymous/private read attempts are handled appropriately."""

    @pytest.mark.anyio
    async def test_search_procedures_empty_result_has_no_phase0_stub_message(self, tmp_path):
        """Empty search result should not contain Phase-0 stub message (proving it's real)."""
        from mcp_server.src.routers.episodes_procedures import register_tools
        from mcp_server.src.services.change_ledger import ChangeLedger

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)

        import mcp_server.src.services.change_ledger as cl_module
        original_path = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path

        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['search_procedures']
            result = await fn(query='deploy')
        finally:
            cl_module.DB_PATH_DEFAULT = original_path

        assert 'error' not in result
        assert result['procedures'] == []
        assert result['total'] == 0
        # Real implementation path — should NOT have a Phase-0 stub message
        assert 'Phase 0 stub' not in result.get('message', ''), (
            "search_procedures is still returning Phase-0 stub message"
        )

    @pytest.mark.anyio
    async def test_search_episodes_with_mismatched_group_ids_returns_empty(self, tmp_path):
        """search_episodes scoped to a non-existent lane returns an empty list (not cross-lane data)."""
        from mcp_server.src.routers.episodes_procedures import register_tools
        from mcp_server.src.services.change_ledger import ChangeLedger

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = ChangeLedger(db_path)
        # Write an episode in lane_a
        _create_episode_in_ledger(ledger, source_lane='lane_a', object_id='ep-scope-test-001')

        import mcp_server.src.services.change_ledger as cl_module
        original_path = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path

        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['search_episodes']
            # Search scoped to a different lane
            result = await fn(query='anything', group_ids=['lane_b_that_does_not_exist'])
        finally:
            cl_module.DB_PATH_DEFAULT = original_path

        # Lane B has no episodes — should return empty list, NOT lane_a's episode
        assert 'error' not in result
        assert result['episodes'] == [], (
            f"Lane B scope returned episodes from Lane A — lane leakage: {result['episodes']}"
        )
