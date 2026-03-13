"""Tests for Phase 0 router architecture: import, initialization, and stub contracts.

Run from mcp_server/ directory:
    pytest tests/test_phase0_routers.py -v
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Import path setup
# ---------------------------------------------------------------------------

_MCP_SRC = Path(__file__).parent.parent / 'src'
if str(_MCP_SRC) not in sys.path:
    sys.path.insert(0, str(_MCP_SRC.parent))
    sys.path.insert(0, str(_MCP_SRC))

_REPO_TESTS = Path(__file__).resolve().parents[2] / 'tests'
if str(_REPO_TESTS) not in sys.path:
    sys.path.insert(0, str(_REPO_TESTS))


def load_graphiti_mcp_server():
    return importlib.import_module('helpers_mcp_import').load_graphiti_mcp_server()


# ---------------------------------------------------------------------------
# Router import tests
# ---------------------------------------------------------------------------


class TestAllRoutersImportable:
    """Verify all four router modules can be imported without errors."""

    def test_memory_router_importable(self):
        try:
            from mcp_server.src.routers import memory
        except ImportError:
            from routers import memory  # type: ignore[no-redef]
        assert hasattr(memory, 'register_tools')

    def test_candidates_router_importable(self):
        try:
            from mcp_server.src.routers import candidates
        except ImportError:
            from routers import candidates  # type: ignore[no-redef]
        assert hasattr(candidates, 'register_tools')

    def test_packs_router_importable(self):
        try:
            from mcp_server.src.routers import packs
        except ImportError:
            from routers import packs  # type: ignore[no-redef]
        assert hasattr(packs, 'register_tools')

    def test_episodes_procedures_router_importable(self):
        try:
            from mcp_server.src.routers import episodes_procedures
        except ImportError:
            from routers import episodes_procedures  # type: ignore[no-redef]
        assert hasattr(episodes_procedures, 'register_tools')


# ---------------------------------------------------------------------------
# MCP server initialization test
# ---------------------------------------------------------------------------


class TestMcpServerInitializesWithAllRouters:
    """Verify the MCP server module imports and registers all routers."""

    def test_mcp_server_importable(self):
        mod = load_graphiti_mcp_server()
        assert mod is not None

    def test_mcp_object_exists(self):
        mod = load_graphiti_mcp_server()
        assert hasattr(mod, 'mcp')
        assert mod.mcp is not None

    def test_debug_flag_attribute_exists(self):
        mod = load_graphiti_mcp_server()
        assert hasattr(mod, '_BICAMERAL_DEBUG_TOOLS')
        assert isinstance(mod._BICAMERAL_DEBUG_TOOLS, bool)

    def test_get_tools_response_precomputed(self):
        mod = load_graphiti_mcp_server()
        assert hasattr(mod, '_GET_TOOLS_RESPONSE')
        assert isinstance(mod._GET_TOOLS_RESPONSE, list)
        assert len(mod._GET_TOOLS_RESPONSE) >= 20


# ---------------------------------------------------------------------------
# Stub return type tests
# ---------------------------------------------------------------------------


class TestAllStubsReturnValidTypes:
    """Verify the branch's current Phase 0 surfaces return coherent typed payloads."""

    @pytest.mark.anyio
    async def test_memory_remember_fact_contract(self):
        from mcp_server.src.routers.memory import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        remember_fact = mock_mcp._tools['remember_fact']
        result = await remember_fact(text='test fact')
        assert result == {
            'status': 'error',
            'error_type': 'validation_error',
            'message': 'missing required field: subject',
        }

    @pytest.mark.anyio
    async def test_memory_get_current_state_contract(self):
        from mcp_server.src.routers.memory import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['get_current_state']
        result = await fn(subject='user')
        assert result == {
            'status': 'ok',
            'facts': [],
        }

    @pytest.mark.anyio
    async def test_memory_get_history_contract(self):
        from mcp_server.src.routers.memory import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['get_history']
        result = await fn(subject='user')
        assert result == {
            'status': 'ok',
            'history': [],
            'scope': 'private',
            'roots_considered': [],
        }

    @pytest.mark.anyio
    async def test_memory_contract_validates_bad_input_instead_of_raising(self):
        from mcp_server.src.routers.memory import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        remember_fact = mock_mcp._tools['remember_fact']
        result = await remember_fact(text=123)
        assert result == {
            'status': 'error',
            'error_type': 'validation_error',
            'message': 'text must be a string',
        }

    @pytest.mark.anyio
    async def test_candidates_list_candidates_contract(self):
        from mcp_server.src.routers.candidates import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['list_candidates']
        result = await fn()
        assert result == {
            'status': 'ok',
            'candidates': [],
        }

    @pytest.mark.anyio
    async def test_candidates_promote_requires_actor_id(self):
        """promote_candidate without actor_id returns unauthorized (auth gate fires first)."""
        from mcp_server.src.routers.candidates import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['promote_candidate']
        result = await fn(candidate_id='cand-001', resolution='supersede')
        assert result['status'] == 'error'
        assert result['error_type'] == 'unauthorized'

    @pytest.mark.anyio
    async def test_candidates_promote_contract(self, monkeypatch):
        """promote_candidate with server-derived principal in allowlist returns not_found for unknown candidate."""
        from mcp_server.src.routers import memory as memory_router
        from mcp_server.src.routers.candidates import register_tools

        monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'system:test')
        # Inject server principal via mock ctx — NOT via caller-supplied actor_id.
        monkeypatch.setattr(memory_router, '_extract_server_principal', lambda ctx: 'system:test')
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['promote_candidate']
        result = await fn(candidate_id='cand-001', resolution='supersede', actor_id='system:test')
        assert result == {
            'status': 'error',
            'error_type': 'not_found',
            'message': 'candidate not found: cand-001',
        }

    @pytest.mark.anyio
    async def test_candidates_reject_requires_actor_id(self):
        """reject_candidate without authenticated context returns unauthorized (auth gate fires first)."""
        from mcp_server.src.routers.candidates import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['reject_candidate']
        # No ctx, no server principal → __anon__ → unauthorized
        result = await fn(candidate_id='cand-001')
        assert result['status'] == 'error'
        assert result['error_type'] == 'unauthorized'

    @pytest.mark.anyio
    async def test_candidates_reject_contract(self, monkeypatch):
        """reject_candidate with server-derived principal in allowlist returns not_found for unknown candidate."""
        from mcp_server.src.routers import memory as memory_router
        from mcp_server.src.routers.candidates import register_tools

        monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'system:test')
        # Inject server principal via mock ctx — NOT via caller-supplied actor_id.
        monkeypatch.setattr(memory_router, '_extract_server_principal', lambda ctx: 'system:test')
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['reject_candidate']
        result = await fn(candidate_id='cand-001', actor_id='system:test')
        assert result == {
            'status': 'error',
            'error_type': 'not_found',
            'message': 'candidate not found: cand-001',
        }

    @pytest.mark.anyio
    async def test_candidates_reject_contract_treats_unmatched_identifier_as_not_found(self, monkeypatch):
        """reject_candidate with authorized principal and unrecognized ID returns not_found."""
        from mcp_server.src.routers import memory as memory_router
        from mcp_server.src.routers.candidates import register_tools

        monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'system:test')
        # Inject server principal via mock ctx — NOT via caller-supplied actor_id.
        monkeypatch.setattr(memory_router, '_extract_server_principal', lambda ctx: 'system:test')
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['reject_candidate']
        result = await fn(candidate_id='Bad Candidate Id', actor_id='system:test')
        assert result == {
            'status': 'error',
            'error_type': 'not_found',
            'message': 'candidate not found: Bad Candidate Id',
        }

    @pytest.mark.anyio
    async def test_packs_list_packs_stub(self):
        from mcp_server.src.routers.packs import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['list_packs']
        result = await fn()
        assert result['packs'] == []
        assert 'status' not in result

    @pytest.mark.anyio
    async def test_packs_get_context_pack_stub(self):
        from mcp_server.src.routers.packs import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['get_context_pack']
        result = await fn(pack_id='my-pack')
        assert result['error'] == 'not_implemented'

    @pytest.mark.anyio
    async def test_packs_get_workflow_pack_stub(self):
        from mcp_server.src.routers.packs import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['get_workflow_pack']
        result = await fn(pack_id='my-pack')
        assert result['error'] == 'not_implemented'

    @pytest.mark.anyio
    async def test_packs_describe_pack_stub(self):
        from mcp_server.src.routers.packs import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['describe_pack']
        result = await fn(pack_id='my-pack')
        assert result['error'] == 'not_implemented'

    @pytest.mark.anyio
    async def test_packs_create_workflow_pack_stub_validates_definition_schema(self):
        from mcp_server.src.routers.packs import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['create_workflow_pack']
        result = await fn(definition={'id': 'test', 'steps': []})
        assert result['error'] == 'validation_error'

    @pytest.mark.anyio
    async def test_packs_create_workflow_pack_stub_rejects_context_scope(self):
        from mcp_server.src.routers.packs import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['create_workflow_pack']
        result = await fn(
            definition={
                'pack_id': 'coding-context',
                'scope': 'context',
                'intent': 'coding defaults',
                'consumer': 'archibald',
                'version': '1.0',
                'context_rules': ['prefer pytest'],
            }
        )
        assert result['error'] == 'validation_error'
        assert 'definition.scope' in (result.get('details') or {}).get('field', '')

    @pytest.mark.anyio
    async def test_episodes_procedures_search_episodes_stub(self):
        from mcp_server.src.routers.episodes_procedures import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['search_episodes']
        result = await fn(query='last deployment')
        assert result['episodes'] == []
        assert 'status' not in result

    @pytest.mark.anyio
    async def test_episodes_procedures_search_episodes_rejects_invalid_time_range(self):
        from mcp_server.src.routers.episodes_procedures import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['search_episodes']
        result = await fn(query='last deployment', time_range={'start': 'not-a-date'})
        assert result['error'] == 'validation_error'

    @pytest.mark.anyio
    async def test_episodes_procedures_search_episodes_rejects_reversed_time_range(self):
        from mcp_server.src.routers.episodes_procedures import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['search_episodes']
        result = await fn(
            query='last deployment',
            time_range={
                'start': '2026-03-11T12:00:00Z',
                'end': '2026-03-10T12:00:00Z',
            },
        )
        assert result['error'] == 'validation_error'

    @pytest.mark.anyio
    async def test_episodes_procedures_get_episode_stub(self):
        from mcp_server.src.routers.episodes_procedures import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['get_episode']
        result = await fn(episode_id='ep-001')
        assert result['error'] == 'not_implemented'

    @pytest.mark.anyio
    async def test_episodes_procedures_get_episode_stub_validates_identifier_pattern(self):
        from mcp_server.src.routers.episodes_procedures import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['get_episode']
        result = await fn(episode_id='Episode With Spaces')
        assert result['error'] == 'validation_error'

    @pytest.mark.anyio
    async def test_episodes_procedures_search_procedures_stub(self):
        from mcp_server.src.routers.episodes_procedures import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['search_procedures']
        result = await fn(query='how to deploy')
        assert result['procedures'] == []
        assert 'status' not in result

    @pytest.mark.anyio
    async def test_episodes_procedures_stub_validates_bad_query_instead_of_raising(self):
        from mcp_server.src.routers.episodes_procedures import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['search_procedures']
        result = await fn(query=123)
        assert result['error'] == 'validation_error'

    @pytest.mark.anyio
    async def test_episodes_procedures_get_procedure_stub(self):
        from mcp_server.src.routers.episodes_procedures import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['get_procedure']
        result = await fn(trigger_or_id='deploy to production')
        assert result['error'] == 'not_implemented'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_mcp() -> MagicMock:
    tools: dict[str, object] = {}
    mock = MagicMock()

    def _tool_decorator(*_args, **kwargs):
        def decorator(fn):
            tools[kwargs.get('name', fn.__name__)] = fn
            return fn

        return decorator

    mock.tool = _tool_decorator
    mock._tools = tools
    return mock
