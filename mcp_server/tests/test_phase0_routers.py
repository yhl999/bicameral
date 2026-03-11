"""Tests for Phase 0 router architecture: import, initialization, and stub types.

Run from mcp_server/ directory:
    pytest tests/test_phase0_routers.py -v
"""

from __future__ import annotations

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
        """graphiti_mcp_server can be imported without errors."""
        try:
            import mcp_server.src.graphiti_mcp_server as _mod
        except ImportError:
            import graphiti_mcp_server as _mod  # type: ignore[no-redef]
        assert _mod is not None

    def test_mcp_object_exists(self):
        """The 'mcp' FastMCP instance exists in the module."""
        try:
            from mcp_server.src import graphiti_mcp_server as _mod
        except ImportError:
            import graphiti_mcp_server as _mod  # type: ignore[no-redef]
        assert hasattr(_mod, 'mcp')
        assert _mod.mcp is not None

    def test_debug_flag_attribute_exists(self):
        """_BICAMERAL_DEBUG_TOOLS attribute exists as a bool."""
        try:
            from mcp_server.src import graphiti_mcp_server as _mod
        except ImportError:
            import graphiti_mcp_server as _mod  # type: ignore[no-redef]
        assert hasattr(_mod, '_BICAMERAL_DEBUG_TOOLS')
        assert isinstance(_mod._BICAMERAL_DEBUG_TOOLS, bool)

    def test_get_tools_response_precomputed(self):
        """_GET_TOOLS_RESPONSE is precomputed at module load (not None or empty)."""
        try:
            from mcp_server.src import graphiti_mcp_server as _mod
        except ImportError:
            import graphiti_mcp_server as _mod  # type: ignore[no-redef]
        assert hasattr(_mod, '_GET_TOOLS_RESPONSE')
        assert isinstance(_mod._GET_TOOLS_RESPONSE, list)
        assert len(_mod._GET_TOOLS_RESPONSE) >= 16


# ---------------------------------------------------------------------------
# Stub return type tests
# ---------------------------------------------------------------------------


class TestAllStubsReturnValidTypes:
    """Verify all stub methods return dict (not raise) and include expected keys."""

    @pytest.mark.asyncio
    async def test_memory_remember_fact_stub(self):
        try:
            from mcp_server.src.routers.memory import register_tools
        except ImportError:
            from routers.memory import register_tools  # type: ignore[no-redef]

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        remember_fact = mock_mcp._tools['remember_fact']
        result = await remember_fact(text='test fact')
        assert isinstance(result, dict)
        assert 'status' in result

    @pytest.mark.asyncio
    async def test_memory_get_current_state_stub(self):
        try:
            from mcp_server.src.routers.memory import register_tools
        except ImportError:
            from routers.memory import register_tools  # type: ignore[no-redef]

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['get_current_state']
        result = await fn(subject='user')
        assert isinstance(result, dict)
        assert 'facts' in result
        assert isinstance(result['facts'], list)

    @pytest.mark.asyncio
    async def test_memory_get_history_stub(self):
        try:
            from mcp_server.src.routers.memory import register_tools
        except ImportError:
            from routers.memory import register_tools  # type: ignore[no-redef]

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['get_history']
        result = await fn(subject='user')
        assert isinstance(result, dict)
        assert 'history' in result
        assert isinstance(result['history'], list)

    @pytest.mark.asyncio
    async def test_candidates_list_candidates_stub(self):
        try:
            from mcp_server.src.routers.candidates import register_tools
        except ImportError:
            from routers.candidates import register_tools  # type: ignore[no-redef]

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['list_candidates']
        result = await fn()
        assert isinstance(result, dict)
        assert 'candidates' in result
        assert isinstance(result['candidates'], list)

    @pytest.mark.asyncio
    async def test_candidates_promote_stub(self):
        try:
            from mcp_server.src.routers.candidates import register_tools
        except ImportError:
            from routers.candidates import register_tools  # type: ignore[no-redef]

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['promote_candidate']
        result = await fn(candidate_id='cand-001', resolution='looks good')
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_candidates_reject_stub(self):
        try:
            from mcp_server.src.routers.candidates import register_tools
        except ImportError:
            from routers.candidates import register_tools  # type: ignore[no-redef]

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['reject_candidate']
        result = await fn(candidate_id='cand-001')
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_packs_list_packs_stub(self):
        try:
            from mcp_server.src.routers.packs import register_tools
        except ImportError:
            from routers.packs import register_tools  # type: ignore[no-redef]

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['list_packs']
        result = await fn()
        assert isinstance(result, dict)
        assert 'packs' in result
        assert isinstance(result['packs'], list)

    @pytest.mark.asyncio
    async def test_packs_get_context_pack_stub(self):
        try:
            from mcp_server.src.routers.packs import register_tools
        except ImportError:
            from routers.packs import register_tools  # type: ignore[no-redef]

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['get_context_pack']
        result = await fn(pack_id='my-pack')
        assert isinstance(result, dict)
        assert 'items' in result

    @pytest.mark.asyncio
    async def test_packs_get_workflow_pack_stub(self):
        try:
            from mcp_server.src.routers.packs import register_tools
        except ImportError:
            from routers.packs import register_tools  # type: ignore[no-redef]

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['get_workflow_pack']
        result = await fn(pack_id='my-pack')
        assert isinstance(result, dict)
        assert 'steps' in result

    @pytest.mark.asyncio
    async def test_packs_describe_pack_stub(self):
        try:
            from mcp_server.src.routers.packs import register_tools
        except ImportError:
            from routers.packs import register_tools  # type: ignore[no-redef]

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['describe_pack']
        result = await fn(pack_id='my-pack')
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_packs_create_workflow_pack_stub(self):
        try:
            from mcp_server.src.routers.packs import register_tools
        except ImportError:
            from routers.packs import register_tools  # type: ignore[no-redef]

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['create_workflow_pack']
        result = await fn(definition={'id': 'test', 'steps': []})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_episodes_procedures_search_episodes_stub(self):
        try:
            from mcp_server.src.routers.episodes_procedures import register_tools
        except ImportError:
            from routers.episodes_procedures import register_tools  # type: ignore[no-redef]

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['search_episodes']
        result = await fn(query='last deployment')
        assert isinstance(result, dict)
        assert 'episodes' in result
        assert isinstance(result['episodes'], list)

    @pytest.mark.asyncio
    async def test_episodes_procedures_get_episode_stub(self):
        try:
            from mcp_server.src.routers.episodes_procedures import register_tools
        except ImportError:
            from routers.episodes_procedures import register_tools  # type: ignore[no-redef]

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['get_episode']
        result = await fn(episode_id='ep-001')
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_episodes_procedures_search_procedures_stub(self):
        try:
            from mcp_server.src.routers.episodes_procedures import register_tools
        except ImportError:
            from routers.episodes_procedures import register_tools  # type: ignore[no-redef]

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['search_procedures']
        result = await fn(query='how to deploy')
        assert isinstance(result, dict)
        assert 'procedures' in result
        assert isinstance(result['procedures'], list)

    @pytest.mark.asyncio
    async def test_episodes_procedures_get_procedure_stub(self):
        try:
            from mcp_server.src.routers.episodes_procedures import register_tools
        except ImportError:
            from routers.episodes_procedures import register_tools  # type: ignore[no-redef]

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['get_procedure']
        result = await fn(trigger_or_id='deploy to production')
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_mcp() -> MagicMock:
    """Create a mock MCP object that captures registered tools."""
    tools: dict = {}

    mock = MagicMock()

    def _tool_decorator():
        def decorator(fn):
            tools[fn.__name__] = fn
            return fn
        return decorator

    mock.tool = _tool_decorator
    mock._tools = tools
    return mock
