"""Startup smoke test that exercises server startup registration path.

This test intentionally exercises the real graphiti_mcp_server startup function and
checks tool registration through FastMCP, rather than only validating a parallel
in-memory registry used by earlier unit-test helpers.
"""

from __future__ import annotations

import asyncio
import sys
from importlib import import_module
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

_MCP_SRC = Path(__file__).parent.parent / 'src'
if str(_MCP_SRC) not in sys.path:
    sys.path.insert(0, str(_MCP_SRC.parent))
    sys.path.insert(0, str(_MCP_SRC))

_REPO_TESTS = Path(__file__).resolve().parents[2] / 'tests'
if str(_REPO_TESTS) not in sys.path:
    sys.path.insert(0, str(_REPO_TESTS))


class _DummyClient:
    pass


class _FakeGraphitiService:
    """Minimal GraphitiService replacement to avoid external DB dependencies."""

    def __init__(self, _config: Any, semaphore_limit: int):
        self.semaphore = asyncio.Semaphore(semaphore_limit)
        self._client = _DummyClient()

    async def initialize(self) -> None:
        return None

    async def get_client(self) -> Any:
        return self._client

    async def get_client_for_group(self, _group_id: str) -> Any:
        return self._client

    def resolve_ontology(self, _group_id: str):
        return (None, '', None, 'permissive')


def _load_graphiti_module():
    pytest.importorskip('mcp')

    sys.modules.pop('mcp_server.src.graphiti_mcp_server', None)
    return import_module('helpers_mcp_import').load_graphiti_mcp_server()


def test_mcp_server_starts_and_registers_runtime_tools(monkeypatch):
    """Smoke-test startup registration and invocation with the real FastMCP registry."""

    module = _load_graphiti_module()
    monkeypatch.setattr(module, 'GraphitiService', _FakeGraphitiService)

    remember_fact = AsyncMock(return_value={'status': 'ok', 'tool': 'remember_fact'})
    get_current_state = AsyncMock(return_value={'status': 'ok', 'tool': 'get_current_state'})
    get_history = AsyncMock(return_value={'status': 'ok', 'tool': 'get_history'})
    monkeypatch.setattr(module._memory_router, 'remember_fact', remember_fact)
    monkeypatch.setattr(module._memory_router, 'get_current_state', get_current_state)
    monkeypatch.setattr(module._memory_router, 'get_history', get_history)

    run_stdio = AsyncMock()
    monkeypatch.setattr(module.mcp, 'run_stdio_async', run_stdio)

    original_argv = list(sys.argv)
    monkeypatch.setattr(sys, 'argv', ['graphiti_mcp_server', '--transport', 'stdio'])

    async def _run() -> None:
        await module.run_mcp_server()

        registered_tools = await module.mcp.list_tools()
        registered_names = {tool.name for tool in registered_tools}

        # Public contract tools are registered with FastMCP at runtime.
        assert set(module._PHASE0_PUBLIC_TOOL_CALLABLES).issubset(registered_names)
        assert {'remember_fact', 'get_current_state', 'get_history'}.issubset(registered_names)
        assert 'remember_fact_tool' not in registered_names
        assert 'get_current_state_tool' not in registered_names
        assert 'get_history_tool' not in registered_names

        remember_result = await module.mcp._tool_manager.call_tool(  # noqa: SLF001
            'remember_fact',
            {'text': 'Tabs over spaces', 'hint': {'subject': 'editor prefs'}},
        )
        current_state_result = await module.mcp._tool_manager.call_tool(  # noqa: SLF001
            'get_current_state',
            {'subject': 'editor prefs', 'predicate': 'theme', 'scope': 'public'},
        )
        history_result = await module.mcp._tool_manager.call_tool(  # noqa: SLF001
            'get_history',
            {'subject': 'editor prefs', 'predicate': 'theme', 'scope': 'public'},
        )

        assert remember_result == {'status': 'ok', 'tool': 'remember_fact'}
        assert current_state_result == {'status': 'ok', 'tool': 'get_current_state'}
        assert history_result == {'status': 'ok', 'tool': 'get_history'}
        remember_fact.assert_awaited_once_with(
            text='Tabs over spaces',
            hint={'subject': 'editor prefs'},
        )
        get_current_state.assert_awaited_once_with(
            subject='editor prefs',
            predicate='theme',
            scope='public',
        )
        get_history.assert_awaited_once_with(
            subject='editor prefs',
            predicate='theme',
            scope='public',
        )

        assert 'get_tools' in registered_names
        assert module.config.server.transport == 'stdio'
        assert module.graphiti_service is not None
        run_stdio.assert_awaited_once()

    try:
        asyncio.run(_run())
    finally:
        monkeypatch.setattr(sys, 'argv', original_argv)
