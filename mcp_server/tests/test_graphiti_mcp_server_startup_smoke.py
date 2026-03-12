"""Startup smoke test that exercises server startup registration path.

This test intentionally exercises the real graphiti_mcp_server startup function and
checks tool registration through FastMCP, rather than only validating a parallel
in-memory registry used by earlier unit-test helpers.
"""

from __future__ import annotations

import asyncio
import sys
from importlib import import_module, reload
from typing import Any
from unittest.mock import AsyncMock

import pytest


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

    module = import_module('mcp_server.src.graphiti_mcp_server')
    return reload(module)


def test_mcp_server_starts_and_registers_runtime_tools(monkeypatch):
    """Smoke-test startup registration with real FastMCP transport setup."""

    module = _load_graphiti_module()
    monkeypatch.setattr(module, 'GraphitiService', _FakeGraphitiService)

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
        assert 'get_tools' in registered_names
        assert module.config.server.transport == 'stdio'
        assert module.graphiti_service is not None
        run_stdio.assert_awaited_once()

    try:
        asyncio.run(_run())
    finally:
        monkeypatch.setattr(sys, 'argv', original_argv)
