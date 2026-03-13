"""Focused contract tests for the integrated MCP surface.

Proves:
  1. Memory router (get_current_state / get_history) returns the documented
     {'status': 'ok', ...} envelope, not the old {'message': ...} shape.
  2. episodes/procedures integrated search responses have no spurious 'message'
     key and always carry the pagination fields.
  3. create_workflow_pack TOOL_CONTRACTS example validates against the
     PackRegistryService schema rules (version semver, required fields, steps).
  4. Ledger context-manager cleanup in get_episode / get_procedure /
     search_procedures does not regress behaviour (no functional change).
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------


def _make_mock_mcp() -> MagicMock:
    tools: dict[str, Any] = {}
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


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# A1 — memory router response contract (status envelope)
# ---------------------------------------------------------------------------


class TestMemoryResponseContract:
    """Prove get_current_state and get_history use {'status': 'ok', ...} envelope."""

    @pytest.fixture(autouse=True)
    def _isolated_ledger(self, tmp_path, monkeypatch):
        db_path = tmp_path / 'change_ledger.db'
        monkeypatch.setenv('BICAMERAL_CHANGE_LEDGER_PATH', str(db_path))
        # Reset cached ledger so tests get a fresh one
        import mcp_server.src.routers.memory as mem_module
        mem_module._change_ledger = None
        yield db_path
        mem_module._change_ledger = None

    def test_get_current_state_returns_status_ok_not_message(self, _isolated_ledger):
        """get_current_state must return {'status': 'ok', 'facts': [...]} not {'message': ...}."""
        from mcp_server.src.routers.memory import get_current_state

        result = _run(get_current_state(subject='test-subject'))

        assert 'status' in result, f"Missing 'status' key: {result}"
        assert result['status'] == 'ok', f"Expected status='ok', got: {result}"
        assert 'facts' in result, f"Missing 'facts' key: {result}"
        assert isinstance(result['facts'], list)
        # Must NOT use the old 'message' envelope
        assert 'message' not in result, (
            f"get_current_state returned legacy 'message' envelope (contract drift): {result}"
        )

    def test_get_history_returns_status_ok_with_scope_and_roots(self, _isolated_ledger):
        """get_history must return {'status': 'ok', 'history': [...], 'scope': ..., 'roots_considered': [...]}."""
        from mcp_server.src.routers.memory import get_history

        result = _run(get_history(subject='test-subject'))

        assert 'status' in result, f"Missing 'status' key: {result}"
        assert result['status'] == 'ok', f"Expected status='ok', got: {result}"
        assert 'history' in result, f"Missing 'history' key: {result}"
        assert 'scope' in result, f"Missing 'scope' key: {result}"
        assert 'roots_considered' in result, f"Missing 'roots_considered' key: {result}"
        assert isinstance(result['roots_considered'], list)
        # Must NOT use the old 'message' envelope
        assert 'message' not in result, (
            f"get_history returned legacy 'message' envelope (contract drift): {result}"
        )

    def test_get_current_state_error_returns_status_error(self, _isolated_ledger):
        """Validation errors must use {'status': 'error', ...} not ErrorResponse dict."""
        from mcp_server.src.routers.memory import get_current_state

        result = _run(get_current_state(subject=''))  # empty subject → error

        assert result.get('status') == 'error'
        assert 'error_type' in result or 'message' in result


# ---------------------------------------------------------------------------
# A2 — episodes/procedures integrated search response shape
# ---------------------------------------------------------------------------


class TestEpisodeProcedureResponseShape:
    """Prove search_episodes / search_procedures return the integrated pagination shape."""

    def _make_ledger(self, db_path: Path):
        from mcp_server.src.services.change_ledger import ChangeLedger
        return ChangeLedger(db_path)

    @pytest.mark.anyio
    async def test_search_episodes_has_pagination_fields_not_message(self, tmp_path):
        """search_episodes response must have pagination fields and no 'message' on success."""
        from mcp_server.src.routers.episodes_procedures import register_tools
        from mcp_server.src.services.change_ledger import ChangeLedger

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)

        import mcp_server.src.services.change_ledger as cl_module
        original = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path
        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['search_episodes']
            result = await fn(query='anything')
        finally:
            cl_module.DB_PATH_DEFAULT = original

        # Integrated surface must always return pagination envelope
        assert 'episodes' in result, f"Missing 'episodes': {result}"
        assert 'limit' in result, f"Missing 'limit': {result}"
        assert 'offset' in result, f"Missing 'offset': {result}"
        assert 'total' in result, f"Missing 'total': {result}"
        assert 'has_more' in result, f"Missing 'has_more': {result}"
        assert 'next_offset' in result, f"Missing 'next_offset': {result}"
        # Must NOT carry 'message' on the success path
        assert 'message' not in result, (
            f"search_episodes integrated surface leaked 'message' key (contract drift): {result}"
        )

    @pytest.mark.anyio
    async def test_search_procedures_has_pagination_fields_not_message(self, tmp_path):
        """search_procedures response must have pagination fields and no 'message' on success."""
        from mcp_server.src.routers.episodes_procedures import register_tools

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)

        import mcp_server.src.services.change_ledger as cl_module
        original = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path
        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['search_procedures']
            result = await fn(query='anything')
        finally:
            cl_module.DB_PATH_DEFAULT = original

        assert 'procedures' in result, f"Missing 'procedures': {result}"
        assert 'limit' in result, f"Missing 'limit': {result}"
        assert 'total' in result, f"Missing 'total': {result}"
        assert 'message' not in result, (
            f"search_procedures integrated surface leaked 'message' key (contract drift): {result}"
        )


# ---------------------------------------------------------------------------
# B — create_workflow_pack example validates against schema rules
# ---------------------------------------------------------------------------


class TestCreateWorkflowPackExampleConsistency:
    """Prove the TOOL_CONTRACTS example for create_workflow_pack is a valid input."""

    @pytest.fixture
    def registry_path(self, tmp_path, monkeypatch):
        path = tmp_path / 'user_packs.json'
        monkeypatch.setenv('BICAMERAL_USER_PACK_REGISTRY_PATH', str(path))
        monkeypatch.delenv('BICAMERAL_PACK_REGISTRY_PATH', raising=False)
        return path

    def test_tool_contracts_example_is_valid_create_input(self, registry_path):
        """The documented example must actually be accepted by create_workflow_pack."""
        from mcp_server.src.routers import packs
        from mcp_server.src.routers.packs import TOOL_CONTRACTS

        # Find the create_workflow_pack contract
        contract = next(c for c in TOOL_CONTRACTS if c['name'] == 'create_workflow_pack')
        assert contract['examples'], "create_workflow_pack contract has no examples"

        example = contract['examples'][0]
        definition_arg = example['definition']

        result = asyncio.run(packs.create_workflow_pack(definition_arg))

        assert 'error' not in result, (
            f"TOOL_CONTRACTS example for create_workflow_pack is invalid: {result}\n"
            f"Example was: {definition_arg}"
        )
        assert result.get('scope') == 'workflow'
        assert result.get('version', '').count('.') == 2, (
            f"Version must be semver X.Y.Z, got: {result.get('version')!r}"
        )

    def test_tool_contracts_example_version_is_semver(self):
        """Version in the example must be X.Y.Z format (three-part semver)."""
        import re
        from mcp_server.src.routers.packs import TOOL_CONTRACTS

        contract = next(c for c in TOOL_CONTRACTS if c['name'] == 'create_workflow_pack')
        example_def = contract['examples'][0]['definition']
        version = example_def.get('version', '')
        assert re.match(r'^\d+\.\d+\.\d+$', version), (
            f"Example version {version!r} is not semver X.Y.Z — "
            f"PackRegistryService rejects non-three-part versions"
        )

    def test_tool_contracts_example_has_predicates(self):
        """Example must include 'predicates' (required by PackRegistryService)."""
        from mcp_server.src.routers.packs import TOOL_CONTRACTS

        contract = next(c for c in TOOL_CONTRACTS if c['name'] == 'create_workflow_pack')
        example_def = contract['examples'][0]['definition']
        assert 'predicates' in example_def, (
            "Example is missing required 'predicates' field"
        )
        assert example_def['predicates'], "Example 'predicates' must be non-empty"

    def test_tool_contracts_example_has_valid_definition_steps(self):
        """Workflow pack example must have definition.steps with step/action objects."""
        from mcp_server.src.routers.packs import TOOL_CONTRACTS

        contract = next(c for c in TOOL_CONTRACTS if c['name'] == 'create_workflow_pack')
        example_def = contract['examples'][0]['definition']
        inner_def = example_def.get('definition', {})
        steps = inner_def.get('steps', [])
        assert steps, "Example workflow pack must have definition.steps (required for scope=workflow)"
        for i, step in enumerate(steps):
            assert isinstance(step, dict), f"steps[{i}] must be a dict"
            assert step.get('step'), f"steps[{i}] missing 'step' field"
            assert step.get('action'), f"steps[{i}] missing 'action' field"


# ---------------------------------------------------------------------------
# C — ledger lifecycle regression (context manager cleanup)
# ---------------------------------------------------------------------------


class TestLedgerLifecycleRegression:
    """Prove ledger context-manager cleanup does not break get_episode / get_procedure
    / search_procedures behaviour."""

    def _make_episode(self, ledger, *, title='Test ep', object_id='ep-cm-001', source_lane=None):
        from mcp_server.src.models.typed_memory import Episode

        ep = Episode.model_validate({
            'object_id': object_id,
            'root_id': object_id,
            'title': title,
            'summary': 'content',
            'source_lane': source_lane,
            'policy_scope': 'private',
            'visibility_scope': 'private',
            'evidence_refs': [{
                'kind': 'event_log',
                'source_system': 'test',
                'locator': {'system': 'test', 'stream': 's', 'event_id': 'e'},
            }],
        })
        ledger.append_event('assert', actor_id='test', reason='test', payload=ep)
        return ep

    def _make_procedure(self, ledger, *, trigger='how to run tests', name='Test proc'):
        from mcp_server.src.services.procedure_service import ProcedureService
        svc = ProcedureService(ledger)
        return svc.create_procedure(
            actor_id='test',
            name=name,
            trigger=trigger,
            steps=['Step 1'],
            expected_outcome='Done',
            evidence_refs=[{
                'uri': 'mem://test/1',
                'kind': 'message',
                'content_summary': 'test',
            }],
            promote=True,
        )

    @pytest.mark.anyio
    async def test_get_episode_context_manager_returns_same_result(self, tmp_path):
        """get_episode with context-managed ledger returns the episode correctly."""
        from mcp_server.src.routers.episodes_procedures import register_tools
        from mcp_server.src.services.change_ledger import ChangeLedger

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = ChangeLedger(db_path)
        self._make_episode(ledger, object_id='ep-cm-001', title='Context managed episode')

        import mcp_server.src.services.change_ledger as cl_module
        original = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path
        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['get_episode']
            result = await fn(episode_id='ep-cm-001')
        finally:
            cl_module.DB_PATH_DEFAULT = original

        assert 'error' not in result, f"get_episode regressed after ledger lifecycle fix: {result}"
        assert result.get('title') == 'Context managed episode' or result.get('object_id') == 'ep-cm-001'

    @pytest.mark.anyio
    async def test_get_procedure_context_manager_returns_same_result(self, tmp_path):
        """get_procedure with context-managed ledger finds a procedure by trigger."""
        from mcp_server.src.routers.episodes_procedures import register_tools
        from mcp_server.src.services.change_ledger import ChangeLedger

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = ChangeLedger(db_path)
        self._make_procedure(ledger, trigger='how to run tests cm', name='CM Procedure')

        import mcp_server.src.services.change_ledger as cl_module
        original = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path
        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['get_procedure']
            result = await fn(trigger_or_id='how to run tests cm')
        finally:
            cl_module.DB_PATH_DEFAULT = original

        assert 'error' not in result, f"get_procedure regressed after ledger lifecycle fix: {result}"
        assert result.get('name') == 'CM Procedure'

    @pytest.mark.anyio
    async def test_search_procedures_context_manager_returns_same_result(self, tmp_path):
        """search_procedures with context-managed ledger returns procedures normally."""
        from mcp_server.src.routers.episodes_procedures import register_tools
        from mcp_server.src.services.change_ledger import ChangeLedger

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = ChangeLedger(db_path)
        self._make_procedure(ledger, trigger='how to deploy cm', name='CM Deploy Procedure')

        import mcp_server.src.services.change_ledger as cl_module
        original = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path
        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['search_procedures']
            result = await fn(query='deploy cm')
        finally:
            cl_module.DB_PATH_DEFAULT = original

        assert 'error' not in result, f"search_procedures regressed after ledger lifecycle fix: {result}"
        assert len(result['procedures']) >= 1
        names = [p.get('name', '') for p in result['procedures']]
        assert any('CM Deploy Procedure' in n for n in names), f"Expected CM Deploy Procedure, got: {names}"

    @pytest.mark.anyio
    async def test_get_episode_not_found_still_works_after_cleanup(self, tmp_path):
        """get_episode not_found path works correctly after context-manager refactor."""
        from mcp_server.src.routers.episodes_procedures import register_tools

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)

        import mcp_server.src.services.change_ledger as cl_module
        original = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path
        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['get_episode']
            result = await fn(episode_id='does-not-exist')
        finally:
            cl_module.DB_PATH_DEFAULT = original

        assert result.get('error') in ('not_found', 'retrieval_error'), (
            f"Expected not_found/retrieval_error after lifecycle fix, got: {result}"
        )
