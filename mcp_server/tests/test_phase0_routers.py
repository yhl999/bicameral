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
        assert hasattr(memory, 'TOOL_CONTRACTS')

    def test_candidates_router_importable(self):
        try:
            from mcp_server.src.routers import candidates
        except ImportError:
            from routers import candidates  # type: ignore[no-redef]
        assert hasattr(candidates, 'register_tools')
        assert hasattr(candidates, 'TOOL_CONTRACTS')

    def test_packs_router_importable(self):
        try:
            from mcp_server.src.routers import packs
        except ImportError:
            from routers import packs  # type: ignore[no-redef]
        assert hasattr(packs, 'register_tools')
        assert hasattr(packs, 'TOOL_CONTRACTS')

    def test_episodes_procedures_router_importable(self):
        try:
            from mcp_server.src.routers import episodes_procedures
        except ImportError:
            from routers import episodes_procedures  # type: ignore[no-redef]
        assert hasattr(episodes_procedures, 'register_tools')
        assert hasattr(episodes_procedures, 'TOOL_CONTRACTS')


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

    def test_router_contracts_match_registered_router_tools(self):
        mod = load_graphiti_mcp_server()
        assert {tool['name'] for tool in mod._ROUTER_TOOL_CONTRACTS} == set(mod._REGISTERED_ROUTER_TOOLS)


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
        """list_candidates requires auth — anonymous callers must be rejected."""
        from mcp_server.src.routers.candidates import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['list_candidates']
        result = await fn()
        assert result['status'] == 'error'
        assert result['error_type'] == 'unauthorized'

    @pytest.mark.anyio
    async def test_candidates_list_candidates_authorized_contract(self, tmp_path, monkeypatch):
        """list_candidates returns an authenticated wrapper response."""
        from mcp_server.src.routers import candidates as candidates_router
        from mcp_server.src.routers import memory as memory_router
        from mcp_server.src.routers.candidates import register_tools

        # Isolate candidates DB to avoid contamination from other tests
        monkeypatch.setenv('BICAMERAL_CANDIDATES_DB_PATH', str(tmp_path / 'candidates.db'))
        monkeypatch.setattr(candidates_router, '_candidate_store', None)
        monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'system:test')
        monkeypatch.setattr(memory_router, '_extract_server_principal', lambda ctx: 'system:test')
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['list_candidates']
        result = await fn()
        assert result['status'] == 'ok'
        assert result['candidates'] == []

    @pytest.mark.anyio
    async def test_candidates_promote_requires_actor_id(self):
        """promote_candidate without authenticated context returns unauthorized."""
        from mcp_server.src.routers.candidates import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['promote_candidate']
        result = await fn(candidate_id='cand-001', resolution='supersede')
        assert result['status'] == 'error'
        assert result['error_type'] == 'unauthorized'

    @pytest.mark.anyio
    async def test_candidates_promote_contract(self, tmp_path, monkeypatch):
        """promote_candidate with server-derived principal in allowlist returns not_found for unknown candidate."""
        from mcp_server.src.routers import candidates as candidates_router
        from mcp_server.src.routers import memory as memory_router
        from mcp_server.src.routers.candidates import register_tools

        # Isolate candidates DB
        monkeypatch.setenv('BICAMERAL_CANDIDATES_DB_PATH', str(tmp_path / 'candidates.db'))
        monkeypatch.setattr(candidates_router, '_candidate_store', None)
        monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'system:test')
        monkeypatch.setattr(memory_router, '_extract_server_principal', lambda ctx: 'system:test')
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['promote_candidate']
        result = await fn(candidate_id='cand-001', resolution='supersede')
        assert result['status'] == 'error'
        assert result['error_type'] == 'not_found'

    @pytest.mark.anyio
    async def test_candidates_reject_requires_actor_id(self):
        """reject_candidate without authenticated context returns unauthorized."""
        from mcp_server.src.routers.candidates import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['reject_candidate']
        result = await fn(candidate_id='cand-001')
        assert result['status'] == 'error'
        assert result['error_type'] == 'unauthorized'

    @pytest.mark.anyio
    async def test_candidates_reject_stub_requires_identifier(self):
        pass

    @pytest.mark.anyio
    async def test_candidates_reject_contract(self, monkeypatch):
        """reject_candidate validates the candidate identifier before proceeding."""
        from mcp_server.src.routers import memory as memory_router
        from mcp_server.src.routers.candidates import register_tools

        monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'system:test')
        monkeypatch.setattr(memory_router, '_extract_server_principal', lambda ctx: 'system:test')
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['reject_candidate']
        result = await fn(candidate_id='')
        assert result['status'] == 'error'
        assert result['error_type'] == 'validation_error'

    @pytest.mark.anyio
    async def test_candidates_reject_contract_treats_unmatched_identifier_as_not_found(self, monkeypatch):
        """reject_candidate with authorized principal and unrecognized ID returns not_found."""
        from mcp_server.src.routers import memory as memory_router
        from mcp_server.src.routers.candidates import register_tools

        monkeypatch.setenv('BICAMERAL_TRUSTED_ACTOR_IDS', 'system:test')
        monkeypatch.setattr(memory_router, '_extract_server_principal', lambda ctx: 'system:test')
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['reject_candidate']
        result = await fn(candidate_id='Bad Candidate Id')
        assert result['status'] == 'error'
        assert result['error_type'] == 'not_found'

    @pytest.mark.anyio
    async def test_packs_list_packs_stub(self):
        from mcp_server.src.routers.packs import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['list_packs']
        result = await fn()
        assert isinstance(result, list)

    @pytest.mark.anyio
    async def test_packs_list_packs_invalid_filter_returns_validation_error(self):
        from mcp_server.src.routers.packs import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['list_packs']
        result = await fn(filter={'scope': 'not-a-scope'})
        assert result['error'] == 'validation_error'

    @pytest.mark.anyio
    async def test_packs_get_context_pack_stub(self):
        from mcp_server.src.routers.packs import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['get_context_pack']
        result = await fn(pack_id='missing-pack')
        assert isinstance(result, dict)
        assert 'error' in result

    @pytest.mark.anyio
    async def test_packs_get_workflow_pack_stub(self):
        from mcp_server.src.routers.packs import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['get_workflow_pack']
        result = await fn(pack_id='missing-pack')
        assert isinstance(result, dict)
        assert 'error' in result

    @pytest.mark.anyio
    async def test_packs_describe_pack_stub(self):
        from mcp_server.src.routers.packs import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['describe_pack']
        result = await fn(pack_id='context-vc-deal-brief')
        assert result.get('pack_registry', {}).get('id') == 'context-vc-deal-brief'

    @pytest.mark.anyio
    async def test_packs_create_workflow_pack_stub(self, tmp_path, monkeypatch):
        from mcp_server.src.routers.packs import register_tools

        monkeypatch.setenv('BICAMERAL_USER_PACK_REGISTRY_PATH', str(tmp_path / 'runtime_user_pack_registry.json'))
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['create_workflow_pack']
        result = await fn(
            definition={
                'id': 'workflow-test-pack',
                'scope': 'workflow',
                'intent': 'verifier',
                'consumer': 'planner',
                'predicates': ['risk'],
                'definition': {'steps': [{'step': 'review', 'action': 'inspect risk facts'}]},
            }
        )
        assert result.get('id') == 'workflow-test-pack'

    @pytest.mark.anyio
    async def test_packs_create_workflow_pack_stub_rejects_context_scope(self, tmp_path, monkeypatch):
        from mcp_server.src.routers.packs import register_tools

        monkeypatch.setenv('BICAMERAL_USER_PACK_REGISTRY_PATH', str(tmp_path / 'runtime_user_pack_registry.json'))
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['create_workflow_pack']
        result = await fn(
            definition={
                'pack_id': 'coding-context',
                'scope': 'context',
                'intent': 'coding defaults',
                'consumer': 'archibald',
                'version': '1.0.0',
                'predicates': ['rule'],
                'definition': {'steps': [{'step': 'review', 'action': 'inspect risk facts'}]},
            }
        )
        assert result['error'] == 'validation_error'
        assert 'scope=workflow' in result.get('message', '')

    @pytest.mark.anyio
    async def test_packs_router_accepts_dotted_pack_ids_for_describe_and_get(self, tmp_path, monkeypatch):
        from mcp_server.src.routers.packs import register_tools

        monkeypatch.setenv('BICAMERAL_USER_PACK_REGISTRY_PATH', str(tmp_path / 'runtime_user_pack_registry.json'))
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)

        create = mock_mcp._tools['create_workflow_pack']
        describe = mock_mcp._tools['describe_pack']
        get_workflow = mock_mcp._tools['get_workflow_pack']

        created = await create(
            definition={
                'pack_id': 'workflow.earnings.review',
                'scope': 'workflow',
                'intent': 'verifier',
                'consumer': 'planner',
                'version': '1.0.0',
                'predicates': ['risk'],
                'definition': {'steps': [{'step': 'review', 'action': 'inspect risk facts'}]},
            }
        )
        assert created.get('id') == 'workflow.earnings.review'

        described = await describe(pack_id='workflow.earnings.review')
        assert described.get('pack_id') == 'workflow.earnings.review'

        fetched = await get_workflow(pack_id='workflow.earnings.review')
        assert fetched.get('pack_id') == 'workflow.earnings.review'

    @pytest.mark.anyio
    async def test_packs_create_workflow_pack_stub_rejects_scope_literal_type(self, tmp_path, monkeypatch):
        from mcp_server.src.routers.packs import register_tools

        monkeypatch.setenv('BICAMERAL_USER_PACK_REGISTRY_PATH', str(tmp_path / 'runtime_user_pack_registry.json'))
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['create_workflow_pack']
        result = await fn(
            definition={
                'pack_id': 'workflow-invalid-scope',
                'scope': 'type',
                'intent': 'verifier',
                'consumer': 'planner',
                'version': '1.0.0',
                'predicates': ['risk'],
                'definition': {'steps': [{'step': 'review', 'action': 'inspect risk facts'}]},
            }
        )
        assert result['error'] == 'validation_error'
        assert "invalid scope 'type'" in result.get('message', '')

    @pytest.mark.anyio
    async def test_packs_create_workflow_pack_stub_requires_steps(self, tmp_path, monkeypatch):
        from mcp_server.src.routers.packs import register_tools

        monkeypatch.setenv('BICAMERAL_USER_PACK_REGISTRY_PATH', str(tmp_path / 'runtime_user_pack_registry.json'))
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['create_workflow_pack']
        result = await fn(
            definition={
                'pack_id': 'workflow-missing-steps',
                'scope': 'workflow',
                'intent': 'verifier',
                'consumer': 'planner',
                'version': '1.0.0',
                'predicates': ['risk'],
                'definition': {'instructions': 'Review the risk facts.'},
            }
        )
        assert result['error'] == 'validation_error'
        assert 'definition.steps' in result.get('message', '')

    @pytest.mark.anyio
    async def test_episodes_procedures_search_episodes_stub(self):
        from mcp_server.src.routers.episodes_procedures import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['search_episodes']
        result = await fn(query='last deployment')
        assert result['episodes'] == []
        assert result['limit'] == 10
        assert result['offset'] == 0
        assert result['total'] == 0
        assert result['has_more'] is False
        assert result['next_offset'] is None
        # Fully-integrated implementation: result includes status:ok envelope
        assert result.get('status') == 'ok'

    @pytest.mark.anyio
    async def test_episodes_procedures_search_episodes_rejects_invalid_time_range(self):
        from mcp_server.src.routers.episodes_procedures import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['search_episodes']
        result = await fn(query='last deployment', time_range={'start': 'not-a-date'})
        assert result['error'] == 'validation_error'

    @pytest.mark.anyio
    async def test_episodes_procedures_search_episodes_accepts_empty_query_and_pagination(self):
        from mcp_server.src.routers.episodes_procedures import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['search_episodes']
        result = await fn(query='', limit=3, offset=2)
        assert result['episodes'] == []
        assert result['limit'] == 3
        assert result['offset'] == 2

    @pytest.mark.anyio
    async def test_episodes_procedures_search_episodes_rejects_invalid_group_ids(self):
        from mcp_server.src.routers.episodes_procedures import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['search_episodes']
        result = await fn(query='last deployment', group_ids=['ok', ''])
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
    async def test_episodes_procedures_get_episode_not_found_on_empty_ledger(self):
        """get_episode returns not_found (real impl) for a non-existent ID, not not_implemented."""
        from mcp_server.src.routers.episodes_procedures import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['get_episode']
        result = await fn(episode_id='ep-001')
        # Real implementation: not_found when episode doesn't exist
        assert result['error'] in ('not_found', 'retrieval_error'), (
            f"Expected not_found or retrieval_error, got: {result}"
        )

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
        assert result['limit'] == 10
        assert result['offset'] == 0
        assert result['total'] == 0
        assert result['has_more'] is False
        assert result['next_offset'] is None
        # Fully-integrated implementation: result includes status:ok envelope
        assert result.get('status') == 'ok'

    @pytest.mark.anyio
    async def test_episodes_procedures_stub_validates_bad_query_instead_of_raising(self):
        from mcp_server.src.routers.episodes_procedures import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['search_procedures']
        result = await fn(query=123)
        assert result['error'] == 'validation_error'

    @pytest.mark.anyio
    async def test_episodes_procedures_search_procedures_accepts_empty_query_and_pagination(self):
        from mcp_server.src.routers.episodes_procedures import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['search_procedures']
        result = await fn(query='', include_all=True, limit=5, offset=1)
        assert result['procedures'] == []
        assert result['limit'] == 5
        assert result['offset'] == 1

    @pytest.mark.anyio
    async def test_episodes_procedures_search_procedures_rejects_invalid_offset(self):
        from mcp_server.src.routers.episodes_procedures import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['search_procedures']
        result = await fn(query='deploy', offset=-1)
        assert result['error'] == 'validation_error'

    @pytest.mark.anyio
    async def test_episodes_procedures_get_procedure_not_found_on_empty_ledger(self):
        """get_procedure returns not_found (real impl) when no procedure matches, not not_implemented."""
        from mcp_server.src.routers.episodes_procedures import register_tools

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp)
        fn = mock_mcp._tools['get_procedure']
        result = await fn(trigger_or_id='deploy to production')
        # Real implementation: not_found when no matching procedure exists
        assert result['error'] in ('not_found', 'retrieval_error'), (
            f"Expected not_found or retrieval_error, got: {result}"
        )


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
