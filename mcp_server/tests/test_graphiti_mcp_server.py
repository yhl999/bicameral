"""Tests for Phase 0 skeleton: schema validation, get_tools(), and debug flag.

Run from mcp_server/ directory:
    pytest tests/test_graphiti_mcp_server.py -v
"""

from __future__ import annotations

import importlib
import inspect
import json
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
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_mcp() -> MagicMock:
    tools: dict[str, object] = {}
    mock = MagicMock()

    def _tool_decorator():
        def decorator(fn):
            tools[fn.__name__] = fn
            return fn

        return decorator

    mock.tool = _tool_decorator
    mock._tools = tools
    return mock


def _registered_router_tools() -> dict[str, object]:
    try:
        from mcp_server.src.routers import candidates, episodes_procedures, memory, packs
    except ImportError:
        from routers import candidates, episodes_procedures, memory, packs  # type: ignore[no-redef]

    mock_mcp = _make_mock_mcp()
    memory.register_tools(mock_mcp)
    candidates.register_tools(mock_mcp)
    packs.register_tools(mock_mcp)
    episodes_procedures.register_tools(mock_mcp)
    return mock_mcp._tools


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Tests for _validate_typed_object() in schema_validation.py."""

    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            schema_validation_module = importlib.import_module(
                'mcp_server.src.services.schema_validation'
            )
        except ImportError:
            schema_validation_module = importlib.import_module('services.schema_validation')
        self.module = schema_validation_module
        self.validate = schema_validation_module._validate_typed_object
        self.registry = schema_validation_module.SCHEMA_REGISTRY

    def test_schema_registry_loaded(self):
        """SCHEMA_REGISTRY should contain at least the core schema types."""
        required = {'TypedFact', 'Candidate', 'Preference', 'Commitment', 'OperationalRule'}
        missing = required - set(self.registry.keys())
        assert not missing, f'Missing schemas in registry: {missing}'

    def test_valid_typed_fact(self):
        obj = {
            'subject': 'user',
            'predicate': 'prefers_dark_mode',
            'value': True,
            'fact_type': 'preference',
            'timestamp': '2026-03-11T12:34:56Z',
        }
        ok, err = self.validate(obj, 'TypedFact')
        assert ok is True
        assert err is None

    def test_missing_required_field(self):
        obj = {
            'predicate': 'prefers_dark_mode',
            'value': True,
            'fact_type': 'preference',
            'timestamp': '2026-03-11T12:34:56Z',
        }
        ok, err = self.validate(obj, 'TypedFact')
        assert ok is False
        assert 'subject' in (err or '')

    def test_empty_required_string_field(self):
        obj = {
            'subject': '',
            'predicate': 'prefers_dark_mode',
            'value': True,
            'fact_type': 'preference',
            'timestamp': '2026-03-11T12:34:56Z',
        }
        ok, err = self.validate(obj, 'TypedFact')
        assert ok is False
        assert 'subject' in (err or '')

    def test_invalid_fact_type_enum(self):
        obj = {
            'subject': 'user',
            'predicate': 'something',
            'value': 'foo',
            'fact_type': 'not_a_valid_type',
            'timestamp': '2026-03-11T12:34:56Z',
        }
        ok, err = self.validate(obj, 'TypedFact')
        assert ok is False
        assert 'fact_type' in (err or '')

    def test_wrong_type_for_string_field(self):
        obj = {
            'subject': 123,
            'predicate': 'something',
            'value': 'foo',
            'fact_type': 'preference',
            'timestamp': '2026-03-11T12:34:56Z',
        }
        ok, err = self.validate(obj, 'TypedFact')
        assert ok is False
        assert 'string' in (err or '')

    def test_missing_timestamp_is_rejected(self):
        obj = {
            'subject': 'user',
            'predicate': 'prefers_dark_mode',
            'value': True,
            'fact_type': 'preference',
        }
        ok, err = self.validate(obj, 'TypedFact')
        assert ok is False
        assert 'timestamp' in (err or '')

    def test_overlong_typed_fact_value_is_rejected(self):
        obj = {
            'subject': 'user',
            'predicate': 'bio',
            'value': 'x' * 9000,
            'fact_type': 'preference',
            'timestamp': '2026-03-11T12:34:56Z',
        }
        ok, err = self.validate(obj, 'TypedFact')
        assert ok is False
        assert 'value' in (err or '')

    def test_over_nested_typed_fact_value_is_rejected(self):
        obj = {
            'subject': 'user',
            'predicate': 'profile',
            'value': {'a': {'b': {'c': {'too': 'deep'}}}},
            'fact_type': 'preference',
            'timestamp': '2026-03-11T12:34:56Z',
        }
        ok, err = self.validate(obj, 'TypedFact')
        assert ok is False
        assert 'value' in (err or '')

    def test_unknown_schema_type(self):
        ok, err = self.validate({'foo': 'bar'}, 'NonExistentSchema')
        assert ok is False
        assert 'Unknown schema type' in (err or '')

    def test_non_dict_input(self):
        ok, err = self.validate('not a dict', 'TypedFact')  # type: ignore[arg-type]
        assert ok is False
        assert err is not None

    def test_valid_candidate(self):
        obj = {
            'candidate_id': 'cand-001',
            'fact_type': 'preference',
            'subject': 'user',
            'predicate': 'editor',
            'value': 'vim',
        }
        ok, err = self.validate(obj, 'Candidate')
        assert ok is True
        assert err is None

    def test_string_min_length_violation(self):
        obj = {
            'candidate_id': '',
            'fact_type': 'preference',
            'subject': 'user',
            'predicate': 'editor',
            'value': 'vim',
        }
        ok, err = self.validate(obj, 'Candidate')
        assert ok is False
        assert 'candidate_id' in (err or '')

    def test_candidate_id_pattern_validation(self):
        obj = {
            'candidate_id': 'Bad Candidate Id',
            'fact_type': 'preference',
            'subject': 'user',
            'predicate': 'editor',
            'value': 'vim',
        }
        ok, err = self.validate(obj, 'Candidate')
        assert ok is False
        assert 'candidate_id' in (err or '')

    def test_pack_definition_requires_workflow_fields(self):
        obj = {
            'pack_id': 'deploy-workflow',
            'scope': 'workflow',
            'intent': 'deploy',
            'consumer': 'archibald',
            'version': '1.0',
        }
        ok, err = self.validate(obj, 'PackDefinition')
        assert ok is False
        assert 'workflow_steps' in (err or '')

    def test_pack_definition_requires_context_rules_for_context_scope(self):
        obj = {
            'pack_id': 'coding-context',
            'scope': 'context',
            'intent': 'coding defaults',
            'consumer': 'archibald',
            'version': '1.0',
        }
        ok, err = self.validate(obj, 'PackDefinition')
        assert ok is False
        assert 'context_rules' in (err or '')

    def test_load_schemas_is_atomic_on_failure(self, tmp_path, monkeypatch):
        good_schema = json.dumps(self.registry['TypedFact'])
        broken_dir = tmp_path / 'schemas'
        broken_dir.mkdir()
        (broken_dir / 'TypedFact.json').write_text(good_schema, encoding='utf-8')
        (broken_dir / 'Broken.json').write_text('{"type": ', encoding='utf-8')

        previous_registry = dict(self.module.SCHEMA_REGISTRY)
        previous_validator_keys = set(self.module._VALIDATORS)
        monkeypatch.setattr(self.module, '_SCHEMAS_DIR', broken_dir)

        with pytest.raises(RuntimeError, match='Broken.json'):
            self.module._load_schemas()

        assert previous_registry == self.module.SCHEMA_REGISTRY
        assert set(self.module._VALIDATORS) == previous_validator_keys

    def test_load_schemas_missing_directory_raises(self, tmp_path, monkeypatch):
        missing_dir = tmp_path / 'missing-schemas'
        monkeypatch.setattr(self.module, '_SCHEMAS_DIR', missing_dir)

        with pytest.raises(RuntimeError, match='Schemas directory not found'):
            self.module._load_schemas()

    def test_unknown_fields_fail_closed_in_strict_mode(self):
        obj = {
            'subject': 'user',
            'predicate': 'prefers_dark_mode',
            'value': True,
            'fact_type': 'preference',
            'timestamp': '2026-03-11T12:34:56Z',
            'surprise': 'nope',
        }
        ok, err = self.validate(obj, 'TypedFact')
        assert ok is False
        assert 'unknown field' in (err or '').lower()

    def test_unknown_fields_allowed_when_strict_false(self):
        obj = {
            'subject': 'user',
            'predicate': 'prefers_dark_mode',
            'value': True,
            'fact_type': 'preference',
            'timestamp': '2026-03-11T12:34:56Z',
            'surprise': 'ok in loose mode',
        }
        ok, err = self.validate(obj, 'TypedFact', strict=False)
        assert ok is True
        assert err is None

    def test_pattern_validation(self):
        obj = {
            'pack_id': 'Bad Pack Id',
            'scope': 'workflow',
            'intent': 'deploy',
            'consumer': 'archibald',
            'version': '1.0',
            'workflow_steps': ['ship it'],
        }
        ok, err = self.validate(obj, 'PackDefinition')
        assert ok is False
        assert 'pack_id' in (err or '')

    def test_pack_id_length_matches_schema_bound(self):
        valid_obj = {
            'pack_id': 'a' * 128,
            'scope': 'workflow',
            'intent': 'deploy',
            'consumer': 'archibald',
            'version': '1.0',
            'workflow_steps': ['ship it'],
        }
        ok, err = self.validate(valid_obj, 'PackDefinition')
        assert ok is True, err

        too_long_obj = valid_obj.copy()
        too_long_obj['pack_id'] = 'a' * 129
        ok, err = self.validate(too_long_obj, 'PackDefinition')
        assert ok is False
        assert 'pack_id' in (err or '')

    def test_episode_id_pattern_validation(self):
        obj = {
            'object_id': 'Episode With Spaces',
            'object_type': 'episode',
        }
        ok, err = self.validate(obj, 'Episode')
        assert ok is False
        assert 'object_id' in (err or '')

    def test_format_validation(self):
        obj = {
            'object_id': 'ep-001',
            'object_type': 'episode',
            'started_at': 'not-a-date',
        }
        ok, err = self.validate(obj, 'Episode')
        assert ok is False
        assert 'started_at' in (err or '')


# ---------------------------------------------------------------------------
# get_tools() tests
# ---------------------------------------------------------------------------


class TestGetToolsSignature:
    """Tests for the get_tools() MCP tool."""

    @pytest.fixture(autouse=True)
    def _import(self, monkeypatch):
        monkeypatch.delenv('BICAMERAL_DEBUG_TOOLS', raising=False)
        self.module = load_graphiti_mcp_server()
        self.module._BICAMERAL_DEBUG_TOOLS = False

    @pytest.mark.anyio
    async def test_get_tools_returns_list(self):
        result = await self.module.get_tools()
        assert isinstance(result, list)

    @pytest.mark.anyio
    async def test_get_tools_required_fields(self):
        result = await self.module.get_tools()
        for tool in result:
            assert 'name' in tool, f'Tool missing name: {tool}'
            assert 'description' in tool, f'Tool {tool.get("name")} missing description'
            assert 'mode_hint' in tool, f'Tool {tool.get("name")} missing mode_hint'
            assert 'schema' in tool, f'Tool {tool.get("name")} missing schema'
            assert 'inputs' in tool['schema'], f'Tool {tool.get("name")} missing input schema'
            assert 'output' in tool['schema'], f'Tool {tool.get("name")} missing output schema'

    @pytest.mark.anyio
    async def test_get_tools_mode_hints_valid(self):
        result = await self.module.get_tools()
        valid_hints = {'typed', 'facts', 'both'}
        for tool in result:
            assert tool.get('mode_hint') in valid_hints

    @pytest.mark.anyio
    async def test_get_tools_exact_public_tool_coverage_without_debug(self):
        result = await self.module.get_tools()
        names = {tool['name'] for tool in result}
        assert names == set(self.module._PHASE0_PUBLIC_TOOL_CALLABLES)

    @pytest.mark.anyio
    async def test_get_tools_schema_inputs_match_runtime_callable_registry(self):
        result = await self.module.get_tools()
        tools_by_name = {tool['name']: tool for tool in result}

        assert set(self.module._PHASE0_PUBLIC_TOOL_CALLABLES) == set(tools_by_name)
        for tool_name, tool_fn in self.module._PHASE0_PUBLIC_TOOL_CALLABLES.items():
            expected_param_names = [
                name
                for name in inspect.signature(tool_fn).parameters
                if name != 'ctx'
            ]
            actual_param_names = list(tools_by_name[tool_name]['schema']['inputs'].keys())
            assert actual_param_names == expected_param_names, (
                f'{tool_name} schema inputs drifted from callable signature: '
                f'expected {expected_param_names}, got {actual_param_names}'
            )

    @pytest.mark.anyio
    async def test_search_memory_facts_has_both_mode_hint(self):
        result = await self.module.get_tools()
        tool = next((t for t in result if t['name'] == 'search_memory_facts'), None)
        assert tool is not None
        assert tool['mode_hint'] == 'both'

    @pytest.mark.anyio
    async def test_router_contract_slice_matches_registered_router_tools(self):
        assert {tool['name'] for tool in self.module._ROUTER_TOOL_CONTRACTS} == set(self.module._REGISTERED_ROUTER_TOOLS)

    @pytest.mark.anyio
    async def test_episode_and_procedure_contracts_advertise_pagination_inputs(self):
        result = await self.module.get_tools()
        tools_by_name = {tool['name']: tool for tool in result}

        assert list(tools_by_name['search_episodes']['schema']['inputs']) == [
            'query',
            'time_range',
            'include_history',
            'group_ids',
            'lane_alias',
            'limit',
            'offset',
        ]
        assert list(tools_by_name['search_procedures']['schema']['inputs']) == [
            'query',
            'include_all',
            'group_ids',
            'lane_alias',
            'limit',
            'offset',
        ]
        assert tools_by_name['search_episodes']['schema']['output'] == 'EpisodeSearchResponse | ErrorResponse'
        assert tools_by_name['search_procedures']['schema']['output'] == 'ProcedureSearchResponse | ErrorResponse'

    @pytest.mark.anyio
    async def test_debug_tools_absent_without_flag(self):
        result = await self.module.get_tools()
        names = {t['name'] for t in result}
        assert 'get_entity_edge' not in names
        assert 'get_episodes' not in names


# ---------------------------------------------------------------------------
# Debug flag tests
# ---------------------------------------------------------------------------


class TestDebugToolsHidden:
    @pytest.fixture(autouse=True)
    def _import_mod(self):
        self.module = load_graphiti_mcp_server()

    @pytest.mark.anyio
    async def test_get_entity_edge_blocked_without_flag(self):
        original = self.module._BICAMERAL_DEBUG_TOOLS
        try:
            self.module._BICAMERAL_DEBUG_TOOLS = False
            result = await self.module.get_entity_edge('some-uuid')
            assert result.get('error') == 'method_unavailable'
        finally:
            self.module._BICAMERAL_DEBUG_TOOLS = original

    @pytest.mark.anyio
    async def test_get_episodes_blocked_without_flag(self):
        original = self.module._BICAMERAL_DEBUG_TOOLS
        try:
            self.module._BICAMERAL_DEBUG_TOOLS = False
            result = await self.module.get_episodes()
            assert result.get('error') == 'method_unavailable'
        finally:
            self.module._BICAMERAL_DEBUG_TOOLS = original

    @pytest.mark.anyio
    async def test_debug_tools_in_get_tools_when_flag_set(self):
        original = self.module._BICAMERAL_DEBUG_TOOLS
        try:
            self.module._BICAMERAL_DEBUG_TOOLS = True
            result = await self.module.get_tools()
            names = {t['name'] for t in result}
            assert 'get_entity_edge' in names
            assert 'get_episodes' in names
        finally:
            self.module._BICAMERAL_DEBUG_TOOLS = original
