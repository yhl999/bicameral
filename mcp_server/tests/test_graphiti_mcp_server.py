"""Tests for Phase 0 skeleton: schema validation, get_tools(), and debug flag.

Run from mcp_server/ directory:
    pytest tests/test_graphiti_mcp_server.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import path setup
# ---------------------------------------------------------------------------

_MCP_SRC = Path(__file__).parent.parent / 'src'
if str(_MCP_SRC) not in sys.path:
    sys.path.insert(0, str(_MCP_SRC.parent))
    sys.path.insert(0, str(_MCP_SRC))


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Tests for _validate_typed_object() in schema_validation.py."""

    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            from mcp_server.src.services.schema_validation import (
                SCHEMA_REGISTRY,
                _validate_typed_object,
            )
        except ImportError:
            from services.schema_validation import (
                SCHEMA_REGISTRY,
                _validate_typed_object,
            )
        self.validate = _validate_typed_object
        self.registry = SCHEMA_REGISTRY

    def test_schema_registry_loaded(self):
        """SCHEMA_REGISTRY should contain at least the core schema types."""
        required = {'TypedFact', 'Candidate', 'Preference', 'Commitment', 'OperationalRule'}
        missing = required - set(self.registry.keys())
        assert not missing, f'Missing schemas in registry: {missing}'

    def test_valid_typed_fact(self):
        """Valid TypedFact should pass validation."""
        obj = {
            'subject': 'user',
            'predicate': 'prefers_dark_mode',
            'value': True,
            'fact_type': 'preference',
        }
        ok, err = self.validate(obj, 'TypedFact')
        assert ok is True
        assert err is None

    def test_missing_required_field(self):
        """TypedFact missing 'subject' should fail validation."""
        obj = {
            'predicate': 'prefers_dark_mode',
            'value': True,
            'fact_type': 'preference',
        }
        ok, err = self.validate(obj, 'TypedFact')
        assert ok is False
        assert err is not None
        assert 'subject' in err

    def test_empty_required_string_field(self):
        """TypedFact with empty string for required field should fail."""
        obj = {
            'subject': '',
            'predicate': 'prefers_dark_mode',
            'value': True,
            'fact_type': 'preference',
        }
        ok, err = self.validate(obj, 'TypedFact')
        assert ok is False
        assert err is not None

    def test_invalid_fact_type_enum(self):
        """TypedFact with invalid fact_type enum should fail."""
        obj = {
            'subject': 'user',
            'predicate': 'something',
            'value': 'foo',
            'fact_type': 'not_a_valid_type',
        }
        ok, err = self.validate(obj, 'TypedFact')
        assert ok is False
        assert err is not None

    def test_wrong_type_for_string_field(self):
        """TypedFact with integer for 'subject' (string field) should fail."""
        obj = {
            'subject': 123,
            'predicate': 'something',
            'value': 'foo',
            'fact_type': 'preference',
        }
        ok, err = self.validate(obj, 'TypedFact')
        assert ok is False
        assert err is not None

    def test_unknown_schema_type(self):
        """Validating against an unknown schema type should return False."""
        ok, err = self.validate({'foo': 'bar'}, 'NonExistentSchema')
        assert ok is False
        assert 'Unknown schema type' in (err or '')

    def test_non_dict_input(self):
        """Validating a non-dict should return False."""
        ok, err = self.validate('not a dict', 'TypedFact')  # type: ignore[arg-type]
        assert ok is False
        assert err is not None

    def test_valid_candidate(self):
        """Valid Candidate should pass validation."""
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
        """Field violating minLength constraint should fail."""
        obj = {
            'candidate_id': '',  # minLength 1
            'fact_type': 'preference',
            'subject': 'user',
            'predicate': 'editor',
            'value': 'vim',
        }
        ok, err = self.validate(obj, 'Candidate')
        assert ok is False


# ---------------------------------------------------------------------------
# get_tools() tests
# ---------------------------------------------------------------------------


class TestGetToolsSignature:
    """Tests for the get_tools() MCP tool."""

    @pytest.fixture(autouse=True)
    def _import(self, monkeypatch):
        # Ensure debug tools are off for base tests
        monkeypatch.delenv('BICAMERAL_DEBUG_TOOLS', raising=False)
        try:
            import mcp_server.src.graphiti_mcp_server as _mod
        except ImportError:
            import graphiti_mcp_server as _mod  # type: ignore[no-redef]
        self.module = _mod

    @pytest.mark.asyncio
    async def test_get_tools_returns_list(self):
        """get_tools() should return a list."""
        result = await self.module.get_tools()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_tools_minimum_count(self):
        """get_tools() should return at least 16 tools (all new methods)."""
        result = await self.module.get_tools()
        assert len(result) >= 16, f'Expected >= 16 tools, got {len(result)}'

    @pytest.mark.asyncio
    async def test_get_tools_required_fields(self):
        """Each tool dict must have name, description, mode_hint, and schema."""
        result = await self.module.get_tools()
        for tool in result:
            assert 'name' in tool, f'Tool missing name: {tool}'
            assert 'description' in tool, f'Tool {tool.get("name")} missing description'
            assert 'mode_hint' in tool, f'Tool {tool.get("name")} missing mode_hint'
            assert 'schema' in tool, f'Tool {tool.get("name")} missing schema'

    @pytest.mark.asyncio
    async def test_get_tools_mode_hints_valid(self):
        """All mode_hint values must be one of: 'typed', 'facts', 'both'."""
        result = await self.module.get_tools()
        valid_hints = {'typed', 'facts', 'both'}
        for tool in result:
            hint = tool.get('mode_hint')
            assert hint in valid_hints, (
                f"Tool {tool.get('name')} has invalid mode_hint={hint!r}"
            )

    @pytest.mark.asyncio
    async def test_search_memory_facts_has_both_mode_hint(self):
        """search_memory_facts should have mode_hint='both'."""
        result = await self.module.get_tools()
        tool = next((t for t in result if t['name'] == 'search_memory_facts'), None)
        assert tool is not None, 'search_memory_facts not found in get_tools() result'
        assert tool['mode_hint'] == 'both'

    @pytest.mark.asyncio
    async def test_new_typed_methods_present(self):
        """All new typed methods from the epic must appear in get_tools()."""
        expected = {
            'remember_fact',
            'get_current_state',
            'get_history',
            'list_candidates',
            'promote_candidate',
            'reject_candidate',
            'list_packs',
            'get_context_pack',
            'get_workflow_pack',
            'describe_pack',
            'create_workflow_pack',
            'search_episodes',
            'get_episode',
            'search_procedures',
            'get_procedure',
        }
        result = await self.module.get_tools()
        names = {t['name'] for t in result}
        missing = expected - names
        assert not missing, f'get_tools() missing methods: {missing}'

    @pytest.mark.asyncio
    async def test_debug_tools_absent_without_flag(self):
        """Without BICAMERAL_DEBUG_TOOLS, get_entity_edge and get_episodes are hidden."""
        result = await self.module.get_tools()
        names = {t['name'] for t in result}
        assert 'get_entity_edge' not in names
        assert 'get_episodes' not in names


# ---------------------------------------------------------------------------
# Debug flag tests
# ---------------------------------------------------------------------------


class TestDebugToolsHidden:
    """Tests for BICAMERAL_DEBUG_TOOLS environment flag."""

    @pytest.fixture
    def _import_mod(self):
        try:
            import mcp_server.src.graphiti_mcp_server as _mod
        except ImportError:
            import graphiti_mcp_server as _mod  # type: ignore[no-redef]
        return _mod

    @pytest.mark.asyncio
    async def test_get_entity_edge_blocked_without_flag(self, _import_mod):
        """get_entity_edge returns ErrorResponse('method_unavailable') when flag is off."""
        import importlib
        mod = _import_mod
        # Force flag to False
        original = mod._BICAMERAL_DEBUG_TOOLS
        try:
            mod._BICAMERAL_DEBUG_TOOLS = False
            result = await mod.get_entity_edge('some-uuid')
            assert isinstance(result, dict)
            assert result.get('error') == 'method_unavailable'
        finally:
            mod._BICAMERAL_DEBUG_TOOLS = original

    @pytest.mark.asyncio
    async def test_get_episodes_blocked_without_flag(self, _import_mod):
        """get_episodes returns ErrorResponse('method_unavailable') when flag is off."""
        mod = _import_mod
        original = mod._BICAMERAL_DEBUG_TOOLS
        try:
            mod._BICAMERAL_DEBUG_TOOLS = False
            result = await mod.get_episodes()
            assert isinstance(result, dict)
            assert result.get('error') == 'method_unavailable'
        finally:
            mod._BICAMERAL_DEBUG_TOOLS = original

    @pytest.mark.asyncio
    async def test_debug_tools_in_get_tools_when_flag_set(self, _import_mod):
        """With BICAMERAL_DEBUG_TOOLS=1, get_tools() includes debug methods."""
        mod = _import_mod
        original = mod._BICAMERAL_DEBUG_TOOLS
        try:
            mod._BICAMERAL_DEBUG_TOOLS = True
            result = await mod.get_tools()
            names = {t['name'] for t in result}
            assert 'get_entity_edge' in names
            assert 'get_episodes' in names
        finally:
            mod._BICAMERAL_DEBUG_TOOLS = original
