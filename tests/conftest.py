"""
Pytest configuration for MCP server tests.

Provides test isolation, fixtures, and shared utilities.
"""

import sys

import pytest


# Save clean module state at import time
_CLEAN_MODULES = set(sys.modules.keys())


@pytest.fixture(scope="function", autouse=True)
def isolate_test_module_state():
    """
    Isolate sys.modules between test functions.
    
    Cleans up modules added during test execution to prevent cross-test
    contamination (e.g., test_search_memory_facts_typed_mode.py polluting
    test_tool_result_scope.py when run in sequence).
    
    Preserves core modules (mcp_server, graphiti_core, etc.) that should
    persist across tests.
    """
    yield
    
    # Find modules added during this test
    current_modules = set(sys.modules.keys())
    added_modules = current_modules - _CLEAN_MODULES
    
    # Remove added modules that are not part of the core codebase
    # Keep: mcp_server.*, graphiti_core.*, built-ins
    preserve_prefixes = ('mcp_server', 'graphiti', 'graphiti_core')
    
    for mod_name in added_modules:
        # Skip built-in and standard library modules
        if mod_name.startswith('_'):
            continue
        
        # Skip modules we want to preserve
        if any(mod_name.startswith(prefix) for prefix in preserve_prefixes):
            continue
        
        # Remove the module
        sys.modules.pop(mod_name, None)


@pytest.fixture
def optional_import():
    """
    Helper fixture for tests that need optional dependencies.
    
    Usage:
        def test_something(optional_import):
            psutil = optional_import('psutil')
            if psutil is None:
                pytest.skip("psutil not installed")
            # test code
    """
    def _import(module_name):
        try:
            return __import__(module_name)
        except ImportError:
            return None
    
    return _import
