"""
Pytest configuration for MCP server tests.
This file prevents pytest from loading the parent project's conftest.py
"""

import sys
from pathlib import Path

import pytest

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Ensure the local (patched) graphiti_core takes precedence over the installed package.
# The repo root contains the hotfix overlay of graphiti_core; tests must import from
# there — not from the venv-installed upstream version — so patched helpers
# (_canonicalize_edge_name, _should_filter_constrained_edge, extraction_mode, etc.)
# are available.
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

# Allow tests to use `from mcp_server.src.services...` import style.
# When pytest runs from inside mcp_server/, the parent directory (repo root)
# doubles as the package root for `mcp_server` imports.
mcp_parent = Path(__file__).parent.parent.parent
if str(mcp_parent) not in sys.path:
    sys.path.insert(0, str(mcp_parent))

from config.schema import GraphitiConfig  # noqa: E402


@pytest.fixture
def config():
    """Provide a default GraphitiConfig for tests."""
    return GraphitiConfig()
