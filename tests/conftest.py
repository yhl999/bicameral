"""
Pytest configuration for graphiti_core unit tests.

Provides shared fixtures and utilities.
"""

import pytest


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
