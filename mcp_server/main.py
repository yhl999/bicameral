#!/usr/bin/env python3
"""
Main entry point for Graphiti MCP Server

This is a backwards-compatible wrapper around the original graphiti_mcp_server.py
to maintain compatibility with existing deployment scripts and documentation.

Usage:
    python main.py [args...]

All arguments are passed through to the original server implementation.
"""

# Import and run the original server via the package path so
# intra-package relative imports (for example ``from .routers ...`` and
# ``from ..models ...``) continue to resolve correctly when this wrapper is
# launched as ``python mcp_server/main.py``.
if __name__ == '__main__':
    from mcp_server.src.graphiti_mcp_server import main

    # Pass all command line arguments to the original main function
    main()
