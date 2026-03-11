"""Router modules for Bicameral MCP Surface v1.

Each module owns a specific set of MCP tools and registers them
via register_tools(mcp_instance). This architecture allows parallel
development of Exec tasks without merge conflicts.
"""
