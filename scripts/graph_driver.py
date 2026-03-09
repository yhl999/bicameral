#!/usr/bin/env python3
"""Small graph client shim for standalone maintenance scripts.

These post-ingest maintenance scripts are intentionally runnable from the public
repo without private helper modules. Imports are lazy so `--help` works even
when optional backend dependencies are not installed yet.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any


class GraphDriverSetupError(RuntimeError):
    """Raised when a requested backend is not installed or not configured."""


@dataclass
class QueryResult:
    result_set: list[list[Any]]


class Neo4jGraphClient:
    def __init__(self, host: str | None = None, port: int | None = None):
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError as exc:  # pragma: no cover - exercised via subprocess tests
            raise GraphDriverSetupError(
                'Neo4j support is unavailable because the `neo4j` package is not installed. '
                'Install project dependencies with `uv sync` (or `pip install neo4j`) and retry.'
            ) from exc

        self._database = os.environ.get('NEO4J_DATABASE', 'neo4j')
        self._user = os.environ.get('NEO4J_USER', 'neo4j')
        self._password = os.environ.get('NEO4J_PASSWORD')
        if not self._password:
            raise GraphDriverSetupError(
                'Neo4j backend requested but `NEO4J_PASSWORD` is not set. '\
                'Export `NEO4J_PASSWORD` (and optionally `NEO4J_USER` / `NEO4J_DATABASE`) '\
                'before running these maintenance scripts, or rerun with `--backend falkordb`.'
            )

        resolved_host = host or os.environ.get('NEO4J_HOST') or 'localhost'
        resolved_port = int(port or os.environ.get('NEO4J_PORT') or 7687)
        self._driver = AsyncGraphDatabase.driver(
            f'bolt://{resolved_host}:{resolved_port}',
            auth=(self._user, self._password),
        )

    async def query(self, cypher: str, params: dict[str, Any] | None = None) -> QueryResult:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, params or {})
            rows = await result.values()
            await result.consume()
            return QueryResult(result_set=[list(row) for row in rows])

    async def run_in_transaction(self, queries: list[tuple[str, dict[str, Any]]]) -> None:
        async with self._driver.session(database=self._database) as session:
            async def _execute(tx):
                for cypher, params in queries:
                    result = await tx.run(cypher, params or {})
                    await result.consume()
            await session.execute_write(_execute)

    async def close(self) -> None:
        await self._driver.close()


class FalkorGraphClient:
    def __init__(self, group_id: str, host: str | None = None, port: int | None = None):
        try:
            from falkordb import Graph
            from falkordb.asyncio import FalkorDB
        except ImportError as exc:  # pragma: no cover - exercised via subprocess tests
            raise GraphDriverSetupError(
                'FalkorDB backend requested but the optional `falkordb` package is not installed. '
                'Install the extra with `uv sync --extra falkordb` or '
                '`pip install "graphiti-core[falkordb]"`, then retry.'
            ) from exc

        resolved_host = host or os.environ.get('FALKORDB_HOST') or 'localhost'
        resolved_port = int(port or os.environ.get('FALKORDB_PORT') or 6379)
        self._client = FalkorDB(host=resolved_host, port=resolved_port)
        self._graph = Graph(group_id, self._client)

    async def query(self, cypher: str, params: dict[str, Any] | None = None) -> QueryResult:
        result = await self._graph.query(cypher, params or {})
        return QueryResult(result_set=[list(row) for row in result.result_set])

    async def run_in_transaction(self, queries: list[tuple[str, dict[str, Any]]]) -> None:
        for cypher, params in queries:
            await self._graph.query(cypher, params or {})

    async def close(self) -> None:
        if hasattr(self._client, 'aclose'):
            await self._client.aclose()  # type: ignore[attr-defined]
            return
        connection = getattr(self._client, 'connection', None)
        if connection is not None and hasattr(connection, 'aclose'):
            await connection.aclose()
            return
        if connection is not None and hasattr(connection, 'close'):
            await connection.close()


def add_backend_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        '--backend',
        choices=['neo4j', 'falkordb'],
        default=os.environ.get('GRAPHITI_BACKEND', 'neo4j'),
        help='Graph backend to target (default: neo4j; set GRAPHITI_BACKEND to override).',
    )
    return parser


async def get_graph_client(
    backend: str,
    *,
    group_id: str,
    host: str | None = None,
    port: int | None = None,
):
    backend = str(backend or '').strip().lower()
    if backend == 'neo4j':
        return Neo4jGraphClient(host=host, port=port)
    if backend == 'falkordb':
        return FalkorGraphClient(group_id=group_id, host=host, port=port)
    raise GraphDriverSetupError(
        f'Unsupported backend {backend!r}. Expected one of: neo4j, falkordb.'
    )
