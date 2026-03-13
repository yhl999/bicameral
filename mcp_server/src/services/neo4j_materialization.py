"""Neo4j materialization service for remember_fact typed writes."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

try:
    from ..models.typed_memory import StateFact
except ImportError:  # pragma: no cover
    from models.typed_memory import StateFact

logger = logging.getLogger(__name__)


class Neo4jMaterializationService:
    """Project a typed state fact into Neo4j as simple Entity + HAS_FACT graph."""

    async def materialize_typed_fact(
        self,
        *,
        fact: StateFact,
        source: str = 'caller_asserted_unverified',
        superseded_fact_id: str | None = None,
        graphiti_client: Any | None = None,
    ) -> tuple[bool, str | None]:
        """Write/merge a typed StateFact into Neo4j.

        Returns:
            (success, error)
        """
        try:
            if graphiti_client is None:
                return False, 'graphiti_client_unavailable'

            driver = self._extract_driver(graphiti_client)
            if driver is None:
                return False, 'graphiti_driver_unavailable'

            timestamp = self._timestamp()
            subject = self._coerce_text(getattr(fact, 'subject', ''), fallback='')
            predicate = self._coerce_text(getattr(fact, 'predicate', ''), fallback='')
            value = getattr(fact, 'value', None)
            fact_id = self._coerce_text(getattr(fact, 'object_id', ''), fallback='')
            fact_type = self._coerce_text(getattr(fact, 'fact_type', ''), fallback='')

            if not subject or not predicate or not fact_id:
                return False, 'invalid_fact_payload'

            params = {
                'subject': subject,
                'predicate': predicate,
                'fact_id': fact_id,
                'value_text': _coerce_json_scalar(value),
                'fact_type': fact_type,
                'status': 'active',
                'source': source,
                'timestamp': timestamp,
            }

            upsert_query = (
                "MERGE (s:Entity {name: $subject})\n"
                "SET s.last_seen = $timestamp\n"
                "MERGE (s)-[r:HAS_FACT {fact_id: $fact_id}]->(v:FactValue)\n"
                "SET r.predicate = $predicate, r.fact_type = $fact_type, r.status = $status, r.source = $source, r.timestamp = $timestamp\n"
                "SET v.subject = $subject, v.value = $value_text, v.fact_type = $fact_type, v.fact_id = $fact_id\n"
                "RETURN s, r"
            )
            await self._run_query(driver, upsert_query, params)

            if superseded_fact_id:
                supersede_query = (
                    "MATCH (s:Entity {name: $subject})-[sr:HAS_FACT]->(sv:FactValue)\n"
                    "WHERE sr.fact_id = $superseded_fact_id\n"
                    "SET sr.status = 'superseded'"
                )
                await self._run_query(
                    driver,
                    supersede_query,
                    {
                        **params,
                        'superseded_fact_id': str(superseded_fact_id),
                    },
                )

            return True, None
        except Exception as exc:
            logger.warning('Neo4j materialization failed for fact=%s: %s', getattr(fact, 'object_id', None), exc)
            return False, str(exc)

    async def _run_query(self, driver: Any, query: str, params: Mapping[str, Any]) -> None:
        if hasattr(driver, 'execute_query'):
            result = driver.execute_query(query, parameters=dict(params))
            if inspect.isawaitable(result):
                await result
            return

        session = driver.session()
        try:
            cursor = session.run(query, dict(params))
            if inspect.isawaitable(cursor):
                cursor = await cursor
            if hasattr(cursor, 'consume'):
                maybe = cursor.consume()
                if inspect.isawaitable(maybe):
                    await maybe
        finally:
            close_result = session.close()
            if inspect.isawaitable(close_result):
                await close_result

    def _extract_driver(self, graphiti_client: Any) -> Any | None:
        if graphiti_client is None:
            return None
        if hasattr(graphiti_client, 'driver'):
            return graphiti_client.driver
        if hasattr(graphiti_client, 'client') and hasattr(graphiti_client.client, 'driver'):
            return graphiti_client.client.driver
        return None

    def _coerce_text(self, value: Any, fallback: str = '') -> str:
        if value is None:
            return fallback
        text = str(value).strip()
        return text if text else fallback

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _coerce_json_scalar(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ''
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)
