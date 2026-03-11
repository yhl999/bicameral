"""Search adapter service for OM-lane retrieval from OM primitives."""

from __future__ import annotations

from datetime import datetime
from typing import Any

try:
    from .neo4j_service import Neo4jService
    from .om_group_scope import (
        DEFAULT_OM_GROUP_ID,
        includes_om_native_group,
        om_native_groups_in_scope,
    )
except ImportError:  # pragma: no cover - top-level import fallback
    from services.neo4j_service import Neo4jService
    from services.om_group_scope import (
        DEFAULT_OM_GROUP_ID,
        includes_om_native_group,
        om_native_groups_in_scope,
    )


def _provider_name(service: Any) -> str:
    try:
        return str(service.config.database.provider).lower()
    except Exception:
        return ''


def _allows_om_node_label(entity_types: list[str] | None) -> bool:
    # Keep search_nodes(entity_types=...) contract: OM adapter should return
    # no rows when OMNode label is not in scope.
    if not entity_types:
        return True

    normalized_labels = {
        str(label).strip().lower()
        for label in entity_types
        if str(label).strip()
    }
    return 'omnode' in normalized_labels


def _row_uuid(row: dict[str, Any]) -> str:
    return str(row.get('uuid') or '').strip()


def _row_group_id(row: dict[str, Any]) -> str:
    return str(row.get('group_id') or '').strip()


def _combined_om_row_identity(row: dict[str, Any]) -> tuple[str, str]:
    return (_row_group_id(row), _row_uuid(row))


def _row_lexical_score(row: dict[str, Any]) -> float:
    try:
        return float(row.get('lexical_score') or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _row_created_at_timestamp(row: dict[str, Any]) -> float:
    raw_value = row.get('created_at')
    text = str(raw_value).strip() if raw_value not in (None, '') else ''
    if not text:
        return 0.0

    try:
        return datetime.fromisoformat(text.replace('Z', '+00:00')).timestamp()
    except ValueError:
        return 0.0


def _combined_om_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -_row_lexical_score(row),
        -_row_created_at_timestamp(row),
        _row_group_id(row),
        _row_uuid(row),
    )


def _rank_combined_om_rows(rows: list[dict[str, Any]], *, max_items: int) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str], dict[str, Any]] = {}

    for row in rows:
        row_id = _combined_om_row_identity(row)
        if not row_id[1]:
            continue

        existing = deduped.get(row_id)
        if existing is None or _combined_om_sort_key(row) < _combined_om_sort_key(existing):
            deduped[row_id] = row

    ranked = sorted(deduped.values(), key=_combined_om_sort_key)
    return ranked[:max_items]


class SearchService:
    """Adapter that routes OM-lane search requests to OM primitives."""

    def __init__(
        self,
        *,
        om_group_id: str = DEFAULT_OM_GROUP_ID,
        neo4j_service: Neo4jService | None = None,
    ):
        self.om_group_id = om_group_id
        self.neo4j_service = neo4j_service or Neo4jService()

    def includes_observational_memory(self, group_ids: list[str]) -> bool:
        # Empty group scope means "all lanes" in the MCP server contract.
        return includes_om_native_group(group_ids, default_group_id=self.om_group_id)

    def _om_groups_in_scope(self, group_ids: list[str]) -> list[str]:
        return om_native_groups_in_scope(group_ids, default_group_id=self.om_group_id)

    async def search_observational_nodes(
        self,
        *,
        graphiti_service: Any,
        query: str,
        group_ids: list[str],
        max_nodes: int,
        entity_types: list[str] | None,
    ) -> list[dict[str, Any]]:
        if not self.includes_observational_memory(group_ids):
            return []
        if not _allows_om_node_label(entity_types):
            return []
        if _provider_name(graphiti_service) != 'neo4j':
            return []

        client = await graphiti_service.get_client()
        candidate_rows: list[dict[str, Any]] = []

        for group_id in self._om_groups_in_scope(group_ids):
            rows = await self.neo4j_service.search_om_nodes(
                client.driver,
                group_id=group_id,
                query=query,
                limit=max_nodes,
            )
            candidate_rows.extend(
                {
                    **row,
                    'group_id': str(row.get('group_id') or group_id),
                }
                for row in rows
            )

        nodes: list[dict[str, Any]] = []
        for row in _rank_combined_om_rows(candidate_rows, max_items=max_nodes):
            node_id = _row_uuid(row)
            content = str(row.get('content') or '').strip()
            created_at = row.get('created_at')
            nodes.append(
                {
                    'uuid': node_id,
                    'name': (content[:120] if content else node_id),
                    'labels': ['OMNode'],
                    'created_at': str(created_at) if created_at is not None else None,
                    'summary': content or None,
                    'group_id': str(row.get('group_id') or self.om_group_id),
                    'attributes': {
                        'source': 'om_primitive',
                        'status': row.get('status'),
                        'semantic_domain': row.get('semantic_domain'),
                        'urgency_score': row.get('urgency_score'),
                        'lexical_score': row.get('lexical_score'),
                    },
                }
            )
        return nodes

    async def search_observational_facts(
        self,
        *,
        graphiti_service: Any,
        query: str,
        group_ids: list[str],
        max_facts: int,
        center_node_uuid: str | None,
    ) -> list[dict[str, Any]]:
        if not self.includes_observational_memory(group_ids):
            return []
        if _provider_name(graphiti_service) != 'neo4j':
            return []

        client = await graphiti_service.get_client()
        candidate_rows: list[dict[str, Any]] = []

        for group_id in self._om_groups_in_scope(group_ids):
            rows = await self.neo4j_service.search_om_facts(
                client.driver,
                group_id=group_id,
                query=query,
                limit=max_facts,
                center_node_uuid=center_node_uuid,
            )
            candidate_rows.extend(
                {
                    **row,
                    'group_id': str(row.get('group_id') or group_id),
                }
                for row in rows
            )

        facts: list[dict[str, Any]] = []
        for row in _rank_combined_om_rows(candidate_rows, max_items=max_facts):
            relation_type = str(row.get('relation_type') or '').strip()
            source_node_id = str(row.get('source_node_id') or '').strip()
            target_node_id = str(row.get('target_node_id') or '').strip()
            fact_uuid = _row_uuid(row)
            if not relation_type or not source_node_id or not target_node_id or not fact_uuid:
                continue

            source_content = str(row.get('source_content') or '').strip()
            target_content = str(row.get('target_content') or '').strip()
            created_at = row.get('created_at')
            valid_at = row.get('valid_at')
            invalid_at = row.get('invalid_at')
            relation_properties = row.get('relation_properties')
            if not isinstance(relation_properties, dict):
                relation_properties = {}

            facts.append(
                {
                    'uuid': fact_uuid,
                    'name': relation_type,
                    'fact': f'{relation_type}: {source_content} -> {target_content}',
                    'group_id': str(row.get('group_id') or self.om_group_id),
                    'source_node_uuid': source_node_id,
                    'target_node_uuid': target_node_id,
                    'created_at': str(created_at) if created_at is not None else None,
                    'valid_at': str(valid_at) if valid_at is not None else None,
                    'invalid_at': str(invalid_at) if invalid_at is not None else None,
                    'expired_at': None,
                    'episodes': [],
                    'attributes': {
                        'source': 'om_primitive',
                        'lexical_score': row.get('lexical_score'),
                        'source_content': source_content,
                        'target_content': target_content,
                        'relation_properties': relation_properties,
                    },
                }
            )
        return facts
