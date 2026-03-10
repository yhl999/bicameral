from __future__ import annotations

from typing import Any

try:
    from ..models.typed_memory import Episode, EvidenceRef, StateFact, TypedMemoryObject
    from .search_service import SearchService
except ImportError:  # pragma: no cover - top-level import fallback
    from models.typed_memory import Episode, EvidenceRef, StateFact, TypedMemoryObject
    from services.search_service import SearchService


def _coerce_timestamp(value: Any) -> str | None:
    text = str(value).strip() if value not in (None, '') else ''
    return text or None


class OMTypedProjectionService:
    """Project OM-native retrieval results into shared typed-memory objects.

    This is a read-time adapter only. It deliberately preserves OM-native storage
    and search primitives while letting the product retrieval layer surface OM
    content through the shared typed buckets.
    """

    def __init__(
        self,
        *,
        search_service: SearchService | None = None,
        graphiti_service: Any | None = None,
    ) -> None:
        self.search_service = search_service or SearchService()
        self.graphiti_service = graphiti_service

    async def project(
        self,
        *,
        query: str,
        effective_group_ids: list[str] | None,
        object_types: set[str],
        max_results: int,
    ) -> tuple[list[TypedMemoryObject], dict[str, str], dict[str, Any]]:
        scope = list(effective_group_ids or [])
        normalized_max_results = max(1, int(max_results or 1))

        if self.graphiti_service is None:
            return [], {}, {'enabled': False, 'reason': 'graphiti_service_unavailable'}
        if not self.search_service.includes_observational_memory(scope):
            return [], {}, {'enabled': False, 'reason': 'om_not_in_scope'}

        objects: list[TypedMemoryObject] = []
        search_text_overrides: dict[str, str] = {}
        episode_count = 0
        state_count = 0

        if not object_types or 'episode' in object_types:
            node_rows = await self.search_service.search_observational_nodes(
                graphiti_service=self.graphiti_service,
                query=query,
                group_ids=scope,
                max_nodes=normalized_max_results,
                entity_types=['OMNode'],
            )
            for row in node_rows:
                obj = self._episode_from_node(row)
                if obj is None:
                    continue
                objects.append(obj)
                episode_count += 1
                search_text_overrides[obj.object_id] = self._node_search_text(row)

        if not object_types or 'state_fact' in object_types:
            fact_rows = await self.search_service.search_observational_facts(
                graphiti_service=self.graphiti_service,
                query=query,
                group_ids=scope,
                max_facts=normalized_max_results,
                center_node_uuid=None,
            )
            for row in fact_rows:
                obj = self._state_fact_from_relation(row)
                if obj is None:
                    continue
                objects.append(obj)
                state_count += 1
                search_text_overrides[obj.object_id] = self._fact_search_text(row)

        limits = {
            'enabled': True,
            'reason': 'projected',
            'groups_considered': self.search_service._om_groups_in_scope(scope),
            'episodes_projected': episode_count,
            'state_projected': state_count,
            'max_results': normalized_max_results,
        }
        return objects, search_text_overrides, limits

    def _episode_from_node(self, row: dict[str, Any]) -> Episode | None:
        node_id = str(row.get('uuid') or '').strip()
        group_id = str(row.get('group_id') or '').strip()
        if not node_id or not group_id:
            return None

        summary = str(row.get('summary') or '').strip()
        title = str(row.get('name') or '').strip() or (summary[:120] if summary else node_id)
        created_at = _coerce_timestamp(row.get('created_at'))
        attributes = row.get('attributes') if isinstance(row.get('attributes'), dict) else {}
        annotations = [
            'om_native',
            str(attributes.get('semantic_domain') or '').strip(),
            str(attributes.get('status') or '').strip(),
        ]
        annotations = [item for item in annotations if item]

        return Episode(
            object_id=f'om_episode:{group_id}:{node_id}',
            root_id=f'om_episode:{group_id}:{node_id}',
            version=1,
            is_current=True,
            source_lane=group_id,
            source_key=f'om:{group_id}:node:{node_id}',
            policy_scope='private',
            visibility_scope='private',
            title=title,
            summary=summary or title,
            annotations=annotations,
            created_at=created_at or '2026-01-01T00:00:00Z',
            valid_at=created_at,
            evidence_refs=[
                EvidenceRef(
                    kind='event_log',
                    source_system='om',
                    locator={'system': 'om', 'stream': f'{group_id}:node', 'event_id': node_id},
                    title=title,
                    snippet=summary or title,
                    observed_at=created_at,
                )
            ],
        )

    def _state_fact_from_relation(self, row: dict[str, Any]) -> StateFact | None:
        relation_id = str(row.get('uuid') or '').strip()
        group_id = str(row.get('group_id') or '').strip()
        relation_type = str(row.get('name') or '').strip()
        source_node_id = str(row.get('source_node_uuid') or '').strip()
        target_node_id = str(row.get('target_node_uuid') or '').strip()
        if not relation_id or not group_id or not relation_type or not source_node_id or not target_node_id:
            return None

        attributes = row.get('attributes') if isinstance(row.get('attributes'), dict) else {}
        source_content = str(attributes.get('source_content') or '').strip()
        target_content = str(attributes.get('target_content') or '').strip()
        fact_text = str(row.get('fact') or '').strip()
        created_at = _coerce_timestamp(row.get('created_at'))

        return StateFact(
            object_id=f'om_state:{group_id}:{relation_id}',
            root_id=f'om_state:{group_id}:{relation_id}',
            version=1,
            is_current=True,
            source_lane=group_id,
            source_key=f'om:{group_id}:relation:{relation_id}',
            policy_scope='private',
            visibility_scope='private',
            fact_type='relationship',
            subject=f'om_node:{source_node_id}',
            predicate=f'om_relation:{relation_type.lower()}',
            value={
                'fact': fact_text,
                'source_node_id': source_node_id,
                'target_node_id': target_node_id,
                'source_content': source_content,
                'target_content': target_content,
            },
            scope='private',
            created_at=created_at or '2026-01-01T00:00:00Z',
            valid_at=created_at,
            evidence_refs=[
                EvidenceRef(
                    kind='event_log',
                    source_system='om',
                    locator={'system': 'om', 'stream': f'{group_id}:relation', 'event_id': relation_id},
                    title=relation_type,
                    snippet=fact_text or f'{source_content} -> {target_content}',
                    observed_at=created_at,
                )
            ],
        )

    def _node_search_text(self, row: dict[str, Any]) -> str:
        attributes = row.get('attributes') if isinstance(row.get('attributes'), dict) else {}
        parts = [
            str(row.get('name') or ''),
            str(row.get('summary') or ''),
            str(attributes.get('semantic_domain') or ''),
            str(attributes.get('status') or ''),
            str(row.get('group_id') or ''),
        ]
        return ' '.join(part for part in parts if part).strip()

    def _fact_search_text(self, row: dict[str, Any]) -> str:
        attributes = row.get('attributes') if isinstance(row.get('attributes'), dict) else {}
        parts = [
            str(row.get('name') or ''),
            str(row.get('fact') or ''),
            str(attributes.get('source_content') or ''),
            str(attributes.get('target_content') or ''),
            str(row.get('group_id') or ''),
        ]
        return ' '.join(part for part in parts if part).strip()
