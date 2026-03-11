from __future__ import annotations

from typing import Any

try:
    from ..models.typed_memory import Episode, EvidenceRef, StateFact, TypedMemoryObject
    from .search_service import SearchService
except ImportError:  # pragma: no cover - top-level import fallback
    from models.typed_memory import Episode, EvidenceRef, StateFact, TypedMemoryObject
    from services.search_service import SearchService

_MAX_HISTORY_LINEAGE_DEPTH = 32


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
        query_mode: str = 'all',
    ) -> tuple[list[TypedMemoryObject], dict[str, str], dict[str, Any]]:
        scope = list(effective_group_ids or [])
        normalized_max_results = max(1, int(max_results or 1))

        if self.graphiti_service is None:
            return [], {}, {'enabled': False, 'reason': 'graphiti_service_unavailable'}
        if not self.search_service.includes_observational_memory(scope):
            return [], {}, {'enabled': False, 'reason': 'om_not_in_scope'}

        if query_mode == 'history':
            return await self._project_history(
                query=query,
                scope=scope,
                object_types=object_types,
                max_results=normalized_max_results,
            )

        return await self._project_non_history(
            query=query,
            scope=scope,
            object_types=object_types,
            max_results=normalized_max_results,
        )

    async def _project_non_history(
        self,
        *,
        query: str,
        scope: list[str],
        object_types: set[str],
        max_results: int,
    ) -> tuple[list[TypedMemoryObject], dict[str, str], dict[str, Any]]:
        objects: list[TypedMemoryObject] = []
        search_text_overrides: dict[str, str] = {}
        episode_count = 0
        state_count = 0

        if not object_types or 'episode' in object_types:
            node_rows = await self.search_service.search_observational_nodes(
                graphiti_service=self.graphiti_service,
                query=query,
                group_ids=scope,
                max_nodes=max_results,
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
                max_facts=max_results,
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
            'max_results': max_results,
            'history_mode': False,
        }
        return objects, search_text_overrides, limits

    async def _project_history(
        self,
        *,
        query: str,
        scope: list[str],
        object_types: set[str],
        max_results: int,
    ) -> tuple[list[TypedMemoryObject], dict[str, str], dict[str, Any]]:
        groups_considered = self.search_service._om_groups_in_scope(scope)
        requested_object_types = sorted(object_types) if object_types else []
        unsupported_object_types = [
            object_type
            for object_type in requested_object_types
            if object_type != 'episode'
        ]

        if object_types and 'episode' not in object_types:
            return [], {}, {
                'enabled': True,
                'reason': 'history_projection_requires_episode_scope',
                'groups_considered': groups_considered,
                'episodes_projected': 0,
                'state_projected': 0,
                'max_results': max_results,
                'history_mode': True,
                'history_candidates': 0,
                'history_lineages_projected': 0,
                'history_state_projection_supported': False,
                'unsupported_object_types': unsupported_object_types,
                'skipped_candidates': [],
            }

        node_rows = await self.search_service.search_observational_nodes(
            graphiti_service=self.graphiti_service,
            query=query,
            group_ids=scope,
            max_nodes=max_results,
            entity_types=['OMNode'],
        )

        objects: list[TypedMemoryObject] = []
        search_text_overrides: dict[str, str] = {}
        skipped_candidates: list[dict[str, str]] = []
        history_lineages_projected = 0
        seen_component_nodes: set[tuple[str, str]] = set()

        driver = await self._graphiti_driver()
        if driver is None:
            return [], {}, {
                'enabled': False,
                'reason': 'graphiti_driver_unavailable',
                'groups_considered': groups_considered,
                'episodes_projected': 0,
                'state_projected': 0,
                'max_results': max_results,
                'history_mode': True,
                'history_candidates': len(node_rows),
                'history_lineages_projected': 0,
                'history_state_projection_supported': False,
                'unsupported_object_types': unsupported_object_types,
                'skipped_candidates': [],
            }

        for row in node_rows:
            group_id = str(row.get('group_id') or '').strip()
            seed_node_id = str(row.get('uuid') or '').strip()
            if not group_id or not seed_node_id:
                continue
            if (group_id, seed_node_id) in seen_component_nodes:
                continue

            lineage = await self._load_linear_node_lineage(
                driver=driver,
                group_id=group_id,
                seed_node_id=seed_node_id,
            )
            seen_component_nodes.update(lineage['member_keys'])
            if lineage['status'] != 'ok':
                skipped_candidates.append(
                    {
                        'group_id': group_id,
                        'node_id': seed_node_id,
                        'reason': str(lineage['reason']),
                    }
                )
                continue

            members = lineage['members']
            if len(members) < 2:
                skipped_candidates.append(
                    {
                        'group_id': group_id,
                        'node_id': seed_node_id,
                        'reason': 'no_explicit_supersession_lineage',
                    }
                )
                continue

            history_lineages_projected += 1
            for member in members:
                obj = self._episode_from_lineage_member(member)
                if obj is None:
                    continue
                objects.append(obj)
                search_text_overrides[obj.object_id] = self._node_search_text(
                    {
                        'name': member.get('title'),
                        'summary': member.get('summary'),
                        'group_id': member.get('group_id'),
                        'attributes': {
                            'semantic_domain': member.get('semantic_domain'),
                            'status': member.get('status'),
                        },
                    }
                )

        limits = {
            'enabled': True,
            'reason': 'projected_history',
            'groups_considered': groups_considered,
            'episodes_projected': len(objects),
            'state_projected': 0,
            'max_results': max_results,
            'history_mode': True,
            'history_candidates': len(node_rows),
            'history_lineages_projected': history_lineages_projected,
            'history_state_projection_supported': False,
            'unsupported_object_types': unsupported_object_types,
            'skipped_candidates': skipped_candidates,
        }
        return objects, search_text_overrides, limits

    async def _graphiti_driver(self) -> Any | None:
        if self.graphiti_service is None:
            return None
        client = await self.graphiti_service.get_client()
        return getattr(client, 'driver', None)

    async def _load_linear_node_lineage(
        self,
        *,
        driver: Any,
        group_id: str,
        seed_node_id: str,
    ) -> dict[str, Any]:
        if driver is None:
            return {
                'status': 'skip',
                'reason': 'graphiti_driver_unavailable',
                'members': [],
                'member_keys': {(group_id, seed_node_id)},
            }

        records, _, _ = await driver.execute_query(
            """
            MATCH (seed:OMNode)
            WHERE seed.group_id = $group_id
              AND (
                seed.node_id = $seed_node_id
                OR seed.uuid = $seed_node_id
              )
            CALL {
                WITH seed
                MATCH path = (seed)-[:SUPERSEDES*0..32]-(member:OMNode)
                WHERE member.group_id = $group_id
                RETURN collect(DISTINCT member) AS members
            }
            UNWIND members AS member
            OPTIONAL MATCH (member)-[rel:SUPERSEDES]->(older:OMNode)
            WHERE older IN members
            RETURN coalesce(member.node_id, member.uuid, '') AS node_id,
                   coalesce(member.uuid, member.node_id, '') AS uuid,
                   coalesce(member.group_id, $group_id) AS group_id,
                   coalesce(member.content, '') AS content,
                   coalesce(member.last_observed_at, member.created_at) AS created_at,
                   coalesce(member.status, 'open') AS status,
                   coalesce(member.semantic_domain, '') AS semantic_domain,
                   collect(
                       DISTINCT CASE
                           WHEN older IS NULL THEN NULL
                           ELSE {
                               target_id: coalesce(older.node_id, older.uuid, ''),
                               created_at: coalesce(rel.created_at, member.last_observed_at, member.created_at),
                               relation_uuid: coalesce(rel.uuid, '')
                           }
                       END
                   ) AS supersedes
            ORDER BY created_at ASC, node_id ASC
            """,
            group_id=group_id,
            seed_node_id=seed_node_id,
            routing_='r',
        )
        rows = [record.data() if hasattr(record, 'data') else dict(record) for record in records]
        if not rows:
            return {
                'status': 'skip',
                'reason': 'no_explicit_supersession_lineage',
                'members': [],
                'member_keys': {(group_id, seed_node_id)},
            }

        members: dict[str, dict[str, Any]] = {}
        outgoing: dict[str, list[dict[str, Any]]] = {}
        member_keys: set[tuple[str, str]] = set()

        for row in rows:
            node_id = str(row.get('node_id') or row.get('uuid') or '').strip()
            row_group_id = str(row.get('group_id') or group_id).strip() or group_id
            if not node_id:
                continue
            member_keys.add((row_group_id, node_id))
            supersedes = []
            seen_targets: set[str] = set()
            for item in row.get('supersedes') or []:
                if not isinstance(item, dict):
                    continue
                target_id = str(item.get('target_id') or '').strip()
                if not target_id or target_id in seen_targets:
                    continue
                seen_targets.add(target_id)
                supersedes.append(
                    {
                        'target_id': target_id,
                        'created_at': _coerce_timestamp(item.get('created_at')),
                        'relation_uuid': str(item.get('relation_uuid') or '').strip() or None,
                    }
                )
            member = {
                'node_id': node_id,
                'uuid': str(row.get('uuid') or node_id).strip() or node_id,
                'group_id': row_group_id,
                'summary': str(row.get('content') or '').strip(),
                'title': (str(row.get('content') or '').strip()[:120] or node_id),
                'created_at': _coerce_timestamp(row.get('created_at')),
                'status': str(row.get('status') or '').strip() or None,
                'semantic_domain': str(row.get('semantic_domain') or '').strip() or None,
            }
            members[node_id] = member
            outgoing[node_id] = supersedes

        if not members:
            return {
                'status': 'skip',
                'reason': 'no_explicit_supersession_lineage',
                'members': [],
                'member_keys': member_keys or {(group_id, seed_node_id)},
            }

        incoming: dict[str, list[dict[str, Any]]] = {node_id: [] for node_id in members}
        for node_id, relations in outgoing.items():
            if len(relations) > 1:
                return {
                    'status': 'skip',
                    'reason': 'ambiguous_supersession_graph',
                    'members': [],
                    'member_keys': member_keys,
                }
            for relation in relations:
                target_id = relation['target_id']
                if target_id not in members:
                    return {
                        'status': 'skip',
                        'reason': 'ambiguous_supersession_graph',
                        'members': [],
                        'member_keys': member_keys,
                    }
                incoming[target_id].append(
                    {
                        'source_id': node_id,
                        'created_at': relation.get('created_at') or members[node_id].get('created_at'),
                    }
                )

        if any(len(relations) > 1 for relations in incoming.values()):
            return {
                'status': 'skip',
                'reason': 'ambiguous_supersession_graph',
                'members': [],
                'member_keys': member_keys,
            }

        current_candidates = [node_id for node_id, relations in incoming.items() if not relations]
        root_candidates = [node_id for node_id, relations in outgoing.items() if not relations]
        if len(current_candidates) != 1 or len(root_candidates) != 1:
            return {
                'status': 'skip',
                'reason': 'ambiguous_supersession_graph',
                'members': [],
                'member_keys': member_keys,
            }

        newest_to_oldest: list[str] = []
        visited: set[str] = set()
        current_id = current_candidates[0]
        while True:
            if current_id in visited:
                return {
                    'status': 'skip',
                    'reason': 'ambiguous_supersession_graph',
                    'members': [],
                    'member_keys': member_keys,
                }
            visited.add(current_id)
            newest_to_oldest.append(current_id)
            relations = outgoing.get(current_id) or []
            if not relations:
                break
            current_id = relations[0]['target_id']
            if len(newest_to_oldest) > _MAX_HISTORY_LINEAGE_DEPTH:
                return {
                    'status': 'skip',
                    'reason': 'ambiguous_supersession_graph',
                    'members': [],
                    'member_keys': member_keys,
                }

        if len(visited) != len(members):
            return {
                'status': 'skip',
                'reason': 'ambiguous_supersession_graph',
                'members': [],
                'member_keys': member_keys,
            }

        oldest_to_newest = list(reversed(newest_to_oldest))
        root_object_id = self._episode_object_id(group_id, oldest_to_newest[0])
        superseded_by_meta: dict[str, dict[str, Any]] = {}
        for newer_id, relations in outgoing.items():
            if not relations:
                continue
            relation = relations[0]
            superseded_by_meta[relation['target_id']] = {
                'newer_id': newer_id,
                'invalid_at': relation.get('created_at') or members[newer_id].get('created_at'),
            }

        projected_members: list[dict[str, Any]] = []
        total_versions = len(oldest_to_newest)
        for index, node_id in enumerate(oldest_to_newest, start=1):
            member = dict(members[node_id])
            parent_node_id = oldest_to_newest[index - 2] if index > 1 else None
            superseded = superseded_by_meta.get(node_id) or {}
            newer_node_id = superseded.get('newer_id')
            member.update(
                {
                    'object_id': self._episode_object_id(group_id, node_id),
                    'root_id': root_object_id,
                    'parent_id': self._episode_object_id(group_id, parent_node_id) if parent_node_id else None,
                    'version': index,
                    'is_current': index == total_versions,
                    'superseded_by': self._episode_object_id(group_id, newer_node_id) if newer_node_id else None,
                    'invalid_at': superseded.get('invalid_at'),
                }
            )
            projected_members.append(member)

        return {
            'status': 'ok',
            'reason': 'linear_supersession_lineage',
            'members': projected_members,
            'member_keys': member_keys,
        }

    def _episode_object_id(self, group_id: str, node_id: str | None) -> str | None:
        normalized_node_id = str(node_id or '').strip()
        if not normalized_node_id:
            return None
        return f'om_episode:{group_id}:{normalized_node_id}'

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

    def _episode_from_lineage_member(self, member: dict[str, Any]) -> Episode | None:
        group_id = str(member.get('group_id') or '').strip()
        node_id = str(member.get('node_id') or '').strip()
        object_id = str(member.get('object_id') or '').strip()
        root_id = str(member.get('root_id') or '').strip()
        if not group_id or not node_id or not object_id or not root_id:
            return None

        title = str(member.get('title') or '').strip() or node_id
        summary = str(member.get('summary') or '').strip() or title
        created_at = _coerce_timestamp(member.get('created_at'))
        annotations = [
            'om_native',
            str(member.get('semantic_domain') or '').strip(),
            str(member.get('status') or '').strip(),
            'history_lineage',
        ]
        annotations = [item for item in annotations if item]

        return Episode(
            object_id=object_id,
            root_id=root_id,
            parent_id=member.get('parent_id'),
            version=int(member.get('version') or 1),
            is_current=bool(member.get('is_current')),
            source_lane=group_id,
            source_key=f'om:{group_id}:node:{node_id}',
            policy_scope='private',
            visibility_scope='private',
            title=title,
            summary=summary,
            annotations=annotations,
            created_at=created_at or '2026-01-01T00:00:00Z',
            valid_at=created_at,
            invalid_at=_coerce_timestamp(member.get('invalid_at')),
            superseded_by=member.get('superseded_by'),
            evidence_refs=[
                EvidenceRef(
                    kind='event_log',
                    source_system='om',
                    locator={'system': 'om', 'stream': f'{group_id}:node', 'event_id': node_id},
                    title=title,
                    snippet=summary,
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
