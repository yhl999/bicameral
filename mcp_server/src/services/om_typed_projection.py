from __future__ import annotations

import hashlib
from collections import Counter, defaultdict, deque
from typing import Any

try:
    from ..models.typed_memory import Episode, EvidenceRef, StateFact, TypedMemoryObject
    from .search_service import SearchService
except ImportError:  # pragma: no cover - top-level import fallback
    from models.typed_memory import Episode, EvidenceRef, StateFact, TypedMemoryObject
    from services.search_service import SearchService

_MAX_HISTORY_COMPONENT_DEPTH = 256
_MAX_RELATION_NEIGHBOR_LIMIT = 64
_ALLOWED_HISTORY_OBJECT_TYPES = {'episode', 'state_fact'}
_CLOSURE_RELATION_TYPES = {'SUPERSEDES', 'RESOLVES'}


def _coerce_timestamp(value: Any) -> str | None:
    text = str(value).strip() if value not in (None, '') else ''
    return text or None


def _timestamp_sort_key(value: Any) -> tuple[int, str]:
    text = _coerce_timestamp(value)
    if text is None:
        return (1, '')
    return (0, text)


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

        driver = await self._graphiti_driver()
        if driver is None:
            return [], {}, {'enabled': False, 'reason': 'graphiti_driver_unavailable'}

        return await self._project_history_aware_query(
            query=query,
            scope=scope,
            object_types=object_types,
            max_results=normalized_max_results,
            query_mode=query_mode,
            driver=driver,
        )

    async def _project_history_aware_query(
        self,
        *,
        query: str,
        scope: list[str],
        object_types: set[str],
        max_results: int,
        query_mode: str,
        driver: Any,
    ) -> tuple[list[TypedMemoryObject], dict[str, str], dict[str, Any]]:
        groups_considered = self.search_service._om_groups_in_scope(scope)
        requested_object_types = sorted(object_types) if object_types else []
        unsupported_object_types = [
            object_type
            for object_type in requested_object_types
            if object_type not in _ALLOWED_HISTORY_OBJECT_TYPES
        ]
        want_episodes = not object_types or 'episode' in object_types
        want_state = not object_types or 'state_fact' in object_types

        node_seed_rows = await self.search_service.search_observational_nodes(
            graphiti_service=self.graphiti_service,
            query=query,
            group_ids=scope,
            max_nodes=max_results,
            entity_types=['OMNode'],
        )

        direct_fact_rows: list[dict[str, Any]] = []
        if want_state or query_mode == 'history':
            direct_fact_rows = await self.search_service.search_observational_facts(
                graphiti_service=self.graphiti_service,
                query=query,
                group_ids=scope,
                max_facts=max_results,
                center_node_uuid=None,
            )

        seed_keys: list[tuple[str, str]] = []
        seen_seed_keys: set[tuple[str, str]] = set()
        for row in node_seed_rows:
            group_id = str(row.get('group_id') or '').strip()
            node_id = str(row.get('uuid') or '').strip()
            if not group_id or not node_id:
                continue
            key = (group_id, node_id)
            if key in seen_seed_keys:
                continue
            seen_seed_keys.add(key)
            seed_keys.append(key)
        for row in direct_fact_rows:
            group_id = str(row.get('group_id') or '').strip()
            for node_id in (
                str(row.get('source_node_uuid') or row.get('source_node_id') or '').strip(),
                str(row.get('target_node_uuid') or row.get('target_node_id') or '').strip(),
            ):
                if not group_id or not node_id:
                    continue
                key = (group_id, node_id)
                if key in seen_seed_keys:
                    continue
                seen_seed_keys.add(key)
                seed_keys.append(key)

        components: list[dict[str, Any]] = []
        skipped_candidates: list[dict[str, str]] = []
        seen_component_nodes: set[tuple[str, str]] = set()
        for group_id, seed_node_id in seed_keys:
            if (group_id, seed_node_id) in seen_component_nodes:
                continue
            component = await self._load_node_history_component(
                driver=driver,
                group_id=group_id,
                seed_node_id=seed_node_id,
            )
            seen_component_nodes.update(component['member_keys'])
            if component['status'] != 'ok':
                skipped_candidates.append(
                    {
                        'group_id': group_id,
                        'node_id': seed_node_id,
                        'reason': str(component['reason']),
                    }
                )
                continue
            components.append(component)

        relation_rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for row in direct_fact_rows:
            self._store_relation_row(relation_rows_by_key, row)

        if components:
            neighbor_limit = min(_MAX_RELATION_NEIGHBOR_LIMIT, max(16, max_results * 4))
            for component in components:
                for member in component['members']:
                    member_group_id = str(member.get('group_id') or '').strip()
                    member_node_id = str(member.get('node_id') or '').strip()
                    if not member_group_id or not member_node_id:
                        continue
                    rows = await self.search_service.search_observational_facts(
                        graphiti_service=self.graphiti_service,
                        query='',
                        group_ids=[member_group_id],
                        max_facts=neighbor_limit,
                        center_node_uuid=member_node_id,
                    )
                    for row in rows:
                        self._store_relation_row(relation_rows_by_key, row)

        relation_rows = list(relation_rows_by_key.values())
        closure_rows_by_target = self._closure_rows_by_target_node(relation_rows)

        node_history_index: dict[tuple[str, str], dict[str, Any]] = {}
        episode_objects: list[Episode] = []
        search_text_overrides: dict[str, str] = {}
        topology_counts: Counter[str] = Counter()
        component_projection_count = 0

        for component in components:
            projected_episodes = self._project_component_episodes(
                component=component,
                closure_rows_by_target=closure_rows_by_target,
            )
            if not projected_episodes:
                skipped_candidates.append(
                    {
                        'group_id': str(component.get('group_id') or ''),
                        'node_id': str(component.get('seed_node_id') or ''),
                        'reason': 'component_projection_failed',
                    }
                )
                continue
            component_projection_count += 1
            topology_counts[str(component.get('topology') or 'unknown')] += 1
            for episode in projected_episodes:
                episode_objects.append(episode)
                search_text_overrides[episode.object_id] = self._episode_search_text(episode)
                node_key = self._node_key_from_source_key(episode.source_key)
                if node_key is not None:
                    node_history_index[node_key] = {
                        'object_id': episode.object_id,
                        'root_id': episode.root_id,
                        'version': int(episode.version),
                        'is_current': bool(episode.is_current),
                        'invalid_at': _coerce_timestamp(episode.invalid_at),
                        'topology': self._annotation_value(episode.annotations, 'history_topology'),
                        'ordering': self._annotation_value(episode.annotations, 'history_ordering'),
                        'source_lane': episode.source_lane,
                    }

        state_objects: list[StateFact] = []
        relation_lineage_count = 0
        if want_state and relation_rows:
            state_objects, relation_lineage_count = self._project_relation_history(
                relation_rows=relation_rows,
                node_history_index=node_history_index,
            )
            for fact in state_objects:
                search_text_overrides[fact.object_id] = self._fact_search_text_from_object(fact)

        objects: list[TypedMemoryObject] = []
        if want_episodes:
            objects.extend(episode_objects)
        if want_state:
            objects.extend(state_objects)

        reason = 'projected_history' if query_mode == 'history' else 'projected'
        limits = {
            'enabled': True,
            'reason': reason,
            'groups_considered': groups_considered,
            'episodes_projected': len(episode_objects) if want_episodes else 0,
            'state_projected': len(state_objects) if want_state else 0,
            'max_results': max_results,
            'history_mode': query_mode == 'history',
            'history_candidates': len(seed_keys),
            'history_lineages_projected': component_projection_count,
            'history_relation_lineages_projected': relation_lineage_count,
            'history_state_projection_supported': want_state,
            'unsupported_object_types': unsupported_object_types,
            'skipped_candidates': skipped_candidates,
            'history_topology_counts': dict(sorted(topology_counts.items())),
            'history_relation_candidates': len(relation_rows),
        }
        return objects, search_text_overrides, limits

    async def _graphiti_driver(self) -> Any | None:
        if self.graphiti_service is None:
            return None
        client = await self.graphiti_service.get_client()
        return getattr(client, 'driver', None)

    async def _load_node_history_component(
        self,
        *,
        driver: Any,
        group_id: str,
        seed_node_id: str,
    ) -> dict[str, Any]:
        records, _, _ = await driver.execute_query(
            f"""
            MATCH (seed:OMNode)
            WHERE seed.group_id = $group_id
              AND (
                seed.node_id = $seed_node_id
                OR seed.uuid = $seed_node_id
              )
            CALL {{
                WITH seed
                MATCH path = (seed)-[:SUPERSEDES*0..{_MAX_HISTORY_COMPONENT_DEPTH}]-(member:OMNode)
                WHERE member.group_id = $group_id
                RETURN collect(DISTINCT member) AS members
            }}
            UNWIND members AS member
            OPTIONAL MATCH (member)-[rel:SUPERSEDES]->(older:OMNode)
            WHERE older IN members
            RETURN coalesce(member.node_id, member.uuid, '') AS node_id,
                   coalesce(member.uuid, member.node_id, '') AS uuid,
                   coalesce(member.group_id, $group_id) AS group_id,
                   coalesce(member.content, '') AS content,
                   coalesce(member.last_observed_at, member.first_observed_at, member.created_at) AS created_at,
                   coalesce(member.status, 'open') AS status,
                   coalesce(member.semantic_domain, '') AS semantic_domain,
                   collect(
                       DISTINCT CASE
                           WHEN older IS NULL THEN NULL
                           ELSE {{
                               target_id: coalesce(older.node_id, older.uuid, ''),
                               created_at: coalesce(rel.created_at, member.last_observed_at, member.first_observed_at, member.created_at),
                               relation_uuid: coalesce(rel.uuid, '')
                           }}
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
                'reason': 'missing_seed_node',
                'seed_node_id': seed_node_id,
                'group_id': group_id,
                'members': [],
                'member_keys': {(group_id, seed_node_id)},
            }

        members_by_id: dict[str, dict[str, Any]] = {}
        outgoing: dict[str, list[dict[str, Any]]] = {}
        incoming: dict[str, list[dict[str, Any]]] = {}
        member_keys: set[tuple[str, str]] = set()

        for row in rows:
            node_id = str(row.get('node_id') or row.get('uuid') or '').strip()
            row_group_id = str(row.get('group_id') or group_id).strip() or group_id
            if not node_id:
                continue
            member_keys.add((row_group_id, node_id))
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
            members_by_id[node_id] = member
            outgoing[node_id] = []
            incoming.setdefault(node_id, [])

        if not members_by_id:
            return {
                'status': 'skip',
                'reason': 'missing_component_members',
                'seed_node_id': seed_node_id,
                'group_id': group_id,
                'members': [],
                'member_keys': member_keys or {(group_id, seed_node_id)},
            }

        for row in rows:
            node_id = str(row.get('node_id') or row.get('uuid') or '').strip()
            if not node_id or node_id not in members_by_id:
                continue
            seen_targets: set[str] = set()
            for item in row.get('supersedes') or []:
                if not isinstance(item, dict):
                    continue
                target_id = str(item.get('target_id') or '').strip()
                if not target_id or target_id not in members_by_id or target_id in seen_targets:
                    continue
                seen_targets.add(target_id)
                edge = {
                    'source_id': node_id,
                    'target_id': target_id,
                    'created_at': _coerce_timestamp(item.get('created_at')) or members_by_id[node_id].get('created_at'),
                    'relation_uuid': str(item.get('relation_uuid') or '').strip() or None,
                }
                outgoing[node_id].append(edge)
                incoming.setdefault(target_id, []).append(edge)

        for node_id in members_by_id:
            outgoing.setdefault(node_id, [])
            incoming.setdefault(node_id, [])

        ordered_node_ids, topology, ordering = self._ordered_component_node_ids(
            members_by_id=members_by_id,
            outgoing=outgoing,
            incoming=incoming,
        )
        anchor_node_id = ordered_node_ids[0]
        anchor_object_id = self._episode_object_id(group_id, anchor_node_id)
        root_id = (
            anchor_object_id
            if topology in {'singleton', 'linear'}
            else self._history_component_root_id(group_id=group_id, anchor_node_id=anchor_node_id)
        )

        members = [members_by_id[node_id] for node_id in ordered_node_ids]
        return {
            'status': 'ok',
            'reason': 'projectable_history_component',
            'seed_node_id': seed_node_id,
            'group_id': group_id,
            'members': members,
            'members_by_id': members_by_id,
            'outgoing': outgoing,
            'incoming': incoming,
            'ordered_node_ids': ordered_node_ids,
            'topology': topology,
            'ordering': ordering,
            'anchor_node_id': anchor_node_id,
            'root_id': root_id,
            'member_keys': member_keys,
        }

    def _ordered_component_node_ids(
        self,
        *,
        members_by_id: dict[str, dict[str, Any]],
        outgoing: dict[str, list[dict[str, Any]]],
        incoming: dict[str, list[dict[str, Any]]],
    ) -> tuple[list[str], str, str]:
        if len(members_by_id) == 1 and not any(outgoing.values()) and not any(incoming.values()):
            return list(members_by_id.keys()), 'singleton', 'singleton'

        older_to_newer: dict[str, list[str]] = {node_id: [] for node_id in members_by_id}
        indegree: dict[str, int] = {node_id: 0 for node_id in members_by_id}
        for newer_id, edges in outgoing.items():
            for edge in edges:
                older_id = edge['target_id']
                older_to_newer.setdefault(older_id, []).append(newer_id)
                indegree[newer_id] = indegree.get(newer_id, 0) + 1

        queue = deque(
            sorted(
                (node_id for node_id, degree in indegree.items() if degree == 0),
                key=lambda node_id: self._member_sort_key(members_by_id[node_id]),
            )
        )
        ordered_node_ids: list[str] = []
        working_indegree = dict(indegree)
        while queue:
            node_id = queue.popleft()
            ordered_node_ids.append(node_id)
            neighbors = sorted(
                older_to_newer.get(node_id, []),
                key=lambda neighbor_id: self._member_sort_key(members_by_id[neighbor_id]),
            )
            for neighbor_id in neighbors:
                working_indegree[neighbor_id] -= 1
                if working_indegree[neighbor_id] == 0:
                    queue.append(neighbor_id)

        if len(ordered_node_ids) != len(members_by_id):
            ordered_node_ids = sorted(members_by_id.keys(), key=lambda node_id: self._member_sort_key(members_by_id[node_id]))
            return ordered_node_ids, 'cyclic', 'timestamp_fallback'

        current_candidates = [node_id for node_id, edges in incoming.items() if not edges]
        oldest_candidates = [node_id for node_id, edges in outgoing.items() if not edges]
        linear = (
            len(current_candidates) == 1
            and len(oldest_candidates) == 1
            and all(len(edges) <= 1 for edges in outgoing.values())
            and all(len(edges) <= 1 for edges in incoming.values())
        )
        topology = 'linear' if linear else 'branching'
        return ordered_node_ids, topology, 'topological'

    def _project_component_episodes(
        self,
        *,
        component: dict[str, Any],
        closure_rows_by_target: dict[tuple[str, str], list[dict[str, Any]]],
    ) -> list[Episode]:
        group_id = str(component.get('group_id') or '').strip()
        root_id = str(component.get('root_id') or '').strip()
        topology = str(component.get('topology') or 'unknown').strip() or 'unknown'
        ordering = str(component.get('ordering') or 'unknown').strip() or 'unknown'
        members_by_id = component.get('members_by_id') or {}
        outgoing = component.get('outgoing') or {}
        incoming = component.get('incoming') or {}
        ordered_node_ids = component.get('ordered_node_ids') or []
        if not group_id or not root_id or not members_by_id or not ordered_node_ids:
            return []

        member_version = {node_id: index for index, node_id in enumerate(ordered_node_ids, start=1)}
        total_versions = len(ordered_node_ids)
        version_basis = 'lineage_sequence' if ordering == 'topological' else 'chronology_only'
        projected: list[Episode] = []

        for node_id in ordered_node_ids:
            member = members_by_id[node_id]
            object_id = self._episode_object_id(group_id, node_id)
            if object_id is None:
                continue

            older_edges = sorted(outgoing.get(node_id, []), key=lambda edge: (_timestamp_sort_key(edge.get('created_at')), edge.get('target_id') or ''))
            newer_edges = sorted(incoming.get(node_id, []), key=lambda edge: (_timestamp_sort_key(edge.get('created_at')), edge.get('source_id') or ''))
            closure_edges = sorted(
                list(newer_edges) + list(closure_rows_by_target.get((group_id, node_id), [])),
                key=lambda edge: (_timestamp_sort_key(edge.get('created_at')), edge.get('relation_type') or '', edge.get('source_id') or ''),
            )
            closure_invalid_at = self._first_timestamp(edge.get('created_at') for edge in closure_edges)
            invalidation_kind = self._episode_invalidation_kind(newer_edges=newer_edges, closure_edges=closure_edges)
            parent_candidates = [
                candidate
                for candidate in (self._episode_object_id(group_id, edge.get('target_id')) for edge in older_edges)
                if candidate is not None
            ]
            successor_candidates = [
                candidate
                for candidate in (self._episode_object_id(group_id, edge.get('source_id')) for edge in newer_edges)
                if candidate is not None
            ]
            direct_parent_id = parent_candidates[0] if len(parent_candidates) == 1 else None
            direct_superseded_by = successor_candidates[0] if len(successor_candidates) == 1 else None

            lifecycle_status = 'asserted'
            if invalidation_kind == 'superseded':
                lifecycle_status = 'superseded'
            elif invalidation_kind == 'invalidated':
                lifecycle_status = 'invalidated'

            annotations = [
                'om_native',
                str(member.get('semantic_domain') or '').strip(),
                str(member.get('status') or '').strip(),
                'history_projected',
                f'history_topology:{topology}',
                f'history_ordering:{ordering}',
                f'history_component_size:{total_versions}',
                f'history_predecessor_count:{len(older_edges)}',
                f'history_successor_count:{len(newer_edges)}',
                f'history_closure_count:{len(closure_edges)}',
                f'history_version_basis:{version_basis}',
            ]
            if topology != 'linear':
                annotations.append('history_ambiguous')
            if invalidation_kind:
                annotations.append(f'history_invalidation:{invalidation_kind}')
            annotations = [item for item in annotations if item]

            transition_refs = self._transition_evidence_refs(
                group_id=group_id,
                edges=older_edges + newer_edges + closure_rows_by_target.get((group_id, node_id), []),
            )
            evidence_refs = [
                EvidenceRef(
                    kind='event_log',
                    source_system='om',
                    locator={'system': 'om', 'stream': f'{group_id}:node', 'event_id': node_id},
                    title=str(member.get('title') or node_id),
                    snippet=str(member.get('summary') or member.get('title') or node_id),
                    observed_at=_coerce_timestamp(member.get('created_at')),
                )
            ]
            evidence_refs.extend(transition_refs)

            history_meta = self._compact_history_meta(
                {
                    'lineage_kind': 'om_node',
                    'lineage_basis': 'om_supersession_component',
                    'topology': topology,
                    'ordering': ordering,
                    'version_basis': version_basis,
                    'component_size': total_versions,
                    'predecessor_count': len(parent_candidates),
                    'successor_count': len(successor_candidates),
                    'closure_count': len(closure_edges),
                    'parent_candidates': parent_candidates,
                    'successor_candidates': successor_candidates,
                    'root_id': root_id,
                    'parent_id': direct_parent_id,
                    'version': member_version[node_id],
                    'is_current': closure_invalid_at is None,
                    'superseded_by': direct_superseded_by,
                    'invalid_at': closure_invalid_at,
                    'lifecycle_status': lifecycle_status,
                    'transition_reason': invalidation_kind,
                    'is_ambiguous': topology != 'linear',
                    'topology_flags': [flag for flag in [topology, 'competing_successors' if len(successor_candidates) > 1 else None] if flag],
                    'transition_evidence': self._evidence_payload(transition_refs),
                }
            )

            projected.append(
                Episode(
                    object_id=object_id,
                    root_id=root_id,
                    parent_id=direct_parent_id,
                    version=member_version[node_id],
                    is_current=closure_invalid_at is None,
                    source_lane=group_id,
                    source_key=f'om:{group_id}:node:{node_id}',
                    policy_scope='private',
                    visibility_scope='private',
                    title=str(member.get('title') or node_id),
                    summary=str(member.get('summary') or member.get('title') or node_id),
                    annotations=annotations,
                    history_meta=history_meta,
                    created_at=_coerce_timestamp(member.get('created_at')) or '2026-01-01T00:00:00Z',
                    valid_at=_coerce_timestamp(member.get('created_at')),
                    invalid_at=closure_invalid_at,
                    superseded_by=direct_superseded_by if len(successor_candidates) == 1 else None,
                    lifecycle_status=lifecycle_status,
                    evidence_refs=self._dedupe_evidence_refs(evidence_refs),
                )
            )

        return projected

    def _project_relation_history(
        self,
        *,
        relation_rows: list[dict[str, Any]],
        node_history_index: dict[tuple[str, str], dict[str, Any]],
    ) -> tuple[list[StateFact], int]:
        grouped_rows: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in relation_rows:
            group_id = str(row.get('group_id') or '').strip()
            relation_type = str(row.get('name') or row.get('relation_type') or '').strip().upper()
            source_node_id = str(row.get('source_node_uuid') or row.get('source_node_id') or '').strip()
            target_node_id = str(row.get('target_node_uuid') or row.get('target_node_id') or '').strip()
            relation_id = str(row.get('uuid') or '').strip()
            if not group_id or not relation_type or not source_node_id or not target_node_id or not relation_id:
                continue
            source_anchor = self._relation_anchor_id(node_history_index, group_id=group_id, node_id=source_node_id)
            target_anchor = self._relation_anchor_id(node_history_index, group_id=group_id, node_id=target_node_id)
            grouped_rows[(group_id, relation_type, source_anchor, target_anchor)].append(row)

        projected: list[StateFact] = []
        for group_key, rows in grouped_rows.items():
            group_id, relation_type, source_anchor, target_anchor = group_key
            ordered_rows = sorted(rows, key=lambda row: self._relation_order_key(row, node_history_index=node_history_index))
            if not ordered_rows:
                continue
            lineage_root_id = self._relation_root_id(
                group_id=group_id,
                relation_type=relation_type,
                source_anchor=source_anchor,
                target_anchor=target_anchor,
            )
            lineage_links = self._relation_lineage_links(rows=ordered_rows, group_id=group_id, node_history_index=node_history_index)
            lineage_topology = self._relation_lineage_topology(
                rows=ordered_rows,
                node_history_index=node_history_index,
                lineage_links=lineage_links,
            )
            version_basis = 'lineage_sequence' if lineage_topology == 'linear' else 'chronology_only'
            row_by_relation_id = {
                str(row.get('uuid') or '').strip(): row
                for row in ordered_rows
                if str(row.get('uuid') or '').strip()
            }

            for index, row in enumerate(ordered_rows, start=1):
                relation_id = str(row.get('uuid') or '').strip()
                source_node_id = str(row.get('source_node_uuid') or row.get('source_node_id') or '').strip()
                target_node_id = str(row.get('target_node_uuid') or row.get('target_node_id') or '').strip()
                created_at = _coerce_timestamp(row.get('created_at'))
                valid_at = _coerce_timestamp(row.get('valid_at')) or created_at
                direct_invalid_at = _coerce_timestamp(row.get('invalid_at'))
                object_id = self._state_object_id(group_id, relation_id)
                link_meta = lineage_links.get(relation_id, {})
                parent_candidates = [str(item) for item in (link_meta.get('parent_candidates') or []) if str(item).strip()]
                successor_candidates = [str(item) for item in (link_meta.get('successor_candidates') or []) if str(item).strip()]
                parent_id = parent_candidates[0] if lineage_topology == 'linear' and len(parent_candidates) == 1 else None
                successor_object_id = successor_candidates[0] if lineage_topology == 'linear' and len(successor_candidates) == 1 else None
                successor_row = None
                successor_at = None
                if successor_object_id is not None:
                    successor_relation_id = successor_object_id.rsplit(':', 1)[-1]
                    successor_row = row_by_relation_id.get(successor_relation_id)
                    successor_at = _coerce_timestamp(successor_row.get('created_at')) if successor_row is not None else None

                endpoint_invalidations = self._relation_endpoint_invalidations(
                    group_id=group_id,
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    created_at=created_at,
                    node_history_index=node_history_index,
                )
                invalidation = self._relation_invalidation(
                    created_at=created_at,
                    direct_invalid_at=direct_invalid_at,
                    successor_at=successor_at,
                    successor_object_id=successor_object_id,
                    endpoint_invalidations=endpoint_invalidations,
                )
                invalid_at = invalidation.get('invalid_at')
                invalidation_reason = invalidation.get('reason')
                lifecycle_status = 'asserted'
                if invalidation_reason == 'relation_replaced':
                    lifecycle_status = 'superseded'
                elif invalid_at is not None:
                    lifecycle_status = 'invalidated'

                source_version = self._node_version(node_history_index, group_id=group_id, node_id=source_node_id)
                target_version = self._node_version(node_history_index, group_id=group_id, node_id=target_node_id)
                relation_properties = row.get('attributes', {}).get('relation_properties') if isinstance(row.get('attributes'), dict) else {}
                if not isinstance(relation_properties, dict):
                    relation_properties = {}
                om_history = self._compact_history_meta(
                    {
                        'lineage_kind': 'om_relation',
                        'lineage_basis': 'om_relation_anchor_history',
                        'derivation_level': 'native' if direct_invalid_at is not None else 'hybrid',
                        'topology': lineage_topology,
                        'ordering': 'relation_sequence' if lineage_topology == 'linear' else 'relation_chronology',
                        'version_basis': version_basis,
                        'source_anchor': source_anchor,
                        'target_anchor': target_anchor,
                        'source_version': source_version,
                        'target_version': target_version,
                        'source_is_current': self._node_is_current(node_history_index, group_id=group_id, node_id=source_node_id),
                        'target_is_current': self._node_is_current(node_history_index, group_id=group_id, node_id=target_node_id),
                        'parent_candidates': parent_candidates,
                        'successor_candidates': successor_candidates,
                        'invalidation_reason': invalidation_reason,
                        'invalidation_basis': invalidation.get('basis'),
                        'direct_valid_at': valid_at,
                        'direct_invalid_at': direct_invalid_at,
                        'root_id': lineage_root_id,
                        'parent_id': parent_id,
                        'version': index,
                        'is_current': invalid_at is None,
                        'superseded_by': successor_object_id if invalidation_reason == 'relation_replaced' else None,
                        'invalid_at': invalid_at,
                        'lifecycle_status': lifecycle_status,
                        'is_ambiguous': lineage_topology != 'linear',
                        'topology_flags': [
                            flag
                            for flag in [
                                lineage_topology,
                                'competing_successors' if len(successor_candidates) > 1 else None,
                                'competing_predecessors' if len(parent_candidates) > 1 else None,
                                'missing_endpoint_versions' if link_meta.get('missing_versions') else None,
                            ]
                            if flag
                        ],
                        'relation_properties': relation_properties,
                    }
                )

                value = {
                    'fact': str(row.get('fact') or '').strip(),
                    'source_node_id': source_node_id,
                    'target_node_id': target_node_id,
                    'source_content': str((row.get('attributes') or {}).get('source_content') or row.get('source_content') or '').strip(),
                    'target_content': str((row.get('attributes') or {}).get('target_content') or row.get('target_content') or '').strip(),
                    'om_history': om_history,
                }

                evidence_refs = [
                    EvidenceRef(
                        kind='event_log',
                        source_system='om',
                        locator={'system': 'om', 'stream': f'{group_id}:relation', 'event_id': relation_id},
                        title=relation_type,
                        snippet=str(row.get('fact') or '').strip() or f"{value['source_content']} -> {value['target_content']}",
                        observed_at=created_at,
                    )
                ]
                endpoint_refs = self._endpoint_evidence_refs(
                    group_id=group_id,
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    source_content=value['source_content'],
                    target_content=value['target_content'],
                    created_at=created_at,
                )
                evidence_refs.extend(endpoint_refs)
                successor_ref = None
                if invalidation_reason == 'relation_replaced' and successor_row is not None:
                    successor_ref = EvidenceRef(
                        kind='event_log',
                        source_system='om',
                        locator={'system': 'om', 'stream': f'{group_id}:relation', 'event_id': str(successor_row.get('uuid') or '')},
                        title=f'{relation_type} successor',
                        snippet=str(successor_row.get('fact') or '').strip(),
                        observed_at=successor_at,
                    )
                    evidence_refs.append(successor_ref)

                transition_refs = list(endpoint_refs)
                if successor_ref is not None:
                    transition_refs.append(successor_ref)
                history_meta = dict(om_history)
                history_meta['transition_evidence'] = self._evidence_payload(
                    transition_refs,
                    roles_by_uri={
                        ref.canonical_uri: role
                        for ref, role in [
                            (endpoint_refs[0], 'source_endpoint') if len(endpoint_refs) > 0 else (None, None),
                            (endpoint_refs[1], 'target_endpoint') if len(endpoint_refs) > 1 else (None, None),
                            (successor_ref, 'successor_relation') if successor_ref is not None else (None, None),
                        ]
                        if ref is not None and role is not None
                    },
                )
                history_meta['endpoint_invalidations'] = endpoint_invalidations

                projected.append(
                    StateFact(
                        object_id=object_id,
                        root_id=lineage_root_id,
                        parent_id=parent_id,
                        version=index,
                        is_current=invalid_at is None,
                        source_lane=group_id,
                        source_key=f'om:{group_id}:relation:{relation_id}',
                        policy_scope='private',
                        visibility_scope='private',
                        fact_type='relationship',
                        subject=f'om_node:{source_node_id}',
                        predicate=f'om_relation:{relation_type.lower()}',
                        value=value,
                        scope='private',
                        history_meta=history_meta,
                        created_at=created_at or '2026-01-01T00:00:00Z',
                        valid_at=valid_at,
                        invalid_at=invalid_at,
                        superseded_by=successor_object_id if invalidation_reason == 'relation_replaced' else None,
                        lifecycle_status=lifecycle_status,
                        evidence_refs=self._dedupe_evidence_refs(evidence_refs),
                    )
                )

        projected.sort(key=lambda obj: (obj.root_id, 0 if obj.is_current else 1, -obj.version, obj.object_id))
        return projected, len(grouped_rows)

    def _relation_version_vector(
        self,
        row: dict[str, Any],
        *,
        node_history_index: dict[tuple[str, str], dict[str, Any]],
    ) -> tuple[int | None, int | None]:
        group_id = str(row.get('group_id') or '').strip()
        source_node_id = str(row.get('source_node_uuid') or row.get('source_node_id') or '').strip()
        target_node_id = str(row.get('target_node_uuid') or row.get('target_node_id') or '').strip()
        return (
            self._node_version(node_history_index, group_id=group_id, node_id=source_node_id),
            self._node_version(node_history_index, group_id=group_id, node_id=target_node_id),
        )

    def _relation_lineage_links(
        self,
        *,
        rows: list[dict[str, Any]],
        group_id: str,
        node_history_index: dict[tuple[str, str], dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for row in rows:
            relation_id = str(row.get('uuid') or '').strip()
            if not relation_id:
                continue
            source_version, target_version = self._relation_version_vector(row, node_history_index=node_history_index)
            entries.append(
                {
                    'relation_id': relation_id,
                    'object_id': self._state_object_id(group_id, relation_id),
                    'source_version': source_version,
                    'target_version': target_version,
                    'created_at': _coerce_timestamp(row.get('created_at')),
                }
            )

        def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
            if left['source_version'] is None or left['target_version'] is None:
                return False
            if right['source_version'] is None or right['target_version'] is None:
                return False
            return (
                left['source_version'] <= right['source_version']
                and left['target_version'] <= right['target_version']
                and (
                    left['source_version'] < right['source_version']
                    or left['target_version'] < right['target_version']
                )
            )

        lineage_links: dict[str, dict[str, Any]] = {}
        for entry in entries:
            predecessors = [candidate for candidate in entries if candidate['relation_id'] != entry['relation_id'] and dominates(candidate, entry)]
            maximal_predecessors = [
                candidate
                for candidate in predecessors
                if not any(
                    dominates(candidate, other)
                    for other in predecessors
                    if other['relation_id'] != candidate['relation_id']
                )
            ]
            successors = [candidate for candidate in entries if candidate['relation_id'] != entry['relation_id'] and dominates(entry, candidate)]
            minimal_successors = [
                candidate
                for candidate in successors
                if not any(
                    dominates(other, candidate)
                    for other in successors
                    if other['relation_id'] != candidate['relation_id']
                )
            ]
            lineage_links[entry['relation_id']] = {
                'parent_candidates': [candidate['object_id'] for candidate in maximal_predecessors if candidate.get('object_id')],
                'successor_candidates': [candidate['object_id'] for candidate in minimal_successors if candidate.get('object_id')],
                'missing_versions': entry['source_version'] is None or entry['target_version'] is None,
            }
        return lineage_links

    def _relation_lineage_topology(
        self,
        *,
        rows: list[dict[str, Any]],
        node_history_index: dict[tuple[str, str], dict[str, Any]],
        lineage_links: dict[str, dict[str, Any]] | None = None,
    ) -> str:
        if len(rows) <= 1:
            return 'singleton'
        if any(
            self._node_topology(node_history_index, group_id=str(row.get('group_id') or '').strip(), node_id=str(row.get('source_node_uuid') or row.get('source_node_id') or '').strip())
            not in {None, 'singleton', 'linear'}
            or self._node_topology(node_history_index, group_id=str(row.get('group_id') or '').strip(), node_id=str(row.get('target_node_uuid') or row.get('target_node_id') or '').strip())
            not in {None, 'singleton', 'linear'}
            for row in rows
        ):
            return 'branching'
        lineage_links = lineage_links or self._relation_lineage_links(
            rows=rows,
            group_id=str(rows[0].get('group_id') or '').strip(),
            node_history_index=node_history_index,
        )
        version_pairs = [self._relation_version_vector(row, node_history_index=node_history_index) for row in rows]
        if any(source_version is None or target_version is None for source_version, target_version in version_pairs):
            return 'branching'
        if len(set(version_pairs)) != len(version_pairs):
            return 'branching'
        roots = 0
        terminals = 0
        for row in rows:
            relation_id = str(row.get('uuid') or '').strip()
            link_meta = lineage_links.get(relation_id, {})
            predecessor_count = len(link_meta.get('parent_candidates') or [])
            successor_count = len(link_meta.get('successor_candidates') or [])
            if predecessor_count == 0:
                roots += 1
            if successor_count == 0:
                terminals += 1
            if predecessor_count > 1 or successor_count > 1:
                return 'branching'
        if roots == 1 and terminals == 1:
            return 'linear'
        return 'branching'

    def _relation_invalidation(
        self,
        *,
        created_at: str | None,
        direct_invalid_at: str | None,
        successor_at: str | None,
        successor_object_id: str | None,
        endpoint_invalidations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        candidates: list[dict[str, Any]] = []
        if direct_invalid_at is not None and (created_at is None or direct_invalid_at >= created_at):
            candidates.append(
                {
                    'invalid_at': direct_invalid_at,
                    'reason': 'relation_invalidated',
                    'basis': 'relation_edge_invalid_at',
                    'native': True,
                }
            )
        if successor_at is not None and successor_object_id is not None and (created_at is None or successor_at >= created_at):
            candidates.append(
                {
                    'invalid_at': successor_at,
                    'reason': 'relation_replaced',
                    'basis': 'relation_successor',
                    'native': False,
                    'successor_object_id': successor_object_id,
                }
            )
        for endpoint in endpoint_invalidations:
            endpoint_at = _coerce_timestamp(endpoint.get('invalid_at'))
            if endpoint_at is None or (created_at is not None and endpoint_at < created_at):
                continue
            candidates.append(
                {
                    'invalid_at': endpoint_at,
                    'reason': str(endpoint.get('reason') or 'endpoint_invalidated'),
                    'basis': 'endpoint_lineage',
                    'native': False,
                    'endpoint_object_id': endpoint.get('object_id'),
                }
            )
        if not candidates:
            return {}
        candidates.sort(
            key=lambda item: (
                _timestamp_sort_key(item.get('invalid_at')),
                0 if item.get('native') else 1,
                str(item.get('reason') or ''),
            )
        )
        return candidates[0]

    def _relation_endpoint_invalidations(
        self,
        *,
        group_id: str,
        source_node_id: str,
        target_node_id: str,
        created_at: str | None,
        node_history_index: dict[tuple[str, str], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for endpoint_node_id, reason_prefix in ((source_node_id, 'source_node'), (target_node_id, 'target_node')):
            meta = node_history_index.get((group_id, endpoint_node_id))
            if meta is None:
                continue
            invalid_at = _coerce_timestamp(meta.get('invalid_at'))
            if invalid_at is None:
                continue
            if created_at is not None and invalid_at <= created_at:
                continue
            candidates.append(
                {
                    'invalid_at': invalid_at,
                    'reason': f'{reason_prefix}_invalidated',
                    'node_id': endpoint_node_id,
                    'object_id': meta.get('object_id'),
                }
            )
        candidates.sort(key=lambda item: (_timestamp_sort_key(item.get('invalid_at')), str(item.get('node_id') or '')))
        return candidates

    def _closure_rows_by_target_node(self, relation_rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
        closure_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in relation_rows:
            relation_type = str(row.get('name') or row.get('relation_type') or '').strip().upper()
            if relation_type not in _CLOSURE_RELATION_TYPES:
                continue
            group_id = str(row.get('group_id') or '').strip()
            target_node_id = str(row.get('target_node_uuid') or row.get('target_node_id') or '').strip()
            source_node_id = str(row.get('source_node_uuid') or row.get('source_node_id') or '').strip()
            if not group_id or not target_node_id:
                continue
            closure_rows[(group_id, target_node_id)].append(
                {
                    'source_id': source_node_id,
                    'target_id': target_node_id,
                    'created_at': _coerce_timestamp(row.get('created_at')),
                    'relation_uuid': str(row.get('uuid') or '').strip() or None,
                    'relation_type': relation_type,
                }
            )
        for key, rows in closure_rows.items():
            rows.sort(key=lambda edge: (_timestamp_sort_key(edge.get('created_at')), edge.get('relation_type') or '', edge.get('source_id') or ''))
        return closure_rows

    def _store_relation_row(
        self,
        relation_rows_by_key: dict[tuple[str, str], dict[str, Any]],
        row: dict[str, Any],
    ) -> None:
        relation_id = str(row.get('uuid') or '').strip()
        group_id = str(row.get('group_id') or '').strip()
        relation_type = str(row.get('name') or row.get('relation_type') or '').strip()
        source_node_id = str(row.get('source_node_uuid') or row.get('source_node_id') or '').strip()
        target_node_id = str(row.get('target_node_uuid') or row.get('target_node_id') or '').strip()
        if not relation_id or not group_id or not relation_type or not source_node_id or not target_node_id:
            return
        attributes = row.get('attributes') if isinstance(row.get('attributes'), dict) else {}
        relation_properties = attributes.get('relation_properties') if isinstance(attributes.get('relation_properties'), dict) else {}
        normalized = {
            'uuid': relation_id,
            'name': relation_type,
            'relation_type': relation_type,
            'fact': str(row.get('fact') or '').strip(),
            'group_id': group_id,
            'source_node_uuid': source_node_id,
            'target_node_uuid': target_node_id,
            'created_at': _coerce_timestamp(row.get('created_at')),
            'valid_at': _coerce_timestamp(row.get('valid_at')),
            'invalid_at': _coerce_timestamp(row.get('invalid_at')),
            'attributes': {
                'source_content': str(attributes.get('source_content') or row.get('source_content') or '').strip(),
                'target_content': str(attributes.get('target_content') or row.get('target_content') or '').strip(),
                'relation_properties': relation_properties,
            },
        }
        relation_rows_by_key[(group_id, relation_id)] = normalized

    def _member_sort_key(self, member: dict[str, Any]) -> tuple[Any, ...]:
        return (_timestamp_sort_key(member.get('created_at')), str(member.get('node_id') or ''))

    def _history_component_root_id(self, *, group_id: str, anchor_node_id: str) -> str:
        return f'om_episode_component:{group_id}:{anchor_node_id}'

    def _relation_root_id(
        self,
        *,
        group_id: str,
        relation_type: str,
        source_anchor: str,
        target_anchor: str,
    ) -> str:
        digest = hashlib.sha256(
            f'{group_id}|{relation_type}|{source_anchor}|{target_anchor}'.encode()
        ).hexdigest()[:24]
        return f'om_state_lineage:{group_id}:{digest}'

    def _relation_anchor_id(
        self,
        node_history_index: dict[tuple[str, str], dict[str, Any]],
        *,
        group_id: str,
        node_id: str,
    ) -> str:
        meta = node_history_index.get((group_id, node_id))
        if meta is not None:
            return str(meta.get('root_id') or self._episode_object_id(group_id, node_id) or node_id)
        return str(self._episode_object_id(group_id, node_id) or f'om_episode:{group_id}:{node_id}')

    def _relation_order_key(
        self,
        row: dict[str, Any],
        *,
        node_history_index: dict[tuple[str, str], dict[str, Any]],
    ) -> tuple[Any, ...]:
        group_id = str(row.get('group_id') or '').strip()
        source_node_id = str(row.get('source_node_uuid') or row.get('source_node_id') or '').strip()
        target_node_id = str(row.get('target_node_uuid') or row.get('target_node_id') or '').strip()
        source_version = self._node_version(node_history_index, group_id=group_id, node_id=source_node_id) or 0
        target_version = self._node_version(node_history_index, group_id=group_id, node_id=target_node_id) or 0
        return (
            _timestamp_sort_key(row.get('created_at')),
            source_version,
            target_version,
            str(row.get('uuid') or ''),
        )

    def _state_object_id(self, group_id: str, relation_id: str | None) -> str | None:
        normalized_relation_id = str(relation_id or '').strip()
        if not normalized_relation_id:
            return None
        return f'om_state:{group_id}:{normalized_relation_id}'

    def _compact_history_meta(self, payload: dict[str, Any]) -> dict[str, Any]:
        compact: dict[str, Any] = {}
        for key, value in payload.items():
            if value is None or value == '' or value == [] or value == {}:
                continue
            compact[key] = value
        return compact

    def _evidence_payload(
        self,
        refs: list[EvidenceRef],
        *,
        roles_by_uri: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        roles_by_uri = roles_by_uri or {}
        for ref in refs:
            uri = str(ref.canonical_uri or '').strip()
            payload.append(
                self._compact_history_meta(
                    {
                        'canonical_uri': uri,
                        'kind': ref.kind,
                        'source_system': ref.source_system,
                        'title': ref.title,
                        'observed_at': ref.observed_at,
                        'role': roles_by_uri.get(uri),
                    }
                )
            )
        return payload

    def _node_version(
        self,
        node_history_index: dict[tuple[str, str], dict[str, Any]],
        *,
        group_id: str,
        node_id: str,
    ) -> int | None:
        meta = node_history_index.get((group_id, node_id))
        if meta is None:
            return None
        try:
            return int(meta.get('version'))
        except (TypeError, ValueError):
            return None

    def _node_is_current(
        self,
        node_history_index: dict[tuple[str, str], dict[str, Any]],
        *,
        group_id: str,
        node_id: str,
    ) -> bool | None:
        meta = node_history_index.get((group_id, node_id))
        if meta is None:
            return None
        return bool(meta.get('is_current'))

    def _node_topology(
        self,
        node_history_index: dict[tuple[str, str], dict[str, Any]],
        *,
        group_id: str,
        node_id: str,
    ) -> str | None:
        meta = node_history_index.get((group_id, node_id))
        if meta is None:
            return None
        topology = str(meta.get('topology') or '').strip()
        return topology or None

    def _episode_invalidation_kind(
        self,
        *,
        newer_edges: list[dict[str, Any]],
        closure_edges: list[dict[str, Any]],
    ) -> str | None:
        if newer_edges:
            return 'superseded'
        if any(str(edge.get('relation_type') or '').upper() == 'RESOLVES' for edge in closure_edges):
            return 'invalidated'
        return None

    def _transition_evidence_refs(self, *, group_id: str, edges: list[dict[str, Any]]) -> list[EvidenceRef]:
        refs: list[EvidenceRef] = []
        for edge in edges:
            relation_uuid = str(edge.get('relation_uuid') or '').strip()
            relation_type = str(edge.get('relation_type') or 'SUPERSEDES').strip().upper() or 'SUPERSEDES'
            if not relation_uuid:
                continue
            refs.append(
                EvidenceRef(
                    kind='event_log',
                    source_system='om',
                    locator={'system': 'om', 'stream': f'{group_id}:relation', 'event_id': relation_uuid},
                    title=relation_type,
                    snippet=f"{edge.get('source_id') or ''} {relation_type} {edge.get('target_id') or ''}".strip(),
                    observed_at=_coerce_timestamp(edge.get('created_at')),
                )
            )
        return refs

    def _endpoint_evidence_refs(
        self,
        *,
        group_id: str,
        source_node_id: str,
        target_node_id: str,
        source_content: str,
        target_content: str,
        created_at: str | None,
    ) -> list[EvidenceRef]:
        refs: list[EvidenceRef] = []
        refs.append(
            EvidenceRef(
                kind='event_log',
                source_system='om',
                locator={'system': 'om', 'stream': f'{group_id}:node', 'event_id': source_node_id},
                title=source_node_id,
                snippet=source_content or source_node_id,
                observed_at=created_at,
            )
        )
        refs.append(
            EvidenceRef(
                kind='event_log',
                source_system='om',
                locator={'system': 'om', 'stream': f'{group_id}:node', 'event_id': target_node_id},
                title=target_node_id,
                snippet=target_content or target_node_id,
                observed_at=created_at,
            )
        )
        return refs

    def _dedupe_evidence_refs(self, refs: list[EvidenceRef]) -> list[EvidenceRef]:
        deduped: dict[str, EvidenceRef] = {}
        for ref in refs:
            uri = str(ref.canonical_uri or '').strip()
            if not uri:
                continue
            deduped.setdefault(uri, ref)
        return list(deduped.values())

    def _first_timestamp(self, values: Any) -> str | None:
        timestamps = [_coerce_timestamp(value) for value in values]
        normalized = [timestamp for timestamp in timestamps if timestamp is not None]
        if not normalized:
            return None
        normalized.sort(key=_timestamp_sort_key)
        return normalized[0]

    def _annotation_value(self, annotations: list[str], prefix: str) -> str | None:
        needle = f'{prefix}:'
        for item in annotations:
            if item.startswith(needle):
                return item[len(needle):]
        return None

    def _node_key_from_source_key(self, source_key: str | None) -> tuple[str, str] | None:
        raw = str(source_key or '').strip()
        prefix = 'om:'
        if not raw.startswith(prefix):
            return None
        parts = raw.split(':')
        if len(parts) < 4 or parts[2] != 'node':
            return None
        return parts[1], ':'.join(parts[3:])

    def _episode_object_id(self, group_id: str, node_id: str | None) -> str | None:
        normalized_node_id = str(node_id or '').strip()
        if not normalized_node_id:
            return None
        return f'om_episode:{group_id}:{normalized_node_id}'

    def _episode_search_text(self, episode: Episode) -> str:
        history = episode.history_meta if isinstance(episode.history_meta, dict) else {}
        parts = [
            str(episode.title or ''),
            str(episode.summary or ''),
            str(episode.source_lane or ''),
            ' '.join(episode.annotations),
            str(episode.lifecycle_status or ''),
            str(history.get('topology') or ''),
            str(history.get('transition_reason') or ''),
            str(history.get('version_basis') or ''),
        ]
        return ' '.join(part for part in parts if part).strip()

    def _fact_search_text_from_object(self, fact: StateFact) -> str:
        value = fact.value if isinstance(fact.value, dict) else {'value': fact.value}
        history = value.get('om_history') if isinstance(value, dict) else {}
        top_level_history = fact.history_meta if isinstance(fact.history_meta, dict) else {}
        parts = [
            str(fact.fact_type or ''),
            str(fact.subject or ''),
            str(fact.predicate or ''),
            str(value.get('fact') or ''),
            str(value.get('source_content') or ''),
            str(value.get('target_content') or ''),
            str(history.get('topology') or ''),
            str(history.get('invalidation_reason') or ''),
            str(top_level_history.get('invalidation_basis') or ''),
            str(top_level_history.get('version_basis') or ''),
            str(fact.source_lane or ''),
        ]
        return ' '.join(part for part in parts if part).strip()
