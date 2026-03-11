"""Provisional episode projection for unpromoted OM-native content.

Phase 2 refactor: OMTypedProjectionService is now a thin read-time adapter
that projects unpromoted OM nodes as simple provisional episodes. It no longer
reconstructs graph topology, lineage chains, or relation-derived state facts.

Architectural contract:
- State facts are ledger-canonical only. They are never produced here.
- Promoted OM content is already in the ledger and retrieved through the
  canonical TypedRetrievalService path (with episodes derived by
  _derive_ledger_backed_om_history).
- This service handles ONLY the residual case: OM nodes that exist in the
  graph but have not yet been promoted. They surface as provisional episodes
  so they remain discoverable while awaiting promotion.
"""

from __future__ import annotations

from typing import Any

try:
    from ..models.typed_memory import Episode, EvidenceRef, TypedMemoryObject
    from .search_service import SearchService
except ImportError:  # pragma: no cover - top-level import fallback
    from models.typed_memory import Episode, EvidenceRef, TypedMemoryObject
    from services.search_service import SearchService


def _coerce_timestamp(value: Any) -> str | None:
    text = str(value).strip() if value not in (None, '') else ''
    return text or None


class OMTypedProjectionService:
    """Project unpromoted OM-native nodes as provisional typed episodes.

    This is a read-time fallback for OM content not yet promoted through the
    change ledger.  It produces simple flat episodes without topology or
    lineage reconstruction — that complexity belongs to the ledger-canonical
    path after promotion.

    State facts are NOT produced.  State is ledger-canonical only.
    """

    def __init__(
        self,
        *,
        search_service: SearchService | None = None,
        graphiti_service: Any | None = None,
    ) -> None:
        self.search_service = search_service or SearchService()
        self.graphiti_service = graphiti_service

    # ------------------------------------------------------------------
    # Public API — unchanged signature for TypedRetrievalService compat
    # ------------------------------------------------------------------

    async def project(
        self,
        *,
        query: str,
        effective_group_ids: list[str] | None,
        object_types: set[str],
        max_results: int,
        query_mode: str = 'all',
    ) -> tuple[list[TypedMemoryObject], dict[str, str], dict[str, Any]]:
        """Return provisional episodes for unpromoted OM nodes."""

        # Only episodes are produced — skip if caller wants only other types
        if object_types and 'episode' not in object_types:
            return [], {}, {'enabled': False, 'reason': 'episodes_not_requested'}

        scope = list(effective_group_ids or [])
        normalized_max = max(1, int(max_results or 1))

        if self.graphiti_service is None:
            return [], {}, {'enabled': False, 'reason': 'graphiti_service_unavailable'}
        if not self.search_service.includes_observational_memory(scope):
            return [], {}, {'enabled': False, 'reason': 'om_not_in_scope'}

        groups_considered = self.search_service._om_groups_in_scope(scope)

        node_rows = await self.search_service.search_observational_nodes(
            graphiti_service=self.graphiti_service,
            query=query,
            group_ids=scope,
            max_nodes=normalized_max,
            entity_types=['OMNode'],
        )

        if not node_rows:
            return [], {}, {
                'enabled': True,
                'reason': 'no_om_nodes_found',
                'groups_considered': groups_considered,
                'episodes_projected': 0,
                'max_results': normalized_max,
            }

        episodes: list[Episode] = []
        search_text_overrides: dict[str, str] = {}

        for row in node_rows:
            episode = self._project_provisional_episode(row)
            if episode is None:
                continue
            episodes.append(episode)
            search_text_overrides[episode.object_id] = self._episode_search_text(episode)

        limits: dict[str, Any] = {
            'enabled': True,
            'reason': 'provisional_projection',
            'groups_considered': groups_considered,
            'episodes_projected': len(episodes),
            'state_projected': 0,
            'max_results': normalized_max,
        }
        return episodes, search_text_overrides, limits

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _project_provisional_episode(self, row: dict[str, Any]) -> Episode | None:
        group_id = str(row.get('group_id') or '').strip()
        node_id = str(row.get('uuid') or '').strip()
        if not group_id or not node_id:
            return None

        content = str(row.get('summary') or row.get('name') or node_id).strip()
        created_at = _coerce_timestamp(row.get('created_at')) or '2026-01-01T00:00:00Z'
        attrs = row.get('attributes') or {}

        status = str(attrs.get('status') or '').strip()
        semantic_domain = str(attrs.get('semantic_domain') or '').strip()

        object_id = f'om_episode:{group_id}:{node_id}'
        annotations = ['om_native', 'provisional', 'unpromoted']
        if semantic_domain:
            annotations.append(semantic_domain)
        if status:
            annotations.append(status)

        return Episode(
            object_id=object_id,
            root_id=object_id,
            parent_id=None,
            version=1,
            is_current=True,
            source_lane=group_id,
            source_key=f'om:{group_id}:node:{node_id}',
            policy_scope='private',
            visibility_scope='private',
            title=content[:120] or node_id,
            summary=content,
            annotations=annotations,
            history_meta={
                'lineage_kind': 'om_node',
                'lineage_basis': 'provisional',
                'derivation_level': 'provisional',
            },
            created_at=created_at,
            valid_at=created_at,
            invalid_at=None,
            superseded_by=None,
            lifecycle_status='asserted',
            evidence_refs=[
                EvidenceRef(
                    kind='event_log',
                    source_system='om',
                    locator={
                        'system': 'om',
                        'stream': f'{group_id}:node',
                        'event_id': node_id,
                    },
                    title=content[:120] or node_id,
                    snippet=content,
                    observed_at=created_at,
                ),
            ],
        )

    @staticmethod
    def _episode_search_text(episode: Episode) -> str:
        parts = [
            str(episode.title or ''),
            str(episode.summary or ''),
            str(episode.source_lane or ''),
            ' '.join(episode.annotations),
        ]
        return ' '.join(part for part in parts if part).strip()
