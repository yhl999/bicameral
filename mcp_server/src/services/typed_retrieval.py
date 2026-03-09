from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models.typed_memory import Episode, EvidenceRef, Procedure, StateFact, TypedMemoryObject
from .change_ledger import DB_PATH_DEFAULT, ChangeLedger, project_objects
from .evidence_callback import EvidenceCallbackRegistry

_OBJECT_TYPE_ALIASES = {
    'state': 'state_fact',
    'state_fact': 'state_fact',
    'states': 'state_fact',
    'episode': 'episode',
    'episodes': 'episode',
    'procedure': 'procedure',
    'procedures': 'procedure',
}
_BUCKET_BY_OBJECT_TYPE = {
    'state_fact': 'state',
    'episode': 'episodes',
    'procedure': 'procedures',
}
_HISTORY_KEYWORDS = (
    'change',
    'changed',
    'history',
    'previous',
    'previously',
    'before',
    'used to',
    'supersede',
    'superseded',
    'what changed',
)
_CURRENT_KEYWORDS = (
    'current',
    'currently',
    'right now',
    'now',
    'true now',
    'active',
)
_TOKEN_RE = re.compile(r'[a-z0-9_./:-]+')
_STOPWORDS = {
    'a',
    'an',
    'and',
    'are',
    'for',
    'from',
    'how',
    'i',
    'in',
    'is',
    'it',
    'me',
    'my',
    'now',
    'of',
    'or',
    'tell',
    'the',
    'to',
    'was',
    'what',
    'when',
    'with',
}


@dataclass(frozen=True)
class ScoredObject:
    obj: TypedMemoryObject
    score: float


@dataclass
class TypedRetrievalService:
    ledger: ChangeLedger | None = None
    evidence_registry: EvidenceCallbackRegistry = field(default_factory=EvidenceCallbackRegistry)
    ledger_path: Path | str = DB_PATH_DEFAULT

    def __post_init__(self) -> None:
        if self.ledger is not None:
            return

        ledger_path = Path(self.ledger_path)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger = ChangeLedger(ledger_path)

    async def search(
        self,
        *,
        query: str,
        object_types: list[str] | None = None,
        metadata_filters: dict[str, Any] | None = None,
        history_mode: str = 'auto',
        current_only: bool | None = None,
        max_results: int = 10,
        max_evidence: int = 20,
    ) -> dict[str, Any]:
        if max_results <= 0:
            raise ValueError('max_results must be a positive integer')
        if max_evidence <= 0:
            raise ValueError('max_evidence must be a positive integer')

        normalized_object_types = _normalize_object_types(object_types)
        metadata_filters = dict(metadata_filters or {})
        query_mode = _resolve_query_mode(
            query=query,
            history_mode=history_mode,
            current_only=current_only,
        )

        all_objects = self._materialize_all_objects()
        filtered_objects = [
            obj
            for obj in all_objects
            if _matches_object_type(obj, normalized_object_types)
            and _matches_metadata_filters(obj, metadata_filters)
        ]

        if query_mode == 'current':
            filtered_objects = [obj for obj in filtered_objects if obj.is_current]

        ranked_objects = _rank_objects(filtered_objects, query)
        selected_objects = self._select_objects(ranked_objects, filtered_objects, query_mode, max_results)
        bucketed = self._bucket_objects(selected_objects)
        evidence = await self._resolve_evidence(selected_objects, max_evidence=max_evidence)

        total_objects = sum(len(bucket) for bucket in bucketed.values())
        if total_objects == 0:
            return {
                'message': 'No relevant typed memory found',
                'query_mode': query_mode,
                'state': [],
                'episodes': [],
                'procedures': [],
                'evidence': [],
                'counts': {'state': 0, 'episodes': 0, 'procedures': 0, 'evidence': 0},
            }

        return {
            'message': 'Typed memory retrieved successfully',
            'query_mode': query_mode,
            'state': bucketed['state'],
            'episodes': bucketed['episodes'],
            'procedures': bucketed['procedures'],
            'evidence': evidence,
            'counts': {
                'state': len(bucketed['state']),
                'episodes': len(bucketed['episodes']),
                'procedures': len(bucketed['procedures']),
                'evidence': len(evidence),
            },
            'filters_applied': {
                'object_types': sorted(normalized_object_types) if normalized_object_types else [],
                'metadata_filters': metadata_filters,
            },
        }

    def _materialize_all_objects(self) -> list[TypedMemoryObject]:
        assert self.ledger is not None
        rows = self.ledger.conn.execute(
            'SELECT * FROM change_events ORDER BY recorded_at, rowid'
        ).fetchall()
        grouped_rows: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in rows:
            root_key = str(row['root_id'] or row['object_id'] or row['target_object_id'] or '').strip()
            if not root_key:
                continue
            grouped_rows[root_key].append(row)

        objects: list[TypedMemoryObject] = []
        for group_rows in grouped_rows.values():
            objects.extend(project_objects(group_rows))
        return sorted(objects, key=_object_sort_key)

    def _select_objects(
        self,
        ranked_objects: list[ScoredObject],
        filtered_objects: list[TypedMemoryObject],
        query_mode: str,
        max_results: int,
    ) -> list[ScoredObject]:
        if not ranked_objects:
            return []

        if query_mode != 'history':
            return ranked_objects[:max_results]

        selected_roots: list[str] = []
        for scored in ranked_objects:
            if scored.obj.root_id not in selected_roots:
                selected_roots.append(scored.obj.root_id)
            if len(selected_roots) >= max_results:
                break

        root_scores = {scored.obj.root_id: scored.score for scored in ranked_objects}
        expanded = [obj for obj in filtered_objects if obj.root_id in selected_roots]
        expanded.sort(key=lambda obj: (root_scores.get(obj.root_id, 0.0), *_object_sort_key(obj)), reverse=True)

        result: list[ScoredObject] = []
        for obj in expanded:
            result.append(ScoredObject(obj=obj, score=root_scores.get(obj.root_id, 0.0)))
        result.sort(key=lambda item: (-item.score, item.obj.root_id, item.obj.version, item.obj.object_id))
        return result

    def _bucket_objects(self, selected_objects: list[ScoredObject]) -> dict[str, list[dict[str, Any]]]:
        bucketed: dict[str, list[dict[str, Any]]] = {
            'state': [],
            'episodes': [],
            'procedures': [],
        }
        seen_object_ids: set[str] = set()
        for scored in selected_objects:
            if scored.obj.object_id in seen_object_ids:
                continue
            seen_object_ids.add(scored.obj.object_id)
            bucket = _BUCKET_BY_OBJECT_TYPE.get(scored.obj.object_type)
            if bucket is None:
                continue
            payload = scored.obj.model_dump(mode='json')
            payload['match_score'] = round(scored.score, 4)
            payload['search_text'] = _searchable_text(scored.obj)
            bucketed[bucket].append(payload)
        return bucketed

    async def _resolve_evidence(
        self,
        selected_objects: list[ScoredObject],
        *,
        max_evidence: int,
    ) -> list[dict[str, Any]]:
        refs_by_uri: dict[str, EvidenceRef] = {}
        object_ids_by_uri: dict[str, list[str]] = defaultdict(list)

        for scored in selected_objects:
            for ref in scored.obj.evidence_refs:
                canonical_uri = str(ref.canonical_uri or '').strip()
                if not canonical_uri:
                    continue
                refs_by_uri.setdefault(canonical_uri, ref)
                object_ids_by_uri[canonical_uri].append(scored.obj.object_id)

        return await self.evidence_registry.resolve_many(
            list(refs_by_uri.values()),
            object_ids_by_uri=object_ids_by_uri,
            max_items=max_evidence,
        )


def _normalize_object_types(object_types: list[str] | None) -> set[str]:
    if not object_types:
        return set()

    normalized: set[str] = set()
    for item in object_types:
        key = _OBJECT_TYPE_ALIASES.get(str(item or '').strip().lower())
        if key is None:
            raise ValueError(f'Unsupported object_type: {item!r}')
        normalized.add(key)
    return normalized


def _resolve_query_mode(*, query: str, history_mode: str, current_only: bool | None) -> str:
    normalized_mode = str(history_mode or 'auto').strip().lower()
    if normalized_mode not in {'auto', 'current', 'history', 'all'}:
        raise ValueError("history_mode must be one of: 'auto', 'current', 'history', 'all'")

    if current_only is True:
        return 'current'
    if normalized_mode in {'current', 'history', 'all'}:
        return normalized_mode

    query_lc = str(query or '').strip().lower()
    if any(keyword in query_lc for keyword in _HISTORY_KEYWORDS):
        return 'history'
    if any(keyword in query_lc for keyword in _CURRENT_KEYWORDS):
        return 'current'
    return 'all'


def _matches_object_type(obj: TypedMemoryObject, normalized_object_types: set[str]) -> bool:
    return not normalized_object_types or obj.object_type in normalized_object_types


def _matches_metadata_filters(obj: TypedMemoryObject, metadata_filters: dict[str, Any]) -> bool:
    if not metadata_filters:
        return True

    payload = obj.model_dump(mode='json')
    for key, expected in metadata_filters.items():
        actual = _lookup_path(payload, key)
        if not _matches_filter_value(actual, expected):
            return False
    return True


def _lookup_path(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in str(path).split('.'):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _matches_filter_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        if 'eq' in expected:
            return actual == expected['eq']
        if 'in' in expected:
            candidates = expected.get('in') or []
            return actual in candidates
        if 'contains' in expected:
            needle = expected.get('contains')
            if isinstance(actual, str):
                return str(needle).lower() in actual.lower()
            if isinstance(actual, list):
                return needle in actual
            return False
        if 'gte' in expected:
            return actual is not None and actual >= expected['gte']
        if 'lte' in expected:
            return actual is not None and actual <= expected['lte']
        return False

    if isinstance(expected, (list, tuple, set)):
        return actual in expected
    return actual == expected


def _tokenize(value: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(str(value or '').lower()) if token not in _STOPWORDS]


def _searchable_text(obj: TypedMemoryObject) -> str:
    parts = [
        obj.object_type,
        obj.source_lane or '',
        obj.source_key or '',
        obj.policy_scope,
        obj.visibility_scope,
    ]

    if isinstance(obj, StateFact):
        parts.extend(
            [
                obj.fact_type,
                obj.subject,
                obj.predicate,
                _stringify(obj.value),
                obj.scope,
                obj.risk_level,
            ]
        )
    elif isinstance(obj, Episode):
        parts.extend(
            [
                obj.title or '',
                obj.summary or '',
                ' '.join(obj.annotations),
            ]
        )
    elif isinstance(obj, Procedure):
        parts.extend(
            [
                obj.name,
                obj.trigger,
                ' '.join(obj.preconditions),
                ' '.join(obj.steps),
                obj.expected_outcome,
                obj.risk_level,
            ]
        )

    for ref in obj.evidence_refs:
        parts.append(str(ref.title or ''))
        parts.append(str(ref.snippet or ''))
        parts.append(str(ref.canonical_uri or ''))

    return ' '.join(part for part in parts if part).strip()


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _rank_objects(objects: list[TypedMemoryObject], query: str) -> list[ScoredObject]:
    query = str(query or '').strip()
    if not objects:
        return []
    if not query:
        return [ScoredObject(obj=obj, score=_default_score(obj)) for obj in objects]

    query_lc = query.lower()
    query_tokens = set(_tokenize(query))

    scored: list[ScoredObject] = []
    for obj in objects:
        haystack = _searchable_text(obj)
        haystack_lc = haystack.lower()
        haystack_tokens = set(_tokenize(haystack))
        overlap = len(query_tokens & haystack_tokens)
        substring_bonus = 3.0 if query_lc in haystack_lc else 0.0
        current_bonus = 0.5 if obj.is_current else 0.0
        version_bonus = min(float(obj.version) * 0.05, 0.5)
        score = float(overlap) + substring_bonus + current_bonus + version_bonus
        if score <= 0:
            continue
        scored.append(ScoredObject(obj=obj, score=score))

    scored.sort(key=lambda item: (-item.score, *_object_sort_key(item.obj)))
    return scored


def _default_score(obj: TypedMemoryObject) -> float:
    score = 1.0
    if obj.is_current:
        score += 0.5
    score += min(float(obj.version) * 0.05, 0.5)
    return score


def _object_sort_key(obj: TypedMemoryObject) -> tuple[Any, ...]:
    return (
        0 if obj.is_current else 1,
        obj.root_id,
        -obj.version,
        obj.object_id,
    )
