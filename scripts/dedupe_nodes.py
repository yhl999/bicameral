#!/usr/bin/env python3
"""
Post-processing script to deduplicate Entity nodes in a graph database.
Required after running parallel ingestion ("Speed Run"), which creates duplicate
entities due to missing unique constraints.

Usage:
    python3 scripts/dedupe_nodes.py --group-id s1_sessions_main --confirm-destructive
    python3 scripts/dedupe_nodes.py --group-id s1_sessions_main --dry-run
    python3 scripts/dedupe_nodes.py --group-id s1_sessions_main --backend falkordb --host localhost --port 6379 --confirm-destructive
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from graph_driver import add_backend_args, get_graph_client

try:
    from graph_driver import GraphDriverSetupError
except ImportError:  # pragma: no cover - test stubs may omit the concrete error type
    class GraphDriverSetupError(RuntimeError):
        pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CORE_ENTITY_LABEL = 'Entity'
_SAFE_LABEL_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_CONFLICTS_JSON_KEY = '_dedupe_conflicts_json'
_CONFLICT_KEYS_KEY = '_dedupe_conflict_keys'
_RESERVED_NODE_KEYS = {
    'uuid',
    'name_embedding',
    _CONFLICTS_JSON_KEY,
    _CONFLICT_KEYS_KEY,
}
_IMMUTABLE_MERGE_KEYS = {'name'}
_HOMONYM_PROOF_KEYS = (
    'canonical_id',
    'external_id',
    'source_id',
    'source_entity_id',
    'resolved_entity_id',
    'employee_id',
    'email',
    'emails',
    'phone',
    'website',
    'url',
    'domain',
    'linkedin_url',
    'twitter_handle',
    'github_username',
    'crunchbase_url',
    'wikidata_id',
)


def _created_at_sort_key(value) -> str:
    if value is None:
        return '9999-12-31T23:59:59.999999+00:00'
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _dynamic_entity_label(group_id: str) -> str:
    return 'Entity_' + group_id.replace('-', '')


def _normalize_typed_labels(labels: Iterable[str] | None, group_id: str) -> frozenset[str]:
    dynamic_label = _dynamic_entity_label(group_id)
    normalized = {
        str(label)
        for label in (labels or [])
        if label and str(label) not in {CORE_ENTITY_LABEL, dynamic_label}
    }
    return frozenset(normalized)


def _is_missing_value(value: Any) -> bool:
    return value in (None, '', [], {})


def _dedupe_list(values: list[Any]) -> list[Any]:
    merged: list[Any] = []
    for value in values:
        if value not in merged:
            merged.append(value)
    return merged


def _append_conflict_values(conflicts: dict[str, list[Any]], path: str, *values: Any) -> None:
    if not path:
        return

    entries = conflicts.setdefault(path, [])
    seen = {_canonical_json(entry) for entry in entries}
    for value in values:
        if _is_missing_value(value):
            continue
        token = _canonical_json(value)
        if token in seen:
            continue
        entries.append(value)
        seen.add(token)


def _merge_values(existing: Any, incoming: Any, path: str, conflicts: dict[str, list[Any]]) -> Any:
    if _is_missing_value(existing):
        return incoming
    if _is_missing_value(incoming) or existing == incoming:
        return existing

    if isinstance(existing, list) and isinstance(incoming, list):
        return _dedupe_list([*existing, *incoming])

    if isinstance(existing, dict) and isinstance(incoming, dict):
        merged = dict(existing)
        for key, value in incoming.items():
            child_path = f'{path}.{key}' if path else str(key)
            if key not in merged:
                merged[key] = value
                continue
            merged[key] = _merge_values(merged[key], value, child_path, conflicts)
        return merged

    _append_conflict_values(conflicts, path, existing, incoming)
    return existing


def _bucket_sort_key(records: list[dict]) -> tuple[str, str]:
    first = min(
        records,
        key=lambda record: (
            _created_at_sort_key(record.get('created_at')),
            str(record.get('uuid') or ''),
        ),
    )
    return _created_at_sort_key(first.get('created_at')), str(first.get('uuid') or '')


def bucket_duplicate_records(records: list[dict], group_id: str) -> list[list[dict]]:
    """Partition same-name duplicates into fail-closed merge buckets.

    Typed lanes add ontology-specific labels to Entity nodes. Exact typed-label
    sets are treated as distinct identities; generic Entity-only nodes may join a
    typed bucket only when there is exactly one typed signature present. This
    avoids transitive bridge-merges such as Person <- generic -> Organization.

    When multiple typed signatures exist for the same name, generic-only nodes are
    left untouched even if there are several of them. Merging those generic nodes
    would be ambiguous and could create a silent bridge between unrelated typed
    identities.
    """

    ordered = sorted(
        records,
        key=lambda record: (
            _created_at_sort_key(record.get('created_at')),
            str(record.get('uuid') or ''),
        ),
    )

    typed_groups: dict[frozenset[str], list[dict]] = defaultdict(list)
    generic_records: list[dict] = []

    for record in ordered:
        typed_labels = _normalize_typed_labels(record.get('labels'), group_id)
        if typed_labels:
            typed_groups[typed_labels].append(record)
        else:
            generic_records.append(record)

    buckets: list[list[dict]] = []

    if typed_groups:
        if len(typed_groups) == 1:
            only_key = next(iter(typed_groups))
            merged_bucket = [*generic_records, *typed_groups[only_key]]
            if len(merged_bucket) > 1:
                buckets.append(
                    sorted(
                        merged_bucket,
                        key=lambda record: (
                            _created_at_sort_key(record.get('created_at')),
                            str(record.get('uuid') or ''),
                        ),
                    )
                )
        else:
            for group_records in typed_groups.values():
                if len(group_records) > 1:
                    buckets.append(group_records)
    elif len(generic_records) > 1:
        buckets.append(generic_records)

    return sorted(buckets, key=_bucket_sort_key)


def _safe_labels_for_query(labels: Iterable[str], group_id: str) -> list[str]:
    dynamic_label = _dynamic_entity_label(group_id)
    safe_labels: list[str] = []
    for label in labels:
        label = str(label)
        if label in {CORE_ENTITY_LABEL, dynamic_label}:
            continue
        if not _SAFE_LABEL_RE.match(label):
            logger.warning('Skipping unsafe dynamic label during dedupe: %r', label)
            continue
        safe_labels.append(label)
    return sorted(set(safe_labels))


def _select_summary(records: list[dict]) -> str:
    summaries = []
    for record in records:
        props = dict(record.get('properties') or {})
        summary = props.get('summary')
        if isinstance(summary, str) and summary.strip():
            summaries.append(summary)
    if not summaries:
        return ''
    return max(summaries, key=lambda value: (len(value.strip()), value))


def _select_name_embedding(records: list[dict]) -> Any:
    for record in records:
        props = dict(record.get('properties') or {})
        embedding = props.get('name_embedding')
        if embedding not in (None, []):
            return embedding
    return None


def _collect_existing_conflicts(records: list[dict]) -> dict[str, list[Any]]:
    conflicts: dict[str, list[Any]] = {}
    for record in records:
        props = dict(record.get('properties') or {})
        raw = props.get(_CONFLICTS_JSON_KEY)
        if isinstance(raw, str) and raw.strip():
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                for path, values in payload.items():
                    if isinstance(values, list):
                        _append_conflict_values(conflicts, str(path), *values)
                    else:
                        _append_conflict_values(conflicts, str(path), values)
    return conflicts


def build_merged_entity_payload(records: list[dict], winner_uuid: str, group_id: str) -> dict[str, Any]:
    by_uuid = {str(record['uuid']): record for record in records if record.get('uuid')}
    if winner_uuid not in by_uuid:
        raise ValueError(f'Winner uuid {winner_uuid!r} missing from merge records')

    winner_record = by_uuid[winner_uuid]
    winner_props = {
        key: value
        for key, value in dict(winner_record.get('properties') or {}).items()
        if key not in _RESERVED_NODE_KEYS
    }
    conflicts = _collect_existing_conflicts(records)

    winner_name = str(winner_props.get('name') or '').strip()
    if not winner_name:
        raise ValueError('Refusing to dedupe an entity bucket whose winner has a blank or missing name.')

    created_candidates = []
    for record in records:
        props = dict(record.get('properties') or {})
        record_name = str(props.get('name') or '').strip()
        if not record_name:
            raise ValueError('Refusing to merge entity records with blank or missing names.')
        if record_name != winner_name:
            raise ValueError(
                f'Refusing to merge non-identical entity names in one bucket: {winner_name!r} vs {record_name!r}'
            )

        record_group_id = props.get('group_id')
        if record_group_id not in (None, '', group_id):
            raise ValueError(
                f'Refusing to merge entity {record.get("uuid")!r} across groups: {record_group_id!r} != {group_id!r}'
            )

        created_at = props.get('created_at')
        if created_at is not None:
            created_candidates.append(created_at)

        for key, value in props.items():
            if key in _RESERVED_NODE_KEYS or key in {'group_id', 'created_at', 'summary'}:
                continue

            if key in _IMMUTABLE_MERGE_KEYS:
                if key not in winner_props or _is_missing_value(winner_props[key]):
                    winner_props[key] = value
                    continue
                if _is_missing_value(value) or winner_props[key] == value:
                    continue
                raise ValueError(
                    f'Refusing to merge entity {winner_uuid!r}: immutable identity field {key!r} conflicts.'
                )

            if key not in winner_props:
                winner_props[key] = value
                continue
            winner_props[key] = _merge_values(winner_props[key], value, key, conflicts)

    winner_props['group_id'] = group_id
    winner_props['summary'] = _select_summary(records)
    if created_candidates:
        winner_props['created_at'] = min(created_candidates, key=_created_at_sort_key)

    if conflicts:
        winner_props[_CONFLICT_KEYS_KEY] = sorted(conflicts)
        winner_props[_CONFLICTS_JSON_KEY] = _canonical_json(
            {key: conflicts[key] for key in sorted(conflicts)}
        )

    return {
        'winner_props': winner_props,
        'name_embedding': _select_name_embedding(records),
    }


async def _find_duplicate_buckets(client, backend: str, group_id: str) -> list[dict]:
    del backend  # queries are intentionally group-scoped for every backend
    query = """
    MATCH (n:Entity)
    WHERE n.group_id = $group_id
      AND n.name IS NOT NULL
      AND trim(toString(n.name)) <> ''
    WITH n.name AS name,
         collect({
             uuid: n.uuid,
             created_at: n.created_at,
             labels: labels(n)
         }) AS nodes
    WHERE size(nodes) > 1
    RETURN name, nodes
    ORDER BY name ASC
    """
    res = await client.query(query, {'group_id': group_id})

    buckets: list[dict] = []
    for name, nodes in res.result_set:
        for bucket in bucket_duplicate_records(list(nodes or []), group_id):
            ordered_bucket = sorted(
                bucket,
                key=lambda record: (
                    _created_at_sort_key(record.get('created_at')),
                    str(record.get('uuid') or ''),
                ),
            )
            buckets.append({'name': name, 'nodes': ordered_bucket})

    return buckets


async def _load_merge_records(client, group_id: str, uuids: list[str]) -> list[dict[str, Any]]:
    query = """
    MATCH (n:Entity)
    WHERE n.group_id = $group_id AND n.uuid IN $uuids
    RETURN n.uuid AS uuid, labels(n) AS labels, properties(n) AS properties
    ORDER BY n.created_at ASC, n.uuid ASC
    """
    res = await client.query(query, {'group_id': group_id, 'uuids': uuids})
    records = [
        {'uuid': row[0], 'labels': list(row[1] or []), 'properties': dict(row[2] or {})}
        for row in res.result_set
    ]
    if len(records) != len(set(uuids)):
        raise ValueError(
            'Refusing to dedupe: failed to load a full, group-scoped record set for '
            f'{group_id!r} (expected {len(set(uuids))}, got {len(records)}).'
        )
    return records


def _validate_merge_bucket(bucket: dict) -> list[dict]:
    bucket_name = str(bucket.get('name') or '')
    if not bucket_name.strip():
        raise ValueError('Refusing to dedupe a blank-name bucket.')

    nodes = list(bucket.get('nodes') or [])
    if len(nodes) < 2:
        return nodes

    seen: set[str] = set()
    for node in nodes:
        uuid = str(node.get('uuid') or '').strip()
        if not uuid:
            raise ValueError(f'Refusing to dedupe {bucket_name!r}: encountered a node with blank uuid.')
        if uuid in seen:
            raise ValueError(f'Refusing to dedupe {bucket_name!r}: duplicate uuid {uuid!r} in one bucket.')
        seen.add(uuid)

    return nodes


def _normalize_proof_value(value: Any) -> str | None:
    if _is_missing_value(value):
        return None
    if isinstance(value, str):
        normalized = value.strip().casefold()
        return _canonical_json(normalized) if normalized else None
    if isinstance(value, (list, tuple, set)):
        normalized_items = []
        for item in value:
            if _is_missing_value(item):
                continue
            if isinstance(item, str):
                item_text = item.strip().casefold()
                if item_text:
                    normalized_items.append(item_text)
            elif not isinstance(item, dict):
                normalized_items.append(item)
        if not normalized_items:
            return None
        return _canonical_json(sorted(normalized_items, key=str))
    if isinstance(value, dict):
        return None
    return _canonical_json(value)


def _bucket_node_debug(records: list[dict[str, Any]] | list[dict], group_id: str) -> list[dict[str, Any]]:
    debug_rows: list[dict[str, Any]] = []
    for record in records:
        props = dict(record.get('properties') or {})
        debug_rows.append(
            {
                'uuid': str(record.get('uuid') or '').strip(),
                'created_at': props.get('created_at', record.get('created_at')),
                'typed_labels': sorted(_normalize_typed_labels(record.get('labels'), group_id)),
            }
        )
    return debug_rows


def _collect_candidate_proof_values(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for record in records:
        props = dict(record.get('properties') or {})
        uuid = str(record.get('uuid') or '').strip()
        for key in _HOMONYM_PROOF_KEYS:
            value = props.get(key)
            if _normalize_proof_value(value) is None:
                continue
            observed.setdefault(key, {})[uuid] = value
    return observed


def _shared_homonym_proofs(records: list[dict[str, Any]]) -> dict[str, Any]:
    proofs: dict[str, Any] = {}
    for key in _HOMONYM_PROOF_KEYS:
        normalized_values: list[str] = []
        representative_value: Any = None
        for record in records:
            props = dict(record.get('properties') or {})
            value = props.get(key)
            normalized = _normalize_proof_value(value)
            if normalized is None:
                break
            normalized_values.append(normalized)
            if representative_value is None:
                representative_value = value
        else:
            if len(set(normalized_values)) == 1 and representative_value is not None:
                proofs[key] = representative_value
    return proofs


def _require_homonym_merge_proof(records: list[dict[str, Any]], group_id: str, bucket_name: str) -> None:
    typed_groups: dict[frozenset[str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        typed_labels = _normalize_typed_labels(record.get('labels'), group_id)
        if typed_labels:
            typed_groups[typed_labels].append(record)

    for typed_labels, group_records in typed_groups.items():
        if len(group_records) < 2:
            continue

        proofs = _shared_homonym_proofs(group_records)
        if proofs:
            continue

        candidate_values = _collect_candidate_proof_values(group_records)
        raise ValueError(
            'Refusing to merge same-name typed entities without stronger identity proof. '
            f'name={bucket_name!r} group_id={group_id!r} typed_labels={sorted(typed_labels)!r} '
            f'nodes={_bucket_node_debug(group_records, group_id)!r} '
            f'candidate_proofs={candidate_values!r}. '
            'Inspect the listed node UUIDs for shared external IDs, emails, domains, or profile URLs and merge manually only if they are truly the same entity.'
        )


async def _inspect_remaining_relationships(client, group_id: str, node_uuids: list[str]) -> list[dict[str, Any]]:
    if not node_uuids:
        return []

    query = """
    MATCH (loser:Entity)
    WHERE loser.group_id = $group_id AND loser.uuid IN $node_uuids
    OPTIONAL MATCH (loser)-[r]-()
    RETURN loser.uuid AS loser_uuid, type(r) AS rel_type, count(r) AS rel_count
    ORDER BY loser_uuid ASC, rel_type ASC
    """
    res = await client.query(query, {'group_id': group_id, 'node_uuids': node_uuids})
    rows: list[dict[str, Any]] = []
    for loser_uuid, rel_type, rel_count in res.result_set:
        if rel_type is None or not rel_count:
            continue
        rows.append(
            {
                'loser_uuid': loser_uuid,
                'rel_type': rel_type,
                'rel_count': rel_count,
            }
        )
    return rows


async def _prepare_bucket_merge(client, group_id: str, bucket: dict) -> dict[str, Any] | None:
    nodes = _validate_merge_bucket(bucket)
    if len(nodes) < 2:
        return None

    winner = nodes[0]
    losers = nodes[1:]
    loser_uuids = [str(node['uuid']) for node in losers]
    if not loser_uuids:
        return None

    winner_uuid = str(winner['uuid'])
    merged_uuids = [winner_uuid, *loser_uuids]
    merged_labels = _safe_labels_for_query(
        {
            label
            for node in nodes
            for label in (node.get('labels') or [])
        },
        group_id,
    )
    merge_records = await _load_merge_records(client, group_id, merged_uuids)
    _require_homonym_merge_proof(merge_records, group_id, bucket_name=str(bucket.get('name') or ''))
    merged_payload = build_merged_entity_payload(merge_records, winner_uuid=winner_uuid, group_id=group_id)

    return {
        'nodes': nodes,
        'winner_uuid': winner_uuid,
        'loser_uuids': loser_uuids,
        'merged_uuids': merged_uuids,
        'merged_labels': merged_labels,
        'merged_payload': merged_payload,
    }


async def _merge_bucket(client, backend: str, group_id: str, bucket: dict) -> int:
    merge_plan = await _prepare_bucket_merge(client, group_id, bucket)
    if merge_plan is None:
        return 0

    winner_uuid = merge_plan['winner_uuid']
    loser_uuids = merge_plan['loser_uuids']
    merged_uuids = merge_plan['merged_uuids']
    merged_labels = merge_plan['merged_labels']
    merged_payload = merge_plan['merged_payload']

    params = {
        'winner_uuid': winner_uuid,
        'loser_uuids': loser_uuids,
        'merged_uuids': merged_uuids,
        'group_id': group_id,
    }

    merge_queries: list[tuple[str, dict]] = [
        (
            """
            MATCH (winner:Entity {uuid: $winner_uuid})
            WHERE winner.group_id = $group_id
            SET winner += $winner_props
            """,
            {
                'winner_uuid': winner_uuid,
                'group_id': group_id,
                'winner_props': merged_payload['winner_props'],
            },
        )
    ]

    name_embedding = merged_payload.get('name_embedding')
    if name_embedding not in (None, []):
        if backend == 'falkordb':
            merge_queries.append(
                (
                    """
                    MATCH (winner:Entity {uuid: $winner_uuid})
                    WHERE winner.group_id = $group_id
                    SET winner.name_embedding = vecf32($name_embedding)
                    """,
                    {
                        'winner_uuid': winner_uuid,
                        'group_id': group_id,
                        'name_embedding': name_embedding,
                    },
                )
            )
        else:
            merge_queries.append(
                (
                    """
                    MATCH (winner:Entity {uuid: $winner_uuid})
                    WHERE winner.group_id = $group_id
                    SET winner.name_embedding = $name_embedding
                    """,
                    {
                        'winner_uuid': winner_uuid,
                        'group_id': group_id,
                        'name_embedding': name_embedding,
                    },
                )
            )

    for label in merged_labels:
        merge_queries.append(
            (
                f"""
                MATCH (winner:Entity {{uuid: $winner_uuid}})
                WHERE winner.group_id = $group_id
                SET winner:{label}
                """,
                {'winner_uuid': winner_uuid, 'group_id': group_id},
            )
        )

    q_move_mentions = """
    MATCH (winner:Entity {uuid: $winner_uuid})
    WHERE winner.group_id = $group_id
    UNWIND $loser_uuids AS loser_uuid
    MATCH (loser:Entity {uuid: loser_uuid})
    WHERE loser.group_id = $group_id
    MATCH (e:Episodic)-[r:MENTIONS]->(loser)
    WHERE e.group_id = $group_id
    WITH winner, loser_uuid, e, r,
         coalesce(r.uuid, 'dedupe:mentions:' + loser_uuid + ':' + e.uuid + ':' + $winner_uuid) AS new_uuid
    MERGE (e)-[nr:MENTIONS {uuid: new_uuid}]->(winner)
    SET nr += properties(r)
    SET nr.uuid = new_uuid,
        nr.group_id = coalesce(nr.group_id, e.group_id, $group_id),
        nr.created_at = coalesce(nr.created_at, e.created_at)
    DELETE r
    """

    q_move_relates_out = """
    MATCH (winner:Entity {uuid: $winner_uuid})
    WHERE winner.group_id = $group_id
    UNWIND $loser_uuids AS loser_uuid
    MATCH (loser:Entity {uuid: loser_uuid})-[r:RELATES_TO]->(target:Entity)
    WHERE loser.group_id = $group_id
      AND NOT target.uuid IN $merged_uuids
      AND target.group_id = $group_id
    WITH winner, loser_uuid, target, r,
         coalesce(r.uuid, 'dedupe:rel-out:' + loser_uuid + ':' + target.uuid + ':' + coalesce(r.name, 'RELATES_TO')) AS new_uuid
    MERGE (winner)-[nr:RELATES_TO {uuid: new_uuid}]->(target)
    SET nr += properties(r)
    SET nr.uuid = new_uuid,
        nr.group_id = coalesce(nr.group_id, $group_id),
        nr.episodes = coalesce(nr.episodes, []),
        nr.created_at = coalesce(nr.created_at, winner.created_at)
    DELETE r
    """

    q_move_relates_in = """
    MATCH (winner:Entity {uuid: $winner_uuid})
    WHERE winner.group_id = $group_id
    UNWIND $loser_uuids AS loser_uuid
    MATCH (source:Entity)-[r:RELATES_TO]->(loser:Entity {uuid: loser_uuid})
    WHERE loser.group_id = $group_id
      AND NOT source.uuid IN $merged_uuids
      AND source.group_id = $group_id
    WITH winner, loser_uuid, source, r,
         coalesce(r.uuid, 'dedupe:rel-in:' + source.uuid + ':' + loser_uuid + ':' + coalesce(r.name, 'RELATES_TO')) AS new_uuid
    MERGE (source)-[nr:RELATES_TO {uuid: new_uuid}]->(winner)
    SET nr += properties(r)
    SET nr.uuid = new_uuid,
        nr.group_id = coalesce(nr.group_id, $group_id),
        nr.episodes = coalesce(nr.episodes, []),
        nr.created_at = coalesce(nr.created_at, winner.created_at)
    DELETE r
    """

    q_move_has_member = """
    MATCH (winner:Entity {uuid: $winner_uuid})
    WHERE winner.group_id = $group_id
    UNWIND $loser_uuids AS loser_uuid
    MATCH (loser:Entity {uuid: loser_uuid})<-[r:HAS_MEMBER]-(c:Community)
    WHERE loser.group_id = $group_id
      AND c.group_id = $group_id
    WITH winner, loser_uuid, c, r,
         coalesce(r.uuid, 'dedupe:member:' + c.uuid + ':' + loser_uuid + ':' + $winner_uuid) AS new_uuid
    MERGE (c)-[nr:HAS_MEMBER {uuid: new_uuid}]->(winner)
    SET nr += properties(r)
    SET nr.uuid = new_uuid,
        nr.group_id = coalesce(nr.group_id, c.group_id, $group_id),
        nr.created_at = coalesce(nr.created_at, c.created_at)
    DELETE r
    """

    q_delete = """
    MATCH (loser:Entity)
    WHERE loser.group_id = $group_id AND loser.uuid IN $loser_uuids
    DELETE loser
    """

    merge_queries.extend(
        [
            (q_move_mentions, params),
            (q_move_relates_out, params),
            (q_move_relates_in, params),
            (q_move_has_member, params),
            (q_delete, {'group_id': group_id, 'loser_uuids': loser_uuids}),
        ]
    )

    try:
        await client.run_in_transaction(merge_queries)
    except Exception as exc:
        remaining = await _inspect_remaining_relationships(client, group_id, loser_uuids)
        raise RuntimeError(
            'Refusing to complete dedupe after relationship rewrites failed. '
            f'name={bucket.get("name")!r} group_id={group_id!r} winner_uuid={winner_uuid!r} '
            f'loser_uuids={loser_uuids!r} typed_labels={sorted(merged_labels)!r} '
            f'remaining_relationships={remaining!r}. '
            'Inspect the listed loser UUIDs and the reported relationship types/counts before attempting a manual merge.'
        ) from exc
    return len(loser_uuids)


async def dedupe_nodes(backend, host, port, group_id, dry_run=False):
    logger.info('Connecting to %s for group_id=%s', backend, group_id)
    if backend == 'falkordb':
        logger.warning(
            'FalkorDB: multi-statement merge is best-effort only (no real transaction). '
            'Use --dry-run first; Neo4j remains the safest backend for destructive dedupe.'
        )
    if dry_run:
        logger.info('DRY RUN mode — no changes will be written.')
    client = await get_graph_client(backend, group_id=group_id, host=host, port=port)

    try:
        logger.info('Scanning for duplicate entities...')
        buckets = await _find_duplicate_buckets(client, backend, group_id)
        logger.info('Found %d merge bucket(s).', len(buckets))

        if not buckets:
            logger.info('No duplicates found. Exiting.')
            return

        if dry_run:
            mergeable_buckets = 0
            for bucket in buckets:
                nodes = bucket['nodes']
                typed_labels = sorted(
                    {
                        label
                        for node in nodes
                        for label in _normalize_typed_labels(node.get('labels'), group_id)
                    }
                )
                label_suffix = f' labels={typed_labels}' if typed_labels else ''
                try:
                    await _prepare_bucket_merge(client, group_id, bucket)
                except ValueError as exc:
                    logger.warning(
                        '  [dry-run] Refusing %r (%d node(s))%s: %s',
                        bucket['name'],
                        len(nodes),
                        label_suffix,
                        exc,
                    )
                    continue

                mergeable_buckets += 1
                logger.info(
                    '  [dry-run] Would merge %d copies of %r (keep oldest)%s',
                    len(nodes),
                    bucket['name'],
                    label_suffix,
                )
            logger.info('Dry run complete. %d merge bucket(s) are mergeable.', mergeable_buckets)
            return

        total_merged = 0
        for bucket in buckets:
            logger.info('Merging %d copies of %r...', len(bucket['nodes']), bucket['name'])
            total_merged += await _merge_bucket(client, backend, group_id, bucket)

        logger.info('Dedup complete. Merged %d duplicate node(s).', total_merged)
    finally:
        await client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_backend_args(parser)
    parser.add_argument('--host', default=None, help='Defaults to localhost unless backend env overrides it.')
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='Defaults to 7687 for Neo4j or 6379 for FalkorDB unless backend env overrides it.',
    )
    parser.add_argument('--group-id', required=True)
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help='Show what would be merged without making changes.',
    )
    parser.add_argument(
        '--confirm-destructive',
        action='store_true',
        default=False,
        help='Required flag to confirm destructive deduplication (deletes and rewires nodes).',
    )
    args = parser.parse_args()

    if not args.dry_run and not args.confirm_destructive:
        logger.error('This script deletes duplicate nodes and rewires their relationships.')
        logger.error('Pass --confirm-destructive to proceed, or --dry-run to preview.')
        sys.exit(1)

    try:
        asyncio.run(dedupe_nodes(args.backend, args.host, args.port, args.group_id, dry_run=args.dry_run))
    except GraphDriverSetupError as exc:
        logger.error('%s', exc)
        sys.exit(2)
    except (RuntimeError, ValueError) as exc:
        logger.error('%s', exc)
        sys.exit(1)
