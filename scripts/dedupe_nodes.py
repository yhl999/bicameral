#!/usr/bin/env python3
"""
Post-processing script to deduplicate Entity nodes in a graph database.
Required after running parallel ingestion ("Speed Run"), which creates duplicate entities due to missing unique constraints.

Usage:
    python3 scripts/dedupe_nodes.py --group-id s1_sessions_main --confirm-destructive
    python3 scripts/dedupe_nodes.py --group-id s1_sessions_main --dry-run
    python3 scripts/dedupe_nodes.py --group-id s1_sessions_main --backend falkordb --host localhost --port 6379 --confirm-destructive
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from graph_driver import add_backend_args, get_graph_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CORE_ENTITY_LABEL = 'Entity'
_SAFE_LABEL_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_RESERVED_NODE_KEYS = {'uuid', 'name_embedding'}


def _created_at_sort_key(value) -> str:
    if value is None:
        return '9999-12-31T23:59:59.999999+00:00'
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    return str(value)


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


def _merge_dict_values(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in incoming.items():
        if key not in merged or _is_missing_value(merged[key]):
            merged[key] = value
            continue
        if _is_missing_value(value) or merged[key] == value:
            continue
        if isinstance(merged[key], list) and isinstance(value, list):
            merged[key] = _dedupe_list([*merged[key], *value])
            continue
        if isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dict_values(merged[key], value)
    return merged


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
    avoids transitive bridge-merges such as Person <- Person,Organization ->
    Organization.
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
            if len(generic_records) > 1:
                buckets.append(generic_records)
    elif len(generic_records) > 1:
        buckets.append(generic_records)

    return sorted(
        buckets,
        key=_bucket_sort_key,
    )


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

    created_candidates = []
    for record in records:
        props = dict(record.get('properties') or {})
        created_at = props.get('created_at')
        if created_at is not None:
            created_candidates.append(created_at)
        for key, value in props.items():
            if key in _RESERVED_NODE_KEYS:
                continue
            if key not in winner_props or _is_missing_value(winner_props[key]):
                winner_props[key] = value
                continue
            if _is_missing_value(value) or winner_props[key] == value:
                continue
            if isinstance(winner_props[key], list) and isinstance(value, list):
                winner_props[key] = _dedupe_list([*winner_props[key], *value])
                continue
            if isinstance(winner_props[key], dict) and isinstance(value, dict):
                winner_props[key] = _merge_dict_values(winner_props[key], value)
                continue
            logger.warning(
                'Conflicting property %r while deduping entity %s; keeping winner value.',
                key,
                winner_uuid,
            )

    winner_props['group_id'] = group_id
    winner_props['summary'] = _select_summary(records)
    if created_candidates:
        winner_props['created_at'] = min(created_candidates, key=_created_at_sort_key)

    return {
        'winner_props': winner_props,
        'name_embedding': _select_name_embedding(records),
    }


async def _find_duplicate_buckets(client, backend: str, group_id: str) -> list[dict]:
    del backend  # queries are intentionally group-scoped for every backend
    query = """
    MATCH (n:Entity)
    WHERE n.group_id = $group_id
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


async def _merge_bucket(client, backend: str, group_id: str, bucket: dict) -> int:
    nodes = list(bucket['nodes'])
    winner = nodes[0]
    losers = nodes[1:]
    loser_uuids = [str(node['uuid']) for node in losers if node.get('uuid')]
    if not loser_uuids:
        return 0

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
    merged_payload = build_merged_entity_payload(merge_records, winner_uuid=winner_uuid, group_id=group_id)

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
    DETACH DELETE loser
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

    await client.run_in_transaction(merge_queries)
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
                logger.info(
                    '  [dry-run] Would merge %d copies of %r (keep oldest)%s',
                    len(nodes),
                    bucket['name'],
                    label_suffix,
                )
            logger.info('Dry run complete. %d merge bucket(s) would be deduped.', len(buckets))
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
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=6379)
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

    asyncio.run(dedupe_nodes(args.backend, args.host, args.port, args.group_id, dry_run=args.dry_run))
