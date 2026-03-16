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
import os
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
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
_STRONG_HOMONYM_PROOF_KEYS = (
    'canonical_id',
    'external_id',
    'source_id',
    'source_entity_id',
    'resolved_entity_id',
    'employee_id',
    'email',
    'emails',
    'wikidata_id',
)
_WEAK_HOMONYM_PROOF_KEYS = (
    'phone',
    'website',
    'url',
    'domain',
    'linkedin_url',
    'twitter_handle',
    'github_username',
    'crunchbase_url',
)
_HOMONYM_PROOF_KEYS = (*_STRONG_HOMONYM_PROOF_KEYS, *_WEAK_HOMONYM_PROOF_KEYS)


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


def _proof_fields_present(record: dict[str, Any]) -> list[str]:
    props = dict(record.get('properties') or {})
    return [
        key
        for key in _HOMONYM_PROOF_KEYS
        if _normalize_proof_value(props.get(key)) is not None
    ]


def _bucket_node_debug(records: list[dict[str, Any]] | list[dict], group_id: str) -> list[dict[str, Any]]:
    debug_rows: list[dict[str, Any]] = []
    for record in records:
        props = dict(record.get('properties') or {})
        debug_rows.append(
            {
                'uuid': str(record.get('uuid') or '').strip(),
                'created_at': props.get('created_at', record.get('created_at')),
                'labels': sorted({str(label) for label in (record.get('labels') or []) if label}),
                'typed_labels': sorted(_normalize_typed_labels(record.get('labels'), group_id)),
                'proof_fields_present': _proof_fields_present(record),
            }
        )
    return debug_rows


def _bucket_identity_shape(records: list[dict[str, Any]], group_id: str) -> dict[str, Any]:
    typed_signatures: set[tuple[str, ...]] = set()
    generic_count = 0
    for record in records:
        typed_labels = tuple(sorted(_normalize_typed_labels(record.get('labels'), group_id)))
        if typed_labels:
            typed_signatures.add(typed_labels)
        else:
            generic_count += 1
    return {
        'generic_count': generic_count,
        'typed_signatures': [list(signature) for signature in sorted(typed_signatures)],
    }


def _collect_candidate_proof_values(
    records: list[dict[str, Any]],
    proof_keys: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    keys = tuple(proof_keys or _HOMONYM_PROOF_KEYS)
    observed: dict[str, dict[str, Any]] = {}
    for record in records:
        props = dict(record.get('properties') or {})
        uuid = str(record.get('uuid') or '').strip()
        for key in keys:
            value = props.get(key)
            if _normalize_proof_value(value) is None:
                continue
            observed.setdefault(key, {})[uuid] = value
    return observed


def _shared_homonym_proofs(
    records: list[dict[str, Any]],
    proof_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    keys = tuple(proof_keys or _HOMONYM_PROOF_KEYS)
    proofs: dict[str, Any] = {}
    for key in keys:
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


def _bucket_proof_policy(records: list[dict[str, Any]], group_id: str) -> dict[str, Any]:
    typed_labels: set[str] = set()
    has_generic_nodes = False
    for record in records:
        normalized_labels = _normalize_typed_labels(record.get('labels'), group_id)
        if normalized_labels:
            typed_labels.update(normalized_labels)
        else:
            has_generic_nodes = True

    has_person_labels = 'Person' in typed_labels
    mode = 'strict_person_or_homonym'
    if typed_labels and not has_generic_nodes and not has_person_labels:
        mode = 'strict_typed_non_person'

    return {
        'mode': mode,
        'allowed_keys': _STRONG_HOMONYM_PROOF_KEYS,
        'blocked_weak_keys': _WEAK_HOMONYM_PROOF_KEYS,
        'typed_labels': sorted(typed_labels),
        'has_generic_nodes': has_generic_nodes,
        'has_person_labels': has_person_labels,
    }


def _render_proof_value(value: Any) -> str:
    rendered = _canonical_json(value)
    if len(rendered) <= 120:
        return rendered
    return rendered[:117] + '...'


def _format_authorizing_proof(proof: dict[str, Any] | None) -> str:
    if not proof:
        return ' proof=<none>'
    return (
        f" proof={proof.get('proof_key')}={_render_proof_value(proof.get('proof_value'))}"
        f" policy={proof.get('policy_mode')}"
    )


def _suggest_homonym_inspect_query() -> str:
    return (
        'MATCH (n:Entity) WHERE n.group_id = $group_id AND n.uuid IN $uuids '
        'RETURN n.uuid, labels(n) AS labels, n.name AS name, n.canonical_id AS canonical_id, '
        'n.external_id AS external_id, n.email AS email, n.emails AS emails, '
        'n.phone AS phone, n.domain AS domain, n.website AS website, n.url AS url '
        'ORDER BY n.created_at ASC, n.uuid ASC'
    )


def _require_homonym_merge_proof(
    records: list[dict[str, Any]],
    group_id: str,
    bucket_name: str,
) -> dict[str, Any]:
    if len(records) < 2:
        return {}

    policy = _bucket_proof_policy(records, group_id)
    shared_authorizing_proofs = _shared_homonym_proofs(records, policy['allowed_keys'])
    if shared_authorizing_proofs:
        for key in policy['allowed_keys']:
            if key in shared_authorizing_proofs:
                return {
                    'proof_key': key,
                    'proof_value': shared_authorizing_proofs[key],
                    'all_shared_proofs': shared_authorizing_proofs,
                    'policy_mode': policy['mode'],
                }

    weak_shared_proofs = _shared_homonym_proofs(records, policy['blocked_weak_keys'])
    candidate_values = _collect_candidate_proof_values(records)
    bucket_uuids = [str(record.get('uuid') or '').strip() for record in records]
    inspect_params = {'group_id': group_id, 'uuids': bucket_uuids}
    proof_fields_seen = {
        str(record.get('uuid') or '').strip(): _proof_fields_present(record)
        for record in records
    }

    raise ValueError(
        'Refusing to merge same-name entities without shared identity proof. '
        f'name={bucket_name!r} group_id={group_id!r} '
        f'bucket_uuids={bucket_uuids!r} '
        f'bucket_shape={_bucket_identity_shape(records, group_id)!r} '
        f'proof_policy={policy!r} '
        f'nodes={_bucket_node_debug(records, group_id)!r} '
        f'proof_fields_seen_by_uuid={proof_fields_seen!r} '
        f'candidate_proofs={candidate_values!r}. '
        f'blocked_shared_weak_proofs={weak_shared_proofs!r}. '
        f'suggested_inspect_query={_suggest_homonym_inspect_query()!r} '
        f'suggested_inspect_params={inspect_params!r}. '
        'Inspect this bucket manually before any override.'
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
    merge_proof = _require_homonym_merge_proof(
        merge_records,
        group_id,
        bucket_name=str(bucket.get('name') or ''),
    )
    merged_payload = build_merged_entity_payload(merge_records, winner_uuid=winner_uuid, group_id=group_id)

    return {
        'nodes': nodes,
        'winner_uuid': winner_uuid,
        'loser_uuids': loser_uuids,
        'merged_uuids': merged_uuids,
        'merged_labels': merged_labels,
        'merged_payload': merged_payload,
        'merge_proof': merge_proof,
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
    merge_proof = merge_plan.get('merge_proof') or {}

    logger.info(
        '  [proof] Authorized %r winner_uuid=%r%s',
        bucket.get('name'),
        winner_uuid,
        _format_authorizing_proof(merge_proof),
    )

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


def _build_ambiguous_record(bucket: dict, group_id: str, error: str) -> dict[str, Any]:
    """Build a machine-readable record for one skipped ambiguous bucket."""
    nodes = list(bucket.get('nodes') or [])
    return {
        'name': str(bucket.get('name') or ''),
        'group_id': group_id,
        'node_count': len(nodes),
        'uuids': [str(n.get('uuid') or '') for n in nodes],
        'typed_labels': sorted(
            {
                str(label)
                for node in nodes
                for label in _normalize_typed_labels(node.get('labels'), group_id)
            }
        ),
        'error': error,
    }


def _write_ambiguous_report(
    skipped: list[dict[str, Any]],
    group_id: str,
    report_path: str | None,
    total_buckets: int,
    merged_count: int,
) -> str | None:
    """Write a JSON report of skipped ambiguous buckets. Returns the path written, or None."""
    if not skipped:
        return None

    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'group_id': group_id,
        'total_buckets': total_buckets,
        'merged_successfully': merged_count,
        'skipped_ambiguous': len(skipped),
        'buckets': skipped,
    }

    if report_path is None:
        report_path = f'dedupe_ambiguous_report_{group_id}.json'

    os.makedirs(os.path.dirname(report_path) if os.path.dirname(report_path) else '.', exist_ok=True)
    with open(report_path, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)
    return report_path


async def dedupe_nodes(backend, host, port, group_id, dry_run=False, skip_ambiguous=False, ambiguous_report=None):
    logger.info('Connecting to %s for group_id=%s', backend, group_id)
    if backend == 'falkordb':
        logger.warning(
            'FalkorDB: multi-statement merge is best-effort only (no real transaction). '
            'Use --dry-run first; Neo4j remains the safest backend for destructive dedupe.'
        )
    if dry_run:
        logger.info('DRY RUN mode — no changes will be written.')
    if skip_ambiguous:
        logger.info('SKIP-AMBIGUOUS mode — ambiguous buckets will be skipped and reported.')
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
            refused_buckets = 0
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
                    merge_plan = await _prepare_bucket_merge(client, group_id, bucket)
                except ValueError as exc:
                    refused_buckets += 1
                    logger.error(
                        '  [dry-run] Refusing %r (%d node(s))%s: %s',
                        bucket['name'],
                        len(nodes),
                        label_suffix,
                        exc,
                    )
                    continue

                mergeable_buckets += 1
                logger.info(
                    '  [dry-run] Would merge %d copies of %r (keep oldest)%s%s',
                    len(nodes),
                    bucket['name'],
                    label_suffix,
                    _format_authorizing_proof(merge_plan.get('merge_proof')),
                )
            logger.info(
                'Dry run complete. %d merge bucket(s) are mergeable; %d bucket(s) were refused.',
                mergeable_buckets,
                refused_buckets,
            )
            return

        total_merged = 0
        skipped_buckets: list[dict[str, Any]] = []
        for bucket in buckets:
            logger.info('Merging %d copies of %r...', len(bucket['nodes']), bucket['name'])
            try:
                total_merged += await _merge_bucket(client, backend, group_id, bucket)
            except ValueError as exc:
                if not skip_ambiguous:
                    raise
                skipped_buckets.append(
                    _build_ambiguous_record(bucket, group_id, str(exc))
                )
                logger.warning(
                    '  [skip-ambiguous] Skipped %r (%d node(s)): %s',
                    bucket.get('name'),
                    len(bucket.get('nodes') or []),
                    exc,
                )
                continue

        if skipped_buckets:
            report_path = _write_ambiguous_report(
                skipped_buckets, group_id, ambiguous_report, len(buckets), total_merged,
            )
            logger.warning(
                'Dedup complete with skips. Merged %d duplicate node(s); '
                'skipped %d ambiguous bucket(s). Report: %s',
                total_merged,
                len(skipped_buckets),
                report_path,
            )
        else:
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
    parser.add_argument(
        '--skip-ambiguous',
        action='store_true',
        default=False,
        help='Continue past ambiguous same-name buckets that lack shared identity proof, '
        'skipping them instead of aborting. Skipped buckets are logged at WARNING level '
        'and written to a JSON report file.',
    )
    parser.add_argument(
        '--ambiguous-report',
        default=None,
        help='Path for the JSON report of skipped ambiguous buckets. '
        'Defaults to dedupe_ambiguous_report_<group_id>.json in the current directory.',
    )
    args = parser.parse_args()

    if not args.dry_run and not args.confirm_destructive:
        logger.error('This script deletes duplicate nodes and rewires their relationships.')
        logger.error('Pass --confirm-destructive to proceed, or --dry-run to preview.')
        sys.exit(1)

    try:
        asyncio.run(
            dedupe_nodes(
                args.backend,
                args.host,
                args.port,
                args.group_id,
                dry_run=args.dry_run,
                skip_ambiguous=args.skip_ambiguous,
                ambiguous_report=args.ambiguous_report,
            )
        )
    except GraphDriverSetupError as exc:
        logger.error('%s', exc)
        sys.exit(2)
    except (RuntimeError, ValueError) as exc:
        logger.error('%s', exc)
        sys.exit(1)
