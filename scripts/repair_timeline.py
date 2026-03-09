#!/usr/bin/env python3
"""
Post-processing script to repair the Episode timeline in a graph database.
Required after running parallel ingestion ("Speed Run"), which fragments the NEXT_EPISODE chain.

Usage:
    python3 scripts/repair_timeline.py --group-id s1_sessions_main --confirm-destructive
    python3 scripts/repair_timeline.py --group-id s1_sessions_main --backend falkordb --host localhost --port 6379 --confirm-destructive
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from collections import defaultdict
from typing import Any

from graph_driver import add_backend_args, get_graph_client

try:
    from graph_driver import GraphDriverSetupError
except ImportError:  # pragma: no cover - test stubs may omit the concrete error type
    class GraphDriverSetupError(RuntimeError):
        pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_SESSION_CHUNK_RE = re.compile(r'session chunk:\s*([^\(]+)')
_SUBCHUNK_SUFFIX_RE = re.compile(r':p\d+$')
_CHUNK_SUFFIX_RE = re.compile(r':c\d+$')


def _time_sort_key(value: Any) -> str:
    if value is None:
        return '9999-12-31T23:59:59.999999+00:00'
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    return str(value)


def sort_episodes_for_timeline(episodes: list[dict]) -> list[dict]:
    return sorted(
        episodes,
        key=lambda episode: (
            _time_sort_key(episode.get('valid_at') or episode.get('created_at')),
            _time_sort_key(episode.get('created_at') or episode.get('valid_at')),
            str(episode.get('uuid') or ''),
        ),
    )


def infer_episode_stream_key(source_description: str | None) -> str:
    if not isinstance(source_description, str) or not source_description.strip():
        return ''

    match = _SESSION_CHUNK_RE.search(source_description)
    if not match:
        return ''

    chunk_key = match.group(1).strip()
    if not chunk_key:
        return ''

    chunk_key = _SUBCHUNK_SUFFIX_RE.sub('', chunk_key)
    chunk_key = _CHUNK_SUFFIX_RE.sub('', chunk_key)
    return chunk_key.strip()


def _episode_debug(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        'uuid': episode.get('uuid'),
        'saga_uuids': sorted(
            {
                str(saga_uuid).strip()
                for saga_uuid in (episode.get('saga_uuids') or [])
                if str(saga_uuid or '').strip()
            }
        ),
        'stream_key': infer_episode_stream_key(episode.get('source_description')),
        'source_description': episode.get('source_description'),
        'valid_at': episode.get('valid_at'),
        'created_at': episode.get('created_at'),
    }



def build_timeline_groups(episodes: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    ungrouped: list[dict] = []

    for episode in sort_episodes_for_timeline(episodes):
        saga_ids = sorted(
            {
                str(saga_uuid).strip()
                for saga_uuid in (episode.get('saga_uuids') or [])
                if str(saga_uuid or '').strip()
            }
        )
        if len(saga_ids) > 1:
            raise ValueError(
                'Refusing to repair timeline: one episode belongs to multiple sagas. '
                f'episode={_episode_debug(episode)!r}. '
                'Inspect HAS_EPISODE links and repair the saga assignment before retrying.'
            )

        timeline_key = ''
        if saga_ids:
            timeline_key = f'saga:{saga_ids[0]}'
        else:
            stream_key = infer_episode_stream_key(episode.get('source_description'))
            if stream_key:
                timeline_key = f'stream:{stream_key}'

        if timeline_key:
            groups[timeline_key].append(episode)
        else:
            ungrouped.append(episode)

    if ungrouped:
        ungrouped_debug = [_episode_debug(episode) for episode in ungrouped]
        if groups:
            raise ValueError(
                'Refusing to repair timeline: some episodes have no saga or stream identity while other '
                'episodes do. Repair would be ambiguous and could cross-link unrelated streams. '
                f'ungrouped_episodes={ungrouped_debug!r} existing_groups={sorted(groups)!r}. '
                'Inspect the listed episode UUIDs, source_description values, and saga membership first.'
            )
        if len(ungrouped) > 1:
            raise ValueError(
                'Refusing to repair timeline: multiple episodes lack saga/stream identity, so the group '
                'cannot be proven to be a single linear stream. '
                f'ungrouped_episodes={ungrouped_debug!r}. '
                'Add explicit saga membership or a recoverable stream key before retrying.'
            )
        groups['ungrouped'] = ungrouped

    return dict(groups)



def build_timeline_pairs(episodes: list[dict], group_id: str) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    for idx in range(len(episodes) - 1):
        prev = episodes[idx]
        curr = episodes[idx + 1]
        pairs.append(
            {
                'prev': prev['uuid'],
                'curr': curr['uuid'],
                'uuid': f"timeline:{prev['uuid']}->{curr['uuid']}",
                'group_id': group_id,
                'created_at': prev.get('created_at') or curr.get('created_at'),
            }
        )
    return pairs


async def _fetch_group_episodes(client, group_id: str) -> list[dict[str, Any]]:
    query = """
    MATCH (e:Episodic)
    WHERE e.group_id = $group_id
    OPTIONAL MATCH (s:Saga)-[:HAS_EPISODE]->(e)
    WHERE s.group_id = $group_id
    RETURN e.uuid,
           e.valid_at,
           e.created_at,
           e.source_description,
           collect(DISTINCT s.uuid) AS saga_uuids
    ORDER BY coalesce(e.valid_at, e.created_at) ASC, e.created_at ASC, e.uuid ASC
    """
    res = await client.query(query, {'group_id': group_id})
    return [
        {
            'uuid': rec[0],
            'valid_at': rec[1],
            'created_at': rec[2],
            'source_description': rec[3],
            'saga_uuids': list(rec[4] or []),
        }
        for rec in res.result_set
    ]


def _edge_debug(edge: dict[str, Any]) -> dict[str, Any]:
    return {
        'prev': edge.get('prev'),
        'curr': edge.get('curr'),
        'uuid': edge.get('uuid'),
    }



def _edge_key(edge: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(edge.get('prev') or ''),
        str(edge.get('curr') or ''),
        str(edge.get('uuid') or ''),
    )


async def _fetch_existing_timeline_edges(client, group_id: str) -> list[dict[str, Any]]:
    query = """
    MATCH (a:Episodic)-[r:NEXT_EPISODE]->(b:Episodic)
    WHERE a.group_id = $group_id AND b.group_id = $group_id
    RETURN a.uuid, b.uuid, r.uuid
    ORDER BY a.uuid ASC, b.uuid ASC, r.uuid ASC
    """
    res = await client.query(query, {'group_id': group_id})
    return [
        {
            'prev': row[0],
            'curr': row[1],
            'uuid': row[2],
        }
        for row in res.result_set
    ]


async def _link_timeline_pairs(client, backend: str, group_id: str, pairs: list[dict[str, object]]) -> None:
    if not pairs:
        return

    batch_size = 500
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i : i + batch_size]

        if backend == 'falkordb':
            for pair in chunk:
                q_single = """
                MATCH (a:Episodic {uuid: $prev}), (b:Episodic {uuid: $curr})
                WHERE a.group_id = $group_id AND b.group_id = $group_id
                MERGE (a)-[e:NEXT_EPISODE {uuid: $uuid}]->(b)
                SET e.group_id = $group_id,
                    e.created_at = $created_at
                """
                await client.query(
                    q_single,
                    {
                        'prev': pair['prev'],
                        'curr': pair['curr'],
                        'uuid': pair['uuid'],
                        'group_id': pair['group_id'],
                        'created_at': pair['created_at'],
                    },
                )
        else:
            q_link = """
            UNWIND $pairs AS pair
            MATCH (a:Episodic {uuid: pair.prev, group_id: $group_id})
            MATCH (b:Episodic {uuid: pair.curr, group_id: $group_id})
            MERGE (a)-[e:NEXT_EPISODE {uuid: pair.uuid}]->(b)
            SET e.group_id = pair.group_id,
                e.created_at = pair.created_at
            """
            await client.query(q_link, {'pairs': chunk, 'group_id': group_id})
        logger.info('Linked desired episodes %d to %d', i, min(i + batch_size, len(pairs)))


async def _delete_stale_timeline_edges(client, backend: str, group_id: str, edges: list[dict[str, Any]]) -> None:
    if not edges:
        return

    batch_size = 500
    if backend == 'falkordb':
        q_single = """
        MATCH (a:Episodic {uuid: $prev})-[r:NEXT_EPISODE]->(b:Episodic {uuid: $curr})
        WHERE a.group_id = $group_id
          AND b.group_id = $group_id
          AND coalesce(toString(r.uuid), '') = $edge_uuid
        DELETE r
        """
        for edge in edges:
            await client.query(
                q_single,
                {
                    'prev': edge['prev'],
                    'curr': edge['curr'],
                    'edge_uuid': str(edge.get('uuid') or ''),
                    'group_id': group_id,
                },
            )
        return

    for i in range(0, len(edges), batch_size):
        chunk = edges[i : i + batch_size]
        q_delete = """
        UNWIND $edges AS edge
        MATCH (a:Episodic {uuid: edge.prev})-[r:NEXT_EPISODE]->(b:Episodic {uuid: edge.curr})
        WHERE a.group_id = $group_id
          AND b.group_id = $group_id
          AND coalesce(toString(r.uuid), '') = edge.edge_uuid
        DELETE r
        """
        await client.query(
            q_delete,
            {
                'group_id': group_id,
                'edges': [
                    {
                        'prev': edge['prev'],
                        'curr': edge['curr'],
                        'edge_uuid': str(edge.get('uuid') or ''),
                    }
                    for edge in chunk
                ],
            },
        )


async def repair_timeline(backend, host, port, group_id):
    logger.info('Connecting to %s for group_id=%s', backend, group_id)
    client = await get_graph_client(backend, group_id=group_id, host=host, port=port)

    try:
        logger.info('Fetching episodes...')
        episodes = sort_episodes_for_timeline(await _fetch_group_episodes(client, group_id))
        logger.info('Found %d episodes. Analysing timeline groups...', len(episodes))

        if not episodes:
            logger.info('No episodes found. Exiting.')
            return

        timeline_groups = build_timeline_groups(episodes)
        pairs: list[dict[str, object]] = []
        for timeline_key, grouped_episodes in sorted(timeline_groups.items()):
            ordered = sort_episodes_for_timeline(grouped_episodes)
            group_pairs = build_timeline_pairs(ordered, group_id)
            pairs.extend(group_pairs)
            logger.info(
                'Prepared timeline %s with %d episode(s) and %d desired link(s).',
                timeline_key,
                len(ordered),
                len(group_pairs),
            )

        desired_keys = {_edge_key(pair) for pair in pairs}
        logger.info('Fetching existing NEXT_EPISODE edges before staged repair...')
        existing_before = await _fetch_existing_timeline_edges(client, group_id)
        logger.info('Found %d existing NEXT_EPISODE edge(s).', len(existing_before))

        await _link_timeline_pairs(client, backend, group_id, pairs)
        existing_after_link = await _fetch_existing_timeline_edges(client, group_id)

        current_keys = {_edge_key(edge) for edge in existing_after_link}
        missing_after_link = [pair for pair in pairs if _edge_key(pair) not in current_keys]
        if missing_after_link:
            raise RuntimeError(
                'Refusing to delete stale NEXT_EPISODE edges because staged desired links did not materialize. '
                f'group_id={group_id!r} missing_pairs={[_edge_debug(pair) for pair in missing_after_link]!r} '
                f'current_edges_sample={[_edge_debug(edge) for edge in existing_after_link[:10]]!r}. '
                'Inspect the listed episode UUIDs and rerun only after the desired links can be created safely.'
            )

        # Only prune edges that existed before this repair pass started.
        # This avoids deleting concurrently-created NEXT_EPISODE edges that may
        # be valid but were not part of the snapshot used to build `pairs`.
        stale_edges = [edge for edge in existing_before if _edge_key(edge) not in desired_keys]
        if stale_edges:
            logger.info('Deleting %d stale NEXT_EPISODE edge(s) after staged validation...', len(stale_edges))
            await _delete_stale_timeline_edges(client, backend, group_id, stale_edges)

        existing_final = await _fetch_existing_timeline_edges(client, group_id)
        final_keys = {_edge_key(edge) for edge in existing_final}
        missing_final = [pair for pair in pairs if _edge_key(pair) not in final_keys]
        unexpected_final = [edge for edge in existing_final if _edge_key(edge) not in desired_keys]
        if missing_final or unexpected_final:
            raise RuntimeError(
                'Timeline repair did not converge to the validated desired edge set. '
                f'group_id={group_id!r} missing_pairs={[_edge_debug(pair) for pair in missing_final]!r} '
                f'unexpected_edges={[_edge_debug(edge) for edge in unexpected_final[:10]]!r} '
                f'expected_edge_count={len(pairs)} actual_edge_count={len(existing_final)}. '
                'Inspect the listed episode/edge UUIDs before attempting another destructive repair.'
            )

        logger.info(
            'Timeline repair complete. desired_edges=%d removed_stale_edges=%d final_edges=%d',
            len(pairs),
            len(stale_edges),
            len(existing_final),
        )
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
        '--confirm-destructive',
        action='store_true',
        default=False,
        help='Required flag to confirm destructive timeline reconciliation (stages desired edges, then prunes stale ones).',
    )
    args = parser.parse_args()

    if not args.confirm_destructive:
        logger.error('This script may delete stale NEXT_EPISODE edges after validating the desired replacement set.')
        logger.error('Pass --confirm-destructive to proceed.')
        sys.exit(1)

    try:
        asyncio.run(repair_timeline(args.backend, args.host, args.port, args.group_id))
    except GraphDriverSetupError as exc:
        logger.error('%s', exc)
        sys.exit(2)
    except (RuntimeError, ValueError) as exc:
        logger.error('%s', exc)
        sys.exit(1)
