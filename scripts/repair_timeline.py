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
                f"Refusing to repair timeline: episode {episode.get('uuid')!r} belongs to multiple sagas {saga_ids!r}."
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
        if groups:
            raise ValueError(
                'Refusing to repair timeline: some episodes have no saga or stream identity while other '
                'episodes do. Repair would be ambiguous and could cross-link unrelated streams.'
            )
        if len(ungrouped) > 1:
            raise ValueError(
                'Refusing to repair timeline: multiple episodes lack saga/stream identity, so the group '
                'cannot be proven to be a single linear stream.'
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
                'Prepared timeline %s with %d episode(s) and %d link(s).',
                timeline_key,
                len(ordered),
                len(group_pairs),
            )

        logger.info('Deleting existing NEXT_EPISODE edges...')
        await client.query(
            'MATCH (e1:Episodic)-[r:NEXT_EPISODE]->(e2:Episodic) '
            'WHERE e1.group_id = $group_id AND e2.group_id = $group_id DELETE r',
            {'group_id': group_id},
        )

        if not pairs:
            logger.info('No timeline links need rebuilding after safe partitioning.')
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
            logger.info('Linked episodes %d to %d', i, min(i + batch_size, len(pairs)))

        logger.info('Timeline repair complete.')
    finally:
        await client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_backend_args(parser)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=6379)
    parser.add_argument('--group-id', required=True)
    parser.add_argument(
        '--confirm-destructive',
        action='store_true',
        default=False,
        help='Required flag to confirm destructive timeline rebuild (deletes all NEXT_EPISODE edges first).',
    )
    args = parser.parse_args()

    if not args.confirm_destructive:
        logger.error('This script deletes ALL existing NEXT_EPISODE edges before rebuilding.')
        logger.error('Pass --confirm-destructive to proceed.')
        sys.exit(1)

    asyncio.run(repair_timeline(args.backend, args.host, args.port, args.group_id))
