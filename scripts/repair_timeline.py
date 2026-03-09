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
import sys
from typing import Any

from graph_driver import add_backend_args, get_graph_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


async def repair_timeline(backend, host, port, group_id):
    logger.info(f'Connecting to {backend} for group_id={group_id}')
    client = await get_graph_client(backend, group_id=group_id, host=host, port=port)

    try:
        logger.info('Deleting existing NEXT_EPISODE edges...')
        await client.query(
            'MATCH (e1:Episodic)-[r:NEXT_EPISODE]->(e2:Episodic) '
            'WHERE e1.group_id = $group_id AND e2.group_id = $group_id DELETE r',
            {'group_id': group_id},
        )

        logger.info('Fetching episodes...')
        query = """
        MATCH (e:Episodic)
        WHERE e.group_id = $group_id
        RETURN e.uuid, e.valid_at, e.created_at
        ORDER BY coalesce(e.valid_at, e.created_at) ASC, e.created_at ASC, e.uuid ASC
        """
        res = await client.query(query, {'group_id': group_id})

        episodes = sort_episodes_for_timeline(
            [
                {'uuid': rec[0], 'valid_at': rec[1], 'created_at': rec[2]}
                for rec in res.result_set
            ]
        )
        logger.info(f'Found {len(episodes)} episodes. Rebuilding chain...')

        if len(episodes) < 2:
            logger.info('Not enough episodes to link.')
            return

        batch_size = 500
        pairs = build_timeline_pairs(episodes, group_id)
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
