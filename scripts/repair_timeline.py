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

from graph_driver import add_backend_args, get_graph_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        if backend == 'neo4j':
            await client.query(
                'MATCH (e1:Episodic)-[r:NEXT_EPISODE]->(e2:Episodic) '
                'WHERE e1.group_id = $group_id AND e2.group_id = $group_id DELETE r',
                {'group_id': group_id},
            )
        else:
            await client.query('MATCH ()-[r:NEXT_EPISODE]->() DELETE r')

        logger.info('Fetching episodes...')
        if backend == 'neo4j':
            query = """
            MATCH (e:Episodic)
            WHERE e.group_id = $group_id
            RETURN e.uuid, e.reference_time, e.created_at
            ORDER BY e.reference_time ASC, e.created_at ASC, e.uuid ASC
            """
            res = await client.query(query, {'group_id': group_id})
        else:
            query = """
            MATCH (e:Episodic)
            RETURN e.uuid, e.reference_time, e.created_at
            ORDER BY e.reference_time ASC, e.created_at ASC, e.uuid ASC
            """
            res = await client.query(query)

        episodes = [
            {'uuid': rec[0], 'reference_time': rec[1], 'created_at': rec[2]}
            for rec in res.result_set
        ]
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
