from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


dedupe_nodes = importlib.import_module('scripts.dedupe_nodes')
repair_timeline = importlib.import_module('scripts.repair_timeline')


def test_bucket_duplicate_records_merges_generic_into_typed_bucket() -> None:
    records = [
        {
            'uuid': 'generic-older',
            'created_at': '2026-03-09T10:00:00Z',
            'labels': ['Entity'],
        },
        {
            'uuid': 'typed-newer',
            'created_at': '2026-03-09T10:05:00Z',
            'labels': ['Entity', 'Person'],
        },
    ]

    buckets = dedupe_nodes.bucket_duplicate_records(records, 's1_sessions_main')

    assert len(buckets) == 1
    assert [row['uuid'] for row in buckets[0]] == ['generic-older', 'typed-newer']


def test_bucket_duplicate_records_keeps_conflicting_typed_nodes_apart() -> None:
    records = [
        {
            'uuid': 'person-node',
            'created_at': '2026-03-09T10:00:00Z',
            'labels': ['Entity', 'Person'],
        },
        {
            'uuid': 'org-node',
            'created_at': '2026-03-09T10:01:00Z',
            'labels': ['Entity', 'Organization'],
        },
    ]

    buckets = dedupe_nodes.bucket_duplicate_records(records, 's1_sessions_main')

    assert buckets == []


def test_dedupe_script_preserves_required_edge_properties_when_rewiring() -> None:
    src = (ROOT / 'scripts' / 'dedupe_nodes.py').read_text(encoding='utf-8')

    assert 'SET nr += properties(r)' in src
    assert 'nr.group_id' in src
    assert 'nr.uuid' in src
    assert 'nr.episodes = coalesce(nr.episodes, [])' in src


def test_build_timeline_pairs_assigns_group_and_deterministic_uuid() -> None:
    episodes = [
        {'uuid': 'ep-1', 'created_at': '2026-03-09T09:00:00Z'},
        {'uuid': 'ep-2', 'created_at': '2026-03-09T09:05:00Z'},
        {'uuid': 'ep-3', 'created_at': '2026-03-09T09:10:00Z'},
    ]

    pairs = repair_timeline.build_timeline_pairs(episodes, 's1_sessions_main')

    assert pairs == [
        {
            'prev': 'ep-1',
            'curr': 'ep-2',
            'uuid': 'timeline:ep-1->ep-2',
            'group_id': 's1_sessions_main',
            'created_at': '2026-03-09T09:00:00Z',
        },
        {
            'prev': 'ep-2',
            'curr': 'ep-3',
            'uuid': 'timeline:ep-2->ep-3',
            'group_id': 's1_sessions_main',
            'created_at': '2026-03-09T09:05:00Z',
        },
    ]


def test_repair_timeline_script_sets_required_next_episode_edge_properties() -> None:
    src = (ROOT / 'scripts' / 'repair_timeline.py').read_text(encoding='utf-8')

    assert 'MERGE (a)-[e:NEXT_EPISODE {uuid: pair.uuid}]->(b)' in src
    assert 'SET e.group_id = pair.group_id' in src
    assert 'e.created_at = pair.created_at' in src
