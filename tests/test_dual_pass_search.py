"""Tests for dual_pass_search module."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'mcp_server' / 'src'))

from services.dual_pass_search import dual_pass_fact_search


def _make_fact(uuid: str, fact_text: str = '') -> dict:
    return {'uuid': uuid, 'fact': fact_text or f'Fact {uuid}', 'name': 'REL'}


@pytest.fixture
def text_facts():
    return [_make_fact('a1'), _make_fact('a2'), _make_fact('a3')]


@pytest.fixture
def center_facts():
    return [_make_fact('b1'), _make_fact('b2'), _make_fact('a1')]  # a1 overlaps


@pytest.fixture
def mock_search(text_facts, center_facts):
    call_count = 0

    async def search_fn(**kwargs):
        nonlocal call_count
        call_count += 1
        if kwargs.get('center_node_uuid'):
            return {'facts': center_facts}
        return {'facts': text_facts}

    search_fn.call_count = lambda: call_count
    return search_fn


@pytest.mark.asyncio
async def test_dual_pass_basic_merge(mock_search):
    result = await dual_pass_fact_search(
        search_fn=mock_search,
        query='test query',
        group_ids=['g1'],
        center_node_uuid='yuan-uuid',
        max_facts=10,
    )
    facts = result['facts']
    uuids = [f['uuid'] for f in facts]

    # Should have 5 unique facts (a1 deduplicated)
    assert len(facts) == 5
    assert len(set(uuids)) == 5
    assert 'a1' in uuids
    assert 'b1' in uuids
    assert 'b2' in uuids


@pytest.mark.asyncio
async def test_dual_pass_no_center_node():
    """Without center_node_uuid, falls back to single-pass."""
    async def search_fn(**kwargs):
        return {'facts': [_make_fact('x1'), _make_fact('x2')]}

    result = await dual_pass_fact_search(
        search_fn=search_fn,
        query='test',
        max_facts=10,
    )
    assert len(result['facts']) == 2
    assert result['_dual_pass_meta']['center_node_new'] == 0


@pytest.mark.asyncio
async def test_dual_pass_center_node_error_failopen():
    """If center-node search errors, text results still returned."""
    call_count = 0

    async def search_fn(**kwargs):
        nonlocal call_count
        call_count += 1
        if kwargs.get('center_node_uuid'):
            raise RuntimeError('Neo4j down')
        return {'facts': [_make_fact('t1')]}

    result = await dual_pass_fact_search(
        search_fn=search_fn,
        query='test',
        center_node_uuid='uid',
        max_facts=5,
    )
    assert len(result['facts']) == 1
    assert result['facts'][0]['uuid'] == 't1'
    assert result['_dual_pass_meta']['center_node_new'] == 0


@pytest.mark.asyncio
async def test_dual_pass_max_facts_cap():
    """Merged results respect max_facts cap."""
    async def search_fn(**kwargs):
        n = kwargs.get('max_facts', 10)
        return {'facts': [_make_fact(f'f{i}') for i in range(n)]}

    result = await dual_pass_fact_search(
        search_fn=search_fn,
        query='test',
        center_node_uuid='uid',
        max_facts=5,
    )
    # Should not exceed max_facts
    assert len(result['facts']) <= 5


@pytest.mark.asyncio
async def test_dual_pass_metadata():
    """Metadata reports correct counts."""
    async def search_fn(**kwargs):
        if kwargs.get('center_node_uuid'):
            return {'facts': [_make_fact('c1'), _make_fact('c2')]}
        return {'facts': [_make_fact('t1')]}

    result = await dual_pass_fact_search(
        search_fn=search_fn,
        query='test',
        center_node_uuid='uid',
        max_facts=10,
    )
    meta = result['_dual_pass_meta']
    assert meta['text_count'] == 1
    assert meta['center_node_new'] == 2
    assert meta['total_merged'] == 3
    assert meta['center_node_uuid'] == 'uid'
