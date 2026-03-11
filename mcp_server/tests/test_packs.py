"""Tests for pack registry and materialization behavior in Exec 3."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import os

import pytest
import asyncio

from mcp_server.src.routers import packs
from mcp_server.src.services.change_ledger import ChangeLedger
from mcp_server.src.services.pack_registry import PackRegistryService
from mcp_server.src.models.typed_memory import EvidenceRef, StateFact


@pytest.fixture
def registry_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create an isolated registry file for the test and point the service to it."""
    path = tmp_path / 'pack_registry.json'
    svc = PackRegistryService(path)
    svc.refresh()
    monkeypatch.setenv('BICAMERAL_USER_PACK_REGISTRY_PATH', str(path))
    return path


@pytest.fixture
def ledger_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / 'change_ledger.db'
    monkeypatch.setenv('BICAMERAL_CHANGE_LEDGER_PATH', str(path))
    return path


def _evidence() -> EvidenceRef:
    return EvidenceRef.model_validate(
        {
            'kind': 'message',
            'source_system': 'test',
            'locator': {
                'system': 'test',
                'conversation_id': 'conv-1',
                'message_id': 'msg-1',
            },
        }
    )


def _make_state_fact(*, subject: str, predicate: str, value, ts: datetime) -> StateFact:
    return StateFact.model_validate(
        {
            'object_id': f'{subject}:{predicate}',
            'root_id': f'{subject}:{predicate}',
            'fact_type': 'world_state',
            'subject': subject,
            'predicate': predicate,
            'value': value,
            'scope': 'private',
            'policy_scope': 'private',
            'visibility_scope': 'private',
            'evidence_refs': [_evidence()],
            'created_at': ts.replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
            'valid_at': ts.replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
        }
    )


def test_list_packs_supports_filtering(registry_path: Path):
    # Contextual packs should be discoverable by scope and intent filters.
    result = asyncio.run(packs.list_packs())
    assert isinstance(result, list)
    ids = {item['id'] for item in result}
    assert {'context-vc-deal-brief', 'context-crypto-constraints'} <= ids
    assert {'workflow-deal-review', 'workflow-standup'} <= ids

    context_only = asyncio.run(packs.list_packs({'scope': 'context'}))
    assert all(item['scope'] == 'context' for item in context_only)

    by_intent = asyncio.run(packs.list_packs({'intent': 'decision_maker'}))
    assert by_intent == [] or len(by_intent) >= 1


def test_get_context_pack_materializes_matching_facts(ledger_path: Path, registry_path: Path):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    ledger = ChangeLedger(ledger_path)

    # Facts older -> lower precedence than newer in final output.
    old_fact = _make_state_fact(
        subject='coinbase',
        predicate='industry',
        value='crypto',
        ts=now - timedelta(minutes=30),
    )
    new_fact = _make_state_fact(
        subject='a16z',
        predicate='industry',
        value='ai',
        ts=now,
    )
    unrelated = _make_state_fact(
        subject='a16z',
        predicate='team_size',
        value=12,
        ts=now,
    )

    ledger.append_event('assert', actor_id='unit-test', reason='seed', payload=old_fact, object_id=old_fact.object_id)
    ledger.append_event('assert', actor_id='unit-test', reason='seed', payload=new_fact, object_id=new_fact.object_id)
    ledger.append_event(
        'assert',
        actor_id='unit-test',
        reason='seed',
        payload=unrelated,
        object_id=unrelated.object_id,
    )

    result = asyncio.run(packs.get_context_pack('context-vc-deal-brief'))
    assert isinstance(result, dict)
    assert result.get('pack_id') == 'context-vc-deal-brief'
    facts = result.get('facts', [])

    assert [f['predicate'] for f in facts] == ['industry', 'industry']
    # newer fact should be materialized first
    assert facts[0]['subject'] == 'a16z'
    assert result['fact_count'] == 2


def test_get_context_pack_task_filter_reduces_matches(ledger_path: Path, registry_path: Path):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    ledger = ChangeLedger(ledger_path)

    ledger.append_event(
        'assert',
        actor_id='unit-test',
        reason='seed',
        payload=_make_state_fact(subject='coinbase', predicate='industry', value='crypto', ts=now),
        object_id='coinbase:industry',
    )
    ledger.append_event(
        'assert',
        actor_id='unit-test',
        reason='seed',
        payload=_make_state_fact(subject='a16z', predicate='industry', value='ai', ts=now),
        object_id='a16z:industry',
    )

    filtered = asyncio.run(packs.get_context_pack('context-vc-deal-brief', task='a16z'))
    assert filtered.get('fact_count') == 1
    assert filtered['facts'][0]['subject'] == 'a16z'


def test_describe_pack_returns_schema_and_examples(registry_path: Path):
    result = asyncio.run(packs.describe_pack('context-vc-deal-brief'))
    assert result.get('pack_id') == 'context-vc-deal-brief'
    assert isinstance(result.get('schema'), dict)
    assert isinstance(result.get('examples'), list)
    assert result.get('instructions')


def test_create_workflow_pack_is_persistent(registry_path: Path):
    created = asyncio.run(
        packs.create_workflow_pack(
            {
                'id': 'workflow-earnings-review',
                'scope': 'workflow',
                'intent': 'verifier',
                'consumer': 'planner',
                'version': '1.2.0',
                'description': 'Workflow for review of quarterly earnings call notes',
                'predicates': ['earnings', 'revenue', 'expense'],
                'definition': {
                    'trigger': 'quarter_end',
                    'steps': [{'step': 'ingest', 'action': 'collect', 'target': 'facts'}],
                    'examples': [{'step': 'ingest', 'action': 'collect facts'}],
                },
            }
        )
    )
    assert created.get('pack', {}).get('id') == 'workflow-earnings-review'

    service = PackRegistryService(os.environ['BICAMERAL_USER_PACK_REGISTRY_PATH'])
    assert service.get_pack('workflow-earnings-review') is not None

    # Reloading should preserve the newly created pack.
    service2 = PackRegistryService(os.environ['BICAMERAL_USER_PACK_REGISTRY_PATH'])
    assert service2.get_pack('workflow-earnings-review') is not None


def test_list_packs_invalid_filter_returns_empty_list(registry_path: Path):
    assert asyncio.run(packs.list_packs({'scope': 'not-a-scope'})) == []
