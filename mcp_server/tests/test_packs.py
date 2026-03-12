"""Tests for pack registry and materialization behavior in Exec 3."""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from mcp_server.src.models.typed_memory import EvidenceRef, StateFact
from mcp_server.src.routers import packs
from mcp_server.src.services.change_ledger import ChangeLedger
from mcp_server.src.services.pack_registry import PackRegistryService
from mcp_server.src.services.schema_validation import _validate_typed_object


@pytest.fixture
def registry_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create an isolated user-pack registry and point the service to it."""
    path = tmp_path / 'runtime_user_pack_registry.json'
    monkeypatch.setenv('BICAMERAL_USER_PACK_REGISTRY_PATH', str(path))
    monkeypatch.delenv('BICAMERAL_PACK_REGISTRY_PATH', raising=False)
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


def _workflow_definition(*, pack_id: str) -> dict[str, object]:
    return {
        'pack_id': pack_id,
        'scope': 'workflow',
        'intent': 'verifier',
        'consumer': 'planner',
        'version': '1.2.0',
        'description': f'Workflow for {pack_id}',
        'predicates': ['earnings', 'revenue', 'expense'],
        'definition': {
            'trigger': 'quarter_end',
            'steps': [{'step': 'ingest', 'action': 'collect facts', 'target': 'facts'}],
            'examples': [{'step': 'ingest', 'action': 'collect facts'}],
            'instructions': 'Collect the latest financial facts before summarizing.',
        },
    }


def _nested_mapping(depth: int) -> dict[str, object]:
    current: dict[str, object] = {'leaf': 'x'}
    for index in range(depth):
        current = {f'level_{index}': current}
    return current


def _assert_schema_valid(obj: dict[str, object], schema_name: str) -> None:
    ok, err = _validate_typed_object(obj, schema_name)
    assert ok, err


def _workflow_default_schema() -> dict[str, object]:
    return {
        'type': 'array',
        'items': {
            'type': 'object',
            'required': ['step', 'action'],
            'properties': {
                'step': {'type': 'string'},
                'action': {'type': 'string'},
            },
        },
    }


def test_list_packs_supports_filtering(registry_path: Path):
    result = asyncio.run(packs.list_packs())
    assert isinstance(result, list)
    ids = {item['id'] for item in result}
    assert {'context-vc-deal-brief', 'context-crypto-constraints'} <= ids
    assert {'workflow-deal-review', 'workflow-standup'} <= ids
    assert all('definition' not in item for item in result)

    context_only = asyncio.run(packs.list_packs({'scope': 'context'}))
    assert context_only
    assert all(item['scope'] == 'context' for item in context_only)

    by_intent = asyncio.run(packs.list_packs({'intent': 'decision_maker'}))
    assert by_intent
    assert all(item['intent'] == 'decision_maker' for item in by_intent)


def test_get_context_pack_materializes_matching_facts(ledger_path: Path, registry_path: Path):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    ledger = ChangeLedger(ledger_path)

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
    assert result.get('pack_id') == 'context-vc-deal-brief'
    assert result['pack_metadata']['id'] == 'context-vc-deal-brief'
    facts = result.get('facts', [])

    assert [fact['predicate'] for fact in facts] == ['industry', 'industry']
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
    assert result.get('pack_registry', {}).get('id') == 'context-vc-deal-brief'
    assert isinstance(result.get('schema'), dict)
    assert isinstance(result.get('examples'), list)
    assert result.get('instructions')


def test_create_workflow_pack_is_persistent_in_private_registry(registry_path: Path):
    created = asyncio.run(packs.create_workflow_pack(_workflow_definition(pack_id='workflow-earnings-review')))
    assert created.get('id') == 'workflow-earnings-review'
    assert created.get('scope') == 'workflow'
    assert 'definition' not in created

    service = PackRegistryService(os.environ['BICAMERAL_USER_PACK_REGISTRY_PATH'])
    assert service.get_pack('workflow-earnings-review') is not None

    persisted = json.loads(registry_path.read_text(encoding='utf-8'))
    persisted_ids = {item['id'] for item in persisted['packs']}
    assert persisted_ids == {'workflow-earnings-review'}


def test_create_workflow_pack_normalizes_schema_valid_metadata_across_create_list_and_describe(
    registry_path: Path,
):
    pack_id = 'workflow-metadata-roundtrip'
    definition = _workflow_definition(pack_id=pack_id)
    definition['created_at'] = '2026-03-11T12:34:56+02:00'
    definition['last_updated'] = '2026-03-11T13:34:56+02:00'
    definition['surprise'] = 'should not persist'

    created = asyncio.run(packs.create_workflow_pack(definition))
    assert created.get('id') == pack_id
    assert created.get('created_at') == '2026-03-11T10:34:56Z'
    assert created.get('last_updated') == '2026-03-11T11:34:56Z'
    assert 'surprise' not in created
    _assert_schema_valid(created, 'PackRegistry')

    listed = next(item for item in asyncio.run(packs.list_packs({'scope': 'workflow'})) if item['id'] == pack_id)
    assert 'surprise' not in listed
    _assert_schema_valid(listed, 'PackRegistry')

    described = asyncio.run(packs.describe_pack(pack_id))
    assert described.get('pack_registry', {}).get('created_at') == '2026-03-11T10:34:56Z'
    assert described.get('pack_registry', {}).get('last_updated') == '2026-03-11T11:34:56Z'
    assert 'surprise' not in described.get('pack_registry', {})
    _assert_schema_valid(described.get('pack_registry', {}), 'PackRegistry')
    _assert_schema_valid(described, 'PackDefinition')

    persisted = json.loads(registry_path.read_text(encoding='utf-8'))
    persisted_pack = persisted['packs'][0]
    assert persisted_pack['created_at'] == '2026-03-11T10:34:56Z'
    assert persisted_pack['last_updated'] == '2026-03-11T11:34:56Z'
    assert 'surprise' not in persisted_pack


def test_create_workflow_pack_rejects_invalid_pack_registry_timestamps(registry_path: Path):
    definition = _workflow_definition(pack_id='workflow-invalid-timestamps')
    definition['created_at'] = 'not-a-date'

    result = asyncio.run(packs.create_workflow_pack(definition))
    assert result.get('error') == 'validation_error'
    assert 'created_at' in result.get('message', '')

    if registry_path.exists():
        persisted = json.loads(registry_path.read_text(encoding='utf-8'))
        assert persisted['packs'] == []


def test_dotted_pack_ids_round_trip_across_create_describe_and_get(registry_path: Path):
    pack_id = 'workflow.earnings.review'

    created = asyncio.run(packs.create_workflow_pack(_workflow_definition(pack_id=pack_id)))
    assert created.get('id') == pack_id

    described = asyncio.run(packs.describe_pack(pack_id))
    assert described.get('pack_id') == pack_id
    assert described.get('pack_registry', {}).get('id') == pack_id

    materialized = asyncio.run(packs.get_workflow_pack(pack_id))
    assert materialized.get('pack_id') == pack_id
    assert materialized.get('pack_metadata', {}).get('id') == pack_id
    assert materialized.get('error') is None


def test_describe_pack_defaults_workflow_schema_when_definition_schema_is_omitted(registry_path: Path):
    pack_id = 'workflow-default-schema'
    created = asyncio.run(packs.create_workflow_pack(_workflow_definition(pack_id=pack_id)))
    assert created.get('id') == pack_id

    described = asyncio.run(packs.describe_pack(pack_id))
    assert described.get('pack_id') == pack_id
    assert described.get('schema') == _workflow_default_schema()
    assert described.get('schema', {}).get('type') == 'array'
    assert 'subject' not in json.dumps(described.get('schema', {}), sort_keys=True)
    _assert_schema_valid(described, 'PackDefinition')



def test_create_workflow_pack_rejects_duplicate_ids_and_builtin_hijack(registry_path: Path):
    original = asyncio.run(packs.create_workflow_pack(_workflow_definition(pack_id='workflow-earnings-review')))
    assert original.get('id') == 'workflow-earnings-review'

    duplicate = asyncio.run(packs.create_workflow_pack(_workflow_definition(pack_id='workflow-earnings-review')))
    assert duplicate.get('error') == 'validation_error'
    assert 'already exists' in duplicate.get('message', '')

    hijack_attempt = asyncio.run(packs.create_workflow_pack(_workflow_definition(pack_id='context-vc-deal-brief')))
    assert hijack_attempt.get('error') == 'validation_error'
    assert 'already exists' in hijack_attempt.get('message', '')

    service = PackRegistryService(registry_path)
    resolved = service.get_pack('workflow-earnings-review')
    assert resolved is not None
    assert resolved['id'] == 'workflow-earnings-review'


def test_create_workflow_pack_rejects_empty_predicates(registry_path: Path):
    definition = _workflow_definition(pack_id='workflow-empty-preds')
    definition['predicates'] = []

    result = asyncio.run(packs.create_workflow_pack(definition))
    assert result.get('error') == 'validation_error'
    assert 'non-empty' in result.get('message', '')


def test_create_workflow_pack_requires_private_registry_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('BICAMERAL_USER_PACK_REGISTRY_PATH', raising=False)
    monkeypatch.delenv('BICAMERAL_PACK_REGISTRY_PATH', raising=False)

    result = asyncio.run(packs.create_workflow_pack(_workflow_definition(pack_id='workflow-public-fallback')))
    assert result.get('error') == 'validation_error'
    assert 'BICAMERAL_USER_PACK_REGISTRY_PATH' in result.get('message', '')


def test_create_workflow_pack_legacy_env_does_not_bypass_user_registry_requirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """BICAMERAL_PACK_REGISTRY_PATH (legacy) must NOT allow creation when BICAMERAL_USER_PACK_REGISTRY_PATH is unset.

    Regression: before the fix, a non-public legacy path passed _is_public_registry() → False,
    so _assert_user_registry_writeable() silently permitted the write.  The dedicated user-path
    flag now closes that bypass regardless of what BICAMERAL_PACK_REGISTRY_PATH points to.
    """
    legacy_path = tmp_path / 'legacy-registry.json'
    monkeypatch.delenv('BICAMERAL_USER_PACK_REGISTRY_PATH', raising=False)
    monkeypatch.setenv('BICAMERAL_PACK_REGISTRY_PATH', str(legacy_path))

    result = asyncio.run(packs.create_workflow_pack(_workflow_definition(pack_id='workflow-legacy-bypass')))

    assert result.get('error') == 'validation_error'
    assert 'BICAMERAL_USER_PACK_REGISTRY_PATH' in result.get('message', '')
    # No file should have been created at the legacy path.
    assert not legacy_path.exists()


@pytest.mark.parametrize(
    ('field', 'value', 'message_fragment'),
    [
        ('pack_id', 'Bad Pack Id', 'invalid pack_id'),
        ('id', 'Bad Pack Id', 'invalid id'),
        ('pack_id', 123, 'pack_id must be a string'),
        ('id', 123, 'id must be a string'),
    ],
)
def test_create_workflow_pack_rejects_malformed_ids_instead_of_normalizing(
    registry_path: Path,
    field: str,
    value: object,
    message_fragment: str,
):
    definition = _workflow_definition(pack_id='workflow-valid-create-id-check')
    definition[field] = value

    result = asyncio.run(packs.create_workflow_pack(definition))
    assert result.get('error') == 'validation_error'
    assert message_fragment in result.get('message', '')

    normalized_alias = 'bad-pack-id'
    assert asyncio.run(packs.describe_pack(normalized_alias)).get('error') == 'not_found'

    if registry_path.exists():
        persisted = json.loads(registry_path.read_text(encoding='utf-8'))
        assert persisted['packs'] == []


def test_create_workflow_pack_rejects_invalid_scope_literal_type(registry_path: Path):
    definition = _workflow_definition(pack_id='workflow-invalid-scope')
    definition['scope'] = 'type'

    result = asyncio.run(packs.create_workflow_pack(definition))
    assert result.get('error') == 'validation_error'
    assert "invalid scope 'type'" in result.get('message', '')


def test_create_workflow_pack_rejects_overly_nested_definitions(registry_path: Path):
    definition = _workflow_definition(pack_id='workflow-too-deep')
    definition['definition'] = {
        'trigger': 'quarter_end',
        'steps': [{'step': 'ingest', 'action': 'collect facts'}],
        'instructions': 'Keep it sane.',
        'nested': _nested_mapping(12),
    }

    result = asyncio.run(packs.create_workflow_pack(definition))
    assert result.get('error') == 'validation_error'
    assert 'max nesting depth' in result.get('message', '')


def test_concurrent_creates_preserve_both_packs(registry_path: Path):
    def create(pack_id: str) -> dict[str, object]:
        return PackRegistryService(registry_path).create_pack(_workflow_definition(pack_id=pack_id))

    pack_ids = ['workflow-concurrent-a', 'workflow-concurrent-b']
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(create, pack_ids))

    assert {result['id'] for result in results} == set(pack_ids)

    service = PackRegistryService(registry_path)
    persisted = {pack['id'] for pack in service.list_packs(filter={'scope': 'workflow'})}
    assert set(pack_ids) <= persisted

    user_payload = json.loads(registry_path.read_text(encoding='utf-8'))
    user_ids = {pack['id'] for pack in user_payload['packs']}
    assert user_ids == set(pack_ids)


def test_list_packs_invalid_filter_returns_validation_error(registry_path: Path):
    result = asyncio.run(packs.list_packs({'scope': 'not-a-scope'}))
    assert result.get('error') == 'validation_error'
    assert 'scope' in result.get('message', '')


def test_list_packs_and_describe_fail_closed_on_invalid_persisted_pack_registry_timestamps(
    registry_path: Path,
):
    registry_path.write_text(
        json.dumps(
            {
                'schema_version': '1.0.0',
                'meta': {
                    'created_at': '2026-03-11T12:34:56Z',
                    'updated_at': '2026-03-11T12:34:56Z',
                },
                'packs': [
                    {
                        'id': 'workflow-invalid-persisted-timestamp',
                        'scope': 'workflow',
                        'intent': 'verifier',
                        'description': 'Workflow with broken metadata',
                        'consumer': 'planner',
                        'version': '1.0.0',
                        'predicates': ['risk'],
                        'created_at': 'not-a-date',
                        'last_updated': '2026-03-11T12:34:56Z',
                        'definition': {'steps': [{'step': 'review', 'action': 'inspect risk facts'}]},
                    }
                ],
            },
            indent=2,
        ),
        encoding='utf-8',
    )

    listed = asyncio.run(packs.list_packs())
    assert listed.get('error') == 'operational_error'
    assert 'created_at' in listed.get('message', '')

    described = asyncio.run(packs.describe_pack('workflow-invalid-persisted-timestamp'))
    assert described.get('error') == 'operational_error'
    assert 'created_at' in described.get('message', '')



def test_list_packs_registry_load_failure_surfaces_operational_error(registry_path: Path):
    registry_path.write_text('{"packs": [', encoding='utf-8')

    result = asyncio.run(packs.list_packs())
    assert result.get('error') == 'operational_error'
    assert 'failed to load registry file' in result.get('message', '')


def test_create_workflow_pack_requires_steps(registry_path: Path):
    definition = _workflow_definition(pack_id='workflow-missing-steps')
    definition['definition'] = {
        'trigger': 'quarter_end',
        'instructions': 'Collect the latest financial facts before summarizing.',
    }

    result = asyncio.run(packs.create_workflow_pack(definition))
    assert result.get('error') == 'validation_error'
    assert 'definition.steps' in result.get('message', '')
