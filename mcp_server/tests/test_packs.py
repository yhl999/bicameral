from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mcp_server.src.routers import packs as packs_router
from mcp_server.src.services.pack_registry import PackRegistryService


@pytest.fixture
def registry_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / 'packs.registry.json'
    monkeypatch.setenv('BICAMERAL_PACK_REGISTRY_PATH', str(path))
    return path


def _seed_registry(path: Path) -> None:
    payload = {
        'schema_version': 1,
        'packs': [
            {
                'pack_id': 'ctx-ops',
                'type': 'context',
                'scope': 'private',
                'intent': 'ops',
                'consumer': 'main',
                'version': '1.0.0',
                'checksum': 'seed-ctx',
                'created_at': '2026-03-10T10:00:00Z',
                'updated_at': '2026-03-10T10:00:00Z',
                'definition': {
                    'pack_id': 'ctx-ops',
                    'type': 'context',
                    'scope': 'private',
                    'intent': 'ops',
                    'consumer': 'main',
                    'version': '1.0.0',
                    'items': ['incident runbook', 'deployment checklist'],
                },
            },
            {
                'pack_id': 'wf-oncall',
                'type': 'workflow',
                'scope': 'private',
                'intent': 'oncall',
                'consumer': 'main',
                'version': '1.0.0',
                'checksum': 'seed-wf',
                'created_at': '2026-03-10T10:00:00Z',
                'updated_at': '2026-03-10T10:00:00Z',
                'definition': {
                    'pack_id': 'wf-oncall',
                    'type': 'workflow',
                    'scope': 'private',
                    'intent': 'oncall',
                    'consumer': 'main',
                    'version': '1.0.0',
                    'steps': ['ack page', 'triage', 'publish update'],
                },
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def test_list_and_filter_packs(registry_path: Path):
    _seed_registry(registry_path)

    all_packs = asyncio.run(packs_router.list_packs())
    assert 'error' not in all_packs
    assert len(all_packs['packs']) == 2

    filtered = asyncio.run(packs_router.list_packs({'type': 'workflow', 'intent': 'oncall'}))
    assert 'error' not in filtered
    assert len(filtered['packs']) == 1
    assert filtered['packs'][0]['pack_id'] == 'wf-oncall'


def test_get_context_pack_with_task_filter(registry_path: Path):
    _seed_registry(registry_path)

    result = asyncio.run(packs_router.get_context_pack('ctx-ops', task='incident response'))
    assert 'error' not in result
    assert result['pack_id'] == 'ctx-ops'
    assert result['count'] == 1
    assert result['items'][0] == 'incident runbook'


def test_describe_pack_not_found(registry_path: Path):
    _seed_registry(registry_path)

    result = asyncio.run(packs_router.describe_pack('missing-pack'))
    assert 'error' in result


def test_create_workflow_pack_validation_and_duplicate(registry_path: Path):
    service = PackRegistryService(registry_path)

    bad_semver = asyncio.run(
        packs_router.create_workflow_pack(
            {
                'pack_id': 'wf-bad',
                'type': 'workflow',
                'scope': 'private',
                'intent': 'ops',
                'consumer': 'main',
                'version': 'v1',
                'steps': ['a'],
            }
        )
    )
    assert 'error' in bad_semver

    created = asyncio.run(
        packs_router.create_workflow_pack(
            {
                'pack_id': 'wf-good',
                'type': 'workflow',
                'scope': 'private',
                'intent': 'ops',
                'consumer': 'main',
                'version': '1.2.3',
                'steps': ['step-1', 'step-2'],
            }
        )
    )
    assert 'error' not in created
    assert created['pack']['pack_id'] == 'wf-good'

    duplicate = asyncio.run(
        packs_router.create_workflow_pack(
            {
                'pack_id': 'wf-good',
                'type': 'workflow',
                'scope': 'private',
                'intent': 'ops',
                'consumer': 'main',
                'version': '1.2.4',
                'steps': ['step-1'],
            }
        )
    )
    assert 'error' in duplicate

    assert service.get_pack('wf-good') is not None
