"""Packs router — list/get/describe/create backed by JSON registry."""

from __future__ import annotations

import logging
import re
from typing import Any

try:
    from ..services.pack_registry import PackRegistryService
except ImportError:  # pragma: no cover - top-level import fallback
    from services.pack_registry import PackRegistryService  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

_SEMVER_RE = re.compile(r'^\d+\.\d+\.\d+$')


def _matches_filter(pack: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for key in ('scope', 'intent', 'consumer', 'type'):
        expected = filters.get(key)
        if expected is None:
            continue
        if str(pack.get(key)) != str(expected):
            return False
    return True


def _filter_context_items(items: list[Any], task: str | None) -> list[Any]:
    if not task:
        return items
    tokens = {token.lower() for token in str(task).split() if token.strip()}
    if not tokens:
        return items

    filtered: list[Any] = []
    for item in items:
        blob = str(item).lower()
        if any(token in blob for token in tokens):
            filtered.append(item)
    return filtered


async def list_packs(filter: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        service = PackRegistryService()
        packs = [pack for pack in service.list_packs() if _matches_filter(pack, filter)]
        return {
            'message': f'Found {len(packs)} pack(s)',
            'packs': [
                {
                    'pack_id': pack['pack_id'],
                    'type': pack.get('type'),
                    'scope': pack.get('scope'),
                    'intent': pack.get('intent'),
                    'consumer': pack.get('consumer'),
                    'version': pack.get('version'),
                    'checksum': pack.get('checksum'),
                    'updated_at': pack.get('updated_at'),
                }
                for pack in packs
            ],
        }
    except Exception as e:
        logger.exception('list_packs failed')
        return {'error': f'list_packs failed: {e}'}


async def get_context_pack(pack_id: str, task: str | None = None) -> dict[str, Any]:
    try:
        service = PackRegistryService()
        pack = service.get_pack(pack_id)
        if pack is None:
            return {'error': f'Pack not found: {pack_id}'}
        if str(pack.get('type')) != 'context':
            return {'error': f'Pack {pack_id} is not a context pack'}

        definition = pack.get('definition') or {}
        items = definition.get('items') or []
        if not isinstance(items, list):
            items = [items]
        materialized = _filter_context_items(items, task)
        return {
            'message': f'Context pack {pack_id} loaded',
            'pack_id': pack_id,
            'items': materialized,
            'count': len(materialized),
        }
    except Exception as e:
        logger.exception('get_context_pack failed')
        return {'error': f'get_context_pack failed: {e}'}


async def get_workflow_pack(pack_id: str, task: str | None = None) -> dict[str, Any]:
    try:
        service = PackRegistryService()
        pack = service.get_pack(pack_id)
        if pack is None:
            return {'error': f'Pack not found: {pack_id}'}
        if str(pack.get('type')) != 'workflow':
            return {'error': f'Pack {pack_id} is not a workflow pack'}

        definition = pack.get('definition') or {}
        steps = definition.get('steps') or []
        if not isinstance(steps, list):
            steps = [steps]
        # v1: task argument is accepted for API parity but does not alter steps.
        _ = task
        return {
            'message': f'Workflow pack {pack_id} loaded',
            'pack_id': pack_id,
            'steps': steps,
            'count': len(steps),
        }
    except Exception as e:
        logger.exception('get_workflow_pack failed')
        return {'error': f'get_workflow_pack failed: {e}'}


async def describe_pack(pack_id: str) -> dict[str, Any]:
    try:
        service = PackRegistryService()
        pack = service.get_pack(pack_id)
        if pack is None:
            return {'error': f'Pack not found: {pack_id}'}
        return {
            'message': f'Pack {pack_id} definition',
            'pack': pack,
        }
    except Exception as e:
        logger.exception('describe_pack failed')
        return {'error': f'describe_pack failed: {e}'}


async def create_workflow_pack(definition: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(definition, dict):
        return {'error': 'definition must be an object/dict'}

    if str(definition.get('type') or 'workflow') != 'workflow':
        return {'error': 'create_workflow_pack only accepts type="workflow"'}

    version = str(definition.get('version') or '').strip()
    if not _SEMVER_RE.match(version):
        return {'error': 'definition.version must be semver (e.g., 1.2.3)'}

    steps = definition.get('steps')
    if not isinstance(steps, list) or not steps:
        return {'error': 'definition.steps must be a non-empty list'}

    try:
        service = PackRegistryService()
        entry = service.create_pack(definition)
        return {
            'message': f'Workflow pack created: {entry["pack_id"]}',
            'pack': {
                'pack_id': entry['pack_id'],
                'type': entry['type'],
                'scope': entry['scope'],
                'intent': entry['intent'],
                'consumer': entry['consumer'],
                'version': entry['version'],
                'checksum': entry['checksum'],
                'updated_at': entry['updated_at'],
            },
        }
    except Exception as e:
        logger.exception('create_workflow_pack failed')
        return {'error': f'create_workflow_pack failed: {e}'}


def register_tools(mcp: Any) -> None:
    mcp.tool()(list_packs)
    mcp.tool()(get_context_pack)
    mcp.tool()(get_workflow_pack)
    mcp.tool()(describe_pack)
    mcp.tool()(create_workflow_pack)
