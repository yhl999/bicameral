"""JSON-backed pack registry service."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

DEFAULT_PACK_REGISTRY_PATH = Path(__file__).resolve().parents[3] / 'state' / 'packs.registry.json'


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _checksum(definition: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(definition).encode('utf-8')).hexdigest()


class PackRegistryService:
    def __init__(self, registry_path: str | Path | None = None):
        env_path = os.getenv('BICAMERAL_PACK_REGISTRY_PATH', '').strip()
        self.path = Path(registry_path or env_path or DEFAULT_PACK_REGISTRY_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {'schema_version': 1, 'packs': []}
        try:
            payload = json.loads(self.path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            return {'schema_version': 1, 'packs': []}
        if not isinstance(payload, dict):
            return {'schema_version': 1, 'packs': []}
        packs = payload.get('packs')
        if not isinstance(packs, list):
            payload['packs'] = []
        payload.setdefault('schema_version', 1)
        return payload

    def persist(self, payload: dict[str, Any]) -> None:
        serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        with NamedTemporaryFile('w', dir=str(self.path.parent), delete=False, encoding='utf-8') as tmp:
            tmp.write(serialized)
            tmp_path = Path(tmp.name)
        tmp_path.replace(self.path)

    def list_packs(self) -> list[dict[str, Any]]:
        payload = self.load()
        packs = payload.get('packs') or []
        if not isinstance(packs, list):
            return []
        return [pack for pack in packs if isinstance(pack, dict)]

    def get_pack(self, pack_id: str) -> dict[str, Any] | None:
        for pack in self.list_packs():
            if str(pack.get('pack_id')) == pack_id:
                return pack
        return None

    def create_pack(self, definition: dict[str, Any]) -> dict[str, Any]:
        payload = self.load()
        packs = payload.get('packs') or []

        pack_id = str(definition.get('pack_id') or '').strip()
        if not pack_id:
            raise ValueError('definition.pack_id is required')

        if any(str(pack.get('pack_id')) == pack_id for pack in packs):
            raise ValueError(f'pack_id already exists: {pack_id}')

        checksum = _checksum(definition)
        if any(str(pack.get('checksum')) == checksum for pack in packs):
            raise ValueError('identical pack definition already exists (checksum match)')

        now = _now_iso()
        entry = {
            'pack_id': pack_id,
            'type': str(definition.get('type') or 'workflow'),
            'scope': str(definition.get('scope') or 'private'),
            'intent': str(definition.get('intent') or ''),
            'consumer': str(definition.get('consumer') or 'main'),
            'version': str(definition.get('version') or '0.1.0'),
            'checksum': checksum,
            'created_at': now,
            'updated_at': now,
            'definition': definition,
        }

        packs.append(entry)
        payload['packs'] = packs
        self.persist(payload)
        return entry
