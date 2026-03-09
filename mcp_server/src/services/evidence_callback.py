from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..models.typed_memory import EvidenceRef


class EvidenceCallback(Protocol):
    name: str

    def supports(self, ref: EvidenceRef) -> bool: ...

    async def resolve(self, ref: EvidenceRef) -> dict[str, Any]: ...


@dataclass
class PassThroughEvidenceCallback:
    """Deterministic evidence resolver for non-QMD sources.

    This callback intentionally treats an ``EvidenceRef`` as a first-class,
    already-grounded pointer. It does not attempt to dereference external
    systems; it normalizes the reference into the surfaced evidence bucket.
    """

    name: str = 'passthrough'

    def supports(self, ref: EvidenceRef) -> bool:
        return True

    async def resolve(self, ref: EvidenceRef) -> dict[str, Any]:
        return {
            'canonical_uri': ref.canonical_uri,
            'kind': ref.kind,
            'source_system': ref.source_system,
            'locator': ref.locator,
            'title': ref.title,
            'snippet': ref.snippet,
            'observed_at': ref.observed_at,
            'retrieved_at': ref.retrieved_at,
            'hash': ref.hash,
            'resolver': self.name,
            'resolution_source': 'reference',
            'status': 'resolved',
        }


@dataclass
class QMDEvidenceCallback:
    """Current evidence adapter for QMD-backed chunk references.

    The adapter is intentionally conservative: it can optionally enrich QMD
    chunk refs through ``qmd query --json`` when the CLI is available, but it
    still resolves the evidence ref deterministically from the pointer itself.
    That keeps the retrieval contract pluggable and fail-closed rather than
    inventing proof when the external adapter is unavailable.
    """

    command: str = 'qmd query --json'
    timeout_seconds: int = 30
    name: str = 'qmd'

    def supports(self, ref: EvidenceRef) -> bool:
        return ref.kind == 'qmd_chunk'

    def _query_text(self, ref: EvidenceRef) -> str:
        locator = ref.locator or {}
        parts = [
            str(locator.get('document_id') or '').strip(),
            str(locator.get('chunk_id') or '').strip(),
            str(ref.title or '').strip(),
            str(ref.snippet or '').strip(),
        ]
        query = ' '.join(part for part in parts if part)
        return query.strip() or str(ref.canonical_uri or '')

    def _run_qmd(self, query: str) -> dict[str, Any] | None:
        if not query:
            return None

        cmd_parts = shlex.split(self.command)
        if not cmd_parts:
            return None
        if shutil.which(cmd_parts[0]) is None:
            return None

        try:
            result = subprocess.run(
                [*cmd_parts, '--', query],
                shell=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return None

    async def resolve(self, ref: EvidenceRef) -> dict[str, Any]:
        payload = {
            'canonical_uri': ref.canonical_uri,
            'kind': ref.kind,
            'source_system': ref.source_system,
            'locator': ref.locator,
            'title': ref.title,
            'snippet': ref.snippet,
            'observed_at': ref.observed_at,
            'retrieved_at': ref.retrieved_at,
            'hash': ref.hash,
            'resolver': self.name,
            'resolution_source': 'reference',
            'status': 'resolved',
        }

        enriched = self._run_qmd(self._query_text(ref))
        if enriched is None:
            return payload

        payload['resolution_source'] = 'qmd_query'
        payload['qmd'] = enriched

        if payload.get('snippet') in (None, ''):
            snippet = enriched.get('snippet') or enriched.get('text')
            if isinstance(snippet, str) and snippet.strip():
                payload['snippet'] = snippet.strip()

        title = enriched.get('title') if isinstance(enriched, dict) else None
        if payload.get('title') in (None, '') and isinstance(title, str) and title.strip():
            payload['title'] = title.strip()

        return payload


@dataclass
class EvidenceCallbackRegistry:
    callbacks: list[EvidenceCallback] = field(
        default_factory=lambda: [QMDEvidenceCallback(), PassThroughEvidenceCallback()]
    )

    def callback_for(self, ref: EvidenceRef) -> EvidenceCallback:
        for callback in self.callbacks:
            if callback.supports(ref):
                return callback
        return PassThroughEvidenceCallback()

    async def resolve_many(
        self,
        refs: list[EvidenceRef],
        *,
        object_ids_by_uri: dict[str, list[str]] | None = None,
        max_items: int | None = None,
    ) -> list[dict[str, Any]]:
        unique_refs: dict[str, EvidenceRef] = {}
        for ref in refs:
            uri = str(ref.canonical_uri or '').strip()
            if not uri or uri in unique_refs:
                continue
            unique_refs[uri] = ref

        resolved_items: list[dict[str, Any]] = []
        for canonical_uri, ref in unique_refs.items():
            callback = self.callback_for(ref)
            payload = await callback.resolve(ref)
            payload['object_ids'] = sorted(set(object_ids_by_uri.get(canonical_uri, []))) if object_ids_by_uri else []
            resolved_items.append(payload)
            if max_items is not None and len(resolved_items) >= max_items:
                break
        return resolved_items
