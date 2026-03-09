from __future__ import annotations

import asyncio
import json
import logging
import shlex
import shutil
from dataclasses import dataclass, field
from typing import Any, Protocol

try:
    from ..models.typed_memory import EvidenceRef
except ImportError:  # pragma: no cover - top-level import fallback
    from models.typed_memory import EvidenceRef

logger = logging.getLogger(__name__)


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
    max_query_chars: int = 512
    max_stdout_bytes: int = 256_000
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
        normalized = ' '.join((query.strip() or str(ref.canonical_uri or '')).split())
        if len(normalized) > self.max_query_chars:
            return normalized[: self.max_query_chars]
        return normalized

    async def _read_stdout_capped(
        self,
        reader: asyncio.StreamReader | None,
        *,
        max_bytes: int,
        chunk_size: int = 65536,
    ) -> bytes | None:
        if reader is None:
            return b''

        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await reader.read(chunk_size)
            if not chunk:
                return b''.join(chunks)
            total += len(chunk)
            if total > max_bytes:
                return None
            chunks.append(chunk)

    async def _run_qmd(self, query: str) -> dict[str, Any] | None:
        if not query:
            return None

        cmd_parts = shlex.split(self.command)
        if not cmd_parts:
            return None
        if shutil.which(cmd_parts[0]) is None:
            return None

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd_parts,
                '--',
                query,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except (FileNotFoundError, OSError):
            return None

        try:
            stdout = await asyncio.wait_for(
                self._read_stdout_capped(process.stdout, max_bytes=self.max_stdout_bytes),
                timeout=float(self.timeout_seconds),
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return None

        if stdout is None:
            process.kill()
            await process.wait()
            return None

        try:
            returncode = await asyncio.wait_for(process.wait(), timeout=float(self.timeout_seconds))
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return None

        if returncode != 0:
            return None

        if not stdout:
            return None

        try:
            return json.loads(stdout.decode('utf-8', errors='replace'))
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

        enriched = await self._run_qmd(self._query_text(ref))
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
    max_concurrency: int = 8

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

        items = list(unique_refs.items())
        if max_items is not None:
            capped_max_items = max(0, int(max_items))
            items = items[:capped_max_items]

        semaphore = asyncio.Semaphore(max(1, int(self.max_concurrency)))
        fallback = PassThroughEvidenceCallback()

        async def _resolve_one(canonical_uri: str, ref: EvidenceRef) -> dict[str, Any]:
            callback = self.callback_for(ref)
            async with semaphore:
                try:
                    payload = await callback.resolve(ref)
                except Exception as exc:  # pragma: no cover - defensive fallback
                    logger.warning(
                        'Evidence resolver failed (resolver=%s, uri=%s): %s',
                        callback.name,
                        canonical_uri,
                        type(exc).__name__,
                    )
                    payload = await fallback.resolve(ref)
                    payload['resolver'] = callback.name
                    payload['status'] = 'resolution_failed'

            payload['object_ids'] = (
                sorted(set(object_ids_by_uri.get(canonical_uri, []))) if object_ids_by_uri else []
            )
            return payload

        tasks = [asyncio.create_task(_resolve_one(uri, ref)) for uri, ref in items]
        if not tasks:
            return []
        return await asyncio.gather(*tasks)
