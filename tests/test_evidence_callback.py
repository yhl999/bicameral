import asyncio

from mcp_server.src.models.typed_memory import EvidenceRef
from mcp_server.src.services.evidence_callback import EvidenceCallbackRegistry, QMDEvidenceCallback


class _TrackingCallback:
    name = 'tracking'

    def __init__(self):
        self.active = 0
        self.max_active = 0

    def supports(self, ref):
        return True

    async def resolve(self, ref):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return {
            'canonical_uri': ref.canonical_uri,
            'resolver': self.name,
            'status': 'resolved',
        }


class _ExplodingCallback:
    name = 'boom'

    def supports(self, ref):
        return True

    async def resolve(self, ref):
        raise RuntimeError('boom')


def _run(coro):
    return asyncio.run(coro)


def _refs(count):
    refs = []
    for idx in range(count):
        refs.append(
            EvidenceRef(
                kind='message',
                source_system='telegram',
                locator={'system': 'telegram', 'conversation_id': 'c1', 'message_id': f'm{idx}'},
                title=f'title {idx}',
                snippet=f'snippet {idx}',
            )
        )
    return refs


def test_resolve_many_runs_callbacks_concurrently_with_bound():
    callback = _TrackingCallback()
    registry = EvidenceCallbackRegistry(callbacks=[callback], max_concurrency=2)

    result = _run(registry.resolve_many(_refs(4), max_items=4))

    assert len(result) == 4
    assert callback.max_active == 2


def test_resolve_many_falls_back_to_passthrough_on_callback_failure():
    refs = _refs(1)
    registry = EvidenceCallbackRegistry(callbacks=[_ExplodingCallback()], max_concurrency=1)

    result = _run(registry.resolve_many(refs, object_ids_by_uri={refs[0].canonical_uri: ['obj_1']}))

    assert result == [
        {
            'canonical_uri': refs[0].canonical_uri,
            'kind': 'message',
            'source_system': 'telegram',
            'locator': {'system': 'telegram', 'conversation_id': 'c1', 'message_id': 'm0'},
            'title': 'title 0',
            'snippet': 'snippet 0',
            'observed_at': None,
            'retrieved_at': None,
            'hash': None,
            'resolver': 'boom',
            'resolution_source': 'reference',
            'status': 'resolution_failed',
            'object_ids': ['obj_1'],
        }
    ]


def test_qmd_query_text_and_stdout_are_bounded():
    callback = QMDEvidenceCallback(
        command='definitely_missing_qmd query --json',
        max_query_chars=16,
        max_stdout_bytes=10,
    )
    ref = EvidenceRef(
        kind='qmd_chunk',
        source_system='qmd',
        locator={'collection': 'docs', 'document_id': 'doc-1', 'chunk_id': 'chunk-1'},
        title='A very long title',
        snippet='This snippet is also very long',
    )

    query_text = callback._query_text(ref)

    assert len(query_text) == 16
    assert _run(callback._run_qmd('x')) is None
