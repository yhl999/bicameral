"""Focused contract / regression tests for the non-blocking cleanup pass.

Covers the four cleanup buckets from the api-gpt/high integrated code review:

  A. Response envelope consistency — search_episodes / search_procedures include
     status='ok' on success; TOOL_CONTRACTS output descriptions reflect this.
  B. Candidate-store / ledger post-promotion skew mitigation — store update
     failure is logged and the readback skew path is detected.
  C. get_history() semantics clarity — TOOL_CONTRACTS description and docstring
     clearly state that only current-root lineages are walked.
  D. search_procedures fetch window quality — TOOL_CONTRACTS includes a
     fetch_window_note; fetch_limit formula is correct for edge cases.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_mcp() -> MagicMock:
    tools: dict[str, Any] = {}
    mock = MagicMock()

    def _tool_decorator():
        def decorator(fn):
            tools[fn.__name__] = fn
            return fn
        mock.tool.return_value = decorator
        return decorator

    mock.tool.side_effect = lambda: _tool_decorator()
    mock._tools = tools
    return mock


def _run(coro):
    return asyncio.run(coro)


def _make_in_memory_ledger():
    from mcp_server.src.services.change_ledger import ChangeLedger
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    return ChangeLedger(conn)


def _create_procedure_in_ledger(ledger, *, name='Test proc', trigger='how to test', promote=True):
    from mcp_server.src.services.procedure_service import ProcedureService
    svc = ProcedureService(ledger)
    return svc.create_procedure(
        actor_id='test',
        name=name,
        trigger=trigger,
        steps=['Step 1'],
        expected_outcome='Done',
        evidence_refs=[{'uri': 'mem://test/1', 'kind': 'message', 'content_summary': 'test'}],
        promote=promote,
    )


# ---------------------------------------------------------------------------
# Bucket A — Response envelope consistency
# ---------------------------------------------------------------------------


class TestSearchEnvelopeHasStatusOk:
    """search_episodes and search_procedures must include status='ok' on success."""

    @pytest.mark.anyio
    async def test_search_episodes_success_includes_status_ok(self, tmp_path):
        """search_episodes success response must include {'status': 'ok'}."""
        from mcp_server.src.routers.episodes_procedures import register_tools
        from mcp_server.src.services.change_ledger import ChangeLedger

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)

        import mcp_server.src.services.change_ledger as cl_module
        original = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path
        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['search_episodes']
            result = await fn(query='anything')
        finally:
            cl_module.DB_PATH_DEFAULT = original

        assert result.get('status') == 'ok', (
            f"search_episodes success response missing status='ok': {result}"
        )
        assert 'episodes' in result
        assert 'limit' in result
        assert 'total' in result

    @pytest.mark.anyio
    async def test_search_procedures_success_includes_status_ok(self, tmp_path):
        """search_procedures success response must include {'status': 'ok'}."""
        from mcp_server.src.routers.episodes_procedures import register_tools

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)

        import mcp_server.src.services.change_ledger as cl_module
        original = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path
        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['search_procedures']
            result = await fn(query='deploy')
        finally:
            cl_module.DB_PATH_DEFAULT = original

        assert result.get('status') == 'ok', (
            f"search_procedures success response missing status='ok': {result}"
        )
        assert 'procedures' in result
        assert 'total' in result

    @pytest.mark.anyio
    async def test_search_episodes_with_results_includes_status_ok(self, tmp_path):
        """search_episodes with actual results must also have status='ok'."""
        from mcp_server.src.models.typed_memory import Episode
        from mcp_server.src.routers.episodes_procedures import register_tools
        from mcp_server.src.services.change_ledger import ChangeLedger

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = ChangeLedger(db_path)
        ep = Episode.model_validate({
            'object_id': 'ep-env-001', 'root_id': 'ep-env-001',
            'title': 'Envelope test episode', 'summary': 'Tests envelope consistency.',
            'policy_scope': 'private', 'visibility_scope': 'private',
            'evidence_refs': [{'kind': 'event_log', 'source_system': 'test',
                               'locator': {'system': 'test', 'stream': 's', 'event_id': 'e'}}],
        })
        ledger.append_event('assert', actor_id='test', reason='test', payload=ep)

        import mcp_server.src.services.change_ledger as cl_module
        original = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path
        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['search_episodes']
            result = await fn(query='envelope test')
        finally:
            cl_module.DB_PATH_DEFAULT = original

        assert result.get('status') == 'ok', (
            f"search_episodes with results missing status='ok': {result}"
        )
        assert isinstance(result.get('episodes'), list)

    @pytest.mark.anyio
    async def test_search_procedures_with_results_includes_status_ok(self, tmp_path):
        """search_procedures with actual results must also have status='ok'."""
        from mcp_server.src.routers.episodes_procedures import register_tools
        from mcp_server.src.services.change_ledger import ChangeLedger

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = ChangeLedger(db_path)
        _create_procedure_in_ledger(ledger, name='Envelope proc', trigger='how to test envelope', promote=True)

        import mcp_server.src.services.change_ledger as cl_module
        original = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path
        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['search_procedures']
            result = await fn(query='test envelope')
        finally:
            cl_module.DB_PATH_DEFAULT = original

        assert result.get('status') == 'ok', (
            f"search_procedures with results missing status='ok': {result}"
        )
        assert isinstance(result.get('procedures'), list)


class TestToolContractsEnvelopeDescriptions:
    """TOOL_CONTRACTS output descriptions must reflect the status='ok' envelope."""

    def test_search_episodes_contract_output_describes_status_ok(self):
        """search_episodes TOOL_CONTRACTS output must mention status='ok'."""
        from mcp_server.src.routers.episodes_procedures import TOOL_CONTRACTS

        contract = next(c for c in TOOL_CONTRACTS if c['name'] == 'search_episodes')
        output = contract['schema']['output']
        assert '"status": "ok"' in output, (
            f"search_episodes TOOL_CONTRACTS output does not describe status='ok': {output!r}"
        )
        assert 'episodes' in output, f"search_episodes output must mention 'episodes': {output!r}"

    def test_search_procedures_contract_output_describes_status_ok(self):
        """search_procedures TOOL_CONTRACTS output must mention status='ok'."""
        from mcp_server.src.routers.episodes_procedures import TOOL_CONTRACTS

        contract = next(c for c in TOOL_CONTRACTS if c['name'] == 'search_procedures')
        output = contract['schema']['output']
        assert '"status": "ok"' in output, (
            f"search_procedures TOOL_CONTRACTS output does not describe status='ok': {output!r}"
        )
        assert 'procedures' in output, f"search_procedures output must mention 'procedures': {output!r}"

    def test_search_procedures_contract_has_fetch_window_note(self):
        """search_procedures TOOL_CONTRACTS must have a fetch_window_note documenting the tradeoff."""
        from mcp_server.src.routers.episodes_procedures import TOOL_CONTRACTS

        contract = next(c for c in TOOL_CONTRACTS if c['name'] == 'search_procedures')
        note = contract['schema'].get('fetch_window_note', '')
        assert note, "search_procedures TOOL_CONTRACTS missing fetch_window_note"
        assert 'has_more' in note or 'pagination' in note.lower(), (
            f"fetch_window_note should mention pagination or has_more: {note!r}"
        )

    def test_list_candidates_contract_output_describes_full_envelope(self):
        """list_candidates TOOL_CONTRACTS output must describe the full {'status':'ok','candidates':...} envelope."""
        from mcp_server.src.routers.candidates import TOOL_CONTRACTS

        contract = next(c for c in TOOL_CONTRACTS if c['name'] == 'list_candidates')
        output = contract['schema']['output']
        assert '"candidates": list[Candidate]' in output, (
            f"list_candidates output does not mention 'candidates': list[Candidate]: {output!r}"
        )
        assert '"status": "ok"' in output, (
            f"list_candidates output does not mention status='ok': {output!r}"
        )


# ---------------------------------------------------------------------------
# Bucket B — Candidate-store / ledger post-promotion skew mitigation
# ---------------------------------------------------------------------------


class TestPromotionSkewMitigation:
    """promote_candidate logs warnings when store update fails or readback shows stale status."""

    def _make_candidate_store_and_ledger(self, tmp_path):
        """Return a (CandidateStore, ChangeLedger, candidate) triple for testing."""
        from mcp_server.src.services.candidate_store import CandidateStore
        from mcp_server.src.services.change_ledger import ChangeLedger, resolve_ledger_path

        db_path = tmp_path / 'change_ledger.db'
        cstore_path = tmp_path / 'candidates.db'

        ledger = ChangeLedger(db_path)
        store = CandidateStore(db_path=cstore_path)

        candidate = store.create_candidate(
            payload={'subject': 'user', 'predicate': 'pref', 'value': 'dark-mode', 'fact_type': 'preference'},
            conflict_with_fact_id=None,
            source='test',
            raw_hint={},
            metadata={'confidence': 0.9, 'scope': 'private', 'fact_type': 'preference'},
        )
        return store, ledger, candidate

    def test_promote_candidate_store_update_none_logs_warning(self, tmp_path, caplog):
        """When store.update_candidate_status returns None, a warning is logged."""
        import os
        from unittest.mock import MagicMock, patch

        import mcp_server.src.routers.candidates as cand_module
        from mcp_server.src.services.candidate_store import CandidateStore

        store, ledger, candidate = self._make_candidate_store_and_ledger(tmp_path)
        cand_id = candidate['candidate_id']

        orig_store = cand_module._candidate_store
        orig_ledger = cand_module._change_ledger
        cand_module._candidate_store = store
        cand_module._change_ledger = ledger

        os.environ['BICAMERAL_TRUSTED_ACTOR_IDS'] = 'test-reviewer'

        mock_promotion = MagicMock()
        mock_promotion.object_id = 'fact-001'
        mock_promotion.root_id = 'root-001'
        mock_promotion.event_id = 'event-001'
        mock_promotion.event_ids = ['event-001']

        ledger.promote_candidate_fact = MagicMock(return_value=mock_promotion)
        ledger.materialize_object = MagicMock(return_value=None)

        import mcp_server.src.routers.memory as mem_mod
        mem_mod._materializer = MagicMock()
        mem_mod._materializer.materialize_typed_fact = MagicMock(return_value=(False, None))

        mock_ctx = MagicMock()
        mock_ctx.client_id = 'test-reviewer'

        try:
            # Patch update_candidate_status at the class level to return None
            # (simulates store failure after successful ledger promotion).
            with patch.object(
                CandidateStore, 'update_candidate_status', return_value=None
            ):
                with caplog.at_level(logging.WARNING, logger='mcp_server.src.routers.candidates'):
                    result = _run(cand_module.promote_candidate(
                        candidate_id=cand_id,
                        resolution='supersede',
                        ctx=mock_ctx,
                    ))
        finally:
            cand_module._candidate_store = orig_store
            cand_module._change_ledger = orig_ledger
            os.environ.pop('BICAMERAL_TRUSTED_ACTOR_IDS', None)

        # The call should succeed (not error) despite the store returning None
        assert result.get('status') == 'ok', (
            f"promote_candidate should still succeed even with store returning None: {result}"
        )
        # A warning should have been logged about the skew
        skew_warnings = [
            r for r in caplog.records
            if ('skew' in r.message.lower() or 'none' in r.message.lower())
            and r.levelno >= logging.WARNING
        ]
        assert skew_warnings, (
            "Expected a WARNING log about store returning None / skew state; got none. "
            f"Log records: {[r.message for r in caplog.records]}"
        )

    def test_promote_candidate_normal_path_succeeds_and_no_skew_warning(self, tmp_path, caplog):
        """Normal promotion (no store failure) must succeed with no skew warning."""
        import mcp_server.src.routers.candidates as cand_module
        import os

        store, ledger, candidate = self._make_candidate_store_and_ledger(tmp_path)
        cand_id = candidate['candidate_id']

        orig_store = cand_module._candidate_store
        orig_ledger = cand_module._change_ledger
        cand_module._candidate_store = store
        cand_module._change_ledger = ledger

        os.environ['BICAMERAL_TRUSTED_ACTOR_IDS'] = 'test-reviewer'

        try:
            from unittest.mock import MagicMock

            mock_promotion = MagicMock()
            mock_promotion.object_id = 'fact-002'
            mock_promotion.root_id = 'root-002'
            mock_promotion.event_id = 'event-002'
            mock_promotion.event_ids = ['event-002']

            ledger.promote_candidate_fact = MagicMock(return_value=mock_promotion)
            ledger.materialize_object = MagicMock(return_value=None)

            import mcp_server.src.routers.memory as mem_mod
            mem_mod._materializer = MagicMock()
            mem_mod._materializer.materialize_typed_fact = MagicMock(return_value=(False, None))

            mock_ctx = MagicMock()
            mock_ctx.client_id = 'test-reviewer'

            with caplog.at_level(logging.WARNING, logger='mcp_server.src.routers.candidates'):
                result = _run(cand_module.promote_candidate(
                    candidate_id=cand_id,
                    resolution='supersede',
                    ctx=mock_ctx,
                ))

        finally:
            cand_module._candidate_store = orig_store
            cand_module._change_ledger = orig_ledger
            os.environ.pop('BICAMERAL_TRUSTED_ACTOR_IDS', None)

        assert result.get('status') == 'ok', f"Normal promotion should succeed: {result}"
        # No skew warnings on the normal path
        skew_warnings = [r for r in caplog.records
                         if 'skew' in r.message.lower() and r.levelno >= logging.WARNING]
        assert not skew_warnings, (
            f"Normal path should not emit skew warnings: {[r.message for r in skew_warnings]}"
        )


# ---------------------------------------------------------------------------
# Bucket C — get_history() semantics clarity
# ---------------------------------------------------------------------------


class TestGetHistoryContractClarity:
    """get_history TOOL_CONTRACTS must make the 'current roots only' narrowing explicit."""

    def test_get_history_contract_description_mentions_current_roots(self):
        """TOOL_CONTRACTS description for get_history must mention current/active roots."""
        from mcp_server.src.routers.memory import TOOL_CONTRACTS

        contract = next(c for c in TOOL_CONTRACTS if c['name'] == 'get_history')
        description = contract.get('description', '')

        assert description, "get_history TOOL_CONTRACTS description is empty"
        # Must communicate the narrowing to current roots
        lower = description.lower()
        assert any(kw in lower for kw in ('current', 'active', 'non-superseded')), (
            f"get_history description must mention 'current', 'active', or 'non-superseded' "
            f"to clarify the narrowed semantics; got: {description!r}"
        )
        assert 'root' in lower, (
            f"get_history description must mention 'root' (lineage walking semantics): {description!r}"
        )

    def test_get_history_contract_schema_has_semantics_note(self):
        """get_history schema must include a semantics_note clarifying narrowed behavior."""
        from mcp_server.src.routers.memory import TOOL_CONTRACTS

        contract = next(c for c in TOOL_CONTRACTS if c['name'] == 'get_history')
        note = contract['schema'].get('semantics_note', '')
        assert note, "get_history schema missing semantics_note"
        lower = note.lower()
        assert 'phase-0' in lower or 'current' in lower, (
            f"semantics_note should mention Phase-0 or current-roots semantics: {note!r}"
        )
        assert 'roots_considered' in note, (
            f"semantics_note should mention roots_considered to help callers: {note!r}"
        )

    def test_get_history_contract_description_not_misleadingly_broad(self):
        """get_history description must NOT claim to return 'all' history for a subject."""
        from mcp_server.src.routers.memory import TOOL_CONTRACTS

        contract = next(c for c in TOOL_CONTRACTS if c['name'] == 'get_history')
        description = contract.get('description', '').lower()

        # These phrasings would be misleadingly broad
        misleading = ['full history', 'all history', 'complete history', 'entire history']
        for phrase in misleading:
            assert phrase not in description, (
                f"get_history description contains misleadingly broad phrase {phrase!r}: {description!r}"
            )

    def test_get_history_response_includes_roots_considered(self, tmp_path, monkeypatch):
        """get_history runtime response must include roots_considered (not just docs)."""
        from mcp_server.src.routers.memory import get_history

        db_path = tmp_path / 'change_ledger.db'
        monkeypatch.setenv('BICAMERAL_CHANGE_LEDGER_PATH', str(db_path))
        import mcp_server.src.routers.memory as mem_module
        mem_module._change_ledger = None

        result = _run(get_history(subject='test-subject'))

        assert result.get('status') == 'ok', f"get_history should return ok: {result}"
        assert 'roots_considered' in result, (
            f"get_history must always include roots_considered in response: {result}"
        )
        assert isinstance(result['roots_considered'], list)
        mem_module._change_ledger = None

    def test_get_history_docstring_clarifies_narrowed_scope(self):
        """get_history function docstring must explain the narrowed scope."""
        from mcp_server.src.routers.memory import get_history

        doc = (get_history.__doc__ or '').lower()
        assert doc, "get_history must have a docstring"
        assert 'current' in doc or 'non-superseded' in doc, (
            "get_history docstring must mention 'current' or 'non-superseded' facts"
        )
        assert 'roots_considered' in doc or 'roots' in doc, (
            "get_history docstring must mention roots_considered or roots"
        )


# ---------------------------------------------------------------------------
# Bucket D — search_procedures fetch window quality documentation
# ---------------------------------------------------------------------------


class TestSearchProceduresFetchWindow:
    """search_procedures fetch_window formula is correct and documented."""

    def test_fetch_window_note_in_tool_contracts(self):
        """TOOL_CONTRACTS for search_procedures must include a fetch_window_note."""
        from mcp_server.src.routers.episodes_procedures import TOOL_CONTRACTS

        contract = next(c for c in TOOL_CONTRACTS if c['name'] == 'search_procedures')
        note = contract['schema'].get('fetch_window_note', '')
        assert note, "search_procedures schema missing fetch_window_note"

    def test_fetch_window_note_mentions_tradeoff(self):
        """fetch_window_note must mention the tradeoff (offset/pagination limitation)."""
        from mcp_server.src.routers.episodes_procedures import TOOL_CONTRACTS

        contract = next(c for c in TOOL_CONTRACTS if c['name'] == 'search_procedures')
        note = contract['schema'].get('fetch_window_note', '').lower()
        assert 'tradeoff' in note or 'trade-off' in note or 'floor' in note, (
            f"fetch_window_note should document the flat-floor tradeoff: {note!r}"
        )

    @pytest.mark.anyio
    async def test_search_procedures_pagination_has_more_is_accurate(self, tmp_path):
        """search_procedures has_more accurately reflects whether more results exist."""
        from mcp_server.src.routers.episodes_procedures import register_tools
        from mcp_server.src.services.change_ledger import ChangeLedger

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = ChangeLedger(db_path)

        # Create 3 promoted procedures
        for i in range(3):
            _create_procedure_in_ledger(
                ledger,
                name=f'Pagination proc {i}',
                trigger=f'how to paginate test {i}',
                promote=True,
            )

        import mcp_server.src.services.change_ledger as cl_module
        original = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path
        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['search_procedures']

            # Fetch all with limit=10, offset=0 (should have_more=False)
            result_all = await fn(query='paginate test', limit=10, offset=0)
            # Fetch page 1 of size 1 (should have has_more=True since there are 3)
            result_p1 = await fn(query='paginate test', limit=1, offset=0)
            # Fetch page 2 of size 1
            result_p2 = await fn(query='paginate test', limit=1, offset=1)
        finally:
            cl_module.DB_PATH_DEFAULT = original

        assert result_all.get('status') == 'ok'
        assert result_p1.get('status') == 'ok'
        assert result_p2.get('status') == 'ok'

        total = result_all.get('total', 0)
        assert total >= 1, "Expected at least 1 procedure result"

        # has_more on the all-results page should be False
        assert result_all.get('has_more') is False, (
            f"has_more should be False when all results fit on one page: {result_all}"
        )

        # If there are >= 2 results, has_more on page 1 (size=1) should be True
        if total >= 2:
            assert result_p1.get('has_more') is True, (
                f"has_more should be True when more results remain: {result_p1}"
            )
            assert result_p1.get('next_offset') is not None

    @pytest.mark.anyio
    async def test_search_procedures_empty_result_has_status_ok_and_no_has_more(self, tmp_path):
        """Empty result must have status='ok', has_more=False, next_offset=None."""
        from mcp_server.src.routers.episodes_procedures import register_tools

        db_path = tmp_path / 'state' / 'change_ledger.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)

        import mcp_server.src.services.change_ledger as cl_module
        original = cl_module.DB_PATH_DEFAULT
        cl_module.DB_PATH_DEFAULT = db_path
        try:
            mock_mcp = _make_mock_mcp()
            register_tools(mock_mcp)
            fn = mock_mcp._tools['search_procedures']
            result = await fn(query='xyzzy-nonexistent-query-abc')
        finally:
            cl_module.DB_PATH_DEFAULT = original

        assert result.get('status') == 'ok'
        assert result.get('has_more') is False
        assert result.get('next_offset') is None
        assert result.get('procedures') == []
        assert result.get('total') == 0
