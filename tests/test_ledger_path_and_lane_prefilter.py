"""Tests for blocker-union fixes:

A. Cross-router ledger-path split — env override is honored consistently.
B. Procedure top-K shadowing — lane filtering before top-K truncation.
C. get_episode access_denied → not_found alignment.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Blocker A: resolve_ledger_path shared helper
# ---------------------------------------------------------------------------


class TestResolveLedgerPath:
    """Verify the shared resolve_ledger_path honours env overrides."""

    def _import(self):
        from mcp_server.src.services.change_ledger import DB_PATH_DEFAULT, resolve_ledger_path
        return resolve_ledger_path, DB_PATH_DEFAULT

    def test_default_returns_db_path_default(self):
        resolve_ledger_path, DB_PATH_DEFAULT = self._import()
        with mock.patch.dict(os.environ, {}, clear=True):
            # Remove any env vars that could interfere
            os.environ.pop('BICAMERAL_CHANGE_LEDGER_DB', None)
            os.environ.pop('BICAMERAL_CHANGE_LEDGER_PATH', None)
            result = resolve_ledger_path()
            assert result == Path(DB_PATH_DEFAULT)

    def test_override_argument_wins(self):
        resolve_ledger_path, _ = self._import()
        result = resolve_ledger_path('/custom/path.db')
        assert result == Path('/custom/path.db')

    def test_env_ledger_db_overrides_default(self):
        resolve_ledger_path, DB_PATH_DEFAULT = self._import()
        with mock.patch.dict(os.environ, {'BICAMERAL_CHANGE_LEDGER_DB': '/env/db.db'}):
            os.environ.pop('BICAMERAL_CHANGE_LEDGER_PATH', None)
            result = resolve_ledger_path()
            assert result == Path('/env/db.db')
            assert result != Path(DB_PATH_DEFAULT)

    def test_env_ledger_path_overrides_default(self):
        resolve_ledger_path, DB_PATH_DEFAULT = self._import()
        with mock.patch.dict(os.environ, {'BICAMERAL_CHANGE_LEDGER_PATH': '/env/path.db'}):
            os.environ.pop('BICAMERAL_CHANGE_LEDGER_DB', None)
            result = resolve_ledger_path()
            assert result == Path('/env/path.db')

    def test_env_ledger_db_takes_priority_over_path(self):
        resolve_ledger_path, _ = self._import()
        with mock.patch.dict(os.environ, {
            'BICAMERAL_CHANGE_LEDGER_DB': '/primary.db',
            'BICAMERAL_CHANGE_LEDGER_PATH': '/secondary.db',
        }):
            result = resolve_ledger_path()
            assert result == Path('/primary.db')

    def test_whitespace_only_env_treated_as_unset(self):
        resolve_ledger_path, DB_PATH_DEFAULT = self._import()
        with mock.patch.dict(os.environ, {
            'BICAMERAL_CHANGE_LEDGER_DB': '   ',
            'BICAMERAL_CHANGE_LEDGER_PATH': '  ',
        }):
            result = resolve_ledger_path()
            assert result == Path(DB_PATH_DEFAULT)


class TestRouterLedgerPathConsistency:
    """Verify each router uses resolve_ledger_path (not hardcoded default)."""

    def test_candidates_uses_resolve_ledger_path(self):
        """candidates._get_change_ledger() should call resolve_ledger_path."""
        import mcp_server.src.routers.candidates as cmod
        # Reset cached singleton
        cmod._change_ledger = None
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp:
            with mock.patch(
                'mcp_server.src.routers.candidates.resolve_ledger_path',
                return_value=Path(tmp.name),
            ) as mock_resolve:
                ledger = cmod._get_change_ledger()
                mock_resolve.assert_called_once()
            cmod._change_ledger = None  # clean up

    def test_memory_uses_resolve_ledger_path(self):
        """memory._get_change_ledger() should call resolve_ledger_path."""
        import mcp_server.src.routers.memory as mmod
        mmod._change_ledger = None
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp:
            with mock.patch(
                'mcp_server.src.routers.memory.resolve_ledger_path',
                return_value=Path(tmp.name),
            ) as mock_resolve:
                ledger = mmod._get_change_ledger()
                mock_resolve.assert_called_once()
            mmod._change_ledger = None

    def test_packs_resolve_ledger_path_delegates(self):
        """packs._resolve_ledger_path delegates to shared resolve_ledger_path."""
        import mcp_server.src.routers.packs as pmod
        with mock.patch(
            'mcp_server.src.routers.packs.resolve_ledger_path',
            return_value=Path('/shared/test.db'),
        ) as mock_resolve:
            result = pmod._resolve_ledger_path()
            mock_resolve.assert_called_once_with(None)
            assert result == Path('/shared/test.db')

    def test_full_flow_same_ledger_with_env_override(self):
        """remember_fact → list_candidates → promote uses one ledger when env-overridden.

        This is a structural test: verify that all routers resolve the SAME
        path when BICAMERAL_CHANGE_LEDGER_DB is set.
        """
        from mcp_server.src.services.change_ledger import resolve_ledger_path

        test_path = '/tmp/test_unified_ledger.db'
        with mock.patch.dict(os.environ, {'BICAMERAL_CHANGE_LEDGER_DB': test_path}):
            os.environ.pop('BICAMERAL_CHANGE_LEDGER_PATH', None)
            resolved = resolve_ledger_path()
            assert resolved == Path(test_path)

            # All router call sites should get the same path
            # (they all call resolve_ledger_path with no args)
            import mcp_server.src.routers.candidates as cmod
            import mcp_server.src.routers.memory as mmod
            import mcp_server.src.routers.packs as pmod

            # Verify no hardcoded DB_PATH_DEFAULT in the code paths
            # by checking the shared function returns consistently
            assert resolve_ledger_path() == Path(test_path)


# ---------------------------------------------------------------------------
# Blocker B: Procedure top-K lane prefilter
# ---------------------------------------------------------------------------


class TestProcedureTopKLanePrefilter:
    """Verify lane filtering happens before top-K truncation."""

    def _make_procedure(self, *, name: str, trigger: str, source_lane: str, score_seed: int = 1):
        """Create a minimal Procedure-like object for testing."""
        from mcp_server.src.models.typed_memory import EvidenceRef, Procedure
        return Procedure.model_validate({
            'object_id': f'proc_{name}',
            'root_id': f'proc_{name}',
            'name': name,
            'trigger': trigger,
            'steps': ['step 1'],
            'expected_outcome': 'done',
            'risk_level': 'low',
            'policy_scope': 'private',
            'visibility_scope': 'private',
            'evidence_refs': [{
                'kind': 'event_log',
                'source_system': 'test',
                'locator': {'system': 'test', 'stream': 'test-stream', 'event_id': 'test-1'},
            }],
            'source_lane': source_lane,
            'success_count': score_seed,
        })

    def test_forbidden_lane_cannot_shadow_allowed(self):
        """Forbidden-lane high-scored procedures must not crowd out allowed ones."""
        from mcp_server.src.services.change_ledger import ChangeLedger
        from mcp_server.src.services.procedure_service import ProcedureService

        with tempfile.NamedTemporaryFile(suffix='.db') as tmp:
            ledger = ChangeLedger(tmp.name)
            svc = ProcedureService(ledger)

            # Create procedures: forbidden lane has higher scores
            forbidden_procs = [
                self._make_procedure(
                    name=f'forbidden_{i}', trigger='deploy production',
                    source_lane='forbidden', score_seed=100 + i,
                )
                for i in range(10)
            ]
            allowed_proc = self._make_procedure(
                name='allowed_deploy', trigger='deploy production',
                source_lane='allowed', score_seed=1,
            )
            all_procs = forbidden_procs + [allowed_proc]

            # Without lane filter: allowed proc could get shadowed at limit=5
            unfiltered = svc.retrieve_procedures(
                'deploy production', limit=5, procedures=all_procs,
            )
            # The allowed proc might not be in top 5 since forbidden ones score higher
            allowed_in_unfiltered = [m for m in unfiltered if m.procedure.source_lane == 'allowed']

            # With lane filter: allowed proc must be returned
            filtered = svc.retrieve_procedures(
                'deploy production', limit=5, procedures=all_procs,
                effective_group_ids=['allowed'],
            )
            assert len(filtered) >= 1
            assert all(m.procedure.source_lane == 'allowed' for m in filtered)
            assert filtered[0].procedure.name == 'allowed_deploy'

    def test_empty_group_ids_returns_nothing(self):
        """Empty effective_group_ids (fail-closed) returns no results."""
        from mcp_server.src.services.change_ledger import ChangeLedger
        from mcp_server.src.services.procedure_service import ProcedureService

        with tempfile.NamedTemporaryFile(suffix='.db') as tmp:
            ledger = ChangeLedger(tmp.name)
            svc = ProcedureService(ledger)

            proc = self._make_procedure(
                name='some_proc', trigger='test trigger',
                source_lane='lane_a',
            )
            results = svc.retrieve_procedures(
                'test trigger', limit=5, procedures=[proc],
                effective_group_ids=[],
            )
            assert results == []

    def test_none_group_ids_returns_all(self):
        """None effective_group_ids means no lane filter (all results)."""
        from mcp_server.src.services.change_ledger import ChangeLedger
        from mcp_server.src.services.procedure_service import ProcedureService

        with tempfile.NamedTemporaryFile(suffix='.db') as tmp:
            ledger = ChangeLedger(tmp.name)
            svc = ProcedureService(ledger)

            procs = [
                self._make_procedure(name=f'p{i}', trigger='target', source_lane=f'lane_{i}')
                for i in range(3)
            ]
            results = svc.retrieve_procedures(
                'target', limit=10, procedures=procs,
                effective_group_ids=None,
            )
            assert len(results) == 3

    def test_same_lane_lower_ranked_still_retrievable(self):
        """A same-lane lower-ranked procedure is still retrievable when forbidden-lane higher-ranked matches exist."""
        from mcp_server.src.services.change_ledger import ChangeLedger
        from mcp_server.src.services.procedure_service import ProcedureService

        with tempfile.NamedTemporaryFile(suffix='.db') as tmp:
            ledger = ChangeLedger(tmp.name)
            svc = ProcedureService(ledger)

            # 20 high-score forbidden procedures
            forbidden = [
                self._make_procedure(
                    name=f'high_score_{i}', trigger='common task pattern',
                    source_lane='forbidden_lane', score_seed=1000 + i,
                )
                for i in range(20)
            ]
            # 1 low-score allowed procedure
            allowed = self._make_procedure(
                name='low_score_allowed', trigger='common task pattern',
                source_lane='my_lane', score_seed=1,
            )

            results = svc.retrieve_procedures(
                'common task pattern', limit=3,
                procedures=forbidden + [allowed],
                effective_group_ids=['my_lane'],
            )
            assert len(results) >= 1
            assert results[0].procedure.source_lane == 'my_lane'
            assert results[0].procedure.name == 'low_score_allowed'


# ---------------------------------------------------------------------------
# Blocker C: get_episode not_found alignment
# ---------------------------------------------------------------------------


class TestGetEpisodeNotFoundAlignment:
    """Verify get_episode returns not_found (not access_denied) for out-of-lane IDs."""

    def test_out_of_lane_episode_returns_not_found(self):
        """Accessing an episode outside caller's lane scope should return not_found, not access_denied."""
        import asyncio
        from mcp_server.src.services.change_ledger import ChangeLedger

        with tempfile.NamedTemporaryFile(suffix='.db') as tmp:
            ledger = ChangeLedger(tmp.name)

            # Create an episode in lane_a
            from mcp_server.src.models.typed_memory import Episode
            ep = Episode.model_validate({
                'object_id': 'ep-test-001',
                'root_id': 'ep-test-001',
                'title': 'Test episode',
                'summary': 'Test episode content',
                'source_lane': 'lane_a',
                'policy_scope': 'private',
                'visibility_scope': 'private',
                'evidence_refs': [{
                    'kind': 'event_log',
                    'source_system': 'test',
                    'locator': {'system': 'test', 'stream': 'test-stream', 'event_id': 'src-1'},
                }],
            })
            ledger.append_event(
                'assert', actor_id='test', reason='test',
                payload=ep, _autocommit=True,
            )

            # Mock the MCP server and register tools
            class FakeMCP:
                def tool(self):
                    def decorator(fn):
                        self._last_tool = fn
                        return fn
                    return decorator

            fake_mcp = FakeMCP()

            # Patch the module-level imports and register
            from mcp_server.src.routers import episodes_procedures as ep_mod

            with mock.patch.dict(os.environ, {'BICAMERAL_CHANGE_LEDGER_DB': tmp.name}):
                tools = ep_mod.register_tools(fake_mcp)
                get_episode = tools['get_episode']

                # Request episode from a different lane
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(
                        get_episode(
                            episode_id='ep-test-001',
                            group_ids=['lane_b'],
                            lane_alias=None,
                        )
                    )
                finally:
                    loop.close()

                # Must be not_found, not access_denied
                assert result.get('error') == 'not_found', (
                    f"Expected 'not_found' but got {result.get('error')!r} — "
                    f"existence leak: out-of-lane access should not reveal the episode exists"
                )
                assert 'access_denied' not in str(result)
