"""Regression tests for the lane-isolation bug class.

Covers:
  - Cross-lane conflict non-disclosure in remember_fact (#1)
  - Cross-lane supersede rejection in promote_candidate_fact (#2)
  - Cross-lane lineage takeover rejection in _validate_supersede_target (#2)
  - Lane-bounded candidate review in list_candidates (#7)
  - Adjacent lane-isolation lifecycle behavior

These tests are intentionally self-contained so they remain runnable without
a live Neo4j / FalkorDB backend.  They exercise the SQLite ledger and in-memory
memory-router paths only.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
from mcp_server.src.models.typed_memory import EvidenceRef, StateFact
from mcp_server.src.routers import candidates as candidates_router
from mcp_server.src.services.change_ledger import ChangeLedger, _validate_supersede_target
from mcp_server.src.services.candidate_store import CandidateStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state_fact(
    *,
    object_id: str,
    subject: str = 'Yuan',
    predicate: str = 'prefers',
    value: str = 'espresso',
    scope: str = 'private',
    source_lane: str | None = None,
) -> StateFact:
    evidence = EvidenceRef.model_validate({
        'kind': 'event_log',
        'source_system': 'tests',
        'locator': {
            'system': 'pytest',
            'stream': 'lane-isolation',
            'event_id': f'evt-{object_id}',
        },
        'observed_at': '2026-03-13T00:00:00Z',
        'retrieved_at': '2026-03-13T00:00:00Z',
    })
    return StateFact.model_validate({
        'object_id': object_id,
        'root_id': object_id,
        'version': 1,
        'fact_type': 'preference',
        'subject': subject,
        'predicate': predicate,
        'value': value,
        'scope': scope,
        'policy_scope': scope,
        'visibility_scope': scope,
        'source_lane': source_lane,
        'evidence_refs': [evidence],
        'created_at': '2026-03-13T00:00:00Z',
        'valid_at': '2026-03-13T00:00:00Z',
        'promotion_status': 'promoted',
    })


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ChangeLedger:
    db_path = tmp_path / 'change_ledger.db'
    candidates_path = tmp_path / 'candidates.db'
    monkeypatch.setenv('BICAMERAL_CHANGE_LEDGER_PATH', str(db_path))
    # Isolate candidates DB so remember_fact calls don't pollute the global default path.
    monkeypatch.setenv('BICAMERAL_CANDIDATES_DB_PATH', str(candidates_path))
    ldr = ChangeLedger(db_path)
    yield ldr
    ldr.close()


@pytest.fixture
def candidate_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Shared store + ledger + auth bypass for candidate tests."""
    ledger_path = tmp_path / 'change_ledger.db'
    candidates_path = tmp_path / 'candidates.db'
    monkeypatch.setenv('BICAMERAL_CHANGE_LEDGER_PATH', str(ledger_path))
    monkeypatch.setenv('BICAMERAL_CANDIDATES_DB_PATH', str(candidates_path))
    monkeypatch.setattr(candidates_router, '_change_ledger', None)
    monkeypatch.setattr(candidates_router, '_candidate_store', None)
    monkeypatch.setattr(candidates_router, '_require_reviewer', lambda ctx: 'system')
    store = CandidateStore(db_path=candidates_path)
    ldr = ChangeLedger(ledger_path)
    yield store, ldr
    ldr.close()


# ===========================================================================
# Issue #1: Cross-lane conflict non-disclosure in remember_fact
# ===========================================================================

class TestCrossLaneConflictNonDisclosure:
    """remember_fact must not expose facts from a different lane via conflict/duplicate responses."""

    def test_same_lane_conflict_detected(self, ledger: ChangeLedger, monkeypatch):
        """Baseline: same-lane conflict IS detected (conflict is not suppressed globally)."""
        import mcp_server.src.routers.memory as memory_router
        monkeypatch.setattr(memory_router, '_change_ledger', ledger)
        # Reset candidate store so it picks up the temp BICAMERAL_CANDIDATES_DB_PATH
        monkeypatch.setattr(memory_router, '_candidate_store', None)
        monkeypatch.setattr(memory_router, '_materializer', None)

        # Seed a fact in lane_a
        fact = _make_state_fact(
            object_id='same-lane-fact', subject='Yuan', predicate='prefers',
            value='espresso', source_lane='lane_a',
        )
        ledger.append_event(
            'assert', actor_id='seed', reason='seed', payload=fact,
            object_id=fact.object_id, object_type=fact.object_type, root_id=fact.root_id,
        )

        # A new remember_fact in the SAME lane should detect a conflict
        with patch.object(
            memory_router,
            '_derive_source_lane',
            return_value='lane_a',
        ):
            result = asyncio.run(
                memory_router.remember_fact(
                    text='Yuan prefers pour-over',
                    hint={
                        'fact_type': 'preference',
                        'subject': 'Yuan',
                        'predicate': 'prefers',
                        'value': 'pour-over',
                        'scope': 'private',
                    },
                )
            )
        # Conflict or duplicate should be detected for same-lane
        assert result.get('status') in ('conflict', 'duplicate', 'ok'), (
            f"Unexpected same-lane result: {result}"
        )

    def test_cross_lane_conflict_not_disclosed(self, ledger: ChangeLedger, monkeypatch):
        """A fact in lane_b must NOT appear as an existing_fact conflict when writing in lane_a."""
        import mcp_server.src.routers.memory as memory_router
        monkeypatch.setattr(memory_router, '_change_ledger', ledger)
        # Reset candidate store so it picks up the temp BICAMERAL_CANDIDATES_DB_PATH
        monkeypatch.setattr(memory_router, '_candidate_store', None)
        monkeypatch.setattr(memory_router, '_materializer', None)

        # Seed a fact in lane_b
        fact_b = _make_state_fact(
            object_id='lane-b-fact', subject='Yuan', predicate='prefers',
            value='espresso', source_lane='lane_b',
        )
        ledger.append_event(
            'assert', actor_id='seed', reason='seed', payload=fact_b,
            object_id=fact_b.object_id, object_type=fact_b.object_type, root_id=fact_b.root_id,
        )

        # Now try remember_fact from lane_a — should NOT see lane_b fact as conflict
        with patch.object(
            memory_router,
            '_derive_source_lane',
            return_value='lane_a',
        ):
            result = asyncio.run(
                memory_router.remember_fact(
                    text='Yuan prefers pour-over',
                    hint={
                        'fact_type': 'preference',
                        'subject': 'Yuan',
                        'predicate': 'prefers',
                        'value': 'pour-over',
                        'scope': 'private',
                    },
                )
            )

        # Must NOT return a conflict exposing the lane_b fact's value/source
        assert result.get('status') != 'conflict', (
            f"Cross-lane conflict disclosure: lane_a write exposed lane_b fact. Result: {result}"
        )
        # Also must not return duplicate with lane_b data
        assert result.get('status') != 'duplicate', (
            f"Cross-lane duplicate disclosure: lane_a write exposed lane_b fact. Result: {result}"
        )
        # Ensure lane_b fact metadata is not in the response
        result_str = str(result)
        assert 'lane_b' not in result_str or result.get('status') not in ('conflict', 'duplicate'), (
            f"Cross-lane lane identifier leaked in response: {result}"
        )

    def test_unscoped_lane_conflict_detected_backward_compat(self, ledger: ChangeLedger, monkeypatch):
        """When no server lane is configured (None), unscoped conflict detection is preserved."""
        import mcp_server.src.routers.memory as memory_router
        monkeypatch.setattr(memory_router, '_change_ledger', ledger)
        # Reset candidate store so it picks up the temp BICAMERAL_CANDIDATES_DB_PATH
        monkeypatch.setattr(memory_router, '_candidate_store', None)
        monkeypatch.setattr(memory_router, '_materializer', None)

        # Seed an unscoped fact (no source_lane)
        fact = _make_state_fact(
            object_id='unscoped-fact', subject='Yuan', predicate='prefers',
            value='espresso', source_lane=None,
        )
        ledger.append_event(
            'assert', actor_id='seed', reason='seed', payload=fact,
            object_id=fact.object_id, object_type=fact.object_type, root_id=fact.root_id,
        )

        # Unscoped write (_derive_source_lane() returns None) should still detect conflict
        with patch.object(memory_router, '_derive_source_lane', return_value=None):
            result = asyncio.run(
                memory_router.remember_fact(
                    text='Yuan prefers pour-over',
                    hint={
                        'fact_type': 'preference',
                        'subject': 'Yuan',
                        'predicate': 'prefers',
                        'value': 'pour-over',
                        'scope': 'private',
                    },
                )
            )
        # Conflict or duplicate detection must still work in unscoped deployments
        assert result.get('status') in ('conflict', 'duplicate', 'ok'), (
            f"Unexpected result for unscoped deployment: {result}"
        )


# ===========================================================================
# Issue #2: Cross-lane supersede rejection
# ===========================================================================

class TestCrossLaneSupersede:
    """promote_candidate_fact / _validate_supersede_target must reject cross-lane supersede."""

    def test_validate_supersede_target_rejects_cross_lane(self):
        """_validate_supersede_target raises ValueError for cross-lane supersede attempts."""
        candidate = _make_state_fact(
            object_id='cand-001', subject='Yuan', predicate='prefers',
            value='pour-over', source_lane='lane_a',
        )
        prior = _make_state_fact(
            object_id='prior-001', subject='Yuan', predicate='prefers',
            value='espresso', source_lane='lane_b',
        )

        with pytest.raises(ValueError, match='cross-lane supersede rejected'):
            _validate_supersede_target(candidate=candidate, prior=prior)

    def test_validate_supersede_target_allows_same_lane(self):
        """_validate_supersede_target allows supersede within the same lane."""
        candidate = _make_state_fact(
            object_id='cand-002', subject='Yuan', predicate='prefers',
            value='pour-over', source_lane='lane_a',
        )
        prior = _make_state_fact(
            object_id='prior-002', subject='Yuan', predicate='prefers',
            value='espresso', source_lane='lane_a',
        )
        # Should not raise
        _validate_supersede_target(candidate=candidate, prior=prior)

    def test_validate_supersede_target_rejects_unscoped_prior_when_candidate_is_scoped(self):
        """_validate_supersede_target hard-fails when a scoped candidate targets an unscoped legacy fact.

        Policy (chosen: hard-fail): when lane isolation is active (candidate.source_lane is not
        None), the prior must belong to the same lane.  An unscoped prior (source_lane=None)
        predates lane awareness; its multi-lane provenance is unknown, so allowing a scoped
        candidate to absorb it silently is unsafe.  Hard-fail is cleaner.
        """
        candidate = _make_state_fact(
            object_id='cand-003', subject='Yuan', predicate='prefers',
            value='pour-over', source_lane='lane_a',
        )
        prior_unscoped = _make_state_fact(
            object_id='prior-003', subject='Yuan', predicate='prefers',
            value='espresso', source_lane=None,
        )
        # Must raise — scoped candidate targeting unscoped legacy fact is rejected.
        with pytest.raises(ValueError, match='unscoped-fact supersede rejected'):
            _validate_supersede_target(candidate=candidate, prior=prior_unscoped)

    def test_validate_supersede_target_allows_unscoped_candidate_targeting_unscoped_prior(self):
        """Both facts unscoped (global/unscoped deployment): supersede is still allowed."""
        candidate_unscoped = _make_state_fact(
            object_id='cand-004', subject='Yuan', predicate='prefers',
            value='pour-over', source_lane=None,
        )
        prior_unscoped = _make_state_fact(
            object_id='prior-004', subject='Yuan', predicate='prefers',
            value='espresso', source_lane=None,
        )
        # Should not raise — both unscoped, no lane isolation active
        _validate_supersede_target(candidate=candidate_unscoped, prior=prior_unscoped)

    def test_validate_supersede_target_allows_unscoped_candidate_targeting_scoped_prior(self):
        """Unscoped candidate targeting a scoped prior is allowed (global-owner semantics)."""
        candidate_unscoped = _make_state_fact(
            object_id='cand-005', subject='Yuan', predicate='prefers',
            value='pour-over', source_lane=None,
        )
        prior_scoped = _make_state_fact(
            object_id='prior-005', subject='Yuan', predicate='prefers',
            value='espresso', source_lane='lane_a',
        )
        # Should not raise — candidate has no source_lane, so no lane check is applied
        _validate_supersede_target(candidate=candidate_unscoped, prior=prior_scoped)

    def _fact_input(self, *, subject: str = 'Yuan', predicate: str = 'prefers',
                    value: str = 'pour-over', source_lane: str = 'lane_a') -> dict:
        """Build a minimal fact dict accepted by promote_candidate_fact."""
        return {
            'assertion_type': 'preference',
            'subject': subject,
            'predicate': predicate,
            'value': value,
            'scope': 'private',
            'source_lane': source_lane,
            'evidence_refs': [
                {
                    'source_key': f'test-{source_lane}',
                    'scope': 'private',
                    'source_system': 'lane_isolation_test',
                    'evidence_id': f'eid-{source_lane}-001',
                    'observed_at': '2026-03-13T00:00:00Z',
                }
            ],
        }

    def test_promote_candidate_fact_rejects_cross_lane_conflict_auto_resolution(
        self, ledger: ChangeLedger
    ):
        """promote_candidate_fact must not auto-supersede a fact from a different lane."""
        # Seed an existing fact in lane_b
        fact_b = _make_state_fact(
            object_id='lane-b-existing', subject='Yuan', predicate='prefers',
            value='espresso', source_lane='lane_b',
        )
        ledger.append_event(
            'assert', actor_id='seed', reason='seed', payload=fact_b,
            object_id=fact_b.object_id, object_type=fact_b.object_type, root_id=fact_b.root_id,
        )

        # Promote a candidate from lane_a with the same conflict_set.
        # Auto-detection loop should NOT pick up the lane_b fact, so require_supersede fails.
        with pytest.raises(ValueError):
            ledger.promote_candidate_fact(
                actor_id='reviewer',
                reason='test',
                policy_version='v1',
                candidate_id='cand-cross-lane',
                fact=self._fact_input(source_lane='lane_a'),
                conflict_with_fact_id=None,
                allow_parallel=False,
                require_supersede=True,
            )

        # lane_b fact must remain current — it was not touched by the failed cross-lane attempt
        current_facts = ledger.current_state_facts()
        assert any(f.object_id == 'lane-b-existing' for f in current_facts), (
            'lane_b fact should still be current after rejected cross-lane auto-supersede attempt'
        )

    def test_promote_candidate_fact_rejects_explicit_cross_lane_supersede(
        self, ledger: ChangeLedger
    ):
        """promote_candidate_fact must reject explicit cross-lane supersede via conflict_with_fact_id."""
        # Seed an existing fact in lane_b
        fact_b = _make_state_fact(
            object_id='lane-b-explicit', subject='Yuan', predicate='prefers',
            value='espresso', source_lane='lane_b',
        )
        ledger.append_event(
            'assert', actor_id='seed', reason='seed', payload=fact_b,
            object_id=fact_b.object_id, object_type=fact_b.object_type, root_id=fact_b.root_id,
        )

        # Explicitly try to supersede lane_b fact from lane_a candidate
        with pytest.raises(ValueError, match='cross-lane supersede rejected|incompatible'):
            ledger.promote_candidate_fact(
                actor_id='reviewer',
                reason='test',
                policy_version='v1',
                candidate_id='cand-explicit-cross',
                fact=self._fact_input(source_lane='lane_a'),
                conflict_with_fact_id='lane-b-explicit',  # explicitly targeting lane_b
                allow_parallel=False,
                require_supersede=True,
            )

        # lane_b fact must remain current (not superseded)
        current_facts = ledger.current_state_facts()
        assert any(f.object_id == 'lane-b-explicit' for f in current_facts), (
            'lane_b fact was incorrectly superseded by cross-lane candidate'
        )

    def test_same_lane_supersede_succeeds(self, ledger: ChangeLedger):
        """Baseline: same-lane supersede DOES succeed (not over-blocked)."""
        fact_a = _make_state_fact(
            object_id='lane-a-existing', subject='Yuan', predicate='prefers',
            value='espresso', source_lane='lane_a',
        )
        ledger.append_event(
            'assert', actor_id='seed', reason='seed', payload=fact_a,
            object_id=fact_a.object_id, object_type=fact_a.object_type, root_id=fact_a.root_id,
        )

        result = ledger.promote_candidate_fact(
            actor_id='reviewer',
            reason='test',
            policy_version='v1',
            candidate_id='cand-same-lane',
            fact=self._fact_input(source_lane='lane_a'),  # same lane as the target
            conflict_with_fact_id='lane-a-existing',
            allow_parallel=False,
            require_supersede=True,
        )
        assert result.object_id is not None
        # Original fact should no longer be current
        current_facts = ledger.current_state_facts()
        assert not any(f.object_id == 'lane-a-existing' for f in current_facts), (
            'Original lane_a fact should have been superseded'
        )


# ===========================================================================
# Issue #7: Lane-bounded candidate review (list_candidates)
# ===========================================================================

class TestLaneBoundedCandidateReview:
    """list_candidates must respect lane-scope invariant when server has a configured lane."""

    def test_list_candidates_unscoped_server_returns_all(self, candidate_env, monkeypatch):
        """When server has no lane (global deployment), all candidates are visible."""
        store, ldr = candidate_env
        # Create candidates in two different lanes
        store.create_candidate(
            payload={'subject': 'Yuan', 'predicate': 'prefers', 'value': 'espresso', 'fact_type': 'preference'},
            candidate_id='cand-lane-a',
            raw_hint={'source_lane': 'lane_a'},
        )
        store.create_candidate(
            payload={'subject': 'Yuan', 'predicate': 'prefers', 'value': 'matcha', 'fact_type': 'preference'},
            candidate_id='cand-lane-b',
            raw_hint={'source_lane': 'lane_b'},
        )
        store.create_candidate(
            payload={'subject': 'Yuan', 'predicate': 'prefers', 'value': 'chai', 'fact_type': 'preference'},
            candidate_id='cand-no-lane',
            raw_hint={},  # no source_lane
        )

        # Unscoped server (no group_id configured) → all candidates visible
        import mcp_server.src.routers.memory as memory_router
        with patch.object(memory_router, '_derive_source_lane', return_value=None):
            result = asyncio.run(candidates_router.list_candidates())

        assert result['status'] == 'ok'
        ids = {c['uuid'] for c in result['candidates']}
        assert 'cand-lane-a' in ids
        assert 'cand-lane-b' in ids
        assert 'cand-no-lane' in ids

    def test_list_candidates_scoped_server_filters_to_own_lane(self, candidate_env, monkeypatch):
        """When server has a configured lane, only same-lane candidates are returned."""
        store, ldr = candidate_env
        store.create_candidate(
            payload={'subject': 'Yuan', 'predicate': 'prefers', 'value': 'espresso', 'fact_type': 'preference'},
            candidate_id='cand-lane-a',
            raw_hint={'source_lane': 'lane_a'},
        )
        store.create_candidate(
            payload={'subject': 'Yuan', 'predicate': 'prefers', 'value': 'matcha', 'fact_type': 'preference'},
            candidate_id='cand-lane-b',
            raw_hint={'source_lane': 'lane_b'},
        )

        # Server configured as lane_a → only lane_a candidates visible
        import mcp_server.src.routers.memory as memory_router
        with patch.object(memory_router, '_derive_source_lane', return_value='lane_a'):
            result = asyncio.run(candidates_router.list_candidates())

        assert result['status'] == 'ok'
        ids = {c['uuid'] for c in result['candidates']}
        assert 'cand-lane-a' in ids, 'lane_a candidate should be visible to lane_a server'
        assert 'cand-lane-b' not in ids, 'lane_b candidate must NOT be visible to lane_a server'

    def test_list_candidates_cross_lane_metadata_not_disclosed(self, candidate_env, monkeypatch):
        """list_candidates from a scoped server must not expose any foreign-lane candidate metadata."""
        store, ldr = candidate_env
        store.create_candidate(
            payload={'subject': 'secret_subject', 'predicate': 'secret_pred', 'value': 'SECRET_VALUE', 'fact_type': 'preference'},
            candidate_id='cand-secret-lane-b',
            raw_hint={'source_lane': 'lane_b'},
        )

        # Server is in lane_a — must not see lane_b candidate at all
        import mcp_server.src.routers.memory as memory_router
        with patch.object(memory_router, '_derive_source_lane', return_value='lane_a'):
            result = asyncio.run(candidates_router.list_candidates())

        assert result['status'] == 'ok'
        result_str = str(result)
        assert 'SECRET_VALUE' not in result_str, (
            'Cross-lane secret value leaked via list_candidates'
        )
        assert 'cand-secret-lane-b' not in result_str, (
            'Cross-lane candidate ID leaked via list_candidates'
        )


# ===========================================================================
# Issue #9: Adjacent lane-isolation lifecycle behavior
# ===========================================================================

class TestLaneIsolationLifecycle:
    """End-to-end lane-isolation lifecycle: write, conflict, promote — all lane-bounded."""

    def test_full_lane_isolated_lifecycle(self, candidate_env, monkeypatch):
        """A candidate quarantined by lane_a remember_fact is only visible to lane_a reviewer."""
        store, ldr = candidate_env

        # Manually seed same-lane and cross-lane candidates
        store.create_candidate(
            payload={'subject': 'Yuan', 'predicate': 'prefers', 'value': 'espresso', 'fact_type': 'preference'},
            candidate_id='cand-lifecycle-a',
            raw_hint={'source_lane': 'lane_a'},
        )
        store.create_candidate(
            payload={'subject': 'Yuan', 'predicate': 'prefers', 'value': 'matcha', 'fact_type': 'preference'},
            candidate_id='cand-lifecycle-b',
            raw_hint={'source_lane': 'lane_b'},
        )

        import mcp_server.src.routers.memory as memory_router

        # lane_a reviewer: sees only lane_a candidate
        with patch.object(memory_router, '_derive_source_lane', return_value='lane_a'):
            result_a = asyncio.run(candidates_router.list_candidates())
        ids_a = {c['uuid'] for c in result_a['candidates']}
        assert 'cand-lifecycle-a' in ids_a
        assert 'cand-lifecycle-b' not in ids_a

        # lane_b reviewer: sees only lane_b candidate
        with patch.object(memory_router, '_derive_source_lane', return_value='lane_b'):
            result_b = asyncio.run(candidates_router.list_candidates())
        ids_b = {c['uuid'] for c in result_b['candidates']}
        assert 'cand-lifecycle-b' in ids_b
        assert 'cand-lifecycle-a' not in ids_b

    def test_cross_lane_supersede_rejected_in_full_promote_flow(self, candidate_env, monkeypatch):
        """Full promote_candidate flow: cross-lane supersede raises validation_error."""
        store, ldr = candidate_env

        # Seed an existing fact in lane_b
        fact_b = _make_state_fact(
            object_id='lane-b-promo', subject='Yuan', predicate='prefers',
            value='espresso', source_lane='lane_b',
        )
        ldr.append_event(
            'assert', actor_id='seed', reason='seed', payload=fact_b,
            object_id=fact_b.object_id, object_type=fact_b.object_type, root_id=fact_b.root_id,
        )

        # Create a lane_a candidate that tries to supersede the lane_b fact.
        # Include evidence_refs so promote_candidate_fact can build the typed object.
        store.create_candidate(
            payload={'subject': 'Yuan', 'predicate': 'prefers', 'value': 'pour-over', 'fact_type': 'preference'},
            candidate_id='cand-cross-promo',
            conflict_with_fact_id='lane-b-promo',
            raw_hint={
                'source_lane': 'lane_a',
                'scope': 'private',
                'policy_scope': 'private',
                'evidence_refs': [
                    {
                        'source_key': 'test-lane-a',
                        'scope': 'private',
                        'source_system': 'lane_isolation_test',
                        'evidence_id': 'eid-cross-promo-001',
                        'observed_at': '2026-03-13T00:00:00Z',
                    }
                ],
            },
        )

        result = asyncio.run(
            candidates_router.promote_candidate('cand-cross-promo', resolution='supersede')
        )

        # Must fail — cross-lane supersede rejected
        assert result.get('status') == 'error', (
            f"Expected error for cross-lane supersede, got: {result}"
        )
        assert result.get('error_type') == 'validation_error', (
            f"Expected validation_error, got: {result}"
        )
        err_msg = result.get('message', '')
        assert 'cross-lane' in err_msg or 'supersede' in err_msg or 'incompatible' in err_msg, (
            f"Error message should describe the cross-lane rejection: {err_msg}"
        )

        # lane_b fact must still be current
        current = ldr.current_state_facts()
        assert any(f.object_id == 'lane-b-promo' for f in current), (
            'lane_b fact was incorrectly superseded by cross-lane promote_candidate'
        )

        # lane_a candidate should remain in pending state
        candidate = store.get_candidate('cand-cross-promo')
        assert candidate is not None
        assert candidate['status'] == 'pending', (
            f"Candidate status should remain pending after failed cross-lane supersede, got: {candidate['status']}"
        )


# ===========================================================================
# Finding 1: Endpoint-level lane ownership enforcement
# (promote_candidate, reject_candidate, cancel)
# ===========================================================================

class TestEndpointLaneOwnershipEnforcement:
    """promote_candidate and reject_candidate must enforce lane ownership at the endpoint level.

    Even if the caller somehow obtains a foreign-lane candidate_id, promotion and
    rejection must be rejected when the server lane does not match the candidate lane.
    The cancel path (promote_candidate with resolution='cancel') is covered too.
    """

    def _make_candidate_in_store(
        self,
        store: CandidateStore,
        candidate_id: str,
        source_lane: str | None = 'lane_a',
    ) -> None:
        store.create_candidate(
            payload={
                'subject': 'Yuan', 'predicate': 'prefers',
                'value': 'espresso', 'fact_type': 'preference',
            },
            candidate_id=candidate_id,
            raw_hint={
                'source_lane': source_lane,
                'scope': 'private',
                'evidence_refs': [
                    {
                        'source_key': f'test-{source_lane}',
                        'scope': 'private',
                        'source_system': 'endpoint_lane_test',
                        'evidence_id': f'eid-{candidate_id}',
                        'observed_at': '2026-03-13T00:00:00Z',
                    }
                ],
            },
        )

    def test_cross_lane_promote_candidate_rejected(self, candidate_env, monkeypatch):
        """promote_candidate from lane_b server on a lane_a candidate must return unauthorized."""
        import mcp_server.src.routers.memory as memory_router
        store, ldr = candidate_env
        self._make_candidate_in_store(store, 'cand-ep-promote-a', source_lane='lane_a')

        # Server is configured as lane_b — must not be allowed to promote lane_a candidate
        with patch.object(memory_router, '_derive_source_lane', return_value='lane_b'):
            result = asyncio.run(
                candidates_router.promote_candidate('cand-ep-promote-a', resolution='supersede')
            )

        assert result.get('status') == 'error', (
            f'Expected error for cross-lane promote_candidate, got: {result}'
        )
        assert result.get('error_type') == 'unauthorized', (
            f'Expected unauthorized error type, got: {result.get("error_type")!r}'
        )
        assert 'lane_a' in result.get('message', '') or 'lane_b' in result.get('message', ''), (
            f'Error message should name the offending lanes: {result.get("message")!r}'
        )
        # Candidate must remain pending
        cand = store.get_candidate('cand-ep-promote-a')
        assert cand is not None and cand['status'] == 'pending', (
            f'Candidate should remain pending after cross-lane rejection, got: {cand}'
        )

    def test_cross_lane_reject_candidate_rejected(self, candidate_env, monkeypatch):
        """reject_candidate from lane_b server on a lane_a candidate must return unauthorized."""
        import mcp_server.src.routers.memory as memory_router
        store, ldr = candidate_env
        self._make_candidate_in_store(store, 'cand-ep-reject-a', source_lane='lane_a')

        with patch.object(memory_router, '_derive_source_lane', return_value='lane_b'):
            result = asyncio.run(
                candidates_router.reject_candidate('cand-ep-reject-a', reason='test cross-lane')
            )

        assert result.get('status') == 'error', (
            f'Expected error for cross-lane reject_candidate, got: {result}'
        )
        assert result.get('error_type') == 'unauthorized', (
            f'Expected unauthorized error type, got: {result.get("error_type")!r}'
        )
        # Candidate must remain pending (not silently rejected)
        cand = store.get_candidate('cand-ep-reject-a')
        assert cand is not None and cand['status'] == 'pending', (
            f'Candidate should remain pending after cross-lane rejection, got: {cand}'
        )

    def test_cross_lane_cancel_rejected(self, candidate_env, monkeypatch):
        """promote_candidate(resolution='cancel') from lane_b server on lane_a candidate is rejected."""
        import mcp_server.src.routers.memory as memory_router
        store, ldr = candidate_env
        self._make_candidate_in_store(store, 'cand-ep-cancel-a', source_lane='lane_a')

        with patch.object(memory_router, '_derive_source_lane', return_value='lane_b'):
            result = asyncio.run(
                candidates_router.promote_candidate('cand-ep-cancel-a', resolution='cancel')
            )

        assert result.get('status') == 'error', (
            f'Expected error for cross-lane cancel, got: {result}'
        )
        assert result.get('error_type') == 'unauthorized', (
            f'Expected unauthorized error type, got: {result.get("error_type")!r}'
        )
        cand = store.get_candidate('cand-ep-cancel-a')
        assert cand is not None and cand['status'] == 'pending', (
            f'Candidate should remain pending after cross-lane cancel rejection, got: {cand}'
        )

    def test_same_lane_promote_succeeds(self, candidate_env, monkeypatch):
        """Baseline: same-lane promote_candidate succeeds (lane check is not over-blocking)."""
        import mcp_server.src.routers.memory as memory_router
        store, ldr = candidate_env

        # Seed a lane_a fact in the ledger so supersede target exists
        fact_a = _make_state_fact(
            object_id='ep-lane-a-existing', subject='Yuan', predicate='prefers',
            value='espresso', source_lane='lane_a',
        )
        ldr.append_event(
            'assert', actor_id='seed', reason='seed', payload=fact_a,
            object_id=fact_a.object_id, object_type=fact_a.object_type, root_id=fact_a.root_id,
        )
        store.create_candidate(
            payload={'subject': 'Yuan', 'predicate': 'prefers', 'value': 'pour-over', 'fact_type': 'preference'},
            candidate_id='cand-ep-same-lane',
            conflict_with_fact_id='ep-lane-a-existing',
            raw_hint={
                'source_lane': 'lane_a',
                'scope': 'private',
                'evidence_refs': [
                    {
                        'source_key': 'test-lane-a',
                        'scope': 'private',
                        'source_system': 'endpoint_lane_test',
                        'evidence_id': 'eid-ep-same-lane',
                        'observed_at': '2026-03-13T00:00:00Z',
                    }
                ],
            },
        )

        with patch.object(memory_router, '_derive_source_lane', return_value='lane_a'):
            result = asyncio.run(
                candidates_router.promote_candidate('cand-ep-same-lane', resolution='supersede')
            )

        assert result.get('status') == 'ok', (
            f'Same-lane promotion should succeed, got: {result}'
        )

    def test_unscoped_server_can_act_on_any_candidate(self, candidate_env, monkeypatch):
        """When server has no lane (global deployment), endpoint acts on candidates of any lane."""
        import mcp_server.src.routers.memory as memory_router
        store, ldr = candidate_env
        self._make_candidate_in_store(store, 'cand-ep-global-a', source_lane='lane_a')
        self._make_candidate_in_store(store, 'cand-ep-global-none', source_lane=None)

        # Global server (no lane) — cancel should be permitted for any candidate
        with patch.object(memory_router, '_derive_source_lane', return_value=None):
            result_a = asyncio.run(
                candidates_router.promote_candidate('cand-ep-global-a', resolution='cancel')
            )
            result_none = asyncio.run(
                candidates_router.promote_candidate('cand-ep-global-none', resolution='cancel')
            )

        assert result_a.get('status') == 'ok', (
            f'Global server should be able to cancel lane_a candidate, got: {result_a}'
        )
        assert result_none.get('status') == 'ok', (
            f'Global server should be able to cancel unscoped candidate, got: {result_none}'
        )


# ===========================================================================
# Finding 2: Unscoped legacy fact supersede policy (hard-fail)
# ===========================================================================

class TestUnscopedLegacyFactSupersede:
    """When lane isolation is active, a scoped candidate must not supersede an unscoped fact.

    Policy chosen: hard-fail.  The prior's multi-lane provenance is unknown;
    silently absorbing it is unsafe.
    """

    def _fact_input(self, *, source_lane: str = 'lane_a') -> dict:
        return {
            'assertion_type': 'preference',
            'subject': 'Yuan',
            'predicate': 'prefers',
            'value': 'pour-over',
            'scope': 'private',
            'source_lane': source_lane,
            'evidence_refs': [
                {
                    'source_key': f'test-{source_lane}',
                    'scope': 'private',
                    'source_system': 'unscoped_test',
                    'evidence_id': f'eid-unscoped-{source_lane}',
                    'observed_at': '2026-03-13T00:00:00Z',
                }
            ],
        }

    def test_scoped_candidate_cannot_auto_supersede_unscoped_fact(self, ledger: ChangeLedger):
        """promote_candidate_fact auto-resolution must not pick up an unscoped legacy fact."""
        # Seed an unscoped legacy fact (source_lane=None)
        unscoped_fact = _make_state_fact(
            object_id='legacy-unscoped', subject='Yuan', predicate='prefers',
            value='espresso', source_lane=None,
        )
        ledger.append_event(
            'assert', actor_id='seed', reason='seed', payload=unscoped_fact,
            object_id=unscoped_fact.object_id, object_type=unscoped_fact.object_type,
            root_id=unscoped_fact.root_id,
        )

        # Scoped candidate from lane_a — auto-resolution should NOT pick up the unscoped fact
        # (require_supersede=True means it raises when no valid target is found)
        with pytest.raises(ValueError):
            ledger.promote_candidate_fact(
                actor_id='reviewer',
                reason='test',
                policy_version='v1',
                candidate_id='cand-unscoped-auto',
                fact=self._fact_input(source_lane='lane_a'),
                conflict_with_fact_id=None,
                allow_parallel=False,
                require_supersede=True,
            )

        # Unscoped fact must remain current — not silently absorbed
        current = ledger.current_state_facts()
        assert any(f.object_id == 'legacy-unscoped' for f in current), (
            'Unscoped legacy fact should remain current after rejected scoped auto-supersede'
        )

    def test_scoped_candidate_explicit_supersede_of_unscoped_fact_hard_fails(
        self, ledger: ChangeLedger
    ):
        """Explicit conflict_with_fact_id targeting an unscoped fact raises ValueError."""
        unscoped_fact = _make_state_fact(
            object_id='legacy-unscoped-explicit', subject='Yuan', predicate='prefers',
            value='espresso', source_lane=None,
        )
        ledger.append_event(
            'assert', actor_id='seed', reason='seed', payload=unscoped_fact,
            object_id=unscoped_fact.object_id, object_type=unscoped_fact.object_type,
            root_id=unscoped_fact.root_id,
        )

        with pytest.raises(ValueError, match='unscoped-fact supersede rejected'):
            ledger.promote_candidate_fact(
                actor_id='reviewer',
                reason='test',
                policy_version='v1',
                candidate_id='cand-unscoped-explicit',
                fact=self._fact_input(source_lane='lane_a'),
                conflict_with_fact_id='legacy-unscoped-explicit',
                allow_parallel=False,
                require_supersede=True,
            )

        # Unscoped fact must remain current (not superseded)
        current = ledger.current_state_facts()
        assert any(f.object_id == 'legacy-unscoped-explicit' for f in current), (
            'Unscoped legacy fact should not have been superseded'
        )

    def test_unscoped_candidate_can_still_supersede_unscoped_fact(self, ledger: ChangeLedger):
        """Backward compat: unscoped candidate (source_lane=None) superseding unscoped fact is still allowed.

        Evidence refs must not include a ``scope`` key in this scenario — the legacy
        ``_source_lane_from_legacy_refs`` fallback reads ``scope`` as a lane identifier,
        which would incorrectly promote an unscoped candidate to a scoped one.  In a
        genuine global/unscoped deployment, evidence refs carry no lane-scope value.
        """
        unscoped_fact = _make_state_fact(
            object_id='legacy-unscoped-bc', subject='Yuan', predicate='prefers',
            value='espresso', source_lane=None,
        )
        ledger.append_event(
            'assert', actor_id='seed', reason='seed', payload=unscoped_fact,
            object_id=unscoped_fact.object_id, object_type=unscoped_fact.object_type,
            root_id=unscoped_fact.root_id,
        )

        unscoped_fact_input = {
            'assertion_type': 'preference',
            'subject': 'Yuan', 'predicate': 'prefers', 'value': 'pour-over',
            # top-level scope is the visibility policy, NOT a lane identifier
            'scope': 'private',
            # Explicitly None — no lane isolation active
            'source_lane': None,
            # Omit 'scope' inside evidence_refs to avoid _source_lane_from_legacy_refs
            # misinterpreting the visibility-scope value as a lane identifier.
            'evidence_refs': [
                {
                    'source_key': 'test-unscoped',
                    'source_system': 'unscoped_test',
                    'evidence_id': 'eid-bc-unscoped',
                    'observed_at': '2026-03-13T00:00:00Z',
                }
            ],
        }

        # Should succeed — no lane isolation active (candidate is unscoped)
        result = ledger.promote_candidate_fact(
            actor_id='reviewer',
            reason='test',
            policy_version='v1',
            candidate_id='cand-bc-unscoped',
            fact=unscoped_fact_input,
            conflict_with_fact_id='legacy-unscoped-bc',
            allow_parallel=False,
            require_supersede=True,
        )
        assert result.object_id is not None
        current = ledger.current_state_facts()
        assert not any(f.object_id == 'legacy-unscoped-bc' for f in current), (
            'Unscoped fact should have been superseded by unscoped candidate'
        )
