from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from scripts import om_convergence


class _FakeResult:
    def __init__(self, single_row=None, rows=None):  # noqa: ANN001
        self._single = single_row
        self._rows = rows or []

    def single(self):
        return self._single

    def data(self):
        return self._rows

    def consume(self):
        return None


class _FakeSession:
    def __init__(self):
        self.queries: list[str] = []
        self.calls: list[dict[str, object]] = []

    def run(self, query: str, params=None):  # noqa: ANN001
        self.queries.append(query)
        self.calls.append({'query': query, 'params': params or {}})
        normalized = " ".join(query.split())

        if 'RETURN count(m) AS eligible_messages' in normalized:
            return _FakeResult({'eligible_messages': 3})
        if 'RETURN count(*) AS deleted_messages' in normalized:
            return _FakeResult({'deleted_messages': 2})
        if 'RETURN count(e) AS eligible_episodes' in normalized:
            return _FakeResult({'eligible_episodes': 4})
        if 'RETURN count(*) AS deleted_episodes' in normalized:
            return _FakeResult({'deleted_episodes': 1})
        if 'RETURN DISTINCT coalesce(j.node_id, j.uuid, \'\') AS source_node_id' in normalized:
            return _FakeResult(
                rows=[{'source_node_id': 'judgment_1', 'group_id': 's1_observational_memory'}]
            )
        if 'RETURN count(r) > 0 AS already_exists' in normalized:
            return _FakeResult({'already_exists': False})
        if 'WHERE coalesce(r.relation_root_id, \'\') = $relation_root_id' in normalized:
            return _FakeResult(None)

        return _FakeResult({'ok': True})


class OMConvergenceDeterminismTests(unittest.TestCase):
    NOW = '2026-02-25T20:00:00Z'

    def _base_node(self, **overrides):
        node = {
            'node_id': 'n1',
            'node_type': 'Friction',
            'status': 'monitoring',
            'semantic_domain': 'planning',
            'status_changed_at': '2026-01-20T00:00:00Z',
            'monitoring_started_at': '2026-01-20T00:00:00Z',
            'created_at': '2026-01-01T00:00:00Z',
            'content_embedding': [1.0, 0.0],
        }
        node.update(overrides)
        return node

    def _event(self, **overrides):
        event = {
            'event_id': 'e1',
            'emitted_at': '2026-02-10T00:00:00Z',
            'semantic_domain': 'planning',
            'content_embedding': [1.0, 0.0],
            'emitted_node_type': 'Friction',
        }
        event.update(overrides)
        return event

    def test_precedence_reopened_wins_over_monitoring_aging(self) -> None:
        node = self._base_node(status='monitoring')
        convergence_events = [self._event()]
        status_window_events = [self._event()]
        monitoring_window_events = [self._event()]

        with (
            patch.object(om_convergence, '_has_addresses', return_value=True),
            patch.object(om_convergence, '_has_fresh_addresses', return_value=True),
        ):
            target, reason = om_convergence._determine_transition(
                _FakeSession(),
                node=node,
                convergence_events=convergence_events,
                status_window_events=status_window_events,
                monitoring_window_events=monitoring_window_events,
                now_utc=self.NOW,
            )

        self.assertEqual(target, 'reopened')
        self.assertEqual(reason, 'reappearance_in_convergence_window')

    def test_precedence_open_to_monitoring_wins_before_abandoned(self) -> None:
        node = self._base_node(
            status='open',
            status_changed_at='2025-12-01T00:00:00Z',
            monitoring_started_at='2025-12-01T00:00:00Z',
        )

        with (
            patch.object(om_convergence, '_has_addresses', return_value=True),
            patch.object(om_convergence, '_has_fresh_addresses', return_value=False),
        ):
            target, reason = om_convergence._determine_transition(
                _FakeSession(),
                node=node,
                convergence_events=[],
                status_window_events=[],
                monitoring_window_events=[],
                now_utc=self.NOW,
            )

        self.assertEqual(target, 'monitoring')
        self.assertEqual(reason, 'addresses_link_detected')

    def test_monitoring_ages_to_abandoned_without_mentions(self) -> None:
        node = self._base_node(status='monitoring', monitoring_started_at='2026-01-01T00:00:00Z')

        with (
            patch.object(om_convergence, '_has_addresses', return_value=False),
            patch.object(om_convergence, '_has_fresh_addresses', return_value=False),
        ):
            target, reason = om_convergence._determine_transition(
                _FakeSession(),
                node=node,
                convergence_events=[],
                status_window_events=[],
                monitoring_window_events=[],
                now_utc=self.NOW,
            )

        self.assertEqual(target, 'abandoned')
        self.assertEqual(reason, 'monitoring_aged_without_mentions')

    def test_reopened_ages_to_abandoned_without_mentions_or_fresh_addresses(self) -> None:
        node = self._base_node(
            status='reopened',
            status_changed_at='2025-12-01T00:00:00Z',
            monitoring_started_at='2025-12-01T00:00:00Z',
        )

        with (
            patch.object(om_convergence, '_has_addresses', return_value=False),
            patch.object(om_convergence, '_has_fresh_addresses', return_value=False),
        ):
            target, reason = om_convergence._determine_transition(
                _FakeSession(),
                node=node,
                convergence_events=[],
                status_window_events=[],
                monitoring_window_events=[],
                now_utc=self.NOW,
            )

        self.assertEqual(target, 'abandoned')
        self.assertEqual(reason, 'reopened_aged_without_mentions_or_addresses')


class OMConvergenceLifecycleWriteTests(unittest.TestCase):
    def test_has_addresses_matches_omnode_judgments_not_just_legacy_label(self) -> None:
        session = _FakeSession()

        om_convergence._has_addresses(session, 'issue_1')

        joined = '\n'.join(session.queries)
        self.assertIn('MATCH (j)-[r:ADDRESSES]->(n:OMNode {node_id:$node_id})', joined)
        self.assertIn("j:OMNode AND coalesce(j.node_type, '') = 'Judgment'", joined)
        self.assertIn('r.invalid_at IS NULL', joined)

    def test_apply_transition_closed_writes_native_resolves_relation_shape(self) -> None:
        session = _FakeSession()

        om_convergence._apply_transition(
            session,
            node_id='issue_1',
            node_type='Friction',
            target_status='closed',
            now_utc='2026-03-10T00:00:00Z',
        )

        joined = '\n'.join(session.queries)
        self.assertIn('n.closed_at = $now_iso', joined)
        self.assertIn('n.invalid_at = $now_iso', joined)
        self.assertIn("n.lifecycle_status = 'invalidated'", joined)
        self.assertIn('MATCH (source)-[r:ADDRESSES]->(target:OMNode {node_id:$node_id})', joined)
        self.assertIn('r.invalid_at = $now_iso', joined)
        self.assertIn('MATCH (j)-[a:ADDRESSES]->(n:OMNode {node_id:$node_id})', joined)
        self.assertIn('CREATE (j)-[r:RESOLVES]->(n)', joined)
        self.assertIn('r.relation_root_id = $relation_root_id', joined)
        self.assertIn('r.group_id = $group_id', joined)
        self.assertIn('r.transition_basis = \'convergence_resolution\'', joined)
        self.assertIn('r.lineage_parent_relation_id = CASE', joined)
        self.assertIn('r.source_node_id = coalesce(j.node_id, j.uuid, $source_node_id)', joined)
        self.assertIn('r.target_node_id = coalesce(n.node_id, n.uuid, $node_id)', joined)

    def test_apply_transition_reopened_invalidates_active_resolves_relations(self) -> None:
        session = _FakeSession()

        om_convergence._apply_transition(
            session,
            node_id='issue_1',
            node_type='Friction',
            target_status='reopened',
            now_utc='2026-03-11T00:00:00Z',
        )

        joined = '\n'.join(session.queries)
        self.assertIn('n.previous_status = n.status', joined)
        self.assertIn('n.reopened_at = $now_iso', joined)
        self.assertIn('n.invalid_at = NULL', joined)
        self.assertIn("n.lifecycle_status = 'asserted'", joined)
        self.assertIn("n.transition_cause = 'reopened_after_reappearance'", joined)
        self.assertIn('MATCH (source)-[r:RESOLVES]->(target:OMNode {node_id:$node_id})', joined)
        self.assertIn('r.invalid_at = $now_iso', joined)
        self.assertIn('r.transition_basis = \'node_status_transition\'', joined)
        self.assertIn('r.invalidated_by_node_id = $node_id', joined)


class OMConvergenceNeo4jFallbackPolicyTests(unittest.TestCase):
    def test_non_dev_requires_opt_in_for_fallback_file(self) -> None:
        with patch.dict(os.environ, {"NODE_ENV": "production"}, clear=True):
            self.assertFalse(om_convergence._allow_neo4j_env_fallback())

    def test_non_dev_opt_in_enables_fallback_file(self) -> None:
        with patch.dict(
            os.environ,
            {
                "NODE_ENV": "production",
                "OM_NEO4J_ENV_FALLBACK_NON_DEV": "1",
            },
            clear=True,
        ):
            self.assertTrue(om_convergence._allow_neo4j_env_fallback())

    def test_local_dev_defaults_to_allow_fallback_file(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(om_convergence._allow_neo4j_env_fallback())


class OMConvergenceWatermarkAndGCTests(unittest.TestCase):
    def test_finalize_cycle_partial_updates_cursor_only(self) -> None:
        session = _FakeSession()
        om_convergence._finalize_cycle(
            session,
            cycle_started_at='2026-02-25T20:00:00Z',
            last_processed_node_id='node-0500',
            has_more_nodes=True,
        )

        joined = '\n'.join(session.queries)
        self.assertIn('SET s.next_node_cursor = $next_node_cursor', joined)
        self.assertNotIn('s.last_convergence_at = $cycle_started_at', joined)

    def test_finalize_cycle_complete_advances_last_convergence(self) -> None:
        session = _FakeSession()
        om_convergence._finalize_cycle(
            session,
            cycle_started_at='2026-02-25T20:00:00Z',
            last_processed_node_id='node-0500',
            has_more_nodes=False,
        )

        joined = '\n'.join(session.queries)
        self.assertIn('s.last_convergence_at = $cycle_started_at', joined)
        self.assertIn('s.next_node_cursor = NULL', joined)

    def test_gc_dry_run_never_executes_deletes(self) -> None:
        session = _FakeSession()
        summary = om_convergence.run_gc(session, 90, dry_run=True)

        self.assertTrue(summary['dry_run'])
        self.assertEqual(summary['eligible_messages'], 3)
        self.assertEqual(summary['deleted_messages'], 0)
        self.assertEqual(summary['eligible_episodes'], 4)
        self.assertEqual(summary['deleted_episodes'], 0)

        joined = '\n'.join(session.queries)
        self.assertNotIn('DETACH DELETE m', joined)
        self.assertNotIn('DETACH DELETE e', joined)


if __name__ == '__main__':
    unittest.main()
