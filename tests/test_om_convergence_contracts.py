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

    def run(self, query: str, params=None):  # noqa: ANN001
        self.queries.append(query)
        normalized = " ".join(query.split())

        if 'RETURN count(m) AS eligible_messages' in normalized:
            return _FakeResult({'eligible_messages': 3})
        if 'RETURN count(*) AS deleted_messages' in normalized:
            return _FakeResult({'deleted_messages': 2})
        if 'RETURN count(e) AS eligible_episodes' in normalized:
            return _FakeResult({'eligible_episodes': 4})
        if 'RETURN count(*) AS deleted_episodes' in normalized:
            return _FakeResult({'deleted_episodes': 1})

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
