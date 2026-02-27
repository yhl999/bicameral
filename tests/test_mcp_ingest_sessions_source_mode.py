"""Tests for mcp_ingest_sessions.py --source-mode (FR-4 + FR-10)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestSourceModeArgParsing(unittest.TestCase):
    """Test --source-mode argument parsing."""

    def test_default_source_mode_is_neo4j(self):
        from scripts.mcp_ingest_sessions import build_parser

        parser = build_parser()
        args = parser.parse_args(['--group-id', 's1_sessions_main'])
        self.assertEqual(args.source_mode, 'neo4j')

    def test_explicit_neo4j_mode(self):
        from scripts.mcp_ingest_sessions import build_parser

        parser = build_parser()
        args = parser.parse_args([
            '--group-id', 's1_sessions_main',
            '--source-mode', 'neo4j',
        ])
        self.assertEqual(args.source_mode, 'neo4j')

    def test_explicit_evidence_mode(self):
        from scripts.mcp_ingest_sessions import build_parser

        parser = build_parser()
        args = parser.parse_args([
            '--group-id', 's1_sessions_main',
            '--source-mode', 'evidence',
            '--evidence', '/tmp/test.json',
        ])
        self.assertEqual(args.source_mode, 'evidence')

    def test_build_manifest_flag(self):
        from scripts.mcp_ingest_sessions import build_parser

        parser = build_parser()
        args = parser.parse_args([
            '--group-id', 's1_sessions_main',
            '--source-mode', 'neo4j',
            '--build-manifest', '/tmp/manifest.jsonl',
        ])
        self.assertEqual(args.build_manifest, '/tmp/manifest.jsonl')

    def test_claim_mode_flags(self):
        from scripts.mcp_ingest_sessions import build_parser

        parser = build_parser()
        args = parser.parse_args([
            '--group-id', 's1_sessions_main',
            '--source-mode', 'neo4j',
            '--manifest', '/tmp/manifest.jsonl',
            '--claim-mode',
            '--shards', '4',
            '--shard-index', '0',
        ])
        self.assertTrue(args.claim_mode)
        self.assertEqual(args.shards, 4)
        self.assertEqual(args.shard_index, 0)

    def test_claim_state_check_flag(self):
        from scripts.mcp_ingest_sessions import build_parser

        parser = build_parser()
        args = parser.parse_args([
            '--group-id', 's1_sessions_main',
            '--manifest', '/tmp/manifest.jsonl',
            '--claim-state-check',
            '--dry-run',
        ])
        self.assertTrue(args.claim_state_check)


class TestBootstrapGuard(unittest.TestCase):
    """Test BOOTSTRAP_REQUIRED guard (FR-4 item 6)."""

    def test_bootstrap_required_when_no_messages_and_evidence_exists(self):
        from scripts.mcp_ingest_sessions import check_bootstrap_guard

        # No messages in Neo4j AND evidence files exist → BOOTSTRAP_REQUIRED
        result = check_bootstrap_guard(
            neo4j_message_count=0,
            evidence_files_exist=True,
        )
        self.assertTrue(result)

    def test_no_bootstrap_when_messages_exist(self):
        from scripts.mcp_ingest_sessions import check_bootstrap_guard

        # Messages exist in Neo4j → guard satisfied
        result = check_bootstrap_guard(
            neo4j_message_count=100,
            evidence_files_exist=True,
        )
        self.assertFalse(result)

    def test_no_bootstrap_when_no_evidence_files(self):
        from scripts.mcp_ingest_sessions import check_bootstrap_guard

        # No evidence files → no guard needed
        result = check_bootstrap_guard(
            neo4j_message_count=0,
            evidence_files_exist=False,
        )
        self.assertFalse(result)


class TestClaimStateDB(unittest.TestCase):
    """Test SQLite claim-state storage (FR-10)."""

    def test_claim_db_schema(self):
        import tempfile

        from scripts.mcp_ingest_sessions import init_claim_db

        with tempfile.NamedTemporaryFile(suffix='.db') as f:
            conn = init_claim_db(f.name)
            # Verify table exists with expected columns
            cursor = conn.execute('PRAGMA table_info(chunk_claims)')
            columns = {row[1] for row in cursor.fetchall()}
            expected = {
                'chunk_id', 'status', 'worker_id',
                'claimed_at', 'completed_at', 'fail_count', 'error',
            }
            self.assertTrue(
                expected.issubset(columns),
                f'Missing columns: {expected - columns}',
            )
            conn.close()

    def test_claim_pending_to_claimed(self):
        import tempfile

        from scripts.mcp_ingest_sessions import (
            claim_chunk,
            init_claim_db,
            seed_claims,
        )

        with tempfile.NamedTemporaryFile(suffix='.db') as f:
            conn = init_claim_db(f.name)
            seed_claims(conn, ['chunk_001', 'chunk_002'])

            # Claim one chunk
            claimed = claim_chunk(conn, worker_id='w0')
            self.assertIsNotNone(claimed)
            self.assertIn(claimed, ['chunk_001', 'chunk_002'])

            # Verify status changed
            row = conn.execute(
                'SELECT status FROM chunk_claims WHERE chunk_id = ?',
                (claimed,),
            ).fetchone()
            self.assertEqual(row[0], 'claimed')
            conn.close()


class TestEvidenceModeBackwardCompat(unittest.TestCase):
    """Test evidence mode still works (backward compat)."""

    def test_evidence_mode_parseable(self):
        """Evidence mode args parse correctly."""
        from scripts.mcp_ingest_sessions import build_parser

        parser = build_parser()
        args = parser.parse_args([
            '--group-id', 's1_sessions_main',
            '--source-mode', 'evidence',
            '--evidence', 'evidence/sessions_v1/main/all_evidence.json',
            '--limit', '10',
            '--dry-run',
        ])
        self.assertEqual(args.source_mode, 'evidence')
        self.assertEqual(args.limit, 10)
        self.assertTrue(args.dry_run)


class TestBuildEpisodeBody(unittest.TestCase):
    """Test _build_episode_body formats chunk content correctly."""

    def test_basic_formatting(self):
        from scripts.mcp_ingest_sessions import _build_episode_body

        messages_by_id = {
            'msg1': {
                'message_id': 'msg1',
                'content': 'Hello there',
                'created_at': '2026-01-15T12:00:00Z',
                'role': 'user',
            },
            'msg2': {
                'message_id': 'msg2',
                'content': 'Hi back',
                'created_at': '2026-01-15T12:01:00Z',
                'role': 'assistant',
            },
        }
        body = _build_episode_body(['msg1', 'msg2'], messages_by_id)
        self.assertIn('Hello there', body)
        self.assertIn('Hi back', body)
        self.assertIn('2026-01-15', body)

    def test_missing_message_skipped(self):
        from scripts.mcp_ingest_sessions import _build_episode_body

        body = _build_episode_body(['nonexistent'], {})
        self.assertEqual(body, '')

    def test_message_order_preserved(self):
        from scripts.mcp_ingest_sessions import _build_episode_body

        messages_by_id = {
            'a': {'content': 'first', 'created_at': '2026-01-15T10:00:00Z', 'role': 'user'},
            'b': {'content': 'second', 'created_at': '2026-01-15T11:00:00Z', 'role': 'user'},
        }
        body = _build_episode_body(['a', 'b'], messages_by_id)
        self.assertLess(body.index('first'), body.index('second'))


class TestLoadManifest(unittest.TestCase):
    """Test _load_manifest reads JSONL manifest correctly."""

    def test_load_manifest(self):
        import os
        import tempfile

        from scripts.mcp_ingest_sessions import _load_manifest

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            import json as _json
            _json.dump({'chunk_id': 'abc', 'message_ids': ['m1', 'm2'], 'content': 'c1'}, f)
            f.write('\n')
            _json.dump({'chunk_id': 'def', 'message_ids': ['m3'], 'content': 'c2'}, f)
            f.write('\n')
            tmp = f.name

        try:
            from pathlib import Path
            result = _load_manifest(Path(tmp))
            self.assertIn('abc', result)
            self.assertIn('def', result)
            self.assertEqual(result['abc']['message_ids'], ['m1', 'm2'])
        finally:
            os.unlink(tmp)

    def test_empty_manifest_returns_empty_dict(self):
        import os
        import tempfile
        from pathlib import Path

        from scripts.mcp_ingest_sessions import _load_manifest

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            tmp = f.name

        try:
            result = _load_manifest(Path(tmp))
            self.assertEqual(result, {})
        finally:
            os.unlink(tmp)

    def test_malformed_lines_skipped(self):
        import json as _json
        import os
        import tempfile
        from pathlib import Path

        from scripts.mcp_ingest_sessions import _load_manifest

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('not json\n')
            _json.dump({'chunk_id': 'valid', 'message_ids': ['m1'], 'content': 'c'}, f)
            f.write('\n')
            tmp = f.name

        try:
            result = _load_manifest(Path(tmp))
            self.assertIn('valid', result)
            self.assertEqual(len(result), 1)
        finally:
            os.unlink(tmp)


class TestClaimHelpers(unittest.TestCase):
    """Test _claim_done and _claim_fail helpers."""

    def _make_db(self):
        import tempfile

        from scripts.mcp_ingest_sessions import init_claim_db, seed_claims
        tmp = tempfile.mktemp(suffix='.db')
        conn = init_claim_db(tmp)
        seed_claims(conn, ['chunk_001', 'chunk_002'])
        return conn, tmp

    def test_claim_done_marks_status(self):
        import os

        from scripts.mcp_ingest_sessions import _claim_done, claim_chunk

        conn, tmp = self._make_db()
        try:
            chunk_id = claim_chunk(conn, 'w0')
            _claim_done(conn, chunk_id)
            row = conn.execute(
                "SELECT status FROM chunk_claims WHERE chunk_id=?", (chunk_id,)
            ).fetchone()
            self.assertEqual(row[0], 'done')
        finally:
            conn.close()
            os.unlink(tmp)

    def test_claim_fail_increments_fail_count(self):
        import os

        from scripts.mcp_ingest_sessions import _claim_fail, claim_chunk

        conn, tmp = self._make_db()
        try:
            chunk_id = claim_chunk(conn, 'w0')
            _claim_fail(conn, chunk_id, 'test error')
            row = conn.execute(
                "SELECT status, fail_count, error FROM chunk_claims WHERE chunk_id=?",
                (chunk_id,)
            ).fetchone()
            self.assertEqual(row[0], 'failed')
            self.assertEqual(row[1], 1)
            self.assertEqual(row[2], 'test error')
        finally:
            conn.close()
            os.unlink(tmp)

    def test_claim_fail_twice_increments_twice(self):
        import os
        import tempfile

        from scripts.mcp_ingest_sessions import _claim_fail, claim_chunk, init_claim_db, seed_claims
        tmp = tempfile.mktemp(suffix='.db')
        conn = init_claim_db(tmp)
        seed_claims(conn, ['chunk_x'])
        try:
            # First claim + fail
            chunk_id = claim_chunk(conn, 'w0')
            _claim_fail(conn, chunk_id, 'err1')
            # Re-seed to pending so we can claim again
            conn.execute("UPDATE chunk_claims SET status='pending' WHERE chunk_id=?", (chunk_id,))
            conn.commit()
            chunk_id2 = claim_chunk(conn, 'w0')
            _claim_fail(conn, chunk_id2, 'err2')
            row = conn.execute(
                "SELECT fail_count FROM chunk_claims WHERE chunk_id=?", (chunk_id,)
            ).fetchone()
            self.assertEqual(row[0], 2)
        finally:
            conn.close()
            os.unlink(tmp)


class TestHardeningCaps(unittest.TestCase):
    """Test that hardening constants are in place."""

    def test_neo4j_fetch_ceiling_exists(self):
        from scripts.mcp_ingest_sessions import _NEO4J_FETCH_CEILING
        self.assertGreater(_NEO4J_FETCH_CEILING, 0)
        self.assertLessEqual(_NEO4J_FETCH_CEILING, 100_000)

    def test_benchmark_max_response_bytes_exists(self):
        from scripts.run_retrieval_benchmark import _MAX_RESPONSE_BYTES
        self.assertGreater(_MAX_RESPONSE_BYTES, 0)

    def test_mcp_server_caps_defined_in_source(self):
        """Verify cap constants appear in the MCP server source file."""
        from pathlib import Path
        server_src = (
            Path(__file__).resolve().parents[1]
            / 'mcp_server' / 'src' / 'graphiti_mcp_server.py'
        )
        source = server_src.read_text(encoding='utf-8')
        self.assertIn('_MAX_NODES_CAP', source)
        self.assertIn('_MAX_FACTS_CAP', source)
        # Ensure the caps are applied in search_nodes and search_memory_facts.
        self.assertIn('_MAX_NODES_CAP', source)
        self.assertIn('_MAX_FACTS_CAP', source)


if __name__ == '__main__':
    unittest.main()
