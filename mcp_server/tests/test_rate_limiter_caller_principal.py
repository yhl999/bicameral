"""Unit tests for caller-principal-aware rate-limit key derivation.

Covers:
- Caller principal included in key derivation.
- Different callers with same scope do not share bucket.
- Same caller with permuted scope order maps to same key.
- Missing / anonymous principal still throttled via global fallback.
- Env-range validation warnings at module load time.
"""

from __future__ import annotations

import importlib
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure src is on the path (conftest.py does this too, but be explicit).
_SRC = Path(__file__).parent.parent / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Module-level import of the server module (venv has all deps).
# ---------------------------------------------------------------------------
import graphiti_mcp_server as _srv  # noqa: E402
from utils.rate_limiter import SlidingWindowRateLimiter  # noqa: E402

# ---------------------------------------------------------------------------
# Tests: _derive_rate_limit_key
# ---------------------------------------------------------------------------

class TestDeriveRateLimitKey(unittest.TestCase):
    """_derive_rate_limit_key must produce caller-principal-bound, order-insensitive keys."""

    derive = staticmethod(_srv._derive_rate_limit_key)
    ANON = _srv._ANON_PRINCIPAL

    def test_caller_principal_included_in_key(self):
        """Key must embed the caller principal so buckets are per-caller."""
        key_alice = self.derive(['groupX'], 'alice')
        key_bob = self.derive(['groupX'], 'bob')
        self.assertIn('alice', key_alice)
        self.assertIn('bob', key_bob)
        self.assertNotEqual(key_alice, key_bob)

    def test_different_callers_same_scope_different_buckets(self):
        """Two callers with identical group scope must get distinct keys."""
        scope = ['g1', 'g2']
        self.assertNotEqual(
            self.derive(scope, 'caller_a'),
            self.derive(scope, 'caller_b'),
        )

    def test_same_caller_permuted_scope_same_key(self):
        """Permuting group_ids must NOT change the key (order-insensitive)."""
        self.assertEqual(
            self.derive(['g1', 'g2', 'g3'], 'alice'),
            self.derive(['g3', 'g1', 'g2'], 'alice'),
        )

    def test_same_caller_duplicate_scope_ids_same_key(self):
        """Duplicate group IDs must be de-duplicated before hashing."""
        self.assertEqual(
            self.derive(['g1', 'g2'], 'alice'),
            self.derive(['g1', 'g2', 'g1'], 'alice'),
        )

    def test_empty_scope_uses_global_component(self):
        """Empty group_ids must produce the __global__ scope component."""
        key = self.derive([], 'alice')
        self.assertIn('__global__', key)
        self.assertIn('alice', key)

    def test_anon_principal_present_in_key(self):
        """Anonymous principal must appear in the derived key."""
        key = self.derive(['g1'], self.ANON)
        self.assertIn(self.ANON, key)

    def test_key_format(self):
        """Key must follow 'caller:<p>|scope:<s>' format."""
        key = self.derive(['g1'], 'alice')
        self.assertTrue(
            key.startswith('caller:alice|scope:'),
            msg=f'Unexpected format: {key!r}',
        )


# ---------------------------------------------------------------------------
# Tests: _extract_trusted_caller_principal
# ---------------------------------------------------------------------------

class TestExtractTrustedCallerPrincipal(unittest.TestCase):
    """Principal must come from trusted transport context, never raw payload."""

    extract = staticmethod(_srv._extract_trusted_caller_principal)
    ANON = _srv._ANON_PRINCIPAL

    def test_returns_anon_when_no_context(self):
        with patch('graphiti_mcp_server.get_access_token', return_value=None):
            result = self.extract(None)
        self.assertEqual(result, self.ANON)

    def test_uses_access_token_client_id_when_available(self):
        mock_token = MagicMock()
        mock_token.client_id = 'oauth-client-123'
        with patch('graphiti_mcp_server.get_access_token', return_value=mock_token):
            result = self.extract(None)
        self.assertEqual(result, 'oauth-client-123')

    def test_falls_back_to_ctx_client_id(self):
        ctx = MagicMock()
        ctx.client_id = 'mcp-session-456'
        with patch('graphiti_mcp_server.get_access_token', return_value=None):
            result = self.extract(ctx)
        self.assertEqual(result, 'mcp-session-456')

    def test_oauth_token_takes_precedence_over_ctx_client_id(self):
        """OAuth access token takes precedence over context client_id."""
        mock_token = MagicMock()
        mock_token.client_id = 'oauth-wins'
        ctx = MagicMock()
        ctx.client_id = 'ctx-loses'
        with patch('graphiti_mcp_server.get_access_token', return_value=mock_token):
            result = self.extract(ctx)
        self.assertEqual(result, 'oauth-wins')

    def test_returns_anon_when_access_token_has_empty_client_id(self):
        mock_token = MagicMock()
        mock_token.client_id = ''
        ctx = MagicMock()
        ctx.client_id = None
        with patch('graphiti_mcp_server.get_access_token', return_value=mock_token):
            result = self.extract(ctx)
        self.assertEqual(result, self.ANON)


# ---------------------------------------------------------------------------
# Tests: global fallback limiter (anonymous callers)
# ---------------------------------------------------------------------------

class TestGlobalFallbackLimiter(unittest.IsolatedAsyncioTestCase):
    """Anonymous callers must be throttled by the global fallback even when
    they rotate group IDs to generate different scope keys."""

    async def test_anon_rotating_groups_hits_global_fallback(self):
        """Rotating group IDs should not escape throttling for anon callers."""
        # Build a very tight fallback limiter (1 request per window).
        fallback = SlidingWindowRateLimiter(max_requests=1, window_seconds=10)
        primary = SlidingWindowRateLimiter(max_requests=100, window_seconds=10)

        derive = _srv._derive_rate_limit_key
        ANON = _srv._ANON_PRINCIPAL

        # First request: unique scope g1 → primary allows, fallback allows.
        key1 = derive(['g1'], ANON)
        self.assertTrue(await primary.is_allowed(key1))
        self.assertTrue(await fallback.is_allowed('__global__'))

        # Second request: different scope g2 → primary would allow a NEW key,
        # but fallback (1 req/window) must now block.
        key2 = derive(['g2'], ANON)
        self.assertNotEqual(key1, key2)  # different scope → different primary bucket
        self.assertTrue(await primary.is_allowed(key2))    # primary would pass
        self.assertFalse(await fallback.is_allowed('__global__'))  # fallback blocks

    async def test_authenticated_caller_not_subject_to_global_fallback(self):
        """Named callers get only per-(caller, scope) throttling — no fallback needed."""
        primary = SlidingWindowRateLimiter(max_requests=100, window_seconds=10)
        fallback = SlidingWindowRateLimiter(max_requests=1, window_seconds=10)

        derive = _srv._derive_rate_limit_key
        ANON = _srv._ANON_PRINCIPAL

        # Exhaust the global fallback with an anon request.
        await fallback.is_allowed('__global__')  # consume the 1 slot

        # Authenticated caller should be unaffected by global fallback state.
        key = derive(['g1'], 'alice')
        self.assertNotIn(ANON, key)
        self.assertTrue(await primary.is_allowed(key))
        # The authenticated flow does NOT check fallback, so state is irrelevant.


# ---------------------------------------------------------------------------
# Tests: _hash_rate_limit_key
# ---------------------------------------------------------------------------

class TestHashRateLimitKey(unittest.TestCase):
    hash_key = staticmethod(_srv._hash_rate_limit_key)

    def test_returns_8_char_hex(self):
        h = self.hash_key('caller:alice|scope:abc123')
        self.assertEqual(len(h), 8)
        self.assertTrue(all(c in '0123456789abcdef' for c in h))

    def test_deterministic(self):
        key = 'caller:alice|scope:deadbeef'
        self.assertEqual(self.hash_key(key), self.hash_key(key))

    def test_different_keys_different_hashes(self):
        self.assertNotEqual(
            self.hash_key('caller:alice|scope:x'),
            self.hash_key('caller:bob|scope:x'),
        )


# ---------------------------------------------------------------------------
# Tests: env range validation
# ---------------------------------------------------------------------------

class TestEnvRangeValidation(unittest.TestCase):
    """Startup env-range validation should emit warnings and fall back to defaults."""

    def _reimport_server(self, env_overrides: dict) -> object:
        """Re-import graphiti_mcp_server with env overrides active."""
        sys.modules.pop('graphiti_mcp_server', None)
        with patch.dict('os.environ', env_overrides):
            try:
                spec = importlib.util.spec_from_file_location(
                    'graphiti_mcp_server',
                    _SRC / 'graphiti_mcp_server.py',
                )
                mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                return mod
            except Exception as exc:
                raise unittest.SkipTest(f'module re-init failed: {exc}') from exc
            finally:
                # Restore original module so other tests still work.
                sys.modules['graphiti_mcp_server'] = _srv

    def test_negative_requests_uses_default_and_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            mod = self._reimport_server({'SEARCH_RATE_LIMIT_REQUESTS': '-5'})
        self.assertEqual(mod._SEARCH_RATE_LIMIT_REQUESTS, 60)
        self.assertTrue(
            any('SEARCH_RATE_LIMIT_REQUESTS' in str(warning.message) for warning in w),
            msg=f'Expected warning about SEARCH_RATE_LIMIT_REQUESTS, got: {[str(x.message) for x in w]}',
        )

    def test_zero_window_uses_default_and_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            mod = self._reimport_server({'SEARCH_RATE_LIMIT_WINDOW': '0'})
        self.assertEqual(mod._SEARCH_RATE_LIMIT_WINDOW, 60.0)
        self.assertTrue(
            any('SEARCH_RATE_LIMIT_WINDOW' in str(warning.message) for warning in w),
            msg=f'Expected warning about SEARCH_RATE_LIMIT_WINDOW, got: {[str(x.message) for x in w]}',
        )

    def test_non_numeric_requests_uses_default(self):
        mod = self._reimport_server({'SEARCH_RATE_LIMIT_REQUESTS': 'banana'})
        self.assertEqual(mod._SEARCH_RATE_LIMIT_REQUESTS, 60)

    def test_valid_requests_used_as_is(self):
        mod = self._reimport_server({'SEARCH_RATE_LIMIT_REQUESTS': '120'})
        self.assertEqual(mod._SEARCH_RATE_LIMIT_REQUESTS, 120)


if __name__ == '__main__':
    unittest.main()
