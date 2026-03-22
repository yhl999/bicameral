"""Integration tests for lane registry wiring into typed retrieval.

Verifies that:
1. _intersect_source_lane_filter resolves group_ids through the lane registry
2. Scope values are filtered from source_lane filters
3. The write-path guards in _derive_source_lane and build_object_from_candidate_fact
   reject scope-as-lane
"""

from types import SimpleNamespace
from unittest import mock

from mcp_server.src.services.change_ledger import (
    _validate_source_lane_value,
    build_object_from_candidate_fact,
)
from mcp_server.src.services.lane_registry import (
    LaneRegistry,
    get_lane_registry,
    set_lane_registry,
)
from tests.helpers_mcp_import import load_graphiti_mcp_server

server = load_graphiti_mcp_server()


# ── Helpers ───────────────────────────────────────────────────────────────────

_SAMPLE_ALIASES = {
    'sessions_main': ['s1_sessions_main'],
    'observational_memory': ['s1_observational_memory'],
    'curated': ['s1_curated_refs'],
    'chatgpt': ['s1_chatgpt_history'],
    'all': [],
}

_REGISTRY = LaneRegistry(_SAMPLE_ALIASES)


def _with_registry(fn):
    """Decorator that installs and restores the test lane registry."""
    def wrapper(*args, **kwargs):
        original = get_lane_registry()
        set_lane_registry(_REGISTRY)
        try:
            return fn(*args, **kwargs)
        finally:
            set_lane_registry(original)
    return wrapper


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


# ── _intersect_source_lane_filter tests ───────────────────────────────────────

@_with_registry
def test_intersect_filter_resolves_canonical_group_ids():
    """Canonical group_ids pass through the registry unchanged."""
    result = server._intersect_source_lane_filter(
        metadata_filters=None,
        effective_group_ids=['s1_sessions_main', 's1_curated_refs'],
    )
    assert result['source_lane'] == {'in': ['s1_sessions_main', 's1_curated_refs']}


@_with_registry
def test_intersect_filter_strips_scope_values():
    """Scope values like 'private' are filtered out of the source_lane filter."""
    result = server._intersect_source_lane_filter(
        metadata_filters=None,
        effective_group_ids=['s1_sessions_main', 'private', 's1_curated_refs'],
    )
    assert result['source_lane'] == {'in': ['s1_sessions_main', 's1_curated_refs']}


@_with_registry
def test_intersect_filter_all_scope_returns_no_lane_filter():
    """When all group_ids are scope values, no source_lane filter is applied."""
    result = server._intersect_source_lane_filter(
        metadata_filters=None,
        effective_group_ids=['private', 'public', 'owner'],
    )
    # No source_lane key should be present (or it should be absent from filters)
    assert 'source_lane' not in result


@_with_registry
def test_intersect_filter_empty_group_ids_no_filter():
    """Empty group_ids (all-lanes) returns filters without source_lane constraint."""
    result = server._intersect_source_lane_filter(
        metadata_filters=None,
        effective_group_ids=[],
    )
    assert 'source_lane' not in result


@_with_registry
def test_intersect_filter_preserves_existing_metadata():
    """Non-source_lane metadata filters are preserved."""
    result = server._intersect_source_lane_filter(
        metadata_filters={'policy_scope': {'eq': 'private'}, 'fact_type': 'preference'},
        effective_group_ids=['s1_sessions_main'],
    )
    assert result['policy_scope'] == {'eq': 'private'}
    assert result['fact_type'] == 'preference'
    assert result['source_lane'] == {'in': ['s1_sessions_main']}


@_with_registry
def test_intersect_filter_caller_lane_intersection():
    """When caller supplies source_lane, it is intersected with resolved canonical lanes."""
    result = server._intersect_source_lane_filter(
        metadata_filters={'source_lane': {'in': ['s1_sessions_main', 's1_curated_refs', 'bogus']}},
        effective_group_ids=['s1_sessions_main', 's1_observational_memory'],
    )
    # Only s1_sessions_main is in both the caller's request and the resolved lanes
    assert result['source_lane'] == {'in': ['s1_sessions_main']}


@_with_registry
def test_intersect_filter_unknown_group_ids_pass_through():
    """Unknown-but-valid group_ids pass through for forward-compat."""
    result = server._intersect_source_lane_filter(
        metadata_filters=None,
        effective_group_ids=['s1_sessions_main', 'experimental_v2_replay_42'],
    )
    lanes = result['source_lane']['in']
    assert 's1_sessions_main' in lanes
    assert 'experimental_v2_replay_42' in lanes


# ── _validate_source_lane_value tests (change_ledger.py) ─────────────────────

@_with_registry
def test_validate_source_lane_value_rejects_private():
    assert _validate_source_lane_value('private') is None


@_with_registry
def test_validate_source_lane_value_accepts_canonical():
    assert _validate_source_lane_value('s1_sessions_main') == 's1_sessions_main'


@_with_registry
def test_validate_source_lane_value_accepts_unknown():
    assert _validate_source_lane_value('some_new_lane') == 'some_new_lane'


@_with_registry
def test_validate_source_lane_value_none():
    assert _validate_source_lane_value(None) is None


@_with_registry
def test_validate_source_lane_value_empty():
    assert _validate_source_lane_value('') is None
    assert _validate_source_lane_value('  ') is None


# ── build_object_from_candidate_fact scope guard ──────────────────────────────

@_with_registry
def test_build_object_rejects_scope_as_source_lane():
    """build_object_from_candidate_fact should NOT store 'private' as source_lane."""
    obj = build_object_from_candidate_fact(
        candidate_id='cand-test-001',
        fact={
            'assertion_type': 'state_fact',
            'subject': 'test',
            'predicate': 'pref',
            'value': 'yes',
            'scope': 'private',
            'source_lane': 'private',  # Scope leaked into lane — should be rejected
            'evidence_refs': [
                {
                    'source_key': 'test_source',
                    'source_system': 'test',
                    'evidence_id': 'ev001',
                    'observed_at': _now(),
                }
            ],
        },
        policy_version='v1',
        recorded_at=_now(),
    )
    # source_lane should be None (rejected), not 'private'
    assert obj.source_lane != 'private', (
        f'source_lane should not be "private" (scope leaked into lane); got {obj.source_lane!r}'
    )


@_with_registry
def test_build_object_preserves_valid_source_lane():
    """build_object_from_candidate_fact preserves a valid canonical source_lane."""
    obj = build_object_from_candidate_fact(
        candidate_id='cand-test-002',
        fact={
            'assertion_type': 'state_fact',
            'subject': 'test',
            'predicate': 'pref',
            'value': 'yes',
            'scope': 'private',
            'source_lane': 's1_sessions_main',
            'evidence_refs': [
                {
                    'source_key': 'test_source',
                    'source_system': 'test',
                    'evidence_id': 'ev002',
                    'observed_at': _now(),
                }
            ],
        },
        policy_version='v1',
        recorded_at=_now(),
    )
    assert obj.source_lane == 's1_sessions_main'


# ── _derive_source_lane scope guard ──────────────────────────────────────────

@_with_registry
def test_derive_source_lane_rejects_scope_value():
    """_derive_source_lane should return None when group_id is a scope value."""
    from mcp_server.src.routers.memory import _derive_source_lane

    fake_config = SimpleNamespace(
        graphiti=SimpleNamespace(group_id='private'),
    )
    with (
        mock.patch('mcp_server.src.routers.memory._derive_source_lane.__module__', 'mcp_server.src.routers.memory'),
        mock.patch('mcp_server.src.graphiti_mcp_server.config', fake_config),
    ):
        result = _derive_source_lane()
    assert result is None, f'_derive_source_lane should reject scope value "private"; got {result!r}'


@_with_registry
def test_derive_source_lane_passes_canonical_value():
    """_derive_source_lane should return the group_id when it's a valid lane."""
    from mcp_server.src.routers.memory import _derive_source_lane

    fake_config = SimpleNamespace(
        graphiti=SimpleNamespace(group_id='s1_sessions_main'),
    )
    with mock.patch('mcp_server.src.graphiti_mcp_server.config', fake_config):
        result = _derive_source_lane()
    assert result == 's1_sessions_main'
