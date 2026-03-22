"""Unit tests for the canonical lane identity registry.

Covers the three-way separation:
  1. Canonical semantic lane identity (source_lane)
  2. Graph / corpus / experimental group_ids
  3. Visibility / policy scope (private/public/owner — NOT lane identity)
"""

import logging

from mcp_server.src.services.lane_registry import (
    SCOPE_NOT_LANE,
    LaneRegistry,
    get_lane_registry,
    init_lane_registry,
    set_lane_registry,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

_SAMPLE_ALIASES = {
    'sessions_main': ['s1_sessions_main'],
    'observational_memory': ['s1_observational_memory'],
    'curated': ['s1_curated_refs'],
    'chatgpt': ['s1_chatgpt_history'],
    'learning': ['learning_self_audit'],
    'all': [],  # meta-alias: maps to all lanes
}


def _make_registry(**kwargs) -> LaneRegistry:
    return LaneRegistry(_SAMPLE_ALIASES, **kwargs)


# ── SCOPE_NOT_LANE constant ──────────────────────────────────────────────────

def test_scope_not_lane_contains_expected_values():
    assert 'private' in SCOPE_NOT_LANE
    assert 'public' in SCOPE_NOT_LANE
    assert 'owner' in SCOPE_NOT_LANE
    assert 'global' in SCOPE_NOT_LANE
    assert 'all' in SCOPE_NOT_LANE


def test_scope_not_lane_is_immutable():
    assert isinstance(SCOPE_NOT_LANE, frozenset)


# ── Constructor ───────────────────────────────────────────────────────────────

def test_canonical_lanes_derived_from_aliases():
    registry = _make_registry()
    assert 's1_sessions_main' in registry.canonical_lanes
    assert 's1_observational_memory' in registry.canonical_lanes
    assert 's1_curated_refs' in registry.canonical_lanes
    assert 's1_chatgpt_history' in registry.canonical_lanes
    assert 'learning_self_audit' in registry.canonical_lanes


def test_meta_alias_all_does_not_add_canonical_lane():
    """The 'all' alias maps to [] and should not add any lane to canonical set."""
    registry = _make_registry()
    assert 'all' not in registry.canonical_lanes


def test_scope_values_excluded_from_canonical_lanes():
    """Even if lane_aliases accidentally includes scope values, they are excluded."""
    aliases = {
        'sessions_main': ['s1_sessions_main'],
        'broken': ['private', 'public'],
    }
    registry = LaneRegistry(aliases)
    assert 's1_sessions_main' in registry.canonical_lanes
    assert 'private' not in registry.canonical_lanes
    assert 'public' not in registry.canonical_lanes


def test_extra_canonical_lanes():
    registry = _make_registry(extra_canonical_lanes=frozenset({'engineering_learnings'}))
    assert 'engineering_learnings' in registry.canonical_lanes


def test_extra_canonical_lanes_rejects_scope_values():
    registry = _make_registry(extra_canonical_lanes=frozenset({'private', 'real_lane'}))
    assert 'private' not in registry.canonical_lanes
    assert 'real_lane' in registry.canonical_lanes


def test_empty_aliases_produces_empty_canonical_lanes():
    registry = LaneRegistry()
    assert registry.canonical_lanes == frozenset()


def test_none_aliases_produces_empty_canonical_lanes():
    registry = LaneRegistry(None)
    assert registry.canonical_lanes == frozenset()


# ── is_scope_not_lane ─────────────────────────────────────────────────────────

def test_is_scope_not_lane_recognizes_private():
    assert LaneRegistry.is_scope_not_lane('private') is True


def test_is_scope_not_lane_recognizes_public():
    assert LaneRegistry.is_scope_not_lane('public') is True


def test_is_scope_not_lane_recognizes_owner():
    assert LaneRegistry.is_scope_not_lane('owner') is True


def test_is_scope_not_lane_case_insensitive():
    assert LaneRegistry.is_scope_not_lane('Private') is True
    assert LaneRegistry.is_scope_not_lane('PRIVATE') is True


def test_is_scope_not_lane_strips_whitespace():
    assert LaneRegistry.is_scope_not_lane('  private  ') is True


def test_is_scope_not_lane_rejects_canonical_lane():
    assert LaneRegistry.is_scope_not_lane('s1_sessions_main') is False


# ── validate_source_lane ──────────────────────────────────────────────────────

def test_validate_source_lane_passes_canonical():
    registry = _make_registry()
    assert registry.validate_source_lane('s1_sessions_main') == 's1_sessions_main'


def test_validate_source_lane_rejects_private():
    registry = _make_registry()
    assert registry.validate_source_lane('private') is None


def test_validate_source_lane_rejects_public():
    registry = _make_registry()
    assert registry.validate_source_lane('public') is None


def test_validate_source_lane_rejects_owner():
    registry = _make_registry()
    assert registry.validate_source_lane('owner') is None


def test_validate_source_lane_rejects_global():
    registry = _make_registry()
    assert registry.validate_source_lane('global') is None


def test_validate_source_lane_none_returns_none():
    registry = _make_registry()
    assert registry.validate_source_lane(None) is None


def test_validate_source_lane_empty_returns_none():
    registry = _make_registry()
    assert registry.validate_source_lane('') is None
    assert registry.validate_source_lane('   ') is None


def test_validate_source_lane_passes_unknown_but_valid():
    """Unknown lanes are passed through for forward-compatibility."""
    registry = _make_registry()
    assert registry.validate_source_lane('new_lane_v2') == 'new_lane_v2'


def test_validate_source_lane_strips_whitespace():
    registry = _make_registry()
    assert registry.validate_source_lane('  s1_sessions_main  ') == 's1_sessions_main'


def test_validate_source_lane_logs_warning_on_scope_rejection(caplog):
    registry = _make_registry()
    with caplog.at_level(logging.WARNING, logger='mcp_server.src.services.lane_registry'):
        result = registry.validate_source_lane('private')
    assert result is None
    assert 'rejected scope value' in caplog.text


# ── resolve_typed_source_lanes ────────────────────────────────────────────────

def test_resolve_typed_source_lanes_passes_canonical_ids():
    registry = _make_registry()
    result = registry.resolve_typed_source_lanes(['s1_sessions_main', 's1_curated_refs'])
    assert result == ['s1_sessions_main', 's1_curated_refs']


def test_resolve_typed_source_lanes_filters_scope_values():
    registry = _make_registry()
    result = registry.resolve_typed_source_lanes(['s1_sessions_main', 'private', 's1_curated_refs'])
    assert result == ['s1_sessions_main', 's1_curated_refs']


def test_resolve_typed_source_lanes_all_scope_values_returns_empty():
    registry = _make_registry()
    result = registry.resolve_typed_source_lanes(['private', 'public', 'owner'])
    assert result == []


def test_resolve_typed_source_lanes_empty_input_returns_empty():
    registry = _make_registry()
    assert registry.resolve_typed_source_lanes([]) == []


def test_resolve_typed_source_lanes_preserves_order():
    registry = _make_registry()
    result = registry.resolve_typed_source_lanes([
        's1_curated_refs', 's1_sessions_main', 'learning_self_audit',
    ])
    assert result == ['s1_curated_refs', 's1_sessions_main', 'learning_self_audit']


def test_resolve_typed_source_lanes_deduplicates():
    registry = _make_registry()
    result = registry.resolve_typed_source_lanes([
        's1_sessions_main', 's1_sessions_main', 's1_curated_refs',
    ])
    assert result == ['s1_sessions_main', 's1_curated_refs']


def test_resolve_typed_source_lanes_passes_unknown_lanes():
    """Unknown-but-valid group_ids are passed through for forward-compat."""
    registry = _make_registry()
    result = registry.resolve_typed_source_lanes(['s1_sessions_main', 'experimental_v2_replay_42'])
    assert 's1_sessions_main' in result
    assert 'experimental_v2_replay_42' in result


def test_resolve_typed_source_lanes_strips_whitespace():
    registry = _make_registry()
    result = registry.resolve_typed_source_lanes(['  s1_sessions_main  '])
    assert result == ['s1_sessions_main']


# ── Module-level singleton management ─────────────────────────────────────────

def test_get_lane_registry_returns_empty_when_not_initialized():
    """Before init_lane_registry is called, get_lane_registry returns a permissive empty registry."""
    original = get_lane_registry()
    set_lane_registry(None)
    try:
        registry = get_lane_registry()
        assert registry.canonical_lanes == frozenset()
        # Empty registry still validates source_lanes
        assert registry.validate_source_lane('private') is None
        assert registry.validate_source_lane('any_lane') == 'any_lane'
    finally:
        set_lane_registry(original)


def test_init_lane_registry_creates_singleton():
    original = get_lane_registry()
    try:
        registry = init_lane_registry(_SAMPLE_ALIASES)
        assert get_lane_registry() is registry
        assert 's1_sessions_main' in registry.canonical_lanes
    finally:
        set_lane_registry(original)


def test_set_lane_registry_replaces_singleton():
    original = get_lane_registry()
    try:
        custom = LaneRegistry({'test': ['test_lane']})
        set_lane_registry(custom)
        assert get_lane_registry() is custom
    finally:
        set_lane_registry(original)


# ── repr ──────────────────────────────────────────────────────────────────────

def test_repr_includes_canonical_lanes():
    registry = LaneRegistry({'test': ['lane_a']})
    r = repr(registry)
    assert 'LaneRegistry' in r
    assert 'lane_a' in r
