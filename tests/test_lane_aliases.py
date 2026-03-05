from types import SimpleNamespace

from tests.helpers_mcp_import import load_graphiti_mcp_server

server = load_graphiti_mcp_server()
_resolve_effective_group_ids = server._resolve_effective_group_ids


def _ensure_alias_config() -> None:
    server.config = SimpleNamespace(
        database=SimpleNamespace(provider='neo4j'),
        graphiti=SimpleNamespace(
            group_id='s1_sessions_main',
            lane_aliases={
                'sessions_main': ['s1_sessions_main'],
                'observational_memory': ['s1_observational_memory'],
                'curated': ['s1_curated'],
            },
        ),
    )


def test_group_ids_take_precedence_over_lane_alias():
    _ensure_alias_config()
    effective_group_ids, invalid_aliases = _resolve_effective_group_ids(
        group_ids=['s1_sessions_main', 's1_curated'],
        lane_alias=['observational_memory'],
    )

    assert effective_group_ids == ['s1_sessions_main', 's1_curated']
    assert invalid_aliases == []


def test_lane_aliases_map_to_expected_groups_when_no_explicit_group_ids():
    _ensure_alias_config()
    effective_group_ids, invalid_aliases = _resolve_effective_group_ids(
        group_ids=None,
        lane_alias=['sessions_main', 'observational_memory', 'curated'],
    )

    assert invalid_aliases == []
    assert effective_group_ids == [
        's1_sessions_main',
        's1_observational_memory',
        's1_curated',
    ]


def test_unknown_alias_is_reported_and_rejected():
    _ensure_alias_config()
    effective_group_ids, invalid_aliases = _resolve_effective_group_ids(
        group_ids=None,
        lane_alias=['unknown_alias'],
    )

    assert effective_group_ids == []
    assert invalid_aliases == ['unknown_alias']
