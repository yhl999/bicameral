from mcp_server.src.graphiti_mcp_server import _resolve_effective_group_ids


def test_explicit_group_ids_block_implicit_aliases():
    effective_group_ids, invalid_aliases = _resolve_effective_group_ids(
        group_ids=['s1_curated'],
        lane_alias=['sessions_main'],
    )

    assert effective_group_ids == ['s1_curated']
    assert invalid_aliases == []


def test_empty_lane_aliases_are_respected_without_alias_validation_error():
    effective_group_ids, invalid_aliases = _resolve_effective_group_ids(
        group_ids=None,
        lane_alias=[],
    )

    assert isinstance(effective_group_ids, list)
    assert invalid_aliases == []
