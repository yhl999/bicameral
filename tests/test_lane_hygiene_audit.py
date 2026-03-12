from scripts import lane_hygiene_audit


def test_lane_hygiene_decisions_cover_all_current_retrieval_and_corroboration_lanes() -> None:
    lane_ids = set(lane_hygiene_audit.LANE_RETRIEVAL_ELIGIBLE_GLOBAL) | set(
        lane_hygiene_audit.LANE_CORROBORATION_ONLY
    )
    unresolved = sorted(lane_id for lane_id in lane_ids if lane_id not in lane_hygiene_audit.DECISIONS_BY_LANE)
    assert unresolved == []


def test_learning_self_audit_explicitly_classified_as_keep() -> None:
    assert lane_hygiene_audit.DECISIONS_BY_LANE['learning_self_audit'] == 'keep'
