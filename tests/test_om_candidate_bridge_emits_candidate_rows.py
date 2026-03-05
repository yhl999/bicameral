from mcp_server.src.services.queue_service import build_om_candidate_rows


REQUIRED_FIELDS = {
    'source_lane',
    'source_node_id',
    'source_event_id',
    'source_group_id',
    'evidence_refs',
    'created_at',
}


def test_om_candidate_bridge_emits_provenance_complete_rows():
    om_facts = [
        {
            'uuid': 'om-fact-1',
            'source_node_uuid': 'om-node-1',
            'group_id': 's1_observational_memory',
            'created_at': '2026-03-05T00:00:00Z',
        }
    ]

    rows = build_om_candidate_rows(om_facts)

    assert len(rows) == 1
    row = rows[0]
    assert REQUIRED_FIELDS <= set(row.keys())
    assert row['source_group_id'] == 's1_observational_memory'
    assert row['source_lane'] == 's1_observational_memory'
    assert isinstance(row['evidence_refs'], list) and row['evidence_refs']
