from __future__ import annotations

import sqlite3

from mcp_server.src.models.typed_memory import EvidenceRef, Procedure
from mcp_server.src.services.change_ledger import ChangeLedger
from mcp_server.src.services.procedure_service import ProcedureService


def _ledger() -> ChangeLedger:
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    return ChangeLedger(conn)


def _message_ref(message_id: str) -> EvidenceRef:
    return EvidenceRef.model_validate(
        {
            'kind': 'message',
            'source_system': 'tests',
            'locator': {
                'system': 'tests',
                'conversation_id': 'procedure-service',
                'message_id': message_id,
            },
            'snippet': f'message {message_id}',
        }
    )


def test_low_risk_candidate_auto_promotes_after_threshold_and_is_retrievable():
    ledger = _ledger()
    service = ProcedureService(ledger)
    procedure = service.create_procedure(
        name='Context map preflight',
        trigger='before planning non-trivial work',
        steps=[
            'read projects/bicameral-runtime/state/context.map',
            'treat it as required preflight',
        ],
        expected_outcome='the canonical context map is consulted before planning',
        actor_id='extractor:self-audit',
        reason='seed from self-audit',
        evidence_refs=[_message_ref('seed')],
        risk_level='low',
        source_key='rule.context-map-preflight',
        source_lane='learning_self_audit',
        event_type='derive',
        recorded_at='2026-03-08T00:00:00Z',
    )

    for idx, episode_id in enumerate(['ep-1', 'ep-2', 'ep-2'], start=1):
        result = service.record_feedback(
            procedure.object_id,
            outcome='success',
            actor_id='runner',
            episode_id=episode_id,
            evidence_refs=[_message_ref(f's{idx}')],
            reason='successful execution',
            recorded_at=f'2026-03-08T0{idx}:00:00Z',
        )

    materialized = ledger.materialize_object(procedure.object_id)
    assert isinstance(materialized, Procedure)
    assert materialized.promotion_status == 'promoted'
    assert materialized.success_count == 3
    assert result.promotion_event_id is not None

    matches = service.find_relevant_procedures('what should I do when planning non-trivial work?')
    assert matches
    assert matches[0].procedure_id == procedure.object_id
    assert matches[0].promotion_status == 'promoted'


def test_failure_feedback_can_trigger_refined_candidate_version():
    ledger = _ledger()
    service = ProcedureService(ledger)
    procedure = service.create_procedure(
        name='Guardrail launchd label check',
        trigger='when validating the bicameral Neo4j launchd service',
        steps=['check for com.clawd.graphiti-mcp-neo4j'],
        expected_outcome='the guardrail detects the active launchd label',
        actor_id='extractor:self-audit',
        reason='seed from self-audit',
        evidence_refs=[_message_ref('seed-failure')],
        risk_level='low',
        source_key='tool.guardrail-launchd-label-drift',
        source_lane='learning_self_audit',
        event_type='derive',
        recorded_at='2026-03-08T00:00:00Z',
    )

    result = service.record_feedback(
        procedure.object_id,
        outcome='failure',
        actor_id='runner',
        episode_id='ep-failure-1',
        evidence_refs=[_message_ref('failure-1')],
        reason='false guardrail failure',
        recorded_at='2026-03-08T03:00:00Z',
        revision={
            'steps': [
                'check for com.clawd.bicameral-mcp-neo4j',
                'accept it as the canonical launchd label',
            ],
            'expected_outcome': 'guardrail checks accept the canonical service label',
        },
        revision_reason='refine procedure after failure',
    )

    assert result.evolved_procedure_id is not None
    evolved = ledger.materialize_object(result.evolved_procedure_id)
    assert isinstance(evolved, Procedure)
    assert evolved.version == 2
    assert evolved.parent_id == procedure.object_id
    assert evolved.root_id == procedure.root_id
    assert evolved.promotion_status == 'proposed'
    assert evolved.fail_count == 0

    old_version = ledger.materialize_object(procedure.object_id)
    assert isinstance(old_version, Procedure)
    assert old_version.is_current is False
    assert old_version.fail_count == 1
