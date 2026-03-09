# ruff: noqa: E402, I001
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_server.src.models.typed_memory import EvidenceRef, StateFact
from mcp_server.src.services.change_ledger import ChangeLedger
from mcp_server.src.services.procedure_evolution import ProcedureEvolutionService
from mcp_server.src.services.procedure_service import ProcedureService
from scripts.mcp_ingest_self_audit import SelfAuditEntry, build_emissions, ingest_emissions, load_ontology


@pytest.fixture()
def ledger() -> ChangeLedger:
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    return ChangeLedger(conn)


@pytest.fixture()
def service(ledger: ChangeLedger) -> ProcedureService:
    return ProcedureService(ledger)



def _legacy_ref(evidence_id: str) -> dict[str, str]:
    return {
        'source_key': 'learning:self-audit',
        'evidence_id': evidence_id,
        'scope': 'learning_self_audit',
    }



def _message_ref(message_id: str) -> EvidenceRef:
    return EvidenceRef.model_validate(
        {
            'kind': 'message',
            'source_system': 'telegram',
            'locator': {
                'system': 'telegram',
                'conversation_id': 'chat-1',
                'message_id': message_id,
            },
        }
    )



def test_create_promote_evolve_and_retrieve_procedure(service: ProcedureService):
    created = service.create_procedure(
        actor_id='tester',
        name='OpenClaw hotfix update',
        trigger='When updating OpenClaw on this machine',
        preconditions=['Repo is clean'],
        steps=['Read TOOLS.md', 'Run update-with-hotfixes.sh', 'Verify gateway health'],
        expected_outcome='OpenClaw updates without dropping local hotfixes',
        evidence_refs=[_message_ref('m1')],
        risk_level='low',
        source_lane='learning_self_audit',
        source_key='memory:self-audit',
    )
    assert created.promotion_status == 'proposed'

    promoted = service.promote_procedure(created.root_id, actor_id='owner')
    assert promoted.promotion_status == 'promoted'
    assert promoted.version == 1

    result = service.record_feedback(
        promoted.root_id,
        outcome='failure',
        actor_id='owner',
        episode_id='ep_fail_1',
        evidence_refs=[_legacy_ref('ev-fail-1')],
        revised_steps=['Read TOOLS.md', 'Run update-with-hotfixes.sh', 'Restart gateway', 'Verify gateway health'],
        reason='adjust_after_restart_gap',
    )
    assert result.evolved_from == promoted.object_id
    assert result.evolved_to is not None
    assert result.procedure.version == 2
    assert result.procedure.parent_id == promoted.object_id
    assert result.procedure.promotion_status == 'proposed'

    re_promoted = service.promote_procedure(result.procedure.root_id, actor_id='owner')
    matches = service.retrieve_procedures('what should I do when updating OpenClaw hotfixes?', limit=3)
    assert matches
    assert matches[0].procedure.object_id == re_promoted.object_id
    assert 'updating' in ' '.join(matches[0].matched_terms) or 'openclaw' in ' '.join(matches[0].matched_terms)



def test_low_risk_auto_promotion_uses_locked_thresholds(ledger: ChangeLedger, service: ProcedureService):
    procedure = service.create_procedure(
        actor_id='tester',
        name='Check browser service',
        trigger='When Playwright or CDP tools fail to connect',
        steps=['Run browser status', 'Start browser if stopped', 'Retry the task'],
        expected_outcome='Browser service is available before automation runs',
        evidence_refs=[_legacy_ref('ev-create')],
        risk_level='low',
    )

    for idx, episode_id in enumerate(['ep-1', 'ep-1', 'ep-2'], start=1):
        outcome = service.record_feedback(
            procedure.root_id,
            outcome='success',
            actor_id='runner',
            episode_id=episode_id,
            evidence_refs=[_legacy_ref(f'ev-success-{idx}')],
        )

    current = outcome.procedure
    assert current.promotion_status == 'promoted'
    assert current.success_count == 3
    assert current.fail_count == 0

    stats = ProcedureEvolutionService(ledger).feedback_stats(current.object_id)
    assert stats.evidence_linked_successes == 3
    assert stats.distinct_episode_count == 2



def test_high_risk_procedure_never_auto_promotes(service: ProcedureService):
    procedure = service.create_procedure(
        actor_id='tester',
        name='Rotate production credential',
        trigger='When a credential leak is suspected',
        steps=['Confirm the leak', 'Rotate the credential', 'Revoke old sessions'],
        expected_outcome='Compromised credential is retired safely',
        evidence_refs=[_legacy_ref('ev-cred-create')],
        risk_level='high',
    )

    for idx, episode_id in enumerate(['ep-1', 'ep-2', 'ep-3', 'ep-4', 'ep-5', 'ep-6'], start=1):
        outcome = service.record_feedback(
            procedure.root_id,
            outcome='success',
            actor_id='runner',
            episode_id=episode_id,
            evidence_refs=[_legacy_ref(f'ev-cred-success-{idx}')],
        )

    assert outcome.procedure.promotion_status == 'proposed'
    assert outcome.procedure.success_count == 6



def test_self_audit_emissions_create_episode_state_fact_and_procedure_candidate(ledger: ChangeLedger):
    ontology = load_ontology(WORKSPACE_ROOT / 'projects' / 'bicameral-private' / 'config' / 'procedure_extraction_ontology.yaml')
    entries = [
        SelfAuditEntry(
            ts='2026-03-08T23:10:00Z',
            src='memory/self-audit.jsonl',
            key='update.hotfixes',
            mode='nightly',
            kind='ops_miss',
            miss='Updates kept dropping local patches after upstream sync.',
            fix='Read TOOLS.md; run update-with-hotfixes.sh; verify gateway health.',
        )
    ]

    emissions = build_emissions(entries, ontology=ontology)
    assert len(emissions) == 1

    emission = emissions[0]
    assert emission.episode.object_type == 'episode'
    assert any(isinstance(fact, StateFact) and fact.predicate.startswith('lesson.self_audit.') for fact in emission.state_facts)
    assert any(isinstance(fact, StateFact) and fact.predicate.startswith('rule.self_audit.') for fact in emission.state_facts)
    assert len(emission.procedure_candidates) == 1
    assert emission.procedure_candidates[0]['risk_level'] == 'low'
    assert emission.procedure_candidates[0]['steps'] == [
        'Read TOOLS.md',
        'run update-with-hotfixes.sh',
        'verify gateway health',
    ] or len(emission.procedure_candidates[0]['steps']) >= 2

    counts = ingest_emissions(ledger, emissions)
    assert counts == {'episodes': 1, 'state_facts': 2, 'procedure_candidates': 1}

    service = ProcedureService(ledger)
    procedures = service.list_current_procedures(include_proposed=True)
    assert len(procedures) == 1
    assert procedures[0].source_episode_id == emission.episode.object_id
    assert procedures[0].promotion_status == 'proposed'
