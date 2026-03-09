# ruff: noqa: E402, I001
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_server.src.models.typed_memory import Episode, EvidenceRef, StateFact
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


def _append_feedback_episode(ledger: ChangeLedger, episode_id: str, evidence_refs: list[EvidenceRef | dict[str, str]]) -> Episode:
    existing = ledger.materialize_object(episode_id)
    if isinstance(existing, Episode):
        return existing

    normalized_refs = [ref if isinstance(ref, EvidenceRef) else EvidenceRef.from_legacy_ref(ref) for ref in evidence_refs]
    episode = Episode.model_validate(
        {
            'object_id': episode_id,
            'root_id': episode_id,
            'policy_scope': 'private',
            'visibility_scope': 'private',
            'source_lane': 'learning_self_audit',
            'source_key': 'memory:self-audit',
            'evidence_refs': normalized_refs,
            'title': f'episode:{episode_id}',
            'summary': f'feedback anchor for {episode_id}',
            'started_at': '2026-03-08T23:10:00Z',
            'ended_at': '2026-03-08T23:10:00Z',
            'created_at': '2026-03-08T23:10:00Z',
            'valid_at': '2026-03-08T23:10:00Z',
        }
    )
    ledger.append_event('assert', actor_id='tester', reason='feedback_episode', payload=episode)
    return episode


def test_create_promote_evolve_and_retrieve_procedure(ledger: ChangeLedger, service: ProcedureService):
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

    failure_ref = _legacy_ref('ev-fail-1')
    failure_episode = _append_feedback_episode(ledger, 'ep_fail_1', [failure_ref])
    result = service.record_feedback(
        promoted.root_id,
        outcome='failure',
        actor_id='owner',
        episode_id=failure_episode.object_id,
        evidence_refs=[failure_ref],
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

    # 3 distinct episodes required (replay detection rejects episode reuse)
    last_result = None
    for episode_id, evidence_id in [('ep-1', 'ev-success-1'), ('ep-2', 'ev-success-2'), ('ep-3', 'ev-success-3')]:
        success_ref = _legacy_ref(evidence_id)
        _append_feedback_episode(ledger, episode_id, [success_ref])
        last_result = service.record_feedback(
            procedure.root_id,
            outcome='success',
            actor_id='runner',
            episode_id=episode_id,
            evidence_refs=[success_ref],
        )

    assert last_result is not None
    current = last_result.procedure
    assert current.promotion_status == 'promoted'
    assert current.success_count == 3
    assert current.fail_count == 0

    stats = ProcedureEvolutionService(ledger).feedback_stats(current.object_id)
    assert stats.evidence_linked_successes == 3
    assert stats.distinct_episode_count == 3


def test_high_risk_procedure_never_auto_promotes(ledger: ChangeLedger, service: ProcedureService):
    procedure = service.create_procedure(
        actor_id='tester',
        name='Rotate production credential',
        trigger='When a credential leak is suspected',
        steps=['Confirm the leak', 'Rotate the credential', 'Revoke old sessions'],
        expected_outcome='Compromised credential is retired safely',
        evidence_refs=[_legacy_ref('ev-cred-create')],
        risk_level='high',
    )

    last_result = None
    for idx, episode_id in enumerate(['ep-1', 'ep-2', 'ep-3', 'ep-4', 'ep-5', 'ep-6'], start=1):
        success_ref = _legacy_ref(f'ev-cred-success-{idx}')
        _append_feedback_episode(ledger, episode_id, [success_ref])
        last_result = service.record_feedback(
            procedure.root_id,
            outcome='success',
            actor_id='runner',
            episode_id=episode_id,
            evidence_refs=[success_ref],
        )

    assert last_result is not None
    assert last_result.procedure.promotion_status == 'proposed'
    assert last_result.procedure.success_count == 6


def test_untrusted_failure_does_not_poison_auto_promotion(ledger: ChangeLedger, service: ProcedureService):
    procedure = service.create_procedure(
        actor_id='tester',
        name='Check browser service',
        trigger='When Playwright or CDP tools fail to connect',
        steps=['Run browser status', 'Start browser if stopped', 'Retry the task'],
        expected_outcome='Browser service is available before automation runs',
        evidence_refs=[_legacy_ref('ev-create')],
        risk_level='low',
    )

    ledger.append_event(
        'procedure_failure',
        actor_id='runner',
        object_id=procedure.object_id,
        root_id=procedure.root_id,
        metadata={'trusted_feedback': False, 'notes': 'spoofed external feedback'},
    )

    current_after_untrusted = service.evolution.resolve_current(procedure.root_id)
    assert current_after_untrusted.fail_count == 0

    last_result = None
    for episode_id, evidence_id in [('ep-1', 'ev-success-1'), ('ep-2', 'ev-success-2'), ('ep-3', 'ev-success-3')]:
        success_ref = _legacy_ref(evidence_id)
        _append_feedback_episode(ledger, episode_id, [success_ref])
        last_result = service.record_feedback(
            procedure.root_id,
            outcome='success',
            actor_id='runner',
            episode_id=episode_id,
            evidence_refs=[success_ref],
        )

    assert last_result is not None
    current = last_result.procedure
    assert current.promotion_status == 'promoted'
    assert current.success_count == 3
    assert current.fail_count == 0

    stats = ProcedureEvolutionService(ledger).feedback_stats(current.object_id)
    assert stats.evidence_linked_successes == 3
    assert stats.distinct_episode_count == 3
    assert stats.failure_count == 0


def test_feedback_rejects_synthetic_episode_and_mismatched_evidence(service: ProcedureService, ledger: ChangeLedger):
    procedure = service.create_procedure(
        actor_id='tester',
        name='Check browser service',
        trigger='When Playwright or CDP tools fail to connect',
        steps=['Run browser status', 'Start browser if stopped', 'Retry the task'],
        expected_outcome='Browser service is available before automation runs',
        evidence_refs=[_legacy_ref('ev-create')],
        risk_level='low',
    )

    with pytest.raises(ValueError, match='episode_id'):
        service.record_feedback(
            procedure.root_id,
            outcome='success',
            actor_id='runner',
            episode_id='ep-missing',
            evidence_refs=[_legacy_ref('ev-spoof')],
        )

    _append_feedback_episode(ledger, 'ep-real', [_legacy_ref('ev-real')])
    with pytest.raises(ValueError, match='trusted evidence_refs'):
        service.record_feedback(
            procedure.root_id,
            outcome='success',
            actor_id='runner',
            episode_id='ep-real',
        )

    with pytest.raises(ValueError, match='must belong to the referenced episode'):
        service.record_feedback(
            procedure.root_id,
            outcome='failure',
            actor_id='runner',
            episode_id='ep-real',
            evidence_refs=[_legacy_ref('ev-spoof')],
            revised_steps=['Actually do the thing'],
        )


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


def test_self_audit_ingest_is_idempotent_and_redacts_state_fact_values(ledger: ChangeLedger):
    ontology = load_ontology(WORKSPACE_ROOT / 'projects' / 'bicameral-private' / 'config' / 'procedure_extraction_ontology.yaml')
    entries = [
        SelfAuditEntry(
            ts='2026-03-08T23:10:00Z',
            src='memory/self-audit.jsonl',
            key='security.secret-handling',
            mode='nightly',
            kind='security_miss',
            miss='Leaked sk-abcdefghijklmnopqrstuvwxyz123456 during debugging.',
            fix='Rotate token: sk-secretsecretsecretsecret12345; update password=abc123 in config; verify the secret is gone.',
        )
    ]

    emissions = build_emissions(entries, ontology=ontology)
    first_counts = ingest_emissions(ledger, emissions)
    second_counts = ingest_emissions(ledger, emissions)

    assert first_counts == {'episodes': 1, 'state_facts': 2, 'procedure_candidates': 1}
    assert second_counts == {'episodes': 0, 'state_facts': 0, 'procedure_candidates': 0}

    service = ProcedureService(ledger)
    procedures = service.list_current_procedures(include_proposed=True)
    assert len(procedures) == 1

    facts = ledger.current_state_facts()
    assert len(facts) == 2
    serialized_values = json.dumps([fact.value for fact in facts], sort_keys=True)
    assert 'sk-abcdefghijklmnopqrstuvwxyz123456' not in serialized_values
    assert 'sk-secretsecretsecretsecret12345' not in serialized_values
    assert 'password=abc123' not in serialized_values
    assert '<REDACTED:OPENAI_KEY>' in serialized_values
    assert '<REDACTED>' in serialized_values

    tracked_ids = [
        emissions[0].episode.object_id,
        *(fact.object_id for fact in emissions[0].state_facts),
        emissions[0].procedure_candidates[0]['object_id'],
    ]
    placeholders = ', '.join('?' for _ in tracked_ids)
    row_count = ledger.conn.execute(
        f"SELECT COUNT(*) AS count FROM change_events WHERE object_id IN ({placeholders})",
        tracked_ids,
    ).fetchone()['count']
    assert row_count == len(tracked_ids)


def test_emission_transaction_rollback_on_failure(ledger: ChangeLedger):
    """If procedure creation fails mid-emission, episode and state facts are rolled back."""
    from unittest.mock import patch

    ontology = load_ontology(None)
    entries = [
        SelfAuditEntry(
            ts='2026-03-08T23:10:00Z',
            src='memory/self-audit.jsonl',
            key='update.hotfixes',
            mode='nightly',
            kind='ops_miss',
            miss='Lost patches after upstream sync.',
            fix='Read TOOLS.md; run update-with-hotfixes.sh; verify gateway health.',
        )
    ]
    emissions = build_emissions(entries, ontology=ontology)

    with patch.object(
        ProcedureService, 'create_procedure', side_effect=RuntimeError('simulated failure')
    ), pytest.raises(RuntimeError, match='simulated failure'):
        ingest_emissions(ledger, emissions)

    # Nothing should have been committed
    row = ledger.conn.execute('SELECT COUNT(*) as cnt FROM change_events').fetchone()
    assert row['cnt'] == 0, 'Expected zero events after rollback'
