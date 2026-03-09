from __future__ import annotations

import sqlite3
from pathlib import Path

from ingest.kv_registry import KVRegistry
from mcp_server.src.models.typed_memory import Episode, Procedure, StateFact
from mcp_server.src.services.change_ledger import ChangeLedger
from scripts import mcp_ingest_self_audit as ingest

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'self_audit.sample.jsonl'


def _ontology_yaml(tmp_path: Path) -> Path:
    path = tmp_path / 'procedure_extraction_ontology.yaml'
    path.write_text(
        """
defaults:
  subject: agent:archibald
  predicate_prefix: learning.self_audit.
  fact_type: lesson
  risk_level: medium
  procedure_enabled: true
  name_prefix: Procedure
  trigger_prefix: when handling
  expected_outcome: the remediation is applied correctly
  step_split_pattern: '(?:\\n+|;\\s*|\\.\\s+|\\s+then\\s+)'
prefix_rules:
  rule:
    fact_type: operational_rule
    risk_level: low
    name_prefix: Operational rule
    trigger_prefix: when the operating rule matters for
  tool:
    fact_type: lesson
    risk_level: low
    name_prefix: Tool fix
    trigger_prefix: when a tool workflow depends on
""".strip()
        + "\n",
        encoding='utf-8',
    )
    return path


def test_build_plan_marks_only_miss_entries_with_fixes_as_procedures(tmp_path: Path):
    entries = ingest.load_entries_jsonl(FIXTURE)
    ontology = ingest.load_ontology_config(_ontology_yaml(tmp_path))
    plan = ingest.build_plan(entries, cursor_ts=None, cursor_id=None, ontology=ontology)

    assert len(plan) == 3
    assert sum(1 for item in plan if item.will_emit_procedure) == 2
    assert plan[-1].will_emit_procedure is False


def test_ingest_emits_episode_state_fact_and_procedure_candidates(tmp_path: Path):
    ontology = ingest.load_ontology_config(_ontology_yaml(tmp_path))
    entries = ingest.load_entries_jsonl(FIXTURE)
    plan = ingest.build_plan(entries, cursor_ts=None, cursor_id=None, ontology=ontology)

    registry = KVRegistry(tmp_path / 'registry.db')
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    ledger = ChangeLedger(conn)

    summary = ingest.ingest_plan(
        plan,
        input_path=FIXTURE,
        ledger=ledger,
        registry=registry,
        ontology=ontology,
        actor_id='learning_self_audit',
        policy_version='procedural-memory-v0',
    )

    assert summary == {'episodes': 3, 'state_facts': 3, 'procedure_candidates': 2}

    current_objects = []
    for row in ledger.conn.execute(
        "SELECT DISTINCT COALESCE(root_id, object_id) AS root_id FROM change_events ORDER BY root_id"
    ).fetchall():
        current = ledger.current_object(str(row['root_id']))
        if current is not None:
            current_objects.append(current)

    assert sum(isinstance(obj, Episode) for obj in current_objects) == 3
    assert sum(isinstance(obj, StateFact) for obj in current_objects) == 3
    assert sum(isinstance(obj, Procedure) for obj in current_objects) == 2

    context_map_fact = next(
        obj
        for obj in current_objects
        if isinstance(obj, StateFact) and obj.source_key == 'rule.context-map-preflight'
    )
    assert context_map_fact.fact_type == 'operational_rule'
    assert context_map_fact.risk_level == 'low'

    procedure = next(
        obj
        for obj in current_objects
        if isinstance(obj, Procedure) and obj.source_key == 'rule.context-map-preflight'
    )
    assert procedure.trigger.startswith('when the operating rule matters for')
    assert 'context' in ' '.join(procedure.steps).lower()

    cursor_ts, cursor_id = ingest.read_cursor(registry)
    assert cursor_ts == ingest.normalize_iso_z('2026-03-07T02:50:00-05:00')
    assert cursor_id is not None


def test_cli_dry_run_prints_typed_counts(tmp_path: Path, monkeypatch):
    ontology_path = _ontology_yaml(tmp_path)
    monkeypatch.setattr(
        ingest,
        'parse_args',
        lambda: type(
            'Args',
            (),
            {
                'input': str(FIXTURE),
                'fixture': None,
                'ontology_config': str(ontology_path),
                'registry_db': str(tmp_path / 'registry.db'),
                'ledger_db': str(tmp_path / 'ledger.db'),
                'actor_id': 'learning_self_audit',
                'policy_version': 'procedural-memory-v0',
                'dry_run': True,
                'print_plan': True,
                'limit': 0,
            },
        )(),
    )

    assert ingest.main() == 0
