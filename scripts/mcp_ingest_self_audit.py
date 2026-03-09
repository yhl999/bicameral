#!/usr/bin/env python3
"""Ingest self-audit JSONL entries into the typed change ledger.

Phase 4 target:
- emit Episode objects
- emit StateFact objects
- emit Procedure candidates when the audit fix describes a repeatable action

This script is intentionally deterministic and cursor-based. It treats input text as
untrusted data and writes only to the typed-memory substrate (`change_ledger.db`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingest.kv_registry import KVRegistry  # noqa: E402
from mcp_server.src.models.typed_memory import Episode, EvidenceRef, StateFact  # noqa: E402
from mcp_server.src.services.change_ledger import ChangeLedger  # noqa: E402
from mcp_server.src.services.procedure_service import ProcedureService  # noqa: E402

DEFAULT_INPUT_PATH = WORKSPACE_ROOT / 'memory' / 'self-audit.jsonl'
DEFAULT_ONTOLOGY_CONFIG = REPO_ROOT / 'config' / 'procedure_extraction_ontology.yaml'
CURSOR_TS_KEY = 'self_audit.last_ingested_ts'
CURSOR_ID_KEY = 'self_audit.last_ingested_id'
DEFAULT_POLICY_VERSION = 'procedural-memory-v0'
DEFAULT_ACTOR_ID = 'learning_self_audit'
DEFAULT_SOURCE_LANE = 'learning_self_audit'
ROLE_PREFIX_RE = re.compile(r'(?im)^(\s*)(system|developer|assistant|user)\s*:')
TOKEN_RE = re.compile(r'\w+')


@dataclass(frozen=True)
class SelfAuditEntry:
    ts: str
    src: str
    key: str
    mode: str
    kind: str
    miss: str | None = None
    fix: str | None = None
    line_number: int = 0

    def to_canonical_payload(self) -> dict[str, Any]:
        return {
            'ts': self.ts,
            'src': self.src,
            'key': self.key,
            'mode': self.mode,
            'kind': self.kind,
            'miss': self.miss,
            'fix': self.fix,
        }


@dataclass(frozen=True)
class IngestPlanItem:
    entry: SelfAuditEntry
    entry_id: str
    episode_object_id: str
    fact_object_id: str
    will_emit_procedure: bool


def _parse_iso_to_utc(ts: str) -> datetime:
    raw = str(ts or '').strip()
    if raw.endswith('Z'):
        raw = raw[:-1] + '+00:00'
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        raise ValueError(f'timestamp must include timezone: {ts}')
    return dt.astimezone(timezone.utc)


def normalize_iso_z(ts: str) -> str:
    return _parse_iso_to_utc(ts).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def compute_entry_id(entry: SelfAuditEntry) -> str:
    payload = json.dumps(
        entry.to_canonical_payload(),
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def load_entries_jsonl(path: Path) -> list[SelfAuditEntry]:
    if not path.exists():
        raise FileNotFoundError(f'input JSONL not found: {path}')

    entries: list[SelfAuditEntry] = []
    for line_number, raw_line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        entry = SelfAuditEntry(
            ts=str(payload['ts']),
            src=str(payload.get('src') or ''),
            key=str(payload['key']),
            mode=str(payload.get('mode') or ''),
            kind=str(payload.get('kind') or ''),
            miss=str(payload['miss']) if payload.get('miss') not in (None, '') else None,
            fix=str(payload['fix']) if payload.get('fix') not in (None, '') else None,
            line_number=line_number,
        )
        if not entry.key.strip() or entry.key.startswith('.') or entry.key.endswith('.'):
            raise ValueError(f'invalid self-audit key on line {line_number}: {entry.key!r}')
        normalize_iso_z(entry.ts)
        entries.append(entry)

    entries.sort(key=lambda item: (normalize_iso_z(item.ts), compute_entry_id(item)))
    return entries


def read_cursor(registry: KVRegistry) -> tuple[str | None, str | None]:
    return registry.get(CURSOR_TS_KEY), registry.get(CURSOR_ID_KEY)


def entry_is_after_cursor(
    entry: SelfAuditEntry,
    entry_id: str,
    cursor_ts: str | None,
    cursor_id: str | None,
) -> bool:
    if not cursor_ts:
        return True
    entry_ts = normalize_iso_z(entry.ts)
    normalized_cursor_ts = normalize_iso_z(cursor_ts)
    if entry_ts > normalized_cursor_ts:
        return True
    if entry_ts < normalized_cursor_ts:
        return False
    if not cursor_id:
        return False
    return entry_id > cursor_id


def load_ontology_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if not isinstance(payload, dict):
        raise ValueError(f'{path} must contain a YAML mapping')
    return payload


def build_plan(
    entries: list[SelfAuditEntry],
    *,
    cursor_ts: str | None,
    cursor_id: str | None,
    ontology: dict[str, Any],
) -> list[IngestPlanItem]:
    items: list[IngestPlanItem] = []
    for entry in entries:
        entry_id = compute_entry_id(entry)
        if not entry_is_after_cursor(entry, entry_id, cursor_ts, cursor_id):
            continue
        items.append(
            IngestPlanItem(
                entry=entry,
                entry_id=entry_id,
                episode_object_id=f'ep_{entry_id[:24]}',
                fact_object_id=f'fact_{entry_id[:24]}',
                will_emit_procedure=_should_emit_procedure(entry, ontology),
            )
        )
    return items


def print_plan(plan: list[IngestPlanItem]) -> None:
    print('=== self-audit typed ingest plan ===')
    print(f'items={len(plan)}')
    for item in plan:
        print(
            f'- ts={normalize_iso_z(item.entry.ts)} key={item.entry.key} kind={item.entry.kind} '
            f'episode={item.episode_object_id} fact={item.fact_object_id} '
            f'procedure={"yes" if item.will_emit_procedure else "no"}'
        )


def ingest_plan(
    plan: list[IngestPlanItem],
    *,
    input_path: Path,
    ledger: ChangeLedger,
    registry: KVRegistry,
    ontology: dict[str, Any],
    actor_id: str,
    policy_version: str,
) -> dict[str, int]:
    procedure_service = ProcedureService(ledger)
    emitted = {'episodes': 0, 'state_facts': 0, 'procedure_candidates': 0}

    for item in plan:
        entry = item.entry
        evidence_ref = build_evidence_ref(entry, entry_id=item.entry_id, source_path=input_path)
        recorded_at = normalize_iso_z(entry.ts)

        episode = Episode.model_validate(
            {
                'object_id': item.episode_object_id,
                'root_id': item.episode_object_id,
                'title': f'self_audit:{entry.key}',
                'summary': summarize_entry(entry),
                'annotations': [
                    f'kind:{entry.kind.lower()}',
                    f'mode:{entry.mode.lower()}',
                    f'key:{entry.key}',
                ],
                'policy_scope': 'private',
                'visibility_scope': 'private',
                'source_lane': DEFAULT_SOURCE_LANE,
                'source_key': entry.key,
                'evidence_refs': [evidence_ref],
                'created_at': recorded_at,
                'valid_at': recorded_at,
            }
        )
        ledger.append_event(
            'derive',
            actor_id=actor_id,
            reason='self-audit episode materialization',
            recorded_at=recorded_at,
            payload=episode,
            policy_version=policy_version,
        )
        emitted['episodes'] += 1

        state_fact = build_state_fact(
            entry,
            entry_id=item.entry_id,
            episode_id=episode.object_id,
            evidence_ref=evidence_ref,
            ontology=ontology,
        )
        prior_fact = _current_fact_for_predicate(
            ledger,
            subject=state_fact.subject,
            predicate=state_fact.predicate,
            scope=state_fact.scope,
        )
        event_type = 'supersede' if prior_fact is not None else 'derive'
        ledger.append_event(
            event_type,
            actor_id=actor_id,
            reason='self-audit typed state fact materialization',
            recorded_at=recorded_at,
            payload=state_fact,
            target_object_id=prior_fact.object_id if prior_fact is not None else None,
            root_id=prior_fact.root_id if prior_fact is not None else None,
            parent_id=prior_fact.object_id if prior_fact is not None else None,
            policy_version=policy_version,
            metadata={'entry_id': item.entry_id, 'source_key': entry.key},
        )
        emitted['state_facts'] += 1

        if item.will_emit_procedure:
            procedure = build_procedure_payload(
                entry,
                ontology=ontology,
                evidence_ref=evidence_ref,
                source_episode_id=episode.object_id,
            )
            procedure_service.upsert_candidate(
                source_key=entry.key,
                name=procedure['name'],
                trigger=procedure['trigger'],
                steps=procedure['steps'],
                expected_outcome=procedure['expected_outcome'],
                preconditions=procedure['preconditions'],
                actor_id=actor_id,
                reason='self-audit procedure candidate materialization',
                recorded_at=recorded_at,
                evidence_refs=[evidence_ref],
                risk_level=procedure['risk_level'],
                source_lane=DEFAULT_SOURCE_LANE,
                source_episode_id=episode.object_id,
                policy_scope='private',
                visibility_scope='private',
                extractor_version='self-audit-procedure-v0',
            )
            emitted['procedure_candidates'] += 1

        registry.set_many(
            {
                CURSOR_TS_KEY: recorded_at,
                CURSOR_ID_KEY: item.entry_id,
            }
        )

    return emitted


def build_evidence_ref(entry: SelfAuditEntry, *, entry_id: str, source_path: Path) -> EvidenceRef:
    snippet = summarize_entry(entry)
    return EvidenceRef.model_validate(
        {
            'kind': 'file',
            'source_system': 'self_audit_jsonl',
            'locator': {
                'repo': 'workspace',
                'path': source_path.as_posix(),
                'start_line': entry.line_number,
                'end_line': entry.line_number,
            },
            'title': entry.key,
            'snippet': snippet,
            'observed_at': normalize_iso_z(entry.ts),
            'retrieved_at': normalize_iso_z(entry.ts),
            'hash': entry_id,
        }
    )


def build_state_fact(
    entry: SelfAuditEntry,
    *,
    entry_id: str,
    episode_id: str,
    evidence_ref: EvidenceRef,
    ontology: dict[str, Any],
) -> StateFact:
    prefix_rule = _prefix_rule_for_key(entry.key, ontology)
    value = {
        'kind': entry.kind,
        'mode': entry.mode,
        'summary': summarize_entry(entry),
        'miss': sanitize_untrusted_audit_text(entry.miss),
        'fix': sanitize_untrusted_audit_text(entry.fix),
        'source_ref': f'{entry.key}:{entry_id[:12]}',
    }
    return StateFact.model_validate(
        {
            'object_id': f'fact_{entry_id[:24]}',
            'root_id': f'fact_{entry_id[:24]}',
            'fact_type': prefix_rule['fact_type'],
            'subject': prefix_rule.get('subject', 'agent:archibald'),
            'predicate': prefix_rule.get('predicate_prefix', 'learning.self_audit.') + entry.key,
            'value': value,
            'scope': 'private',
            'policy_scope': 'private',
            'visibility_scope': 'private',
            'source_lane': DEFAULT_SOURCE_LANE,
            'source_episode_id': episode_id,
            'source_key': entry.key,
            'evidence_refs': [evidence_ref],
            'created_at': normalize_iso_z(entry.ts),
            'valid_at': normalize_iso_z(entry.ts),
            'risk_level': prefix_rule['risk_level'],
            'promotion_status': 'proposed',
        }
    )


def build_procedure_payload(
    entry: SelfAuditEntry,
    *,
    ontology: dict[str, Any],
    evidence_ref: EvidenceRef,
    source_episode_id: str,
) -> dict[str, Any]:
    prefix_rule = _prefix_rule_for_key(entry.key, ontology)
    human_key = humanize_key(entry.key)
    human_leaf = humanize_key(entry.key.split('.')[-1])
    trigger_prefix = prefix_rule.get('trigger_prefix', 'when handling')
    expected_outcome = prefix_rule.get(
        'expected_outcome',
        'the remediation is applied and the prior self-audit miss does not recur',
    )
    name_prefix = prefix_rule.get('name_prefix', 'Procedure')
    return {
        'name': f'{name_prefix}: {human_leaf}',
        'trigger': f'{trigger_prefix} {human_key}',
        'preconditions': build_preconditions(entry),
        'steps': split_fix_into_steps(entry.fix, ontology),
        'expected_outcome': expected_outcome,
        'risk_level': prefix_rule['risk_level'],
        'source_lane': DEFAULT_SOURCE_LANE,
        'source_episode_id': source_episode_id,
        'source_key': entry.key,
        'evidence_refs': [evidence_ref],
    }


def build_preconditions(entry: SelfAuditEntry) -> list[str]:
    preconditions = [f'self-audit mode: {entry.mode.lower()}']
    if entry.kind.strip():
        preconditions.append(f'self-audit kind: {entry.kind.lower()}')
    return preconditions


def split_fix_into_steps(fix_text: str | None, ontology: dict[str, Any]) -> list[str]:
    if not fix_text or not fix_text.strip():
        return ['review the self-audit evidence before acting']
    pattern = str(ontology.get('defaults', {}).get('step_split_pattern') or '')
    segments = re.split(pattern, sanitize_untrusted_audit_text(fix_text)) if pattern else [fix_text]
    steps = [segment.strip(' .;') for segment in segments if segment and segment.strip(' .;')]
    if not steps:
        return ['review the self-audit evidence before acting']
    return steps[:6]


def summarize_entry(entry: SelfAuditEntry) -> str:
    parts = [f'{entry.kind.upper()} {entry.key}']
    if entry.miss:
        parts.append(f'miss: {sanitize_untrusted_audit_text(entry.miss)}')
    if entry.fix:
        parts.append(f'fix: {sanitize_untrusted_audit_text(entry.fix)}')
    return ' | '.join(parts)


def sanitize_untrusted_audit_text(text: str | None) -> str:
    if text is None:
        return ''
    value = str(text)
    value = redact_sensitive(value)
    value = ROLE_PREFIX_RE.sub(r'\1role_\2:', value)
    value = value.strip()
    if len(value) > 2000:
        value = value[:2000] + ' ...(truncated)'
    return value


def redact_sensitive(text: str) -> str:
    replacements = [
        (re.compile(r'op://[^\s\"\'<>\(\)]+'), '<REDACTED:OP_REF>'),
        (re.compile(r'\bsk-[A-Za-z0-9_-]{20,}\b'), '<REDACTED:OPENAI_KEY>'),
        (
            re.compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', re.IGNORECASE),
            '<REDACTED:EMAIL>',
        ),
        (
            re.compile(r'(?<!\w)(?:\+?\d[\d\s().-]{5,}\d)(?!\w)'),
            '<REDACTED:PHONE>',
        ),
    ]
    value = text
    for pattern, replacement in replacements:
        value = pattern.sub(replacement, value)
    return value


def humanize_key(key: str) -> str:
    return ' '.join(part for part in re.split(r'[._-]+', key) if part)


def _should_emit_procedure(entry: SelfAuditEntry, ontology: dict[str, Any]) -> bool:
    if not entry.fix or not entry.fix.strip():
        return False
    if entry.kind.strip().upper() == 'HIT':
        return False
    prefix_rule = _prefix_rule_for_key(entry.key, ontology)
    return bool(prefix_rule.get('procedure_enabled', True))


def _prefix_rule_for_key(key: str, ontology: dict[str, Any]) -> dict[str, Any]:
    defaults = ontology.get('defaults', {}) if isinstance(ontology.get('defaults'), dict) else {}
    rules = ontology.get('prefix_rules', {}) if isinstance(ontology.get('prefix_rules'), dict) else {}
    first_prefix = str(key.split('.', 1)[0] if key else '').strip()
    payload = dict(defaults)
    payload.update(rules.get(first_prefix, {}))
    payload.setdefault('fact_type', 'lesson')
    payload.setdefault('risk_level', 'medium')
    payload.setdefault('predicate_prefix', 'learning.self_audit.')
    payload.setdefault('subject', 'agent:archibald')
    return payload


def _current_fact_for_predicate(
    ledger: ChangeLedger,
    *,
    subject: str,
    predicate: str,
    scope: str,
) -> StateFact | None:
    for fact in ledger.current_state_facts():
        if fact.subject == subject and fact.predicate == predicate and fact.scope == scope:
            return fact
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=str(DEFAULT_INPUT_PATH))
    parser.add_argument('--fixture', help='alias for --input')
    parser.add_argument('--ontology-config', default=str(DEFAULT_ONTOLOGY_CONFIG))
    parser.add_argument('--registry-db', default=None)
    parser.add_argument('--ledger-db', default=None)
    parser.add_argument('--actor-id', default=DEFAULT_ACTOR_ID)
    parser.add_argument('--policy-version', default=DEFAULT_POLICY_VERSION)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--print-plan', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = Path(args.fixture) if args.fixture else Path(args.input)
    ontology_path = Path(args.ontology_config)
    entries = load_entries_jsonl(source_path)
    ontology = load_ontology_config(ontology_path)
    registry = KVRegistry(args.registry_db)
    cursor_ts, cursor_id = read_cursor(registry)
    plan = build_plan(entries, cursor_ts=cursor_ts, cursor_id=cursor_id, ontology=ontology)
    if args.limit > 0:
        plan = plan[: args.limit]

    if args.print_plan or args.dry_run:
        print_plan(plan)
    if args.dry_run:
        print('DRY RUN: no ledger or registry writes')
        print(f'would_emit_episodes={len(plan)}')
        print(f'would_emit_state_facts={len(plan)}')
        print(
            'would_emit_procedure_candidates='
            f"{sum(1 for item in plan if item.will_emit_procedure)}"
        )
        return 0

    ledger = ChangeLedger(args.ledger_db) if args.ledger_db else ChangeLedger()
    summary = ingest_plan(
        plan,
        input_path=source_path,
        ledger=ledger,
        registry=registry,
        ontology=ontology,
        actor_id=args.actor_id,
        policy_version=args.policy_version,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
