#!/usr/bin/env python3
"""Graduate legacy learning_self_audit entries into typed-memory objects.

This v0 script deliberately reuses the old self-audit entry contract (ts/src/key/mode/kind/miss/fix)
while emitting into the typed-memory ledger substrate:
- Episode anchor per self-audit entry
- derived StateFact lesson / operational_rule candidates
- derived Procedure candidates (represented as proposed Procedure objects)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
DEFAULT_INPUT_PATH = WORKSPACE_ROOT / 'memory' / 'self-audit.jsonl'
DEFAULT_ONTOLOGY_PATH = WORKSPACE_ROOT / 'projects' / 'bicameral-private' / 'config' / 'procedure_extraction_ontology.yaml'

sys.path.insert(0, str(REPO_ROOT))

from mcp_server.src.models.typed_memory import Episode, EvidenceRef, StateFact  # noqa: E402
from mcp_server.src.services.change_ledger import ChangeLedger  # noqa: E402
from mcp_server.src.services.procedure_service import ProcedureService  # noqa: E402


@dataclass(frozen=True)
class SelfAuditEntry:
    ts: str
    src: str
    key: str
    mode: str
    kind: str
    miss: str | None = None
    fix: str | None = None

    def to_c14n_obj(self) -> dict[str, Any]:
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
class SelfAuditEmission:
    entry: SelfAuditEntry
    entry_id: str
    episode: Episode
    state_facts: list[StateFact]
    procedure_candidates: list[dict[str, Any]]


DEFAULT_ONTOLOGY: dict[str, Any] = {
    'version': 'procedure-extraction-ontology-v0',
    'subject': 'agent:archibald',
    'policy_scope': 'private',
    'visibility_scope': 'private',
    'source_lane': 'learning_self_audit',
    'source_key': 'memory:self-audit',
    'low_risk_key_prefixes': ['workflow.', 'ops.', 'tools.', 'update.'],
    'high_risk_key_prefixes': ['security.', 'auth.', 'credentials.', 'legal.', 'finance.', 'health.'],
    'imperative_verbs': ['use', 'read', 'run', 'restart', 'check', 'verify', 'inspect', 'update', 'add', 'remove', 'reload', 'open'],
    'step_splitters': [';', ' then ', ' and then ', ' -> ', '\n- ', '\n* '],
    'min_candidate_steps': 2,
}


def load_ontology(path: Path | None = None) -> dict[str, Any]:
    if path is None or not path.exists():
        return dict(DEFAULT_ONTOLOGY)

    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f'Ontology must be a mapping: {path}')
    merged = dict(DEFAULT_ONTOLOGY)
    merged.update(loaded)
    return merged



def compute_entry_id(entry: SelfAuditEntry) -> str:
    payload = json.dumps(entry.to_c14n_obj(), sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()



def load_entries_jsonl(path: Path) -> list[SelfAuditEntry]:
    entries: list[SelfAuditEntry] = []
    with path.open('r', encoding='utf-8') as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f'Line {line_no} is not a JSON object: {path}')
            entry = SelfAuditEntry(
                ts=str(obj['ts']),
                src=str(obj.get('src') or 'self_audit'),
                key=str(obj['key']),
                mode=str(obj.get('mode') or 'nightly'),
                kind=str(obj.get('kind') or ''),
                miss=str(obj['miss']) if obj.get('miss') is not None else None,
                fix=str(obj['fix']) if obj.get('fix') is not None else None,
            )
            entries.append(entry)
    return sorted(entries, key=lambda item: (item.ts, compute_entry_id(item)))



def build_emissions(
    entries: list[SelfAuditEntry],
    *,
    ontology: dict[str, Any] | None = None,
) -> list[SelfAuditEmission]:
    onto = dict(DEFAULT_ONTOLOGY)
    if ontology:
        onto.update(ontology)

    emissions: list[SelfAuditEmission] = []
    for entry in entries:
        entry_id = compute_entry_id(entry)
        evidence_refs = [_evidence_ref_for_entry(entry, entry_id)]
        episode = Episode.model_validate(
            {
                'object_id': f'ep_{entry_id[:24]}',
                'root_id': f'ep_{entry_id[:24]}',
                'policy_scope': onto['policy_scope'],
                'visibility_scope': onto['visibility_scope'],
                'source_lane': onto['source_lane'],
                'source_key': onto['source_key'],
                'evidence_refs': evidence_refs,
                'title': f'self_audit:{entry.key}',
                'summary': _episode_summary(entry),
                'started_at': entry.ts,
                'ended_at': entry.ts,
                'annotations': [f'kind:{entry.kind}', f'mode:{entry.mode}'],
                'created_at': entry.ts,
                'valid_at': entry.ts,
            }
        )

        state_facts: list[StateFact] = []
        if entry.miss:
            state_facts.append(
                StateFact.model_validate(
                    {
                        'object_id': f'sf_lesson_{entry_id[:20]}',
                        'root_id': f'sf_lesson_{entry_id[:20]}',
                        'policy_scope': onto['policy_scope'],
                        'visibility_scope': onto['visibility_scope'],
                        'source_lane': onto['source_lane'],
                        'source_episode_id': episode.object_id,
                        'source_key': onto['source_key'],
                        'evidence_refs': evidence_refs,
                        'fact_type': 'lesson',
                        'subject': onto['subject'],
                        'predicate': f'lesson.self_audit.{entry.key}',
                        'value': {
                            'kind': entry.kind,
                            'mode': entry.mode,
                            'miss': redact_sensitive(sanitize_untrusted_audit_text(entry.miss)),
                            'fix': redact_sensitive(sanitize_untrusted_audit_text(entry.fix)),
                            'source_ref': f'self-audit://{entry_id}',
                        },
                        'scope': onto['policy_scope'],
                        'created_at': entry.ts,
                        'valid_at': entry.ts,
                    }
                )
            )
        if entry.fix:
            state_facts.append(
                StateFact.model_validate(
                    {
                        'object_id': f'sf_rule_{entry_id[:20]}',
                        'root_id': f'sf_rule_{entry_id[:20]}',
                        'policy_scope': onto['policy_scope'],
                        'visibility_scope': onto['visibility_scope'],
                        'source_lane': onto['source_lane'],
                        'source_episode_id': episode.object_id,
                        'source_key': onto['source_key'],
                        'evidence_refs': evidence_refs,
                        'fact_type': 'operational_rule',
                        'subject': onto['subject'],
                        'predicate': f'rule.self_audit.{entry.key}',
                        'value': {
                            'kind': entry.kind,
                            'mode': entry.mode,
                            'instruction': redact_sensitive(sanitize_untrusted_audit_text(entry.fix)),
                            'source_ref': f'self-audit://{entry_id}',
                        },
                        'scope': onto['policy_scope'],
                        'created_at': entry.ts,
                        'valid_at': entry.ts,
                    }
                )
            )

        procedure_candidates: list[dict[str, Any]] = []
        candidate = _procedure_candidate_for_entry(entry, entry_id, episode.object_id, evidence_refs, onto)
        if candidate is not None:
            procedure_candidates.append(candidate)

        emissions.append(
            SelfAuditEmission(
                entry=entry,
                entry_id=entry_id,
                episode=episode,
                state_facts=state_facts,
                procedure_candidates=procedure_candidates,
            )
        )

    return emissions



def ingest_emissions(
    ledger: ChangeLedger,
    emissions: list[SelfAuditEmission],
    *,
    actor_id: str = 'self_audit:ingest',
) -> dict[str, int]:
    """Ingest self-audit emissions into the ledger.

    Idempotency: each object is checked for existence before writing.
    Atomicity: each emission is ingested under BEGIN IMMEDIATE so a rerun can
    safely fill gaps without racing another writer into duplicate creation.
    """
    service = ProcedureService(ledger)
    counts = {'episodes': 0, 'state_facts': 0, 'procedure_candidates': 0}

    for emission in emissions:
        try:
            ledger.conn.execute('BEGIN IMMEDIATE')
            wrote_anything = False

            if not _object_exists(ledger, emission.episode.object_id):
                ledger.append_event(
                    'assert',
                    actor_id=actor_id,
                    reason='self_audit_episode',
                    payload=emission.episode,
                    recorded_at=emission.entry.ts,
                    _autocommit=False,
                )
                counts['episodes'] += 1
                wrote_anything = True

            for fact in emission.state_facts:
                if not _object_exists(ledger, fact.object_id):
                    ledger.append_event(
                        'derive',
                        actor_id=actor_id,
                        reason='self_audit_state_fact',
                        payload=fact,
                        recorded_at=emission.entry.ts,
                        _autocommit=False,
                    )
                    counts['state_facts'] += 1
                    wrote_anything = True

            for candidate in emission.procedure_candidates:
                proc_id = str(candidate.get('object_id') or f'proc_sa_{candidate["entry_id"][:24]}')
                root_id = str(candidate.get('root_id') or proc_id)
                if not _object_exists(ledger, proc_id):
                    service.create_procedure(
                        actor_id=actor_id,
                        name=str(candidate['name']),
                        trigger=str(candidate['trigger']),
                        preconditions=list(candidate.get('preconditions') or []),
                        steps=list(candidate['steps']),
                        expected_outcome=str(candidate['expected_outcome']),
                        evidence_refs=list(candidate['evidence_refs']),
                        risk_level=str(candidate['risk_level']),
                        policy_scope=str(candidate['policy_scope']),
                        visibility_scope=str(candidate['visibility_scope']),
                        source_lane=str(candidate['source_lane']),
                        source_episode_id=str(candidate['source_episode_id']),
                        source_key=str(candidate['source_key']),
                        reason='self_audit_procedure_candidate',
                        recorded_at=emission.entry.ts,
                        derive=True,
                        promote=False,
                        object_id=proc_id,
                        root_id=root_id,
                        _autocommit=False,
                    )
                    counts['procedure_candidates'] += 1
                    wrote_anything = True

            if wrote_anything:
                ledger.conn.commit()
            else:
                ledger.conn.rollback()
        except Exception:
            ledger.conn.rollback()
            raise

    return counts


def _object_exists(ledger: ChangeLedger, object_id: str) -> bool:
    """Check if a creation event already exists for the given object_id."""
    row = ledger.conn.execute(
        "SELECT 1 FROM change_events WHERE object_id = ? AND event_type IN ('assert', 'derive') LIMIT 1",
        (object_id,),
    ).fetchone()
    return row is not None



def sanitize_untrusted_audit_text(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = str(text)
    cleaned = re.sub(r'(?im)^\s*(system|assistant|developer|user)\s*:\s*', '[redacted_role]: ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned[:2000] if len(cleaned) > 2000 else cleaned



def redact_sensitive(text: str | None) -> str | None:
    if text is None:
        return None
    redacted = str(text)
    redacted = re.sub(r'\bsk-[A-Za-z0-9_-]{20,}\b', '<REDACTED:OPENAI_KEY>', redacted)
    redacted = re.sub(r'\b(api[_-]?key|token|secret|password)\b(\s*[:=]\s*)([^\s,;"\']+)', r'\1\2<REDACTED>', redacted, flags=re.IGNORECASE)
    return redacted



def _episode_summary(entry: SelfAuditEntry) -> str:
    parts = [f'key={entry.key}', f'kind={entry.kind}', f'mode={entry.mode}']
    miss = redact_sensitive(sanitize_untrusted_audit_text(entry.miss))
    fix = redact_sensitive(sanitize_untrusted_audit_text(entry.fix))
    if miss:
        parts.append(f'miss={miss}')
    if fix:
        parts.append(f'fix={fix}')
    return ' | '.join(parts)



def _evidence_ref_for_entry(entry: SelfAuditEntry, entry_id: str) -> EvidenceRef:
    return EvidenceRef.model_validate(
        {
            'kind': 'event_log',
            'source_system': 'self_audit',
            'locator': {
                'system': 'self_audit',
                'stream': f'{entry.src}:{entry.mode}',
                'event_id': entry_id,
            },
            'title': entry.key,
            'snippet': redact_sensitive(sanitize_untrusted_audit_text(entry.miss or entry.fix or entry.kind)),
            'observed_at': entry.ts,
        }
    )



def _procedure_candidate_for_entry(
    entry: SelfAuditEntry,
    entry_id: str,
    episode_id: str,
    evidence_refs: list[EvidenceRef],
    ontology: dict[str, Any],
) -> dict[str, Any] | None:
    fix = redact_sensitive(sanitize_untrusted_audit_text(entry.fix))
    if not fix:
        return None

    steps = _extract_steps(fix, ontology)
    min_candidate_steps = int(ontology.get('min_candidate_steps') or 2)
    if len(steps) < min_candidate_steps:
        return None

    trigger = redact_sensitive(sanitize_untrusted_audit_text(entry.miss)) or f'When self-audit issue {entry.key} recurs'
    name = _humanize_key(entry.key)
    return {
        'name': f'{name} remediation',
        'trigger': trigger,
        'preconditions': [f'self-audit kind: {entry.kind}'],
        'steps': steps,
        'expected_outcome': f'Prevent recurrence of {entry.key}',
        'risk_level': _risk_level_for_entry(entry, ontology),
        'policy_scope': ontology['policy_scope'],
        'visibility_scope': ontology['visibility_scope'],
        'source_lane': ontology['source_lane'],
        'source_episode_id': episode_id,
        'source_key': ontology['source_key'],
        'evidence_refs': evidence_refs,
        'entry_id': entry_id,
        'object_id': f'proc_sa_{entry_id[:24]}',
        'root_id': f'proc_sa_{entry_id[:24]}',
    }



def _extract_steps(text: str, ontology: dict[str, Any]) -> list[str]:
    normalized = text.replace('\r\n', '\n')
    for splitter in ontology.get('step_splitters') or []:
        normalized = normalized.replace(str(splitter), '\n')
    normalized = re.sub(r'(?m)^\s*(?:\d+[.)]|[-*])\s*', '', normalized)
    parts = [part.strip(' .') for part in normalized.split('\n')]
    parts = [part for part in parts if part]

    if len(parts) == 1:
        parts = [piece.strip(' .') for piece in re.split(r'\bthen\b|\band then\b|\bafter that\b', parts[0], flags=re.IGNORECASE) if piece.strip(' .')]

    imperative_verbs = {str(item).strip().lower() for item in (ontology.get('imperative_verbs') or []) if str(item).strip()}
    actionable = [part for part in parts if _looks_actionable(part, imperative_verbs)]
    return actionable if actionable else parts



def _looks_actionable(step: str, imperative_verbs: set[str]) -> bool:
    tokens = re.findall(r'[a-z0-9]+', step.lower())
    if not tokens:
        return False
    return tokens[0] in imperative_verbs or len(tokens) >= 4



def _risk_level_for_entry(entry: SelfAuditEntry, ontology: dict[str, Any]) -> str:
    key = entry.key.strip().lower()
    kind = entry.kind.strip().lower()
    high_prefixes = tuple(str(item).strip().lower() for item in (ontology.get('high_risk_key_prefixes') or []))
    low_prefixes = tuple(str(item).strip().lower() for item in (ontology.get('low_risk_key_prefixes') or []))
    if key.startswith(high_prefixes) or any(term in kind for term in ('security', 'credential', 'legal', 'finance', 'health')):
        return 'high'
    if key.startswith(low_prefixes):
        return 'low'
    return 'medium'



def _humanize_key(key: str) -> str:
    cleaned = re.sub(r'[_.-]+', ' ', key).strip()
    return cleaned if cleaned else 'self audit'



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', default=str(DEFAULT_INPUT_PATH))
    ap.add_argument('--ontology', default=str(DEFAULT_ONTOLOGY_PATH))
    ap.add_argument('--ledger-db', help='Optional SQLite change-ledger path to ingest into')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    entries = load_entries_jsonl(Path(args.input))
    emissions = build_emissions(entries, ontology=load_ontology(Path(args.ontology)))

    if args.dry_run or not args.ledger_db:
        print(json.dumps(
            {
                'entries': len(entries),
                'episodes': len(emissions),
                'state_facts': sum(len(item.state_facts) for item in emissions),
                'procedure_candidates': sum(len(item.procedure_candidates) for item in emissions),
            },
            indent=2,
            sort_keys=True,
        ))
        return 0

    conn = sqlite3.connect(args.ledger_db)
    conn.row_factory = sqlite3.Row
    ledger = ChangeLedger(conn)
    counts = ingest_emissions(ledger, emissions)
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
