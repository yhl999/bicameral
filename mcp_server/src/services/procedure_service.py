from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from typing import Any

from mcp_server.src.models.typed_memory import Episode, EvidenceRef, Procedure
from mcp_server.src.services.change_ledger import ChangeLedger
from mcp_server.src.services.procedure_evolution import ProcedureEvolutionService


@dataclass(frozen=True)
class ProcedureMatch:
    procedure: Procedure
    score: float
    matched_terms: list[str]


@dataclass(frozen=True)
class ProcedureFeedbackResult:
    procedure: Procedure
    auto_promoted: bool
    evolved_from: str | None = None
    evolved_to: str | None = None


class ProcedureService:
    def __init__(self, ledger: ChangeLedger):
        self.ledger = ledger
        self.evolution = ProcedureEvolutionService(ledger)

    def create_procedure(
        self,
        *,
        actor_id: str,
        name: str,
        trigger: str,
        steps: list[str],
        expected_outcome: str,
        evidence_refs: list[EvidenceRef | dict[str, Any]],
        preconditions: list[str] | None = None,
        reason: str = 'procedure_create',
        risk_level: str = 'medium',
        policy_scope: str = 'private',
        visibility_scope: str = 'private',
        source_lane: str | None = None,
        source_episode_id: str | None = None,
        source_message_id: str | None = None,
        source_key: str | None = None,
        recorded_at: str | None = None,
        derive: bool = False,
        promote: bool = False,
        object_id: str | None = None,
        root_id: str | None = None,
        _autocommit: bool = True,
    ) -> Procedure:
        normalized_steps = [step.strip() for step in steps if isinstance(step, str) and step.strip()]
        if not normalized_steps:
            raise ValueError('steps must contain at least one non-empty step')

        procedure_object_id = str(object_id or '').strip() or _new_procedure_id()
        procedure_root_id = str(root_id or '').strip() or procedure_object_id
        procedure = Procedure.model_validate(
            {
                'object_id': procedure_object_id,
                'root_id': procedure_root_id,
                'name': name.strip(),
                'trigger': trigger.strip(),
                'preconditions': [item.strip() for item in (preconditions or []) if isinstance(item, str) and item.strip()],
                'steps': normalized_steps,
                'expected_outcome': expected_outcome.strip(),
                'risk_level': str(risk_level or 'medium').strip().lower() or 'medium',
                'policy_scope': policy_scope,
                'visibility_scope': visibility_scope,
                'evidence_refs': _coerce_evidence_refs(evidence_refs),
                'source_lane': source_lane,
                'source_episode_id': source_episode_id,
                'source_message_id': source_message_id,
                'source_key': source_key,
            }
        )
        procedure = procedure.model_copy(update={'root_id': procedure.root_id or procedure.object_id})

        self.ledger.append_event(
            'derive' if derive else 'assert',
            actor_id=actor_id,
            reason=reason,
            payload=procedure,
            recorded_at=recorded_at,
            _autocommit=_autocommit,
        )
        if promote:
            self.ledger.append_event(
                'promote',
                actor_id=actor_id,
                reason=f'{reason}:promote',
                object_id=procedure.object_id,
                root_id=procedure.root_id,
                recorded_at=recorded_at,
                _autocommit=_autocommit,
            )
        return self.evolution.resolve_current(procedure.root_id)

    def promote_procedure(
        self,
        identifier: str,
        *,
        actor_id: str,
        reason: str = 'procedure_manual_approval',
    ) -> Procedure:
        procedure = self.evolution.resolve_current(identifier)
        if procedure.promotion_status != 'promoted':
            self.ledger.append_event(
                'promote',
                actor_id=actor_id,
                reason=reason,
                object_id=procedure.object_id,
                root_id=procedure.root_id,
            )
        return self.evolution.resolve_current(procedure.root_id)

    def record_feedback(
        self,
        identifier: str,
        *,
        outcome: str,
        actor_id: str,
        episode_id: str | None = None,
        evidence_refs: list[EvidenceRef | dict[str, Any]] | None = None,
        notes: str | None = None,
        reason: str | None = None,
        auto_promote: bool = True,
        revised_trigger: str | None = None,
        revised_preconditions: list[str] | None = None,
        revised_steps: list[str] | None = None,
        revised_expected_outcome: str | None = None,
        promote_evolved_version: bool = False,
    ) -> ProcedureFeedbackResult:
        procedure = self.evolution.resolve_current(identifier)
        normalized_outcome = str(outcome or '').strip().lower()
        if normalized_outcome not in {'success', 'failure'}:
            raise ValueError("outcome must be 'success' or 'failure'")

        trusted_episode_id, trusted_evidence_refs = self._trusted_feedback_provenance(
            episode_id=episode_id,
            evidence_refs=evidence_refs,
        )
        trusted_feedback = bool(trusted_episode_id and trusted_evidence_refs)

        metadata: dict[str, Any] = {'trusted_feedback': trusted_feedback}
        if trusted_feedback:
            metadata['episode_id'] = trusted_episode_id
            metadata['evidence_refs'] = [ref.model_dump(mode='json') for ref in trusted_evidence_refs]
        if notes:
            metadata['notes'] = notes.strip()

        self.ledger.append_event(
            f'procedure_{normalized_outcome}',
            actor_id=actor_id,
            reason=reason or f'procedure_{normalized_outcome}',
            object_id=procedure.object_id,
            root_id=procedure.root_id,
            metadata=metadata or None,
        )

        auto_promoted = False
        evolved = None
        if normalized_outcome == 'success' and auto_promote and trusted_feedback:
            auto_promoted = self.evolution.maybe_auto_promote(procedure.object_id)
        elif normalized_outcome == 'failure' and trusted_feedback:
            evolved = self.evolution.evolve_from_feedback(
                procedure.object_id,
                actor_id=actor_id,
                reason=reason or 'procedure_failure_evolution',
                revised_trigger=revised_trigger,
                revised_preconditions=revised_preconditions,
                revised_steps=revised_steps,
                revised_expected_outcome=revised_expected_outcome,
                promote=promote_evolved_version,
                evidence_refs=trusted_evidence_refs,
                source_episode_id=trusted_episode_id,
            )

        current = self.evolution.resolve_current(procedure.root_id)
        return ProcedureFeedbackResult(
            procedure=current,
            auto_promoted=auto_promoted,
            evolved_from=procedure.object_id if evolved is not None else None,
            evolved_to=evolved.object_id if evolved is not None else None,
        )

    def _trusted_feedback_provenance(
        self,
        *,
        episode_id: str | None,
        evidence_refs: list[EvidenceRef | dict[str, Any]] | None,
    ) -> tuple[str | None, list[EvidenceRef]]:
        normalized_episode_id = str(episode_id or '').strip() or None
        parsed_refs = _coerce_feedback_evidence_refs(evidence_refs or [])

        if not normalized_episode_id and not parsed_refs:
            return None, []
        if not normalized_episode_id:
            raise ValueError('feedback evidence_refs require a valid episode_id')

        candidate = self.ledger.materialize_object(normalized_episode_id)
        if not isinstance(candidate, Episode):
            raise ValueError(f'feedback episode_id not found: {normalized_episode_id}')

        if not parsed_refs:
            return candidate.object_id, []

        episode_evidence_uris = {ref.canonical_uri for ref in candidate.evidence_refs}
        trusted_refs = [ref for ref in parsed_refs if ref.canonical_uri in episode_evidence_uris]
        if len(trusted_refs) != len(parsed_refs):
            raise ValueError('feedback evidence_refs must belong to the referenced episode')
        return candidate.object_id, trusted_refs

    def list_current_procedures(self, *, include_proposed: bool = False) -> list[Procedure]:
        rows = self.ledger.conn.execute(
            """
            SELECT DISTINCT COALESCE(root_id, object_id) AS root_id
              FROM change_events
             WHERE object_type = 'procedure'
             ORDER BY root_id ASC
            """
        ).fetchall()

        procedures: list[Procedure] = []
        for row in rows:
            current = self.ledger.current_object(str(row['root_id']))
            if not isinstance(current, Procedure):
                continue
            if current.promotion_status != 'promoted' and not include_proposed:
                continue
            procedures.append(current)

        return sorted(
            procedures,
            key=lambda item: (
                item.promotion_status != 'promoted',
                item.risk_level,
                -item.success_count,
                item.name.lower(),
            ),
        )

    def retrieve_procedures(
        self,
        query: str,
        *,
        limit: int = 5,
        include_proposed: bool = False,
    ) -> list[ProcedureMatch]:
        query_terms = _tokenize(query)
        matches: list[ProcedureMatch] = []
        for procedure in self.list_current_procedures(include_proposed=include_proposed):
            score, matched_terms = _score_procedure(procedure, query, query_terms)
            if score <= 0:
                continue
            matches.append(ProcedureMatch(procedure=procedure, score=score, matched_terms=matched_terms))

        matches.sort(
            key=lambda item: (
                -item.score,
                item.procedure.promotion_status != 'promoted',
                -item.procedure.success_count,
                item.procedure.fail_count,
                -item.procedure.version,
            )
        )
        return matches[: max(1, limit)]


def _coerce_evidence_refs(raw_refs: list[EvidenceRef | dict[str, Any]]) -> list[EvidenceRef]:
    refs = [item if isinstance(item, EvidenceRef) else EvidenceRef.from_legacy_ref(item) for item in raw_refs or [] if item]
    if not refs:
        raise ValueError('procedure creation requires at least one evidence_ref')
    deduped: dict[str, EvidenceRef] = {ref.canonical_uri: ref for ref in refs}
    return [deduped[key] for key in sorted(deduped)]


def _coerce_feedback_evidence_refs(raw_refs: list[EvidenceRef | dict[str, Any]]) -> list[EvidenceRef]:
    if not raw_refs:
        return []
    deduped: dict[str, EvidenceRef] = {}
    for item in raw_refs:
        if not item:
            continue
        ref = item if isinstance(item, EvidenceRef) else EvidenceRef.from_legacy_ref(item)
        deduped[ref.canonical_uri] = ref
    return [deduped[key] for key in sorted(deduped)]


def _new_procedure_id() -> str:
    return f'proc_{secrets.token_hex(12)}'


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r'[a-z0-9]+', str(text or '').lower()) if len(token) >= 2}


def _score_procedure(procedure: Procedure, query: str, query_terms: set[str]) -> tuple[float, list[str]]:
    fields = {
        'name': procedure.name,
        'trigger': procedure.trigger,
        'preconditions': ' '.join(procedure.preconditions),
        'steps': ' '.join(procedure.steps),
        'expected_outcome': procedure.expected_outcome,
    }
    weights = {
        'name': 4.0,
        'trigger': 5.0,
        'preconditions': 3.0,
        'steps': 2.0,
        'expected_outcome': 1.0,
    }

    normalized_query = str(query or '').strip().lower()
    matched_terms: set[str] = set()
    score = 0.0

    for field_name, text in fields.items():
        normalized_text = str(text or '').strip().lower()
        if not normalized_text:
            continue
        if normalized_query and normalized_query in normalized_text:
            score += weights[field_name] * 2.0

        field_terms = _tokenize(normalized_text)
        overlap = sorted(query_terms & field_terms)
        if overlap:
            matched_terms.update(overlap)
            score += weights[field_name] * len(overlap)

    if procedure.promotion_status == 'promoted':
        score += 0.75
    score += min(procedure.success_count, 5) * 0.1
    score -= min(procedure.fail_count, 5) * 0.2
    return score, sorted(matched_terms)
