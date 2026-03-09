from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from ..models.typed_memory import EvidenceRef, Procedure
from .change_ledger import ChangeLedger
from .procedure_evolution import ProcedureEvolution, ProcedureFeedbackStats

_STOPWORDS = {
    'a',
    'an',
    'and',
    'are',
    'at',
    'be',
    'do',
    'for',
    'how',
    'i',
    'if',
    'in',
    'is',
    'it',
    'my',
    'of',
    'on',
    'or',
    'should',
    'the',
    'to',
    'what',
    'when',
    'with',
    'x',
}

_WHAT_SHOULD_I_DO_PREFIX_RE = re.compile(
    r'^\s*(?:what\s+should\s+i\s+do|what\s+do\s+i\s+do|how\s+should\s+i\s+handle)\s+(?:when|if)?\s*',
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r'[a-z0-9]{2,}')


@dataclass(frozen=True)
class ProcedureMatch:
    procedure_id: str
    root_id: str
    name: str
    trigger: str
    promotion_status: str
    score: float
    matched_terms: tuple[str, ...]


@dataclass(frozen=True)
class ProcedureFeedbackResult:
    procedure: Procedure
    stats: ProcedureFeedbackStats
    promotion_event_id: str | None = None
    evolved_procedure_id: str | None = None


class ProcedureService:
    def __init__(self, ledger_or_path: ChangeLedger | str | Path):
        self.ledger = ledger_or_path if isinstance(ledger_or_path, ChangeLedger) else ChangeLedger(ledger_or_path)
        self.evolution = ProcedureEvolution(self.ledger)

    def current_procedures(self, *, include_candidates: bool = True) -> list[Procedure]:
        procedures: list[Procedure] = []
        for root_id in self._distinct_root_ids():
            current = self.ledger.current_object(root_id)
            if not isinstance(current, Procedure):
                continue
            if not include_candidates and current.promotion_status != 'promoted':
                continue
            procedures.append(current)
        return procedures

    def find_by_source_key(self, source_key: str) -> Procedure | None:
        needle = str(source_key or '').strip()
        if not needle:
            return None
        for procedure in self.current_procedures():
            if procedure.source_key == needle:
                return procedure
        return None

    def create_procedure(
        self,
        *,
        name: str,
        trigger: str,
        steps: list[str],
        expected_outcome: str,
        actor_id: str,
        evidence_refs: list[EvidenceRef | dict[str, Any]],
        preconditions: list[str] | None = None,
        reason: str | None = None,
        recorded_at: str | None = None,
        risk_level: Literal['low', 'medium', 'high'] = 'medium',
        source_lane: str | None = None,
        source_episode_id: str | None = None,
        source_message_id: str | None = None,
        source_key: str | None = None,
        policy_scope: str = 'private',
        visibility_scope: str = 'private',
        extractor_version: str | None = None,
        event_type: str = 'assert',
        target_procedure_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Procedure:
        normalized_event_type = str(event_type or '').strip().lower()
        normalized_evidence = [
            item if isinstance(item, EvidenceRef) else EvidenceRef.model_validate(item)
            for item in evidence_refs
        ]
        object_id = _new_procedure_object_id()
        payload = {
            'object_id': object_id,
            'root_id': object_id,
            'name': name,
            'trigger': trigger,
            'preconditions': preconditions or [],
            'steps': [str(step).strip() for step in steps if str(step).strip()],
            'expected_outcome': expected_outcome,
            'risk_level': risk_level,
            'policy_scope': policy_scope,
            'visibility_scope': visibility_scope,
            'evidence_refs': normalized_evidence,
            'source_lane': source_lane,
            'source_episode_id': source_episode_id,
            'source_message_id': source_message_id,
            'source_key': source_key,
            'extractor_version': extractor_version,
        }

        if target_procedure_id:
            current = self._materialize_procedure(target_procedure_id)
            if normalized_event_type not in {'refine', 'supersede'}:
                raise ValueError(
                    'target_procedure_id requires event_type refine or supersede'
                )
            payload.update(
                {
                    'root_id': current.root_id,
                    'parent_id': current.object_id,
                    'version': current.version + 1,
                    'promotion_status': 'proposed',
                    'success_count': 0,
                    'fail_count': 0,
                }
            )
            self.ledger.append_event(
                normalized_event_type,
                actor_id=actor_id,
                reason=reason,
                recorded_at=recorded_at,
                payload=Procedure.model_validate(payload),
                target_object_id=current.object_id,
                root_id=current.root_id,
                parent_id=current.object_id,
                metadata=metadata,
            )
        else:
            if normalized_event_type not in {'assert', 'derive'}:
                raise ValueError('new procedure creation requires event_type assert or derive')
            self.ledger.append_event(
                normalized_event_type,
                actor_id=actor_id,
                reason=reason,
                recorded_at=recorded_at,
                payload=Procedure.model_validate(payload),
                metadata=metadata,
            )

        materialized = self.ledger.materialize_object(object_id)
        if not isinstance(materialized, Procedure):
            raise ValueError('failed to materialize procedure after creation')
        return materialized

    def upsert_candidate(
        self,
        *,
        source_key: str,
        name: str,
        trigger: str,
        steps: list[str],
        expected_outcome: str,
        actor_id: str,
        evidence_refs: list[EvidenceRef | dict[str, Any]],
        preconditions: list[str] | None = None,
        reason: str | None = None,
        recorded_at: str | None = None,
        risk_level: Literal['low', 'medium', 'high'] = 'medium',
        source_lane: str | None = None,
        source_episode_id: str | None = None,
        policy_scope: str = 'private',
        visibility_scope: str = 'private',
        extractor_version: str | None = None,
    ) -> Procedure:
        current = self.find_by_source_key(source_key)
        signature = _procedure_signature(name, trigger, steps, expected_outcome, preconditions or [], risk_level)
        if current is not None:
            current_signature = _procedure_signature(
                current.name,
                current.trigger,
                current.steps,
                current.expected_outcome,
                current.preconditions,
                current.risk_level,
            )
            if current_signature == signature:
                return current
            return self.create_procedure(
                name=name,
                trigger=trigger,
                steps=steps,
                expected_outcome=expected_outcome,
                preconditions=preconditions,
                actor_id=actor_id,
                reason=reason,
                recorded_at=recorded_at,
                evidence_refs=evidence_refs,
                risk_level=risk_level,
                source_lane=source_lane,
                source_episode_id=source_episode_id,
                source_key=source_key,
                policy_scope=policy_scope,
                visibility_scope=visibility_scope,
                extractor_version=extractor_version,
                event_type='supersede',
                target_procedure_id=current.object_id,
                metadata={'source_key': source_key, 'upserted': True},
            )
        return self.create_procedure(
            name=name,
            trigger=trigger,
            steps=steps,
            expected_outcome=expected_outcome,
            preconditions=preconditions,
            actor_id=actor_id,
            reason=reason,
            recorded_at=recorded_at,
            evidence_refs=evidence_refs,
            risk_level=risk_level,
            source_lane=source_lane,
            source_episode_id=source_episode_id,
            source_key=source_key,
            policy_scope=policy_scope,
            visibility_scope=visibility_scope,
            extractor_version=extractor_version,
            event_type='derive',
            metadata={'source_key': source_key, 'upserted': True},
        )

    def record_feedback(
        self,
        procedure_id: str,
        *,
        outcome: Literal['success', 'failure'],
        actor_id: str,
        evidence_refs: list[EvidenceRef | dict[str, Any]] | None = None,
        episode_id: str | None = None,
        reason: str | None = None,
        recorded_at: str | None = None,
        policy_version: str = 'procedural-memory-v0',
        revision: dict[str, Any] | None = None,
        revision_reason: str | None = None,
        evolution_event_type: str = 'refine',
    ) -> ProcedureFeedbackResult:
        procedure = self._materialize_procedure(procedure_id)
        normalized_evidence = [
            item if isinstance(item, EvidenceRef) else EvidenceRef.model_validate(item)
            for item in (evidence_refs or [])
        ]
        if not normalized_evidence and not episode_id:
            raise ValueError('procedure feedback must be evidence-linked via evidence_refs or episode_id')

        event_type = 'procedure_success' if outcome == 'success' else 'procedure_failure'
        self.ledger.append_event(
            event_type,
            actor_id=actor_id,
            reason=reason,
            recorded_at=recorded_at,
            object_id=procedure.object_id,
            root_id=procedure.root_id,
            object_type=procedure.object_type,
            metadata={
                'episode_id': episode_id,
                'evidence_refs': [item.model_dump(mode='json') for item in normalized_evidence],
                'outcome': outcome,
            },
        )

        current_after_feedback = self._materialize_procedure(procedure.object_id)
        promotion_row = self.evolution.maybe_promote(
            current_after_feedback,
            actor_id='policy:procedure-evolution',
            reason='candidate met repeated-success threshold',
            policy_version=policy_version,
            recorded_at=recorded_at,
        )
        evolved: Procedure | None = None
        if revision:
            evolved = self.evolution.evolve(
                current_after_feedback,
                actor_id=actor_id,
                reason=revision_reason or 'procedure revision after feedback',
                revision=revision,
                recorded_at=recorded_at,
                event_type=evolution_event_type,
                metadata={
                    'triggered_by_feedback': outcome,
                    'episode_id': episode_id,
                },
            )

        final_procedure = evolved or self._materialize_procedure(procedure.object_id)
        final_stats = self.evolution.feedback_stats(
            evolved.object_id if evolved is not None else procedure.object_id
        )
        return ProcedureFeedbackResult(
            procedure=final_procedure,
            stats=final_stats,
            promotion_event_id=promotion_row.event_id if promotion_row else None,
            evolved_procedure_id=evolved.object_id if evolved else None,
        )

    def find_relevant_procedures(
        self,
        query: str,
        *,
        limit: int = 5,
        include_candidates: bool = True,
    ) -> list[ProcedureMatch]:
        normalized_query = _normalize_query(query)
        query_terms = _tokenize(normalized_query)
        results: list[ProcedureMatch] = []
        for procedure in self.current_procedures(include_candidates=include_candidates):
            score, matched_terms = _score_procedure_match(procedure, normalized_query, query_terms)
            if score <= 0:
                continue
            results.append(
                ProcedureMatch(
                    procedure_id=procedure.object_id,
                    root_id=procedure.root_id,
                    name=procedure.name,
                    trigger=procedure.trigger,
                    promotion_status=procedure.promotion_status,
                    score=score,
                    matched_terms=tuple(sorted(matched_terms)),
                )
            )

        results.sort(
            key=lambda item: (
                item.score,
                1 if item.promotion_status == 'promoted' else 0,
                item.name.lower(),
            ),
            reverse=True,
        )
        return results[:limit]

    def _distinct_root_ids(self) -> list[str]:
        rows = self.ledger.conn.execute(
            """
            SELECT DISTINCT COALESCE(root_id, object_id) AS root_id
              FROM change_events
             WHERE root_id IS NOT NULL OR object_id IS NOT NULL
             ORDER BY root_id
            """
        ).fetchall()
        return [str(row['root_id']) for row in rows if row['root_id']]

    def _materialize_procedure(self, procedure_id: str) -> Procedure:
        procedure = self.ledger.materialize_object(procedure_id)
        if not isinstance(procedure, Procedure):
            raise ValueError(f'Procedure not found: {procedure_id}')
        return procedure


def _normalize_query(query: str) -> str:
    return _WHAT_SHOULD_I_DO_PREFIX_RE.sub('', str(query or '').strip()).strip().lower()


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(str(text or '').lower())
        if token not in _STOPWORDS and len(token) >= 2
    }


def _score_procedure_match(
    procedure: Procedure,
    normalized_query: str,
    query_terms: set[str],
) -> tuple[float, set[str]]:
    if not normalized_query:
        return 0.0, set()

    trigger_terms = _tokenize(procedure.trigger)
    name_terms = _tokenize(procedure.name)
    step_terms = _tokenize(' '.join(procedure.steps))
    outcome_terms = _tokenize(procedure.expected_outcome)
    combined_terms = trigger_terms | name_terms | step_terms | outcome_terms
    matched_terms = query_terms & combined_terms

    score = 0.0
    score += len(query_terms & trigger_terms) * 3.0
    score += len(query_terms & name_terms) * 2.0
    score += len(query_terms & step_terms) * 1.5
    score += len(query_terms & outcome_terms) * 1.0

    combined_text = ' '.join(
        [procedure.name, procedure.trigger, procedure.expected_outcome, *procedure.preconditions, *procedure.steps]
    ).lower()
    if normalized_query in combined_text:
        score += 4.0
    elif normalized_query in procedure.trigger.lower():
        score += 5.0

    if procedure.promotion_status == 'promoted':
        score += 0.25

    return score, matched_terms


def _procedure_signature(
    name: str,
    trigger: str,
    steps: list[str],
    expected_outcome: str,
    preconditions: list[str],
    risk_level: str,
) -> tuple[str, str, tuple[str, ...], str, tuple[str, ...], str]:
    return (
        name.strip().lower(),
        trigger.strip().lower(),
        tuple(step.strip().lower() for step in steps),
        expected_outcome.strip().lower(),
        tuple(item.strip().lower() for item in preconditions),
        risk_level.strip().lower(),
    )


def _new_procedure_object_id() -> str:
    return f'proc_{uuid4().hex[:24]}'
