from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from ..models.typed_memory import EvidenceRef, Procedure
from .change_ledger import ChangeEventRow, ChangeLedger


@dataclass(frozen=True)
class PromotionThreshold:
    min_successes: int
    min_distinct_episodes: int
    max_failures: int | None = 0
    auto_promote: bool = True


AUTO_PROMOTION_THRESHOLDS: dict[str, PromotionThreshold] = {
    'low': PromotionThreshold(min_successes=3, min_distinct_episodes=2, max_failures=0),
    'medium': PromotionThreshold(min_successes=5, min_distinct_episodes=3, max_failures=0),
    'high': PromotionThreshold(min_successes=0, min_distinct_episodes=0, auto_promote=False),
}


@dataclass(frozen=True)
class ProcedureFeedbackStats:
    procedure_id: str
    root_id: str
    success_count: int
    fail_count: int
    distinct_success_episode_ids: tuple[str, ...]
    distinct_failure_episode_ids: tuple[str, ...]
    latest_success_at: str | None = None
    latest_failure_at: str | None = None

    @property
    def distinct_success_episode_count(self) -> int:
        return len(self.distinct_success_episode_ids)


@dataclass(frozen=True)
class PromotionDecision:
    eligible: bool
    reason: str
    threshold: PromotionThreshold | None = None
    stats: ProcedureFeedbackStats | None = None


class ProcedureEvolution:
    def __init__(self, ledger: ChangeLedger):
        self.ledger = ledger

    def feedback_stats(self, procedure: Procedure | str) -> ProcedureFeedbackStats:
        proc = self._materialize_procedure(procedure)
        success_episode_ids: set[str] = set()
        failure_episode_ids: set[str] = set()
        success_count = 0
        fail_count = 0
        latest_success_at: str | None = None
        latest_failure_at: str | None = None

        for event in self.ledger.events_for_root(proc.root_id):
            target_id = event.target_object_id or event.object_id
            if target_id != proc.object_id:
                continue

            if event.event_type == 'procedure_success':
                success_count += 1
                latest_success_at = event.recorded_at
                episode_id = _episode_id_from_metadata(event.metadata_json)
                if episode_id:
                    success_episode_ids.add(episode_id)
            elif event.event_type == 'procedure_failure':
                fail_count += 1
                latest_failure_at = event.recorded_at
                episode_id = _episode_id_from_metadata(event.metadata_json)
                if episode_id:
                    failure_episode_ids.add(episode_id)

        return ProcedureFeedbackStats(
            procedure_id=proc.object_id,
            root_id=proc.root_id,
            success_count=success_count,
            fail_count=fail_count,
            distinct_success_episode_ids=tuple(sorted(success_episode_ids)),
            distinct_failure_episode_ids=tuple(sorted(failure_episode_ids)),
            latest_success_at=latest_success_at,
            latest_failure_at=latest_failure_at,
        )

    def promotion_decision(self, procedure: Procedure | str) -> PromotionDecision:
        proc = self._materialize_procedure(procedure)
        stats = self.feedback_stats(proc)

        if proc.promotion_status == 'promoted':
            return PromotionDecision(
                eligible=False,
                reason='procedure already promoted',
                threshold=AUTO_PROMOTION_THRESHOLDS[proc.risk_level],
                stats=stats,
            )

        threshold = AUTO_PROMOTION_THRESHOLDS[proc.risk_level]
        if not threshold.auto_promote:
            return PromotionDecision(
                eligible=False,
                reason='high-risk procedures require explicit approval',
                threshold=threshold,
                stats=stats,
            )

        if threshold.max_failures is not None and stats.fail_count > threshold.max_failures:
            return PromotionDecision(
                eligible=False,
                reason='procedure has failures since the latest candidate version',
                threshold=threshold,
                stats=stats,
            )

        if stats.success_count < threshold.min_successes:
            return PromotionDecision(
                eligible=False,
                reason='insufficient successful executions',
                threshold=threshold,
                stats=stats,
            )

        if stats.distinct_success_episode_count < threshold.min_distinct_episodes:
            return PromotionDecision(
                eligible=False,
                reason='insufficient distinct supporting episodes',
                threshold=threshold,
                stats=stats,
            )

        return PromotionDecision(
            eligible=True,
            reason='candidate met conservative repeated-success threshold',
            threshold=threshold,
            stats=stats,
        )

    def maybe_promote(
        self,
        procedure: Procedure | str,
        *,
        actor_id: str = 'policy:procedure-evolution',
        reason: str = 'automatic procedure promotion',
        policy_version: str = 'procedural-memory-v0',
        recorded_at: str | None = None,
    ) -> ChangeEventRow | None:
        proc = self._materialize_procedure(procedure)
        decision = self.promotion_decision(proc)
        if not decision.eligible:
            return None

        return self.ledger.append_event(
            'promote',
            actor_id=actor_id,
            reason=reason,
            recorded_at=recorded_at,
            object_id=proc.object_id,
            object_type=proc.object_type,
            root_id=proc.root_id,
            policy_version=policy_version,
            metadata={
                'risk_level': proc.risk_level,
                'success_count': decision.stats.success_count if decision.stats else 0,
                'distinct_success_episode_count': (
                    decision.stats.distinct_success_episode_count if decision.stats else 0
                ),
            },
        )

    def evolve(
        self,
        procedure: Procedure | str,
        *,
        actor_id: str,
        reason: str,
        revision: dict[str, Any],
        recorded_at: str | None = None,
        event_type: str = 'refine',
        metadata: dict[str, Any] | None = None,
    ) -> Procedure:
        proc = self._materialize_procedure(procedure)
        normalized_event_type = str(event_type or '').strip().lower()
        if normalized_event_type not in {'refine', 'supersede'}:
            raise ValueError("procedure evolution event_type must be 'refine' or 'supersede'")

        updated_fields = {
            key: value
            for key, value in revision.items()
            if key
            in {
                'name',
                'trigger',
                'preconditions',
                'steps',
                'expected_outcome',
                'risk_level',
                'source_lane',
                'source_episode_id',
                'source_message_id',
                'source_key',
                'policy_scope',
                'visibility_scope',
                'extractor_version',
            }
        }
        if 'evidence_refs' in revision:
            updated_fields['evidence_refs'] = [
                item if isinstance(item, EvidenceRef) else EvidenceRef.model_validate(item)
                for item in (revision.get('evidence_refs') or [])
            ]

        next_procedure = proc.model_copy(
            update={
                **updated_fields,
                'object_id': _new_procedure_object_id(),
                'parent_id': proc.object_id,
                'root_id': proc.root_id,
                'version': proc.version + 1,
                'promotion_status': 'proposed',
                'success_count': 0,
                'fail_count': 0,
                'is_current': True,
            }
        )

        self.ledger.append_event(
            normalized_event_type,
            actor_id=actor_id,
            reason=reason,
            recorded_at=recorded_at,
            payload=next_procedure,
            target_object_id=proc.object_id,
            root_id=proc.root_id,
            parent_id=proc.object_id,
            metadata=metadata,
        )
        materialized = self.ledger.materialize_object(next_procedure.object_id)
        if not isinstance(materialized, Procedure):
            raise ValueError('failed to materialize evolved procedure')
        return materialized

    def _materialize_procedure(self, procedure: Procedure | str) -> Procedure:
        if isinstance(procedure, Procedure):
            return procedure
        materialized = self.ledger.materialize_object(procedure)
        if not isinstance(materialized, Procedure):
            raise ValueError(f'Procedure not found: {procedure}')
        return materialized


def _episode_id_from_metadata(metadata_json: str | None) -> str | None:
    if not metadata_json:
        return None
    try:
        payload = json.loads(metadata_json)
    except json.JSONDecodeError:
        return None
    episode_id = payload.get('episode_id')
    if isinstance(episode_id, str) and episode_id.strip():
        return episode_id.strip()
    return None


def _new_procedure_object_id() -> str:
    return f'proc_{uuid4().hex[:24]}'
