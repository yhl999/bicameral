from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from typing import Any

from mcp_server.src.models.typed_memory import EvidenceRef, Procedure
from mcp_server.src.services.change_ledger import ChangeLedger


@dataclass(frozen=True)
class ProcedureAutoPromotionThreshold:
    successes: int
    distinct_episodes: int
    auto_promote: bool


@dataclass(frozen=True)
class ProcedureFeedbackStats:
    evidence_linked_successes: int
    distinct_episode_count: int
    failure_count: int


AUTO_PROMOTION_THRESHOLDS: dict[str, ProcedureAutoPromotionThreshold] = {
    'low': ProcedureAutoPromotionThreshold(successes=3, distinct_episodes=2, auto_promote=True),
    'medium': ProcedureAutoPromotionThreshold(successes=5, distinct_episodes=3, auto_promote=True),
    'high': ProcedureAutoPromotionThreshold(successes=0, distinct_episodes=0, auto_promote=False),
}


class ProcedureEvolutionService:
    """Policy helpers for procedure promotion + version evolution.

    Phase 0 lock:
    - low-risk procedures auto-promote after 3 evidence-linked successes across 2 episodes, 0 failures
    - medium-risk procedures auto-promote after 5 evidence-linked successes across 3 episodes, 0 failures
    - high-risk procedures never auto-promote
    """

    def __init__(self, ledger: ChangeLedger):
        self.ledger = ledger

    def resolve_current(self, identifier: str) -> Procedure:
        candidate = self.ledger.materialize_object(identifier)
        if isinstance(candidate, Procedure):
            current = self.ledger.current_object(candidate.root_id)
            if isinstance(current, Procedure):
                return current

        current = self.ledger.current_object(identifier)
        if isinstance(current, Procedure):
            return current

        row = self.ledger.conn.execute(
            """
            SELECT COALESCE(root_id, object_id) AS root_id
              FROM change_events
             WHERE object_id = ? OR root_id = ?
             ORDER BY recorded_at DESC, event_id DESC
             LIMIT 1
            """,
            (identifier, identifier),
        ).fetchone()
        if row is not None:
            current = self.ledger.current_object(str(row['root_id']))
            if isinstance(current, Procedure):
                return current

        raise ValueError(f'Procedure not found: {identifier}')

    def threshold_for(self, risk_level: str) -> ProcedureAutoPromotionThreshold:
        return AUTO_PROMOTION_THRESHOLDS.get(str(risk_level or '').strip().lower(), AUTO_PROMOTION_THRESHOLDS['high'])

    def feedback_stats(self, identifier: str) -> ProcedureFeedbackStats:
        procedure = self.resolve_current(identifier)
        rows = self.ledger.conn.execute(
            """
            SELECT event_type, metadata_json
              FROM change_events
             WHERE object_id = ?
               AND event_type IN ('procedure_success', 'procedure_failure')
             ORDER BY recorded_at ASC, event_id ASC
            """,
            (procedure.object_id,),
        ).fetchall()

        evidence_linked_successes = 0
        distinct_episode_ids: set[str] = set()
        failure_count = 0

        for row in rows:
            metadata = _coerce_json(row['metadata_json'])
            trusted_feedback = bool(metadata.get('trusted_feedback')) if isinstance(metadata, dict) else False
            if row['event_type'] == 'procedure_failure':
                failure_count += 1
                continue

            evidence_refs = metadata.get('evidence_refs') if isinstance(metadata, dict) else None
            if trusted_feedback and isinstance(evidence_refs, list) and evidence_refs:
                evidence_linked_successes += 1

            if trusted_feedback:
                for episode_id in _episode_ids_from_metadata(metadata):
                    distinct_episode_ids.add(episode_id)

        return ProcedureFeedbackStats(
            evidence_linked_successes=evidence_linked_successes,
            distinct_episode_count=len(distinct_episode_ids),
            failure_count=failure_count,
        )

    def maybe_auto_promote(
        self,
        identifier: str,
        *,
        actor_id: str = 'policy:procedure-v0',
        reason: str = 'procedure_auto_promote',
    ) -> bool:
        procedure = self.resolve_current(identifier)
        if procedure.promotion_status == 'promoted':
            return False

        threshold = self.threshold_for(procedure.risk_level)
        if not threshold.auto_promote:
            return False

        stats = self.feedback_stats(procedure.object_id)
        if procedure.fail_count > 0 or stats.failure_count > 0:
            return False
        if stats.evidence_linked_successes < threshold.successes:
            return False
        if stats.distinct_episode_count < threshold.distinct_episodes:
            return False

        self.ledger.append_event(
            'promote',
            actor_id=actor_id,
            reason=reason,
            object_id=procedure.object_id,
            root_id=procedure.root_id,
        )
        return True

    def evolve_from_feedback(
        self,
        identifier: str,
        *,
        actor_id: str,
        reason: str,
        revised_trigger: str | None = None,
        revised_preconditions: list[str] | None = None,
        revised_steps: list[str] | None = None,
        revised_expected_outcome: str | None = None,
        promote: bool = False,
        evidence_refs: list[EvidenceRef | dict[str, Any]] | None = None,
        source_episode_id: str | None = None,
    ) -> Procedure | None:
        current = self.resolve_current(identifier)

        updates: dict[str, Any] = {}
        if revised_trigger is not None:
            updates['trigger'] = revised_trigger.strip()
        if revised_preconditions is not None:
            updates['preconditions'] = [item.strip() for item in revised_preconditions if isinstance(item, str) and item.strip()]
        if revised_steps is not None:
            cleaned_steps = [item.strip() for item in revised_steps if isinstance(item, str) and item.strip()]
            if not cleaned_steps:
                raise ValueError('revised_steps must contain at least one non-empty step')
            updates['steps'] = cleaned_steps
        if revised_expected_outcome is not None:
            updates['expected_outcome'] = revised_expected_outcome.strip()

        if not updates:
            return None

        merged_evidence = _merge_evidence_refs(current.evidence_refs, evidence_refs or [])
        next_procedure = current.model_copy(
            update={
                **updates,
                'object_id': _new_procedure_id(),
                'success_count': 0,
                'fail_count': 0,
                'promotion_status': 'promoted' if promote else 'proposed',
                'evidence_refs': merged_evidence,
                'source_episode_id': source_episode_id or current.source_episode_id,
            }
        )

        self.ledger.append_event(
            'supersede',
            actor_id=actor_id,
            reason=reason,
            payload=next_procedure,
            target_object_id=current.object_id,
        )
        if promote:
            self.ledger.append_event(
                'promote',
                actor_id=actor_id,
                reason=f'{reason}:promote',
                object_id=next_procedure.object_id,
                root_id=current.root_id,
            )
        return self.resolve_current(current.root_id)



def _coerce_json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except Exception:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}



def _episode_ids_from_metadata(metadata: dict[str, Any]) -> list[str]:
    episode_ids: list[str] = []
    raw_single = metadata.get('episode_id') if isinstance(metadata, dict) else None
    if isinstance(raw_single, str) and raw_single.strip():
        episode_ids.append(raw_single.strip())

    raw_many = metadata.get('episode_ids') if isinstance(metadata, dict) else None
    if isinstance(raw_many, list):
        episode_ids.extend(
            item.strip() for item in raw_many if isinstance(item, str) and item.strip()
        )
    return sorted(set(episode_ids))



def _merge_evidence_refs(
    existing: list[EvidenceRef],
    incoming: list[EvidenceRef | dict[str, Any]],
) -> list[EvidenceRef]:
    merged: dict[str, EvidenceRef] = {}
    for ref in existing:
        merged[ref.canonical_uri] = ref
    for raw in incoming:
        ref = raw if isinstance(raw, EvidenceRef) else EvidenceRef.from_legacy_ref(raw)
        merged[ref.canonical_uri] = ref
    return [merged[key] for key in sorted(merged)]



def _new_procedure_id() -> str:
    return f'proc_{secrets.token_hex(12)}'
