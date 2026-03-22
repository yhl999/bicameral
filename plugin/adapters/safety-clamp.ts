/**
 * Downstream safety clamp for Intercameral decision payloads.
 *
 * This is the final defensive layer in the Bicameral plugin, applied after
 * the Intercameral adapter has accepted and validated a payload but before
 * the result drives the prompt build phase.
 *
 * The clamp performs:
 *   1. Evidence-envelope schema validation (refs are clean strings)
 *   2. Channel policy enforcement (group abstain, private warn-low-confidence)
 *   3. Contradiction signal safety check
 *   4. Final semantic coherence verification
 *
 * WHEN THE CLAMP FIRES (override policy):
 *   - Channel policy / safety requires it (e.g., group + low confidence)
 *   - Required semantic fields are missing (defensive re-check)
 *   - Evidence envelope contains invalid / potentially unsafe refs
 *   - Payload is otherwise malformed or untrustworthy
 *
 * The clamp does NOT re-implement Intercameral routing logic.  It only
 * validates the output surface and enforces downstream Bicameral invariants.
 * When it fires it emits a deterministic safe_fallback decision with an
 * explicit override reason for observability.
 *
 * OVERRIDE SHAPE:
 *   route:             safe_fallback
 *   reason_code:       safety_malformed_state  (safety conditions)
 *                   or fallback_contract_invalid (schema violations)
 *   fallback_directive: use_simple_local_path | abstain_clarify
 */

import type { ValidatedDecision, RouteClass, ReasonCode, FallbackDirective, ContradictionSignal } from './intercameral-client.ts';
import { SUPPORTED_CONTRACT_VERSION } from './intercameral-client.ts';

// ---------------------------------------------------------------------------
// Clamp context — channel policy parameters
// ---------------------------------------------------------------------------

export interface ClampContext {
  /**
   * Channel type from the OpenClaw message provider.
   * Used to enforce group-channel abstain policy.
   */
  channelType?: 'private' | 'group' | 'thread' | 'unknown';
  /**
   * Minimum confidence threshold for group channels.
   * Below this → override to abstain_clarify.
   * Default: 0.7 (matches spec/intercameral_decision_contract.v1.schema.json default).
   */
  groupAbstainBelowConfidence?: number;
  /**
   * Minimum confidence threshold for private channels.
   * Below this → add warn_low_confidence fallback directive.
   * Default: 0.6 (matches spec default).
   */
  privateWarnMinConfidence?: number;
}

const DEFAULT_GROUP_ABSTAIN_BELOW = 0.7;
const DEFAULT_PRIVATE_WARN_MIN = 0.6;

// ---------------------------------------------------------------------------
// Evidence-envelope validation constraints
// ---------------------------------------------------------------------------

/**
 * Maximum allowed length for a single evidence reference string.
 * Refs beyond this are suspicious and are rejected by the clamp.
 */
const MAX_EVIDENCE_REF_LENGTH = 512;

/**
 * Pattern matching unsafe / injection-risk characters in evidence refs.
 * Refs must be plain identifiers or file-path-like strings — not markup.
 */
const UNSAFE_EVIDENCE_REF_RE = /[<>"'`\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/;

// ---------------------------------------------------------------------------
// Clamp result
// ---------------------------------------------------------------------------

export type ClampPassResult = {
  readonly pass: true;
  /** The decision to use downstream — may be the original or an annotated copy. */
  readonly decision: ValidatedDecision;
};

export type ClampOverrideResult = {
  readonly pass: false;
  /** Human-readable reason the clamp fired. */
  readonly overrideReason: string;
  /** Deterministic safe_fallback decision to use instead of the original. */
  readonly override: ValidatedDecision;
};

export type ClampResult = ClampPassResult | ClampOverrideResult;

// ---------------------------------------------------------------------------
// Safe-fallback override factory
// ---------------------------------------------------------------------------

function makeSafeOverride(
  requestId: string,
  reasonCode: ReasonCode & ('safety_malformed_state' | 'fallback_contract_invalid'),
  fallbackDirective: FallbackDirective,
): ValidatedDecision {
  return {
    version: SUPPORTED_CONTRACT_VERSION,
    request_id: requestId,
    route: 'safe_fallback' as RouteClass,
    reason_code: reasonCode,
    confidence: 0 as number,
    fallback_directive: fallbackDirective,
    evidence_refs: [] as readonly string[],
    contradiction_signal: 'none' as ContradictionSignal,
  };
}

function makeOverride(
  reason: string,
  override: ValidatedDecision,
): ClampOverrideResult {
  return { pass: false, overrideReason: reason, override };
}

// ---------------------------------------------------------------------------
// Evidence-envelope validation
// ---------------------------------------------------------------------------

/**
 * Validates the evidence_refs array from a ValidatedDecision.
 *
 * Returns null if valid, or a rejection reason string if invalid.
 *
 * Evidence refs must be:
 *   - Non-empty strings (after trim)
 *   - No longer than MAX_EVIDENCE_REF_LENGTH chars
 *   - Free of markup / injection-risk characters
 */
function validateEvidenceEnvelope(refs: readonly string[]): string | null {
  for (let i = 0; i < refs.length; i++) {
    const ref = refs[i];
    if (ref.trim().length === 0) {
      return `evidence_refs[${i}] is an empty or whitespace-only string`;
    }
    if (ref.length > MAX_EVIDENCE_REF_LENGTH) {
      return `evidence_refs[${i}] exceeds maximum length (${ref.length} > ${MAX_EVIDENCE_REF_LENGTH})`;
    }
    if (UNSAFE_EVIDENCE_REF_RE.test(ref)) {
      return `evidence_refs[${i}] contains unsafe characters (potential injection vector)`;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Main safety clamp — pure function, no I/O
// ---------------------------------------------------------------------------

/**
 * Applies downstream safety checks to an already-validated Intercameral
 * decision.
 *
 * Call this immediately after `validateIntercameralDecision` returns
 * trusted=true, before the decision influences any prompt build.
 */
export function applyDownstreamSafetyClamp(
  decision: ValidatedDecision,
  context: ClampContext = {},
): ClampResult {
  const {
    channelType,
    groupAbstainBelowConfidence = DEFAULT_GROUP_ABSTAIN_BELOW,
    privateWarnMinConfidence = DEFAULT_PRIVATE_WARN_MIN,
  } = context;

  // 1. Evidence-envelope schema validation
  const evidenceError = validateEvidenceEnvelope(decision.evidence_refs);
  if (evidenceError !== null) {
    return makeOverride(
      `evidence envelope invalid: ${evidenceError}`,
      makeSafeOverride(decision.request_id, 'fallback_contract_invalid', 'use_simple_local_path'),
    );
  }

  // 2. Contradiction signal safety: detected contradiction → escalate
  if (decision.contradiction_signal === 'detected') {
    // If Intercameral already picked a safe route, allow it through.
    // If it picked anything other than safe_fallback or chamberlain_single_chain,
    // the clamp overrides — contradiction must not silently reach a narrow route.
    const safeContradictionRoutes: ReadonlySet<RouteClass> = new Set([
      'safe_fallback',
      'chamberlain_single_chain',
      'clarify_first',
    ]);
    if (!safeContradictionRoutes.has(decision.route)) {
      return makeOverride(
        `contradiction_signal=detected but route="${decision.route}" is not a safe contradiction-handling route`,
        makeSafeOverride(decision.request_id, 'safety_malformed_state', 'abstain_clarify'),
      );
    }
  }

  // 3. Group-channel abstain policy
  if (channelType === 'group') {
    if (decision.confidence < groupAbstainBelowConfidence) {
      return makeOverride(
        `group channel policy: confidence ${decision.confidence.toFixed(3)} < threshold ${groupAbstainBelowConfidence} — abstaining`,
        makeSafeOverride(decision.request_id, 'safety_malformed_state', 'abstain_clarify'),
      );
    }
  }

  // 4. Private-channel low-confidence annotation
  // The clamp does not block the decision, but upgrades the fallback_directive
  // to warn_low_confidence so the caller can surface a degraded-confidence signal.
  if (channelType === 'private' && decision.confidence < privateWarnMinConfidence) {
    const annotated: ValidatedDecision = {
      ...decision,
      fallback_directive: 'warn_low_confidence',
    };
    return { pass: true, decision: annotated };
  }

  // 5. Defensive re-check: safe_fallback route must not carry escalate_chamberlain
  // directive (would create a confusing mixed signal downstream).
  if (
    decision.route === 'safe_fallback' &&
    decision.fallback_directive === 'escalate_chamberlain'
  ) {
    return makeOverride(
      'safe_fallback route cannot carry fallback_directive=escalate_chamberlain (semantic conflict)',
      makeSafeOverride(decision.request_id, 'fallback_contract_invalid', 'use_simple_local_path'),
    );
  }

  // All checks passed — decision is safe to use
  return { pass: true, decision };
}

// ---------------------------------------------------------------------------
// Convenience: run adapter result through safety clamp in one call
// ---------------------------------------------------------------------------

/**
 * Applies the safety clamp to the output of `validateIntercameralDecision`.
 *
 * If the adapter rejected the payload, the fallback decision is returned
 * directly (it is already a safe_fallback shape and needs no further clamping).
 *
 * If the adapter accepted the payload, the clamp is applied and the result
 * returned.
 */
export function clampAdapterResult(
  adapterResult: import('./intercameral-client.ts').AdapterResult,
  context?: ClampContext,
): ClampResult {
  if (!adapterResult.trusted) {
    // Fallback decisions are already safe — pass them through unchanged.
    return { pass: true, decision: adapterResult.fallback as unknown as ValidatedDecision };
  }
  return applyDownstreamSafetyClamp(adapterResult.decision, context);
}
