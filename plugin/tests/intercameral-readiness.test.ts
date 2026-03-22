/**
 * Intercameral Readiness test suite
 *
 * Covers:
 *   - Adapter accepts semantically valid Intercameral payloads (US-001)
 *   - Adapter rejects malformed / semantically incompatible payloads (US-002)
 *   - Fallback behaviour is deterministic and observable (US-003)
 *   - Downstream safety clamp evidence-envelope validation (FR-6)
 *   - Safety clamp channel policy enforcement (FR-3)
 *
 * Task: task-bicameral-intercameral-readiness-v2
 * PRD:  projects/intercameral/prd/EXEC-BICAMERAL-INTERCAMERAL-READINESS-v2.md
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import {
  validateIntercameralDecision,
  callIntercameralBroker,
} from '../adapters/intercameral-client.ts';
import type { AdapterResult, ValidatedDecision } from '../adapters/intercameral-client.ts';

import {
  applyDownstreamSafetyClamp,
  clampAdapterResult,
} from '../adapters/safety-clamp.ts';
import type { ClampContext } from '../adapters/safety-clamp.ts';

// ---------------------------------------------------------------------------
// Helper: canonical smoke payload (matches spec/decision_smoke_payload.v1.json)
// ---------------------------------------------------------------------------

function makeSmoke(overrides: Record<string, unknown> = {}): unknown {
  return {
    version: 'v1',
    request: {
      request_id: 'smoke-001',
      session_key: 'smoke-test',
      prompt: 'ok',
      channel_type: 'private',
      context_budget_tokens: 512,
      policy: {
        allow_cloud_fallback: false,
        worker_mode_enabled: false,
        private_warn_min_confidence: 0.6,
        group_abstain_below_confidence: 0.7,
      },
    },
    response: {
      request_id: 'smoke-001',
      route: 'inject_none',
      reason_code: 'low_info_skip',
      confidence: 1.0,
      fallback_directive: 'none',
      evidence_refs: [],
      contradiction_signal: 'none',
      ...overrides,
    },
  };
}

// Helpers to assert specific result shapes
function assertAccepted(result: AdapterResult): ValidatedDecision {
  assert.equal(result.trusted, true, `Expected trusted=true, got reason: ${result.trusted ? '' : result.reason}`);
  if (!result.trusted) throw new Error('not reached');
  return result.decision;
}

function assertRejected(result: AdapterResult, reasonFragment?: string): void {
  assert.equal(result.trusted, false, 'Expected trusted=false (payload should have been rejected)');
  if (!result.trusted && reasonFragment) {
    assert.ok(
      result.reason.includes(reasonFragment),
      `Rejection reason "${result.reason}" does not include "${reasonFragment}"`,
    );
  }
}

// ── US-001: Valid Intercameral payloads ────────────────────────────────────

test('intercameral-readiness: smoke payload is accepted', () => {
  const result = validateIntercameralDecision(makeSmoke());
  const d = assertAccepted(result);
  assert.equal(d.version, 'v1');
  assert.equal(d.route, 'inject_none');
  assert.equal(d.reason_code, 'low_info_skip');
  assert.equal(d.confidence, 1.0);
  assert.equal(d.fallback_directive, 'none');
  assert.deepEqual(d.evidence_refs, []);
  assert.equal(d.contradiction_signal, 'none');
});

test('intercameral-readiness: task_update_narrow decision is accepted', () => {
  const result = validateIntercameralDecision(makeSmoke({
    route: 'task_state',
    reason_code: 'task_update_narrow',
    confidence: 0.85,
    fallback_directive: 'none',
    evidence_refs: ['task:api-impl', 'step:write-tests'],
    contradiction_signal: 'none',
  }));
  const d = assertAccepted(result);
  assert.equal(d.route, 'task_state');
  assert.equal(d.reason_code, 'task_update_narrow');
  assert.deepEqual(d.evidence_refs.slice(), ['task:api-impl', 'step:write-tests']);
});

test('intercameral-readiness: direct_retrieval decision is accepted', () => {
  const result = validateIntercameralDecision(makeSmoke({
    route: 'direct_retrieval',
    reason_code: 'direct_recall',
    confidence: 0.75,
    fallback_directive: 'none',
    evidence_refs: [],
    contradiction_signal: 'none',
  }));
  assertAccepted(result);
});

test('intercameral-readiness: safe_fallback / fallback_timeout is accepted', () => {
  const result = validateIntercameralDecision(makeSmoke({
    route: 'safe_fallback',
    reason_code: 'fallback_timeout',
    confidence: 0.0,
    fallback_directive: 'use_simple_local_path',
    evidence_refs: [],
    contradiction_signal: 'none',
  }));
  const d = assertAccepted(result);
  assert.equal(d.route, 'safe_fallback');
  assert.equal(d.reason_code, 'fallback_timeout');
  assert.equal(d.confidence, 0.0);
});

test('intercameral-readiness: extra unknown fields in response are allowed (tolerant extension)', () => {
  const payload = makeSmoke() as Record<string, unknown>;
  const resp = (payload['response'] as Record<string, unknown>);
  resp['unknown_future_field'] = 'some_value_from_v2';
  resp['another_extra'] = 42;
  const result = validateIntercameralDecision(payload);
  assertAccepted(result);
});

test('intercameral-readiness: missing optional diagnostics field is allowed', () => {
  const result = validateIntercameralDecision(makeSmoke());
  const d = assertAccepted(result);
  assert.equal(d.diagnostics, undefined);
});

test('intercameral-readiness: present diagnostics block is accepted and decoded', () => {
  const payload = makeSmoke() as Record<string, unknown>;
  const resp = (payload['response'] as Record<string, unknown>);
  resp['diagnostics'] = {
    latency_ms: 120.5,
    timed_out: false,
    retried: false,
    model: 'local-thalamus-v1',
    veto_applied: true,
  };
  const result = validateIntercameralDecision(payload);
  const d = assertAccepted(result);
  assert.ok(d.diagnostics);
  assert.equal(d.diagnostics.latency_ms, 120.5);
  assert.equal(d.diagnostics.veto_applied, true);
  assert.equal(d.diagnostics.model, 'local-thalamus-v1');
});

test('intercameral-readiness: all canonical routes + reason_codes are accepted', () => {
  const validPairs: Array<[string, string, string, string]> = [
    ['inject_none',              'low_info_skip',                  'none',                    'none'],
    ['task_state',               'task_update_narrow',             'none',                    'none'],
    ['direct_retrieval',         'direct_recall',                  'none',                    'none'],
    ['pack_only',                'pack_select',                    'none',                    'none'],
    ['clarify_first',            'ambiguous_needs_clarification',  'abstain_clarify',         'none'],
    ['chamberlain_single_chain', 'contradiction_risk',             'escalate_chamberlain',    'detected'],
    ['chamberlain_single_chain', 'complex_multi_hop',              'escalate_chamberlain',    'none'],
    ['safe_fallback',            'safety_malformed_state',         'use_simple_local_path',   'none'],
    ['safe_fallback',            'fallback_contract_invalid',      'use_simple_local_path',   'none'],
    ['safe_fallback',            'fallback_runtime_error',         'use_simple_local_path',   'none'],
  ];

  for (const [route, reason_code, fallback_directive, contradiction_signal] of validPairs) {
    const result = validateIntercameralDecision(makeSmoke({
      route,
      reason_code,
      confidence: 0.9,
      fallback_directive,
      evidence_refs: [],
      contradiction_signal,
    }));
    assert.equal(
      result.trusted,
      true,
      `Expected route=${route}/reason_code=${reason_code} to be accepted, got: ${result.trusted ? '' : (result as { reason: string }).reason}`,
    );
  }
});

// ── US-002: Malformed / semantically incompatible payloads ────────────────

test('intercameral-readiness: unknown reason_code is rejected', () => {
  const result = validateIntercameralDecision(makeSmoke({
    reason_code: 'some_future_unknown_code',
    route: 'inject_none',
  }));
  assertRejected(result, 'unknown reason_code');
});

test('intercameral-readiness: unknown route is rejected', () => {
  const result = validateIntercameralDecision(makeSmoke({
    route: 'future_route_class',
    reason_code: 'low_info_skip',
  }));
  assertRejected(result, 'unknown route');
});

test('intercameral-readiness: higher contract version without compat marker is rejected', () => {
  const payload = { ...(makeSmoke() as Record<string, unknown>), version: 'v2' };
  const result = validateIntercameralDecision(payload);
  assertRejected(result, 'unsupported contract version');
  assert.ok((result as { reason: string }).reason.includes('v2'));
});

test('intercameral-readiness: v3 without compat marker is rejected', () => {
  const payload = { ...(makeSmoke() as Record<string, unknown>), version: 'v3' };
  const result = validateIntercameralDecision(payload);
  assertRejected(result, 'unsupported contract version');
});

test('intercameral-readiness: higher version WITH compat marker is accepted', () => {
  const payload = { ...(makeSmoke() as Record<string, unknown>), version: 'v2-compat-v1' };
  const result = validateIntercameralDecision(payload);
  const d = assertAccepted(result);
  // Normalised to v1 internally
  assert.equal(d.version, 'v1');
});

test('intercameral-readiness: v3-compat-v1 is accepted', () => {
  const payload = { ...(makeSmoke() as Record<string, unknown>), version: 'v3-compat-v1' };
  const result = validateIntercameralDecision(payload);
  assertAccepted(result);
});

test('intercameral-readiness: null payload is rejected', () => {
  const result = validateIntercameralDecision(null);
  assertRejected(result);
});

test('intercameral-readiness: non-object payload (string) is rejected', () => {
  const result = validateIntercameralDecision('not an object');
  assertRejected(result);
});

test('intercameral-readiness: array payload is rejected', () => {
  const result = validateIntercameralDecision([]);
  assertRejected(result);
});

test('intercameral-readiness: missing response.route is rejected', () => {
  const payload = makeSmoke() as Record<string, unknown>;
  const resp = payload['response'] as Record<string, unknown>;
  delete resp['route'];
  const result = validateIntercameralDecision(payload);
  assertRejected(result, 'route');
});

test('intercameral-readiness: missing response.reason_code is rejected', () => {
  const payload = makeSmoke() as Record<string, unknown>;
  const resp = payload['response'] as Record<string, unknown>;
  delete resp['reason_code'];
  const result = validateIntercameralDecision(payload);
  assertRejected(result, 'reason_code');
});

test('intercameral-readiness: missing response.confidence is rejected', () => {
  const payload = makeSmoke() as Record<string, unknown>;
  const resp = payload['response'] as Record<string, unknown>;
  delete resp['confidence'];
  const result = validateIntercameralDecision(payload);
  assertRejected(result, 'confidence');
});

test('intercameral-readiness: missing response.fallback_directive is rejected', () => {
  const payload = makeSmoke() as Record<string, unknown>;
  const resp = payload['response'] as Record<string, unknown>;
  delete resp['fallback_directive'];
  const result = validateIntercameralDecision(payload);
  assertRejected(result, 'fallback_directive');
});

test('intercameral-readiness: missing response.evidence_refs is rejected', () => {
  const payload = makeSmoke() as Record<string, unknown>;
  const resp = payload['response'] as Record<string, unknown>;
  delete resp['evidence_refs'];
  const result = validateIntercameralDecision(payload);
  assertRejected(result, 'evidence_refs');
});

test('intercameral-readiness: missing response.contradiction_signal is rejected', () => {
  const payload = makeSmoke() as Record<string, unknown>;
  const resp = payload['response'] as Record<string, unknown>;
  delete resp['contradiction_signal'];
  const result = validateIntercameralDecision(payload);
  assertRejected(result, 'contradiction_signal');
});

test('intercameral-readiness: confidence below 0 is rejected', () => {
  const result = validateIntercameralDecision(makeSmoke({ confidence: -0.1 }));
  assertRejected(result, 'confidence');
});

test('intercameral-readiness: confidence above 1 is rejected', () => {
  const result = validateIntercameralDecision(makeSmoke({ confidence: 1.1 }));
  assertRejected(result, 'confidence');
});

test('intercameral-readiness: non-string entry in evidence_refs is rejected', () => {
  const result = validateIntercameralDecision(makeSmoke({ evidence_refs: ['valid-ref', 42] }));
  assertRejected(result, 'non-string');
});

test('intercameral-readiness: route / reason_code semantic mismatch is rejected', () => {
  // reason_code=low_info_skip maps to route=inject_none; using task_state is wrong
  const result = validateIntercameralDecision(makeSmoke({
    route: 'task_state',
    reason_code: 'low_info_skip',
  }));
  assertRejected(result, 'semantic mismatch');
});

test('intercameral-readiness: unknown fallback_directive is rejected', () => {
  const result = validateIntercameralDecision(makeSmoke({
    fallback_directive: 'future_unknown_directive',
  }));
  assertRejected(result, 'unknown fallback_directive');
});

test('intercameral-readiness: unknown contradiction_signal is rejected', () => {
  const result = validateIntercameralDecision(makeSmoke({
    contradiction_signal: 'strongly_possible',
  }));
  assertRejected(result, 'unknown contradiction_signal');
});

// ── US-003: Fallback behaviour ─────────────────────────────────────────────

test('intercameral-readiness: fallback decision uses safe_fallback route', () => {
  const result = validateIntercameralDecision(null);
  assert.equal(result.trusted, false);
  if (!result.trusted) {
    assert.equal(result.fallback.route, 'safe_fallback');
    assert.equal(result.fallback.fallback_directive, 'use_simple_local_path');
    assert.equal(result.fallback.confidence, 0);
    assert.equal(result.fallback.contradiction_signal, 'none');
    assert.deepEqual(result.fallback.evidence_refs.slice(), []);
  }
});

test('intercameral-readiness: contract-invalid payload produces fallback_contract_invalid reason_code', () => {
  const result = validateIntercameralDecision({ version: 'v1', response: 'not-an-object' });
  assert.equal(result.trusted, false);
  if (!result.trusted) {
    assert.equal(result.fallback.reason_code, 'fallback_contract_invalid');
  }
});

test('intercameral-readiness: HTTP client maps timeout to fallback_timeout', async (t: unknown) => {
  const originalFetch = globalThis.fetch;
  (t as { after: (fn: () => void) => void }).after(() => {
    globalThis.fetch = originalFetch;
  });

  // Mock fetch to simulate AbortError (timeout)
  globalThis.fetch = (async () => {
    const err = new Error('The operation was aborted');
    err.name = 'AbortError';
    throw err;
  }) as typeof fetch;

  const result = await callIntercameralBroker(
    { request_id: 'timeout-test', prompt: 'hello' },
    { timeoutMs: 1 },
  );

  assert.equal(result.trusted, false);
  if (!result.trusted) {
    assert.equal(result.fallback.reason_code, 'fallback_timeout');
    assert.equal(result.fallback.route, 'safe_fallback');
  }
});

test('intercameral-readiness: HTTP client maps contract-invalid JSON to fallback_contract_invalid', async (t: unknown) => {
  const originalFetch = globalThis.fetch;
  (t as { after: (fn: () => void) => void }).after(() => {
    globalThis.fetch = originalFetch;
  });

  globalThis.fetch = (async () =>
    new Response('not json', { status: 200, headers: { 'content-type': 'text/plain' } })
  ) as typeof fetch;

  const result = await callIntercameralBroker({ request_id: 'invalid-json', prompt: 'test' });
  assert.equal(result.trusted, false);
  if (!result.trusted) {
    assert.equal(result.fallback.reason_code, 'fallback_contract_invalid');
  }
});

test('intercameral-readiness: HTTP client maps HTTP 400 to fallback_contract_invalid (no retry)', async (t: unknown) => {
  const originalFetch = globalThis.fetch;
  let callCount = 0;
  (t as { after: (fn: () => void) => void }).after(() => {
    globalThis.fetch = originalFetch;
  });

  globalThis.fetch = (async () => {
    callCount += 1;
    return new Response('Bad request', { status: 400 });
  }) as typeof fetch;

  const result = await callIntercameralBroker({ request_id: 'bad-req', prompt: 'test' });
  assert.equal(result.trusted, false);
  if (!result.trusted) {
    assert.equal(result.fallback.reason_code, 'fallback_contract_invalid');
  }
  // Must not retry on 4xx
  assert.equal(callCount, 1, '4xx responses must not be retried');
});

test('intercameral-readiness: HTTP client passes validated payload when server returns valid response', async (t: unknown) => {
  const originalFetch = globalThis.fetch;
  (t as { after: (fn: () => void) => void }).after(() => {
    globalThis.fetch = originalFetch;
  });

  const validPayload = makeSmoke();
  globalThis.fetch = (async () =>
    new Response(JSON.stringify(validPayload), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    })
  ) as typeof fetch;

  const result = await callIntercameralBroker({ request_id: 'smoke-001', prompt: 'ok' });
  const d = assertAccepted(result);
  assert.equal(d.route, 'inject_none');
});

// ── FR-6: Downstream safety clamp — evidence-envelope validation ──────────

test('intercameral-readiness: safety clamp passes valid decision with clean evidence refs', () => {
  const result = validateIntercameralDecision(makeSmoke({
    route: 'task_state',
    reason_code: 'task_update_narrow',
    confidence: 0.9,
    evidence_refs: ['task:deploy-api', 'step:integration-tests'],
    contradiction_signal: 'none',
    fallback_directive: 'none',
  }));
  const d = assertAccepted(result);
  const clamp = applyDownstreamSafetyClamp(d);
  assert.equal(clamp.pass, true);
});

test('intercameral-readiness: safety clamp rejects empty string evidence ref', () => {
  const result = validateIntercameralDecision(makeSmoke({
    evidence_refs: ['valid-ref', '   ', 'another-ref'],
  }));
  const d = assertAccepted(result);
  const clamp = applyDownstreamSafetyClamp(d);
  assert.equal(clamp.pass, false);
  if (!clamp.pass) {
    assert.ok(clamp.overrideReason.includes('empty or whitespace'));
  }
});

test('intercameral-readiness: safety clamp rejects evidence ref with HTML injection chars', () => {
  const result = validateIntercameralDecision(makeSmoke({
    evidence_refs: ['<script>alert(1)</script>'],
  }));
  const d = assertAccepted(result);
  const clamp = applyDownstreamSafetyClamp(d);
  assert.equal(clamp.pass, false);
  if (!clamp.pass) {
    assert.ok(clamp.overrideReason.includes('unsafe characters'));
  }
});

test('intercameral-readiness: safety clamp rejects evidence ref with null byte', () => {
  const result = validateIntercameralDecision(makeSmoke({ evidence_refs: ['ref\x00injected'] }));
  const d = assertAccepted(result);
  const clamp = applyDownstreamSafetyClamp(d);
  assert.equal(clamp.pass, false);
  if (!clamp.pass) {
    assert.ok(clamp.overrideReason.includes('unsafe characters'));
  }
});

test('intercameral-readiness: safety clamp rejects evidence ref exceeding max length', () => {
  const result = validateIntercameralDecision(makeSmoke({ evidence_refs: ['a'.repeat(513)] }));
  const d = assertAccepted(result);
  const clamp = applyDownstreamSafetyClamp(d);
  assert.equal(clamp.pass, false);
  if (!clamp.pass) {
    assert.ok(clamp.overrideReason.includes('maximum length'));
  }
});

// ── FR-3: Safety clamp channel policy enforcement ─────────────────────────

test('intercameral-readiness: safety clamp group-channel low-confidence triggers abstain override', () => {
  const result = validateIntercameralDecision(makeSmoke({
    route: 'direct_retrieval',
    reason_code: 'direct_recall',
    confidence: 0.5,  // below 0.7 group threshold
    fallback_directive: 'none',
    evidence_refs: [],
    contradiction_signal: 'none',
  }));
  const d = assertAccepted(result);
  const clamp = applyDownstreamSafetyClamp(d, { channelType: 'group' });
  assert.equal(clamp.pass, false);
  if (!clamp.pass) {
    assert.ok(clamp.overrideReason.includes('group channel policy'));
    assert.equal(clamp.override.fallback_directive, 'abstain_clarify');
    assert.equal(clamp.override.route, 'safe_fallback');
  }
});

test('intercameral-readiness: safety clamp group-channel sufficient-confidence passes', () => {
  const result = validateIntercameralDecision(makeSmoke({
    route: 'direct_retrieval',
    reason_code: 'direct_recall',
    confidence: 0.85,
    fallback_directive: 'none',
    evidence_refs: [],
    contradiction_signal: 'none',
  }));
  const d = assertAccepted(result);
  const clamp = applyDownstreamSafetyClamp(d, { channelType: 'group' });
  assert.equal(clamp.pass, true);
});

test('intercameral-readiness: safety clamp private-channel low-confidence annotates warn_low_confidence', () => {
  const result = validateIntercameralDecision(makeSmoke({
    route: 'direct_retrieval',
    reason_code: 'direct_recall',
    confidence: 0.4,  // below 0.6 private warn threshold
    fallback_directive: 'none',
    evidence_refs: [],
    contradiction_signal: 'none',
  }));
  const d = assertAccepted(result);
  const clamp = applyDownstreamSafetyClamp(d, { channelType: 'private' });
  assert.equal(clamp.pass, true);
  if (clamp.pass) {
    assert.equal(clamp.decision.fallback_directive, 'warn_low_confidence');
  }
});

test('intercameral-readiness: safety clamp contradiction=detected on unsafe route triggers override', () => {
  const result = validateIntercameralDecision(makeSmoke({
    route: 'inject_none',
    reason_code: 'low_info_skip',
    confidence: 0.99,
    fallback_directive: 'none',
    evidence_refs: [],
    contradiction_signal: 'none',  // starts as none, we'll override below
  }));
  // Simulate a payload that slips through validation somehow with a mismatch.
  // We test the safety clamp directly by constructing a valid-but-unsafe decision:
  const d = assertAccepted(result);
  // Manually construct a decision with contradiction=detected but route=inject_none
  // (this combination is dangerous — inject_none should not be used when detected).
  const unsafeDecision: ValidatedDecision = {
    ...d,
    route: 'pack_only',
    reason_code: 'pack_select',
    contradiction_signal: 'detected',
  };
  const clamp = applyDownstreamSafetyClamp(unsafeDecision);
  assert.equal(clamp.pass, false);
  if (!clamp.pass) {
    assert.ok(clamp.overrideReason.includes('contradiction_signal=detected'));
    assert.equal(clamp.override.route, 'safe_fallback');
    assert.equal(clamp.override.fallback_directive, 'abstain_clarify');
  }
});

test('intercameral-readiness: contradiction=detected on chamberlain_single_chain passes clamp', () => {
  const result = validateIntercameralDecision(makeSmoke({
    route: 'chamberlain_single_chain',
    reason_code: 'contradiction_risk',
    confidence: 0.72,
    fallback_directive: 'escalate_chamberlain',
    evidence_refs: [],
    contradiction_signal: 'detected',
  }));
  const d = assertAccepted(result);
  const clamp = applyDownstreamSafetyClamp(d);
  assert.equal(clamp.pass, true, 'chamberlain route is safe for detected contradiction');
});

test('intercameral-readiness: safe_fallback + escalate_chamberlain is blocked by clamp', () => {
  const result = validateIntercameralDecision(makeSmoke({
    route: 'safe_fallback',
    reason_code: 'safety_malformed_state',
    confidence: 0.0,
    fallback_directive: 'none',
    evidence_refs: [],
    contradiction_signal: 'none',
  }));
  const d = assertAccepted(result);
  // Craft the unsafe combination directly
  const badDecision: ValidatedDecision = {
    ...d,
    fallback_directive: 'escalate_chamberlain',
  };
  const clamp = applyDownstreamSafetyClamp(badDecision);
  assert.equal(clamp.pass, false);
  if (!clamp.pass) {
    assert.ok(clamp.overrideReason.includes('escalate_chamberlain'));
  }
});

// ── clampAdapterResult convenience helper ─────────────────────────────────

test('intercameral-readiness: clampAdapterResult passes rejected fallback through as-is', () => {
  const adapterResult = validateIntercameralDecision(null);
  assert.equal(adapterResult.trusted, false);
  const clamp = clampAdapterResult(adapterResult);
  // Fallback decisions are safe — clamp passes them through
  assert.equal(clamp.pass, true);
  if (clamp.pass) {
    assert.equal(clamp.decision.route, 'safe_fallback');
  }
});

test('intercameral-readiness: clampAdapterResult applies safety clamp to accepted decisions', () => {
  const adapterResult = validateIntercameralDecision(makeSmoke({
    route: 'direct_retrieval',
    reason_code: 'direct_recall',
    confidence: 0.5,
    fallback_directive: 'none',
    evidence_refs: [],
    contradiction_signal: 'none',
  }));
  const clamp = clampAdapterResult(adapterResult, { channelType: 'group' });
  // Group channel with 0.5 confidence should be clamped
  assert.equal(clamp.pass, false);
});

// ── False-pass guards (per PRD) ────────────────────────────────────────────

test('intercameral-readiness: schema-valid payload with unknown reason_code fails (not best-effort)', () => {
  // The payload is structurally valid JSON with all fields present, but
  // reason_code is not in the registry.  Must NOT be accepted.
  const payload = makeSmoke({ reason_code: 'exotic_unregistered_code' });
  const result = validateIntercameralDecision(payload);
  assert.equal(result.trusted, false, 'Unknown reason codes must never be accepted via best-effort');
});

test('intercameral-readiness: missing required fields in otherwise valid schema are rejected', () => {
  // All required fields except evidence_refs — must fail-closed
  const payload = makeSmoke() as Record<string, unknown>;
  const resp = payload['response'] as Record<string, unknown>;
  delete resp['evidence_refs'];
  delete resp['contradiction_signal'];

  const result = validateIntercameralDecision(payload);
  assert.equal(result.trusted, false, 'Schema-valid envelope with missing required semantics must be rejected');
});

// ── P2 findings: NaN guard + retry behaviour ───────────────────────────────

test('intercameral-readiness: NaN confidence is rejected (NaN guard)', () => {
  // NaN is typeof 'number' so it passes the type check, but NaN < 0 and NaN > 1
  // are both false — without Number.isFinite(), NaN would be accepted as valid.
  const result = validateIntercameralDecision(makeSmoke({ confidence: NaN }));
  assertRejected(result, 'confidence');
  if (!result.trusted) {
    assert.equal(result.fallback.reason_code, 'fallback_contract_invalid');
  }
});

test('intercameral-readiness: retries 5xx response up to maxRetries limit then returns fallback_runtime_error', async (t: unknown) => {
  const originalFetch = globalThis.fetch;
  let callCount = 0;
  (t as { after: (fn: () => void) => void }).after(() => {
    globalThis.fetch = originalFetch;
  });

  // Always return HTTP 500 — both initial attempt and the one retry.
  globalThis.fetch = (async () => {
    callCount += 1;
    return new Response('Internal Server Error', { status: 500 });
  }) as typeof fetch;

  const result = await callIntercameralBroker(
    { request_id: '5xx-exhaust', prompt: 'test' },
    { maxRetries: 1, retryBackoffMs: 0 }, // 1 retry = 2 total attempts
  );

  assert.equal(callCount, 2, 'expected exactly 2 fetch calls (initial + 1 retry)');
  assert.equal(result.trusted, false);
  if (!result.trusted) {
    assert.equal(result.fallback.reason_code, 'fallback_runtime_error');
    assert.equal(result.fallback.route, 'safe_fallback');
  }
});

test('intercameral-readiness: retries network error and succeeds on second attempt', async (t: unknown) => {
  const originalFetch = globalThis.fetch;
  let callCount = 0;
  (t as { after: (fn: () => void) => void }).after(() => {
    globalThis.fetch = originalFetch;
  });

  const validPayload = makeSmoke();

  // First call throws a network error; second call returns the valid payload.
  globalThis.fetch = (async () => {
    callCount += 1;
    if (callCount === 1) {
      throw new Error('NetworkError: Failed to fetch');
    }
    return new Response(JSON.stringify(validPayload), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  }) as typeof fetch;

  const result = await callIntercameralBroker(
    { request_id: 'retry-success', prompt: 'test' },
    { maxRetries: 1, retryBackoffMs: 0 },
  );

  assert.equal(callCount, 2, 'expected exactly 2 fetch calls (initial error + 1 successful retry)');
  const d = assertAccepted(result);
  assert.equal(d.route, 'inject_none');
  assert.equal(d.confidence, 1.0);
});
