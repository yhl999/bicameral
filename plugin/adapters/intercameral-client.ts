/**
 * Bicameral adapter for Intercameral Recall Broker decision payloads.
 *
 * This is the adapter interface layer between the Bicameral plugin and the
 * Intercameral Recall Broker (Thalamus router).  It accepts, semantically
 * validates, and trusts — or rejects — incoming decision payloads before
 * they influence downstream routing behaviour.
 *
 * COMPATIBILITY POLICY (strict on semantics, tolerant of harmless extension):
 *   - Required response fields absent          → reject (fail-closed)
 *   - Unknown reason codes                     → reject
 *   - Unknown route/directive/signal values    → reject
 *   - route / reason_code semantic mismatch    → reject (untrustworthy)
 *   - Contract version "v1"                   → accept
 *   - Higher version with "-compat-v1" marker → accept
 *   - Higher version without compat marker    → reject
 *   - Extra unknown fields                     → allow (tolerant extension)
 *   - Missing optional fields                  → allow
 *   - confidence outside [0, 1]                → reject
 *
 * OVERRIDE POLICY:
 *   Bicameral may only override an Intercameral decision when:
 *   - Channel policy / safety requires it
 *   - Required semantic fields are missing
 *   - Payload is malformed or untrustworthy
 *   These cases are handled by the downstream safety clamp (safety-clamp.ts).
 *
 * FALLBACK ORDER (deterministic, observable):
 *   1. Valid + trusted Intercameral decision  → use as-is
 *   2. Adapter transport error / timeout      → reason_code=fallback_timeout
 *   3. Contract-invalid response              → reason_code=fallback_contract_invalid
 *   4. Unrecoverable runtime error            → reason_code=fallback_runtime_error
 */

// ---------------------------------------------------------------------------
// Supported contract version
// ---------------------------------------------------------------------------

/** The only contract version this adapter understands natively. */
export const SUPPORTED_CONTRACT_VERSION = 'v1' as const;

// ---------------------------------------------------------------------------
// Canonical enum types (derived from spec/intercameral_decision_contract.v1.schema.json)
// ---------------------------------------------------------------------------

export type RouteClass =
  | 'inject_none'
  | 'task_state'
  | 'direct_retrieval'
  | 'pack_only'
  | 'clarify_first'
  | 'chamberlain_single_chain'
  | 'safe_fallback';

export type ReasonCode =
  | 'low_info_skip'
  | 'task_update_narrow'
  | 'direct_recall'
  | 'pack_select'
  | 'ambiguous_needs_clarification'
  | 'contradiction_risk'
  | 'complex_multi_hop'
  | 'safety_malformed_state'
  | 'fallback_timeout'
  | 'fallback_contract_invalid'
  | 'fallback_runtime_error';

export type FallbackDirective =
  | 'none'
  | 'use_simple_local_path'
  | 'abstain_clarify'
  | 'warn_low_confidence'
  | 'escalate_chamberlain';

export type ContradictionSignal = 'none' | 'possible' | 'detected';

// ---------------------------------------------------------------------------
// Validated decision shape — strongly typed after passing all checks
// ---------------------------------------------------------------------------

export interface ValidatedDecision {
  /** Contract version (normalised to 'v1' after compatibility check). */
  readonly version: typeof SUPPORTED_CONTRACT_VERSION;
  /** Echoed request identifier from Intercameral. */
  readonly request_id: string;
  /** Canonical routing decision. */
  readonly route: RouteClass;
  /** Registered reason code driving the routing decision. */
  readonly reason_code: ReasonCode;
  /** Intercameral confidence score [0.0, 1.0]. */
  readonly confidence: number;
  /** Action hint for downstream consumers (Bicameral / safety clamp). */
  readonly fallback_directive: FallbackDirective;
  /** Evidence reference strings (may be empty). */
  readonly evidence_refs: readonly string[];
  /** Contradiction status emitted by Intercameral. */
  readonly contradiction_signal: ContradictionSignal;
  /** Optional diagnostics block (fields may be absent). */
  readonly diagnostics?: {
    readonly latency_ms?: number;
    readonly timed_out?: boolean;
    readonly retried?: boolean;
    readonly model?: string;
    readonly veto_applied?: boolean;
  };
}

// ---------------------------------------------------------------------------
// Deterministic fallback decision — emitted when adapter rejects a payload
// ---------------------------------------------------------------------------

export interface FallbackDecision {
  readonly version: typeof SUPPORTED_CONTRACT_VERSION;
  readonly request_id: string;
  readonly route: 'safe_fallback';
  readonly reason_code: 'fallback_timeout' | 'fallback_contract_invalid' | 'fallback_runtime_error';
  readonly confidence: 0;
  readonly fallback_directive: 'use_simple_local_path';
  readonly evidence_refs: readonly [];
  readonly contradiction_signal: 'none';
}

// ---------------------------------------------------------------------------
// Adapter result — discriminated union
// ---------------------------------------------------------------------------

export type AdapterAcceptResult = {
  readonly trusted: true;
  readonly decision: ValidatedDecision;
};

export type AdapterRejectResult = {
  readonly trusted: false;
  /** Human-readable rejection reason (for logging / diagnostics). */
  readonly reason: string;
  /** Deterministic fallback decision to use instead. */
  readonly fallback: FallbackDecision;
};

export type AdapterResult = AdapterAcceptResult | AdapterRejectResult;

// ---------------------------------------------------------------------------
// Internal validation tables
// ---------------------------------------------------------------------------

const VALID_ROUTES: ReadonlySet<string> = new Set<RouteClass>([
  'inject_none',
  'task_state',
  'direct_retrieval',
  'pack_only',
  'clarify_first',
  'chamberlain_single_chain',
  'safe_fallback',
]);

const VALID_REASON_CODES: ReadonlySet<string> = new Set<ReasonCode>([
  'low_info_skip',
  'task_update_narrow',
  'direct_recall',
  'pack_select',
  'ambiguous_needs_clarification',
  'contradiction_risk',
  'complex_multi_hop',
  'safety_malformed_state',
  'fallback_timeout',
  'fallback_contract_invalid',
  'fallback_runtime_error',
]);

const VALID_FALLBACK_DIRECTIVES: ReadonlySet<string> = new Set<FallbackDirective>([
  'none',
  'use_simple_local_path',
  'abstain_clarify',
  'warn_low_confidence',
  'escalate_chamberlain',
]);

const VALID_CONTRADICTION_SIGNALS: ReadonlySet<string> = new Set<ContradictionSignal>([
  'none',
  'possible',
  'detected',
]);

/**
 * Canonical mapping of reason_code → expected route.
 * Derived from spec/reason_codes.v1.json.
 * A payload where route does not match its reason_code's canonical route is
 * semantically inconsistent and is rejected as untrustworthy.
 */
const REASON_CODE_ROUTE_MAP: Readonly<Record<ReasonCode, RouteClass>> = {
  low_info_skip:                   'inject_none',
  task_update_narrow:              'task_state',
  direct_recall:                   'direct_retrieval',
  pack_select:                     'pack_only',
  ambiguous_needs_clarification:   'clarify_first',
  contradiction_risk:              'chamberlain_single_chain',
  complex_multi_hop:               'chamberlain_single_chain',
  safety_malformed_state:          'safe_fallback',
  fallback_timeout:                'safe_fallback',
  fallback_contract_invalid:       'safe_fallback',
  fallback_runtime_error:          'safe_fallback',
};

// ---------------------------------------------------------------------------
// Version compatibility check
// ---------------------------------------------------------------------------

/**
 * Returns true if the payload version is compatible with v1.
 *
 * Accepted:
 *   - "v1"             — current supported version
 *   - "vN-compat-v1"  — higher version explicitly marked backward-compatible
 *
 * Rejected:
 *   - "v2", "v3", …   — higher version without explicit compat marker
 *   - anything else   — not a recognised version string
 */
function isCompatibleVersion(version: unknown): boolean {
  if (typeof version !== 'string') return false;
  if (version === 'v1') return true;
  // Accept "v<N>-compat-v1" for N >= 2 (explicitly backward-compatible)
  return /^v[2-9]\d*-compat-v1$/.test(version) || /^v[1-9]\d+-compat-v1$/.test(version);
}

// ---------------------------------------------------------------------------
// Fallback decision factory
// ---------------------------------------------------------------------------

function makeFallback(
  requestId: string,
  reasonCode: FallbackDecision['reason_code'],
): FallbackDecision {
  return {
    version: 'v1',
    request_id: requestId,
    route: 'safe_fallback',
    reason_code: reasonCode,
    confidence: 0,
    fallback_directive: 'use_simple_local_path',
    evidence_refs: [],
    contradiction_signal: 'none',
  };
}

function makeReject(reason: string, fallback: FallbackDecision): AdapterRejectResult {
  return { trusted: false, reason, fallback };
}

// ---------------------------------------------------------------------------
// Core payload validator — pure function, no I/O
// ---------------------------------------------------------------------------

/**
 * Validates an unknown Intercameral decision payload and returns an
 * AdapterResult.
 *
 * This is the single authoritative entry point for all validation logic.
 * The HTTP client calls this after receiving a raw response body.
 */
export function validateIntercameralDecision(payload: unknown): AdapterResult {
  // Must be a non-null object
  if (payload === null || typeof payload !== 'object' || Array.isArray(payload)) {
    return makeReject(
      'payload is not an object',
      makeFallback('', 'fallback_contract_invalid'),
    );
  }

  const doc = payload as Record<string, unknown>;

  // --- Version compatibility check ---
  if (!isCompatibleVersion(doc['version'])) {
    return makeReject(
      `unsupported contract version: ${JSON.stringify(doc['version'])} (expected "v1" or "vN-compat-v1")`,
      makeFallback('', 'fallback_contract_invalid'),
    );
  }

  // --- Extract response block ---
  const response = doc['response'];
  if (response === null || typeof response !== 'object' || Array.isArray(response)) {
    return makeReject(
      'missing or invalid "response" block',
      makeFallback('', 'fallback_contract_invalid'),
    );
  }

  const resp = response as Record<string, unknown>;

  // --- Extract request_id (prefer from response; fall back to request block) ---
  const rawRequestId =
    typeof resp['request_id'] === 'string'
      ? resp['request_id']
      : typeof (doc['request'] as Record<string, unknown> | undefined)?.['request_id'] === 'string'
        ? (doc['request'] as Record<string, unknown>)['request_id']
        : '';

  const requestId = typeof rawRequestId === 'string' ? rawRequestId : '';

  // --- Required field: route ---
  if (typeof resp['route'] !== 'string') {
    return makeReject('missing required field: response.route', makeFallback(requestId, 'fallback_contract_invalid'));
  }
  if (!VALID_ROUTES.has(resp['route'])) {
    return makeReject(
      `unknown route: "${resp['route']}" — expected one of [${[...VALID_ROUTES].join(', ')}]`,
      makeFallback(requestId, 'fallback_contract_invalid'),
    );
  }

  // --- Required field: reason_code ---
  if (typeof resp['reason_code'] !== 'string') {
    return makeReject('missing required field: response.reason_code', makeFallback(requestId, 'fallback_contract_invalid'));
  }
  if (!VALID_REASON_CODES.has(resp['reason_code'])) {
    return makeReject(
      `unknown reason_code: "${resp['reason_code']}" — unknown reason codes are rejected per compatibility policy`,
      makeFallback(requestId, 'fallback_contract_invalid'),
    );
  }

  // --- Required field: confidence ---
  if (typeof resp['confidence'] !== 'number') {
    return makeReject('missing required field: response.confidence', makeFallback(requestId, 'fallback_contract_invalid'));
  }
  if (resp['confidence'] < 0 || resp['confidence'] > 1) {
    return makeReject(
      `confidence ${resp['confidence']} is out of range [0, 1]`,
      makeFallback(requestId, 'fallback_contract_invalid'),
    );
  }

  // --- Required field: fallback_directive ---
  if (typeof resp['fallback_directive'] !== 'string') {
    return makeReject('missing required field: response.fallback_directive', makeFallback(requestId, 'fallback_contract_invalid'));
  }
  if (!VALID_FALLBACK_DIRECTIVES.has(resp['fallback_directive'])) {
    return makeReject(
      `unknown fallback_directive: "${resp['fallback_directive']}"`,
      makeFallback(requestId, 'fallback_contract_invalid'),
    );
  }

  // --- Required field: evidence_refs ---
  if (!Array.isArray(resp['evidence_refs'])) {
    return makeReject('missing required field: response.evidence_refs (must be an array)', makeFallback(requestId, 'fallback_contract_invalid'));
  }
  // All entries must be strings (non-string entries → contract invalid)
  for (const ref of resp['evidence_refs']) {
    if (typeof ref !== 'string') {
      return makeReject(
        'response.evidence_refs contains a non-string entry',
        makeFallback(requestId, 'fallback_contract_invalid'),
      );
    }
  }

  // --- Required field: contradiction_signal ---
  if (typeof resp['contradiction_signal'] !== 'string') {
    return makeReject('missing required field: response.contradiction_signal', makeFallback(requestId, 'fallback_contract_invalid'));
  }
  if (!VALID_CONTRADICTION_SIGNALS.has(resp['contradiction_signal'])) {
    return makeReject(
      `unknown contradiction_signal: "${resp['contradiction_signal']}"`,
      makeFallback(requestId, 'fallback_contract_invalid'),
    );
  }

  // --- Semantic consistency: route must match reason_code's canonical route ---
  const route = resp['route'] as RouteClass;
  const reasonCode = resp['reason_code'] as ReasonCode;
  const expectedRoute = REASON_CODE_ROUTE_MAP[reasonCode];

  if (route !== expectedRoute) {
    return makeReject(
      `semantic mismatch: reason_code "${reasonCode}" maps to route "${expectedRoute}" but payload declares route "${route}" — payload is untrustworthy`,
      makeFallback(requestId, 'fallback_contract_invalid'),
    );
  }

  // --- Optional: diagnostics block validation ---
  let diagnostics: ValidatedDecision['diagnostics'];
  if (resp['diagnostics'] !== undefined) {
    if (resp['diagnostics'] === null || typeof resp['diagnostics'] !== 'object' || Array.isArray(resp['diagnostics'])) {
      return makeReject('response.diagnostics is present but not an object', makeFallback(requestId, 'fallback_contract_invalid'));
    }
    const d = resp['diagnostics'] as Record<string, unknown>;
    diagnostics = {
      latency_ms:   typeof d['latency_ms'] === 'number'   ? d['latency_ms']   : undefined,
      timed_out:    typeof d['timed_out'] === 'boolean'   ? d['timed_out']    : undefined,
      retried:      typeof d['retried'] === 'boolean'     ? d['retried']      : undefined,
      model:        typeof d['model'] === 'string'        ? d['model']        : undefined,
      veto_applied: typeof d['veto_applied'] === 'boolean' ? d['veto_applied'] : undefined,
    };
  }

  // --- All checks passed: return trusted decision ---
  const decision: ValidatedDecision = {
    version: 'v1',
    request_id: requestId,
    route,
    reason_code: reasonCode,
    confidence: resp['confidence'] as number,
    fallback_directive: resp['fallback_directive'] as FallbackDirective,
    evidence_refs: (resp['evidence_refs'] as string[]).slice(),
    contradiction_signal: resp['contradiction_signal'] as ContradictionSignal,
    ...(diagnostics !== undefined ? { diagnostics } : {}),
  };

  return { trusted: true, decision };
}

// ---------------------------------------------------------------------------
// HTTP client options
// ---------------------------------------------------------------------------

export interface InterCameralClientOptions {
  /** Intercameral decision endpoint (default: http://127.0.0.1:18781/v1/decision). */
  baseUrl?: string;
  /** Request timeout in ms (default: 8000, per spec v1 timeout policy). */
  timeoutMs?: number;
  /** Max retries on transport failure (default: 1). */
  maxRetries?: number;
  /** Fixed retry backoff in ms (default: 250). */
  retryBackoffMs?: number;
}

const DEFAULT_BASE_URL = 'http://127.0.0.1:18781';
const DEFAULT_TIMEOUT_MS = 8000;
const DEFAULT_MAX_RETRIES = 1;
const DEFAULT_RETRY_BACKOFF_MS = 250;

// ---------------------------------------------------------------------------
// HTTP client — wraps validateIntercameralDecision with transport handling
// ---------------------------------------------------------------------------

/**
 * Calls the Intercameral Recall Broker HTTP endpoint, validates the response,
 * and returns an AdapterResult.
 *
 * Transport failure semantics (per spec/EXECUTION_INTERFACE_CONTRACT.md §5):
 *   - Timeout exhaustion      → fallback_timeout
 *   - Contract-invalid body   → fallback_contract_invalid (no retry)
 *   - Runtime exception       → fallback_runtime_error
 */
export async function callIntercameralBroker(
  requestBody: Record<string, unknown>,
  options: InterCameralClientOptions = {},
): Promise<AdapterResult> {
  const baseUrl = options.baseUrl ?? DEFAULT_BASE_URL;
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const maxRetries = options.maxRetries ?? DEFAULT_MAX_RETRIES;
  const retryBackoffMs = options.retryBackoffMs ?? DEFAULT_RETRY_BACKOFF_MS;

  const endpoint = `${baseUrl.replace(/\/$/, '')}/v1/decision`;
  const requestId = typeof requestBody['request_id'] === 'string' ? requestBody['request_id'] : '';

  let lastError: unknown = null;
  let attempts = 0;

  while (attempts <= maxRetries) {
    attempts += 1;

    // Backoff after first attempt
    if (attempts > 1) {
      await new Promise<void>((resolve) => setTimeout(resolve, retryBackoffMs));
    }

    let response: Response;
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);

      try {
        response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });
      } finally {
        clearTimeout(timer);
      }
    } catch (err) {
      lastError = err;
      const isAbort = err instanceof Error && err.name === 'AbortError';
      if (isAbort) {
        // Timeout: no retry per spec (retry_on does not include client-side aborts)
        return makeReject(
          `transport timeout after ${timeoutMs}ms`,
          makeFallback(requestId, 'fallback_timeout'),
        );
      }
      // Network error: eligible for retry
      continue;
    }

    // Non-2xx response with a body that might be contract-invalid
    if (!response.ok) {
      // 5xx: eligible for retry
      if (response.status >= 500) {
        lastError = new Error(`HTTP ${response.status}`);
        continue;
      }
      // 4xx: contract or caller error — no retry
      return makeReject(
        `HTTP ${response.status} from Intercameral broker`,
        makeFallback(requestId, 'fallback_contract_invalid'),
      );
    }

    // Parse JSON body
    let body: unknown;
    try {
      body = await response.json();
    } catch {
      // Body is not valid JSON → contract invalid, no retry
      return makeReject(
        'Intercameral response body is not valid JSON',
        makeFallback(requestId, 'fallback_contract_invalid'),
      );
    }

    // Validate payload — if contract-invalid, no retry
    const result = validateIntercameralDecision(body);
    return result;
  }

  // All retries exhausted
  return makeReject(
    `transport error after ${attempts} attempt(s): ${lastError instanceof Error ? lastError.message : String(lastError)}`,
    makeFallback(requestId, 'fallback_runtime_error'),
  );
}
