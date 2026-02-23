import { createHash } from 'node:crypto';

/**
 * Derive a deterministic, non-reversible Graphiti group lane ID from a raw
 * session key.
 *
 * Raw session keys (e.g. "agent:main:telegram:group:-1003893734334:topic:6529")
 * can embed sensitive platform identifiers that must not be forwarded verbatim
 * to an external service.  This function produces a stable 16-hex-char prefix
 * of the SHA-256 digest — long enough to be globally unique across any
 * realistic number of sessions, short enough to be opaque and log-safe.
 *
 * Properties:
 * - Deterministic: same input always yields the same lane id.
 * - Non-reversible: the original session key cannot be recovered from the id.
 * - Consistent: recall and capture called with the same sessionKey will
 *   always resolve to the same lane.
 * - Prefixed: the `sk:` prefix makes hashed lanes visually distinct from
 *   provider-supplied groupIds in logs and Graphiti data.
 */
export const deriveGroupLane = (sessionKey: string): string => {
  const digest = createHash('sha256').update(sessionKey, 'utf8').digest('hex');
  return `sk:${digest.slice(0, 16)}`;
};
