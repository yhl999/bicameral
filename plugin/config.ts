import fs from 'node:fs';
import path from 'node:path';

import type { CompositionRuleSet } from './composition/types.ts';
import type { IntentRuleSet } from './intent/types.ts';
import { isPathWithinRoot, toCanonicalPath } from './path-utils.ts';

export interface PackRegistryEntry {
  pack_id: string;
  pack_type?: string;
  path: string;
  scope: 'public' | 'group-safe' | 'private';
}

export interface PackRegistry {
  schema_version: number;
  packs: PackRegistryEntry[];
}

export interface PluginConfig {
  graphitiBaseUrl: string;
  graphitiApiKey?: string;
  /** Optional explicit provider override (used by before_model_resolve). */
  providerOverride?: string;
  /** Optional explicit model override (used by before_model_resolve). */
  modelOverride?: string;
  /** Explicit opt-in for model/provider overrides (secure-by-default). */
  allowModelRoutingOverride: boolean;
  /** Allowed provider override values. Required when providerOverride is set. */
  allowedProviderOverrides: string[];
  /** Allowed model override values. Required when modelOverride is set. */
  allowedModelOverrides: string[];
  /** Max allowed length for model/provider override tokens. */
  maxOverrideTokenLength: number;
  recallTimeoutMs: number;
  captureTimeoutMs: number;
  maxFacts: number;
  /**
   * Canonical Graphiti group lane used when sessionKey is unavailable.
   *
   * SAFETY: this override pins every request to a single group lane.
   * It is ONLY safe in single-tenant deployments (i.e. exactly one user /
   * logical namespace behind the plugin instance).  Multi-tenant operators
   * MUST leave this unset and rely on per-session lanes (`provider.groupId`
   * or `sessionKey`) to prevent cross-tenant memory leakage.
   *
   * This field has no effect unless `singleTenant: true` is also set.
   */
  memoryGroupId?: string;
  /**
   * Declare that this plugin instance serves a single tenant.
   *
   * Required to unlock `memoryGroupId` overrides.  When `false` (the safe
   * default), `memoryGroupId` is ignored and per-session lanes are used
   * instead, preventing accidental cross-tenant memory fan-out.
   */
  singleTenant: boolean;
  minPromptChars: number;
  enableSticky: boolean;
  stickyMaxWords: number;
  stickySignals: string[];
  intentRulesPath?: string;
  compositionRulesPath?: string;
  packRegistryPath?: string;
  packRouterCommand?: string | string[];
  packRouterRepoRoot?: string;
  packRouterTimeoutMs: number;
  defaultMinConfidence: number;
  debug: boolean;
  configPathRoots?: string[];
  trustedGroupIds?: string[];
  singleTenant?: boolean;
  memoryGroupId?: string;
}

export const DEFAULT_CONFIG: PluginConfig = {
  graphitiBaseUrl: 'http://localhost:8000',
  recallTimeoutMs: 1500,
  captureTimeoutMs: 1500,
  maxFacts: 8,
  minPromptChars: 6,
  enableSticky: true,
  stickyMaxWords: 20,
  stickySignals: ['also', 'and', 'continue', 'what about', 'follow up'],
  packRouterTimeoutMs: 2000,
  defaultMinConfidence: 0.3,
  allowModelRoutingOverride: false,
  allowedProviderOverrides: [],
  allowedModelOverrides: [],
  maxOverrideTokenLength: 128,
  // Safe default: multi-tenant mode. memoryGroupId overrides are disabled
  // unless the operator explicitly opts in via singleTenant: true.
  singleTenant: false,
  debug: false,
};

const normalizeOptionalString = (value?: string): string | undefined => {
  if (!value) {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
};

export const normalizeConfig = (config?: Partial<PluginConfig>): PluginConfig => {
  return {
    ...DEFAULT_CONFIG,
    ...config,
    memoryGroupId: normalizeOptionalString(config?.memoryGroupId),
    stickySignals: config?.stickySignals ?? DEFAULT_CONFIG.stickySignals,
    allowedProviderOverrides:
      config?.allowedProviderOverrides ?? DEFAULT_CONFIG.allowedProviderOverrides,
    allowedModelOverrides: config?.allowedModelOverrides ?? DEFAULT_CONFIG.allowedModelOverrides,
  };
};

const toCanonicalRoot = (candidate: string): string => {
  const absolute = path.resolve(candidate);
  return toCanonicalPath(absolute, `config root ${absolute}`);
};

const resolveSafePath = (filePath: string, allowedRoots?: string[]): string => {
  const absolute = path.resolve(filePath);
  const canonicalPath = toCanonicalPath(absolute, `config path ${absolute}`);
  const roots = (allowedRoots && allowedRoots.length > 0 ? allowedRoots : [process.cwd()]).map(
    toCanonicalRoot,
  );
  const allowed = roots.some((root) => isPathWithinRoot(root, canonicalPath));

  if (!allowed) {
    throw new Error(
      `Config path ${canonicalPath} is outside allowed roots: ${roots.join(', ')}`,
    );
  }
  return canonicalPath;
};

const readConfigFile = <T>(filePath: string, allowedRoots?: string[]): T => {
  const absolute = resolveSafePath(filePath, allowedRoots);
  const raw = fs.readFileSync(absolute, 'utf8');
  try {
    return JSON.parse(raw) as T;
  } catch (error) {
    throw new Error(
      `Config file ${absolute} must be JSON for this scaffold. ${(error as Error).message}`,
    );
  }
};

export const loadIntentRules = (
  filePath?: string,
  allowedRoots?: string[],
): IntentRuleSet | null => {
  if (!filePath) {
    return null;
  }
  return readConfigFile<IntentRuleSet>(filePath, allowedRoots);
};

export const loadCompositionRules = (
  filePath?: string,
  allowedRoots?: string[],
): CompositionRuleSet | null => {
  if (!filePath) {
    return null;
  }
  return readConfigFile<CompositionRuleSet>(filePath, allowedRoots);
};

export const loadPackRegistry = (
  filePath?: string,
  allowedRoots?: string[],
): PackRegistry | null => {
  if (!filePath) {
    return null;
  }
  return readConfigFile<PackRegistry>(filePath, allowedRoots);
};
