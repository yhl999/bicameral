import { normalizeConfig } from '../config.ts';
import type { PluginConfig } from '../config.ts';

export interface BeforeModelResolveEvent {
  prompt: string;
}

export interface ModelResolveContext {
  agentId?: string;
  sessionKey?: string;
  sessionId?: string;
  workspaceDir?: string;
  messageProvider?: {
    chatType?: string;
    groupId?: string;
  };
}

export interface ModelResolveHookResult {
  modelOverride?: string;
  providerOverride?: string;
}

export type ModelResolveHook = (
  event: BeforeModelResolveEvent,
  ctx: ModelResolveContext,
) => Promise<ModelResolveHookResult>;

export interface ModelResolveHookDeps {
  config?: Partial<PluginConfig>;
}

const OVERRIDE_TOKEN_PATTERN = /^(?!.*\.\.)(?!\/)[A-Za-z0-9._/-]+$/;

const normalizeOverride = (value?: string): string | undefined => {
  if (!value) {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
};

const validateOverrideToken = (token: string, maxLen: number): boolean => {
  if (token.length > maxLen) {
    return false;
  }
  return OVERRIDE_TOKEN_PATTERN.test(token);
};

/**
 * Explicit model-resolution hook for modern OpenClaw runtimes.
 *
 * Why this exists:
 * - `before_model_resolve` is the canonical phase for provider/model routing.
 * - `before_agent_start` remains legacy compatibility and should not carry new
 *   routing behavior.
 */
export const createModelResolveHook = (deps?: ModelResolveHookDeps): ModelResolveHook => {
  const config = normalizeConfig(deps?.config);

  if (!config.allowModelRoutingOverride) {
    return async (_event, _ctx) => ({});
  }

  const normalizedProvider = normalizeOverride(config.providerOverride);
  const normalizedModel = normalizeOverride(config.modelOverride);

  const providerOverride =
    normalizedProvider &&
    validateOverrideToken(normalizedProvider, config.maxOverrideTokenLength) &&
    config.allowedProviderOverrides.includes(normalizedProvider)
      ? normalizedProvider
      : undefined;

  const modelOverride =
    normalizedModel &&
    validateOverrideToken(normalizedModel, config.maxOverrideTokenLength) &&
    config.allowedModelOverrides.includes(normalizedModel)
      ? normalizedModel
      : undefined;

  return async (_event, _ctx) => ({
    providerOverride,
    modelOverride,
  });
};
