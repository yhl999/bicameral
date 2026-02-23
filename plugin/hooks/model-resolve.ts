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
  messageProvider?: string;
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

const normalizeOverride = (value?: string): string | undefined => {
  if (!value) {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
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
  const providerOverride = normalizeOverride(config.providerOverride);
  const modelOverride = normalizeOverride(config.modelOverride);

  return async (_event, _ctx) => ({
    providerOverride,
    modelOverride,
  });
};
