import type { RecallHook } from './recall.ts';

export interface LegacyBeforeAgentStartEvent {
  prompt: string;
  messages?: Array<{ role?: string; content: string }>;
}

export interface LegacyBeforeAgentStartContext {
  agentId?: string;
  sessionKey?: string;
  sessionId?: string;
  workspaceDir?: string;
  messageProvider?: {
    chatType?: 'direct' | 'group' | 'channel';
    groupId?: string;
  };
}

export type LegacyBeforeAgentStartHook = (
  event: LegacyBeforeAgentStartEvent,
  ctx: LegacyBeforeAgentStartContext,
) => Promise<{ prependContext?: string }>;

/**
 * Legacy compatibility hook.
 *
 * In modern OpenClaw, `before_agent_start` is precomputed in the model-resolve
 * phase (often without `messages`) and merged with explicit phase hooks.
 *
 * To avoid duplicate Graphiti injections when both explicit hooks are present,
 * this shim only delegates when a message list exists (older/compat paths).
 */
export const createLegacyBeforeAgentStartHook = (
  promptBuildHook: RecallHook,
): LegacyBeforeAgentStartHook => {
  return async (event, ctx) => {
    if (!Array.isArray(event.messages)) {
      return {};
    }
    return promptBuildHook(event, ctx);
  };
};
