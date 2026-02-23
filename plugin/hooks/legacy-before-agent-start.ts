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

const PROMPT_BUILD_MARK = Symbol.for('graphiti.plugin.prompt-build-ran');
const PROMPT_BUILD_EXECUTED = new WeakSet<object>();

type MarkableContext = LegacyBeforeAgentStartContext & { [PROMPT_BUILD_MARK]?: boolean };

export const markPromptBuildExecuted = (ctx: LegacyBeforeAgentStartContext): void => {
  if (!ctx || typeof ctx !== 'object') {
    return;
  }

  PROMPT_BUILD_EXECUTED.add(ctx as object);
  try {
    (ctx as MarkableContext)[PROMPT_BUILD_MARK] = true;
  } catch {
    // Some runtimes may freeze/lock context objects.
    // WeakSet marker above remains the canonical fallback.
  }
};

export const hasPromptBuildExecuted = (ctx: LegacyBeforeAgentStartContext): boolean => {
  if (!ctx || typeof ctx !== 'object') {
    return false;
  }
  if (PROMPT_BUILD_EXECUTED.has(ctx as object)) {
    return true;
  }

  try {
    return (ctx as MarkableContext)[PROMPT_BUILD_MARK] === true;
  } catch {
    return false;
  }
};

/**
 * Legacy compatibility hook.
 *
 * In modern OpenClaw, `before_agent_start` is precomputed in the model-resolve
 * phase (often without `messages`) and merged with explicit phase hooks.
 *
 * To avoid duplicate Graphiti injections when both explicit hooks are present,
 * this shim only delegates when a non-empty message list exists (older/compat
 * paths) AND prompt-build injection has not already run for the same turn.
 */
export const createLegacyBeforeAgentStartHook = (
  promptBuildHook: RecallHook,
): LegacyBeforeAgentStartHook => {
  return async (event, ctx) => {
    if (!Array.isArray(event.messages) || event.messages.length === 0) {
      return {};
    }
    if (hasPromptBuildExecuted(ctx)) {
      return {};
    }
    markPromptBuildExecuted(ctx);
    return promptBuildHook(event, ctx);
  };
};
