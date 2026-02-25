import type { GraphitiClient, GraphitiMessage } from '../client.ts';
import { normalizeConfig } from '../config.ts';
import type { PluginConfig } from '../config.ts';
import { deriveGroupLane } from '../lane-utils.ts';
import type { PackInjectorContext } from './pack-injector.ts';

export interface AgentEndEvent {
  success?: boolean;
  messages?: Array<{ role?: string; content?: unknown; name?: string }>;
}

export type CaptureHook = (event: AgentEndEvent, ctx: PackInjectorContext) => Promise<void>;

export interface CaptureHookDeps {
  client: GraphitiClient;
  config?: Partial<PluginConfig>;
}

const GRAPHITI_CONTEXT_RE = /<graphiti-context>[\s\S]*?<\/graphiti-context>/gi;
const PACK_CONTEXT_RE = /<pack-context[\s\S]*?<\/pack-context>/gi;
const FALLBACK_CONTEXT_RE = /<graphiti-fallback>[\s\S]*?<\/graphiti-fallback>/gi;

const normalizeMessageContent = (content: unknown): string | null => {
  if (typeof content === 'string') {
    const trimmed = content.trim();
    return trimmed.length > 0 ? trimmed : null;
  }

  if (!Array.isArray(content)) {
    return null;
  }

  const chunks: string[] = [];
  for (const block of content) {
    if (!block || typeof block !== 'object') {
      continue;
    }
    const maybeType = (block as { type?: unknown }).type;
    if (maybeType !== 'text') {
      continue;
    }
    const maybeText = (block as { text?: unknown }).text;
    if (typeof maybeText !== 'string') {
      continue;
    }
    if (maybeText.trim()) {
      chunks.push(maybeText);
    }
  }

  if (chunks.length === 0) {
    return null;
  }
  const joined = chunks.join(' ').replace(/\s+/g, ' ').trim();
  return joined.length > 0 ? joined : null;
};

export const stripInjectedContext = (content: unknown): string => {
  const text = normalizeMessageContent(content) ?? '';
  return text
    .replace(GRAPHITI_CONTEXT_RE, '')
    .replace(PACK_CONTEXT_RE, '')
    .replace(FALLBACK_CONTEXT_RE, '')
    .trim();
};

const resolveGroupId = (ctx: PackInjectorContext, config: PluginConfig): string | null => {
  // SAFETY: memoryGroupId is a single-tenant override that pins all requests
  // to one group lane. Only allow it when the operator has explicitly declared
  // singleTenant: true. In multi-tenant mode (the safe default), fall through
  // to per-session lanes so different users cannot read each other's memories.
  if (config.singleTenant && config.memoryGroupId) {
    return config.memoryGroupId;
  }
  if (ctx.messageProvider?.groupId) {
    return ctx.messageProvider.groupId;
  }
  // SECURITY: never forward the raw sessionKey to Graphiti — it may embed
  // sensitive platform identifiers (e.g. Telegram chat IDs, routing tokens).
  // Derive a deterministic, non-reversible lane id from it instead so that
  // recall and capture are still scoped to the same lane without leaking the
  // original value to an external service.
  if (ctx.sessionKey) {
    return deriveGroupLane(ctx.sessionKey);
  }
  return null;
};

const extractTurn = (messages: Array<{ role?: string; content?: unknown }>): GraphitiMessage[] => {
  const reversed = [...messages].reverse();
  const assistant = reversed.find((message) => message.role === 'assistant');
  const user = reversed.find((message) => message.role === 'user');

  const cleaned: GraphitiMessage[] = [];
  if (user) {
    const userContent = stripInjectedContext(user.content);
    if (userContent) {
      cleaned.push({
        content: userContent,
        role_type: 'user',
      });
    }
  }
  if (assistant) {
    const assistantContent = stripInjectedContext(assistant.content);
    if (assistantContent) {
      cleaned.push({
        content: assistantContent,
        role_type: 'assistant',
      });
    }
  }
  return cleaned;
};

export const createCaptureHook = (deps: CaptureHookDeps): CaptureHook => {
  const config = normalizeConfig(deps.config);
  const logger = config.debug ? (message: string) => console.log(message) : () => undefined;

  return async (event, ctx) => {
    if (!event.success) {
      return;
    }

    const groupId = resolveGroupId(ctx, config);
    if (!groupId) {
      logger('Capture skipped: missing group ID.');
      return;
    }

    const messages = event.messages ?? [];
    if (messages.length === 0) {
      return;
    }

    const turnMessages = extractTurn(messages);
    if (turnMessages.length === 0) {
      return;
    }

    void deps.client
      .ingestMessages(groupId, turnMessages)
      .catch((error) => logger(`Graphiti capture failed: ${(error as Error).message}`));
  };
};
