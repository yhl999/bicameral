import type { GraphitiClient } from '../client.ts';
import type { GraphitiSearchResults } from '../client.ts';
import { normalizeConfig } from '../config.ts';
import type { PluginConfig } from '../config.ts';
import type { PackContextResult } from './pack-injector.ts';
import type { PackInjectorContext } from './pack-injector.ts';

export interface BeforeAgentStartEvent {
  prompt: string;
  messages?: Array<{ role?: string; content: string }>;
}

export type RecallHook = (
  event: BeforeAgentStartEvent,
  ctx: PackInjectorContext,
) => Promise<{ prependContext?: string }>;

export interface RecallHookDeps {
  client: GraphitiClient;
  packInjector: (input: {
    prompt: string;
    graphitiResults?: GraphitiSearchResults | null;
    ctx: PackInjectorContext;
  }) => Promise<PackContextResult | null>;
  config?: Partial<PluginConfig>;
}

const formatGraphitiContext = (results: GraphitiSearchResults): string => {
  const lines: string[] = [];
  lines.push('<graphiti-context>');
  lines.push('## Graphiti Recall');
  if (results.facts.length === 0) {
    lines.push('No relevant facts found.');
  } else {
    for (const fact of results.facts) {
      lines.push(`- ${fact.fact}`);
    }
  }
  lines.push('</graphiti-context>');
  return lines.join('\n');
};

const FALLBACK_ERROR_CODE = 'GRAPHITI_QMD_FAILOVER';

const sanitizeReason = (reason: string): string => {
  const compact = reason.replace(/\s+/g, ' ').trim();
  const chars = Array.from(compact);
  if (chars.length <= 180) {
    return compact;
  }
  return `${chars.slice(0, 180).join('')}...`;
};

const formatFallback = (): string => {
  // Surface only a generic message to the model / user-visible output.
  // The real error reason is logged internally via console.warn so it is
  // never leaked through the model context.
  return [
    '<graphiti-fallback>',
    `ERROR_CODE: ${FALLBACK_ERROR_CODE}`,
    'WARNING: Graphiti recall failed (Service unavailable). This turn is using QMD fallback.',
    'Use memory_search or memory_get if you want to inspect fallback retrieval directly.',
    '</graphiti-fallback>',
  ].join('\n');
};

const resolveGroupIds = (
  ctx: PackInjectorContext,
  config: PluginConfig,
): string[] | undefined => {
  // SAFETY: memoryGroupId is a single-tenant override that pins all requests
  // to one group lane. Only allow it when the operator has explicitly declared
  // singleTenant: true. In multi-tenant mode (the safe default), fall through
  // to per-session lanes so different users cannot read each other's memories.
  const groupId =
    (config.singleTenant && config.memoryGroupId)
      ? config.memoryGroupId
      : (ctx.messageProvider?.groupId ?? ctx.sessionKey);
  return groupId ? [groupId] : undefined;
};

export const createRecallHook = (deps: RecallHookDeps): RecallHook => {
  const config = normalizeConfig(deps.config);
  const logger = config.debug ? (message: string) => console.log(message) : () => undefined;

  return async (event, ctx) => {
    const parts: string[] = [];
    const prompt = event.prompt ?? '';
    let graphitiResults: GraphitiSearchResults | null = null;

    if (prompt.trim().length >= config.minPromptChars) {
      const groupIds = resolveGroupIds(ctx, config);
      try {
        graphitiResults = await deps.client.search(prompt, groupIds);
        parts.push(formatGraphitiContext(graphitiResults));
      } catch (error) {
        const message = (error as Error).message || 'unknown error';
        const safeReason = sanitizeReason(message);
        const effectiveGroup = groupIds?.[0] ?? 'unknown';
        // Always emit failover warnings, even when debug logging is disabled.
        console.warn(
          `[graphiti-openclaw] ${FALLBACK_ERROR_CODE} group=${effectiveGroup} reason=${safeReason}`,
        );
        logger(`Graphiti recall failed: ${safeReason}`);
        parts.push(formatFallback());
      }
    } else {
      parts.push('');
    }

    try {
      const packResult = await deps.packInjector({
        prompt,
        graphitiResults,
        ctx,
      });
      if (packResult?.context) {
        parts.push(packResult.context);
      }
    } catch (error) {
      logger(`Pack injection failed: ${(error as Error).message}`);
    }

    return { prependContext: parts.filter((part) => part.trim().length > 0).join('\n\n') };
  };
};
