import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';

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

interface FastWritePayload {
  source_session_id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
}

export interface CaptureHookDeps {
  client: GraphitiClient;
  config?: Partial<PluginConfig>;
  fastWriteRunner?: (payload: FastWritePayload, runtimeRepoRoot: string) => Promise<void>;
}

const GRAPHITI_CONTEXT_RE = /<graphiti-context>[\s\S]*?<\/graphiti-context>/gi;
const PACK_CONTEXT_RE = /<pack-context[\s\S]*?<\/pack-context>/gi;
const FALLBACK_CONTEXT_RE = /<graphiti-fallback>[\s\S]*?<\/graphiti-fallback>/gi;
const FAST_WRITE_SCRIPT_RELATIVE = path.join('scripts', 'om_fast_write.py');

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

const resolveFastWriteSessionId = (ctx: PackInjectorContext): string | null => {
  const providerGroup = ctx.messageProvider?.groupId?.trim();
  if (providerGroup) {
    return providerGroup;
  }
  const sessionKey = ctx.sessionKey?.trim();
  if (sessionKey) {
    return sessionKey;
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

const defaultFastWriteRunner = async (
  payload: FastWritePayload,
  runtimeRepoRoot: string,
): Promise<void> => {
  const scriptPath = path.join(runtimeRepoRoot, FAST_WRITE_SCRIPT_RELATIVE);
  if (!fs.existsSync(scriptPath)) {
    throw new Error(`fast-write script missing at ${scriptPath}`);
  }

  await new Promise<void>((resolve, reject) => {
    const child = spawn(
      'python3',
      [
        scriptPath,
        'write',
        '--runtime-repo',
        runtimeRepoRoot,
        '--payload-json',
        JSON.stringify(payload),
      ],
      {
        stdio: ['ignore', 'pipe', 'pipe'],
        shell: false,
        cwd: runtimeRepoRoot,
      },
    );

    let stderr = '';
    child.stderr.on('data', (chunk: Buffer) => {
      stderr += chunk.toString('utf8');
    });

    child.on('error', (error) => reject(error as Error));
    child.on('close', (code) => {
      if (code === 0) {
        resolve();
        return;
      }
      const detail = stderr.trim() || `exit code ${code}`;
      reject(new Error(`fast-write failed: ${detail}`));
    });
  });
};

export const createCaptureHook = (deps: CaptureHookDeps): CaptureHook => {
  const config = normalizeConfig(deps.config);
  const logger = config.debug ? (message: string) => console.log(message) : () => undefined;
  const runFastWrite = deps.fastWriteRunner ?? defaultFastWriteRunner;

  return async (event, ctx) => {
    if (!event.success) {
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

    const runtimeRepoRoot = config.packRouterRepoRoot?.trim() || process.env.RUNTIME_REPO_ROOT || process.cwd();
    const fastWriteSessionId = resolveFastWriteSessionId(ctx);
    if (fastWriteSessionId) {
      for (const turn of turnMessages) {
        const role = turn.role_type === 'assistant' ? 'assistant' : 'user';
        try {
          await runFastWrite(
            {
              source_session_id: fastWriteSessionId,
              role,
              content: turn.content,
              created_at: new Date().toISOString(),
            },
            runtimeRepoRoot,
          );
        } catch (error) {
          logger(`OM_FAST_WRITE_FAILED ${(error as Error).message}`);
        }
      }
    } else {
      logger('FAST_WRITE_DISABLED missing_session_id');
    }

    const groupId = resolveGroupId(ctx, config);
    if (!groupId) {
      logger('Capture skipped: missing group ID.');
      return;
    }

    void deps.client
      .ingestMessages(groupId, turnMessages)
      .catch((error) => logger(`Graphiti capture failed: ${(error as Error).message}`));
  };
};
