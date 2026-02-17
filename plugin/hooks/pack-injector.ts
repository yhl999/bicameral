import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';

import { resolveComposition } from '../composition/engine.ts';
import type { CompositionRuleSet, ResolvedComposition } from '../composition/types.ts';
import { normalizeConfig } from '../config.ts';
import type { PackRegistry } from '../config.ts';
import type { GraphitiSearchResults } from '../client.ts';
import { detectIntent } from '../intent/detector.ts';
import type { IntentRuleSet, IntentDecision } from '../intent/types.ts';
import type { PluginConfig } from '../config.ts';

export interface PackContextResult {
  context: string;
  intentId: string;
  primaryPackId: string;
  scope: string;
}

export interface PackInjectorContext {
  sessionKey?: string;
  messageProvider?: {
    chatType?: string;
    groupId?: string;
  };
}

export interface PackInjectorInput {
  prompt: string;
  graphitiResults?: GraphitiSearchResults | null;
  ctx: PackInjectorContext;
}

export interface PackMaterialized {
  packId: string;
  content: string;
  scope: string;
  mode?: string;
}

interface PackPlan {
  consumer: string;
  workflow_id: string;
  step_id: string;
  scope: string;
  task: string;
  injection_text: string;
  packs: { pack_id: string; query: string }[];
}

interface PackInjectorDeps {
  intentRules: IntentRuleSet;
  compositionRules?: CompositionRuleSet | null;
  packRegistry?: PackRegistry | null;
  config?: Partial<PluginConfig>;
}

const isGroupChat = (ctx: PackInjectorContext): boolean => {
  return ctx.messageProvider?.chatType === 'group';
};

const resolveRegistryEntry = (registry: PackRegistry, packType: string) => {
  return registry.packs.find(
    (pack) => pack.pack_type === packType || pack.pack_id === packType,
  );
};

const loadPackContent = (repoRoot: string, packPath: string): string => {
  const resolved = path.resolve(repoRoot, packPath);
  return fs.readFileSync(resolved, 'utf8');
};

const formatPackContext = (
  intentId: string,
  primary: PackMaterialized,
  plan: PackPlan | null,
  additional: PackMaterialized[],
): string => {
  const lines: string[] = [];
  lines.push(
    `<pack-context intent="${intentId}" primary-pack="${primary.packId}" scope="${primary.scope}">`,
  );
  if (plan) {
    lines.push(`## Active Workflow: ${plan.workflow_id}`);
    if (plan.task) {
      lines.push(`Task: ${plan.task}`);
    }
    if (plan.injection_text) {
      lines.push(plan.injection_text);
    }
  } else {
    lines.push(`## Active Workflow: ${primary.packId}`);
  }
  lines.push('');
  lines.push(primary.content.trim());

  for (const pack of additional) {
    lines.push('');
    const modeLabel = pack.mode ? ` (${pack.mode})` : '';
    lines.push(`### Composition: ${pack.packId}${modeLabel}`);
    lines.push(pack.content.trim());
  }

  lines.push('</pack-context>');
  return lines.join('\n');
};

const parseRouterOutput = (raw: string): PackPlan => {
  const parsed = JSON.parse(raw) as PackPlan;
  if (!parsed || !Array.isArray(parsed.packs)) {
    throw new Error('Pack router returned invalid plan');
  }
  return parsed;
};

const runPackRouter = (
  command: string,
  args: string[],
  timeoutMs: number,
): Promise<string> => {
  return new Promise((resolve, reject) => {
    const [cmd, ...baseArgs] = command.split(' ').filter((part) => part.length > 0);
    const child = spawn(cmd, [...baseArgs, ...args], { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';

    const timeout = setTimeout(() => {
      child.kill('SIGKILL');
      reject(new Error('Pack router timed out'));
    }, timeoutMs);

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });
    child.on('error', (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    child.on('close', (code) => {
      clearTimeout(timeout);
      if (code !== 0) {
        reject(new Error(`Pack router failed: ${stderr.trim()}`));
        return;
      }
      resolve(stdout.trim());
    });
  });
};

const resolvePrimaryPackViaRegistry = (
  registry: PackRegistry,
  packType: string,
  repoRoot: string,
): PackMaterialized => {
  const entry = resolveRegistryEntry(registry, packType);
  if (!entry) {
    throw new Error(`Pack registry missing pack type: ${packType}`);
  }
  const content = loadPackContent(repoRoot, entry.path);
  return {
    packId: entry.pack_id,
    scope: entry.scope,
    content,
  };
};

const resolveAdditionalPacks = (
  registry: PackRegistry | null | undefined,
  repoRoot: string,
  additional: ResolvedComposition[],
  logger: (message: string) => void,
): PackMaterialized[] => {
  if (!registry) {
    if (additional.length > 0) {
      logger('Pack registry missing; skipping composition packs.');
    }
    return [];
  }

  const resolved: PackMaterialized[] = [];
  for (const item of additional) {
    const entry = resolveRegistryEntry(registry, item.packType);
    if (!entry) {
      const message = `Composition pack missing: ${item.packType}`;
      if (item.required) {
        logger(`${message} (required)`);
      } else {
        logger(`${message} (optional)`);
      }
      continue;
    }

    resolved.push({
      packId: entry.pack_id,
      scope: entry.scope,
      content: loadPackContent(repoRoot, entry.path),
      mode: item.mode,
    });
  }

  return resolved;
};

export const createPackInjector = (deps: PackInjectorDeps) => {
  const config = normalizeConfig(deps.config);
  const logger = config.debug ? (message: string) => console.log(message) : () => undefined;
  const state = new Map<string, string>();

  return async (input: PackInjectorInput): Promise<PackContextResult | null> => {
    try {
      const sessionKey = input.ctx.sessionKey ?? input.ctx.messageProvider?.groupId ?? 'default';
      const previousIntentId = state.get(sessionKey);
      const { decision } = detectIntent(deps.intentRules, {
        prompt: input.prompt,
        graphitiResults: input.graphitiResults ?? undefined,
        previousIntentId,
        enableSticky: config.enableSticky,
        stickyMaxWords: config.stickyMaxWords,
        stickySignals: config.stickySignals,
        defaultMinConfidence: config.defaultMinConfidence,
      });

      if (!decision.matched || !decision.rule) {
        state.delete(sessionKey);
        return null;
      }

      if (decision.rule.scope === 'private' && isGroupChat(input.ctx)) {
        logger('Skipping private intent in group chat.');
        state.delete(sessionKey);
        return null;
      }

      const repoRoot = config.packRouterRepoRoot ?? process.cwd();
      let primaryPack: PackMaterialized | null = null;
      let plan: PackPlan | null = null;

      if (config.packRouterCommand) {
        const args = [
          '--consumer',
          decision.rule.consumerProfile,
          '--workflow-id',
          decision.rule.workflowId ?? decision.rule.id,
          '--step-id',
          decision.rule.stepId ?? 'draft',
          '--task',
          decision.rule.task ?? '',
          '--injection-text',
          decision.rule.injectionText ?? '',
          '--repo',
          repoRoot,
        ];
        const output = await runPackRouter(
          config.packRouterCommand,
          args,
          config.packRouterTimeoutMs,
        );
        plan = parseRouterOutput(output);
        const primaryEntry = plan.packs[0];
        primaryPack = {
          packId: primaryEntry.pack_id,
          scope: plan.scope,
          content: loadPackContent(repoRoot, primaryEntry.query),
        };
      } else if (deps.packRegistry) {
        const packType = decision.rule.packType ?? decision.rule.id;
        primaryPack = resolvePrimaryPackViaRegistry(deps.packRegistry, packType, repoRoot);
      }

      if (!primaryPack) {
        logger('Primary pack resolution failed.');
        state.delete(sessionKey);
        return null;
      }

      if (primaryPack.scope === 'private' && isGroupChat(input.ctx)) {
        logger('Skipping private pack in group chat.');
        state.delete(sessionKey);
        return null;
      }

      const composition = resolveComposition(deps.compositionRules ?? null, decision.rule.id);
      let additional = resolveAdditionalPacks(
        deps.packRegistry,
        repoRoot,
        composition,
        logger,
      );

      if (isGroupChat(input.ctx)) {
        additional = additional.filter((pack) => pack.scope !== 'private');
      }

      const context = formatPackContext(decision.rule.id, primaryPack, plan, additional);
      state.set(sessionKey, decision.rule.id);

      return {
        context,
        intentId: decision.rule.id,
        primaryPackId: primaryPack.packId,
        scope: primaryPack.scope,
      };
    } catch (error) {
      logger(`Pack injector error: ${(error as Error).message}`);
      return null;
    }
  };
};
