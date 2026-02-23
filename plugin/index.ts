import type { OpenClawPluginApi } from 'openclaw/plugin-sdk';

import { createCaptureHook } from './hooks/capture.ts';
import {
  createLegacyBeforeAgentStartHook,
  hasPromptBuildExecuted,
  markPromptBuildExecuted,
} from './hooks/legacy-before-agent-start.ts';
import { createModelResolveHook } from './hooks/model-resolve.ts';
import { createPackInjector } from './hooks/pack-injector.ts';
import { createRecallHook } from './hooks/recall.ts';
import {
  loadCompositionRules,
  loadIntentRules,
  loadPackRegistry,
  normalizeConfig,
} from './config.ts';
import { GraphitiClient } from './client.ts';
import type { CompositionRuleSet } from './composition/types.ts';
import type { IntentRuleSet } from './intent/types.ts';
import type { PackRegistry, PluginConfig } from './config.ts';

export interface GraphitiPluginOptions {
  config?: Partial<PluginConfig>;
  intentRules?: IntentRuleSet | null;
  compositionRules?: CompositionRuleSet | null;
  packRegistry?: PackRegistry | null;
}

const loadConfigFromEnv = (): Partial<PluginConfig> => {
  const raw = process.env.GRAPHITI_PLUGIN_CONFIG;
  if (!raw) {
    return {};
  }
  try {
    return JSON.parse(raw) as Partial<PluginConfig>;
  } catch {
    return {};
  }
};

export const buildGraphitiHooks = (options?: GraphitiPluginOptions) => {
  const config = normalizeConfig({
    ...loadConfigFromEnv(),
    ...(options?.config ?? {}),
  });
  const logger = config.debug ? (message: string) => console.log(message) : () => undefined;
  const configRoots = config.configPathRoots;

  const safeLoad = <T>(label: string, loader: () => T | null): T | null => {
    try {
      return loader();
    } catch (error) {
      const message = `Config load failed for ${label}: ${(error as Error).message}`;
      console.warn(`[graphiti-openclaw] ${message}`);
      logger(message);
      return null;
    }
  };

  const intentRules =
    options?.intentRules ??
    safeLoad('intent rules', () => loadIntentRules(config.intentRulesPath, configRoots)) ?? {
      schema_version: 1,
      rules: [],
    };
  const compositionRules =
    options?.compositionRules ??
    safeLoad('composition rules', () =>
      loadCompositionRules(config.compositionRulesPath, configRoots),
    );
  const packRegistry =
    options?.packRegistry ??
    safeLoad('pack registry', () => loadPackRegistry(config.packRegistryPath, configRoots));

  const client = new GraphitiClient({
    baseUrl: config.graphitiBaseUrl,
    apiKey: config.graphitiApiKey,
    recallTimeoutMs: config.recallTimeoutMs,
    captureTimeoutMs: config.captureTimeoutMs,
    maxFacts: config.maxFacts,
  });

  const packInjector = createPackInjector({
    intentRules,
    compositionRules,
    packRegistry,
    config,
  });

  const promptBuildHook = createRecallHook({
    client,
    packInjector,
    config,
  });

  const beforePromptBuildHook: ReturnType<typeof createRecallHook> = async (event, ctx) => {
    if (hasPromptBuildExecuted(ctx)) {
      return {};
    }
    markPromptBuildExecuted(ctx);
    return promptBuildHook(event, ctx);
  };

  return {
    before_model_resolve: createModelResolveHook({ config }),
    before_prompt_build: beforePromptBuildHook,
    before_agent_start: createLegacyBeforeAgentStartHook(promptBuildHook),
    agent_end: createCaptureHook({
      client,
      config,
    }),
  };
};

const graphitiPlugin = {
  id: 'graphiti-openclaw',
  name: 'Graphiti OpenClaw',
  description: 'Graphiti runtime context injection plugin',

  register(api: OpenClawPluginApi) {
    const hooks = buildGraphitiHooks({
      config: (api.pluginConfig as Partial<PluginConfig> | undefined) ?? {},
    });

    api.on('before_model_resolve', hooks.before_model_resolve);
    api.on('before_prompt_build', hooks.before_prompt_build);
    api.on('before_agent_start', hooks.before_agent_start);
    api.on('agent_end', hooks.agent_end);
  },
};

export default graphitiPlugin;
