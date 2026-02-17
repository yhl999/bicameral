import { createCaptureHook } from './hooks/capture.ts';
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

export interface OpenClawPlugin {
  name: string;
  hooks: {
    before_agent_start: ReturnType<typeof createRecallHook>;
    agent_end: ReturnType<typeof createCaptureHook>;
  };
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

export const createGraphitiPlugin = (options?: GraphitiPluginOptions): OpenClawPlugin => {
  const config = normalizeConfig({
    ...loadConfigFromEnv(),
    ...(options?.config ?? {}),
  });

  const intentRules =
    options?.intentRules ?? loadIntentRules(config.intentRulesPath) ?? { schema_version: 1, rules: [] };
  const compositionRules =
    options?.compositionRules ?? loadCompositionRules(config.compositionRulesPath);
  const packRegistry = options?.packRegistry ?? loadPackRegistry(config.packRegistryPath);

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

  return {
    name: 'graphiti-openclaw',
    hooks: {
      before_agent_start: createRecallHook({
        client,
        packInjector,
        config,
      }),
      agent_end: createCaptureHook({
        client,
        config,
      }),
    },
  };
};

const plugin = createGraphitiPlugin();
export default plugin;
