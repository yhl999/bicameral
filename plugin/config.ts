import fs from 'node:fs';
import path from 'node:path';

import type { CompositionRuleSet } from './composition/types.ts';
import type { IntentRuleSet } from './intent/types.ts';

export interface PackRegistryEntry {
  pack_id: string;
  pack_type?: string;
  path: string;
  scope: 'public' | 'group-safe' | 'private';
}

export interface PackRegistry {
  schema_version: number;
  packs: PackRegistryEntry[];
}

export interface PluginConfig {
  graphitiBaseUrl: string;
  graphitiApiKey?: string;
  recallTimeoutMs: number;
  captureTimeoutMs: number;
  maxFacts: number;
  minPromptChars: number;
  enableSticky: boolean;
  stickyMaxWords: number;
  stickySignals: string[];
  intentRulesPath?: string;
  compositionRulesPath?: string;
  packRegistryPath?: string;
  packRouterCommand?: string;
  packRouterRepoRoot?: string;
  packRouterTimeoutMs: number;
  defaultMinConfidence: number;
  debug: boolean;
}

export const DEFAULT_CONFIG: PluginConfig = {
  graphitiBaseUrl: 'http://localhost:8000',
  recallTimeoutMs: 1500,
  captureTimeoutMs: 1500,
  maxFacts: 8,
  minPromptChars: 6,
  enableSticky: true,
  stickyMaxWords: 20,
  stickySignals: ['also', 'and', 'continue', 'what about', 'follow up'],
  packRouterTimeoutMs: 2000,
  defaultMinConfidence: 0.3,
  debug: false,
};

export const normalizeConfig = (config?: Partial<PluginConfig>): PluginConfig => {
  return {
    ...DEFAULT_CONFIG,
    ...config,
    stickySignals: config?.stickySignals ?? DEFAULT_CONFIG.stickySignals,
  };
};

const readConfigFile = <T>(filePath: string): T => {
  const absolute = path.resolve(filePath);
  const raw = fs.readFileSync(absolute, 'utf8');
  try {
    return JSON.parse(raw) as T;
  } catch (error) {
    throw new Error(
      `Config file ${absolute} must be JSON for this scaffold. ${(error as Error).message}`,
    );
  }
};

export const loadIntentRules = (filePath?: string): IntentRuleSet | null => {
  if (!filePath) {
    return null;
  }
  return readConfigFile<IntentRuleSet>(filePath);
};

export const loadCompositionRules = (filePath?: string): CompositionRuleSet | null => {
  if (!filePath) {
    return null;
  }
  return readConfigFile<CompositionRuleSet>(filePath);
};

export const loadPackRegistry = (filePath?: string): PackRegistry | null => {
  if (!filePath) {
    return null;
  }
  return readConfigFile<PackRegistry>(filePath);
};
