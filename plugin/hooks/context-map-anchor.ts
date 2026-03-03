import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

import { normalizeConfig } from '../config.ts';
import type { PluginConfig } from '../config.ts';

const CONTEXT_MAP_ANCHOR_STATE = Symbol.for('bicameral.context-map-anchor.state');
const CONTEXT_MAP_ANCHOR_STATE_CACHE = new WeakMap<object, ContextMapAnchorState>();
const DEFAULT_ANCHOR_TEXT =
  'Use the context map as authoritative structure for this session when relevant.';

interface ContextMapAnchorState {
  pendingTrigger: boolean;
  lastFingerprint?: string;
}

type ContextMapAnchorContext = object;

type ContextMapAnchorHookResult = { prependContext?: string };

type ContextMapAnchorHook = (
  event: unknown,
  ctx: ContextMapAnchorContext,
) => Promise<ContextMapAnchorHookResult>;

interface ContextMapAnchorHooks {
  session_start: ContextMapAnchorHook;
  after_compaction: ContextMapAnchorHook;
  before_reset: ContextMapAnchorHook;
  before_prompt_build: ContextMapAnchorHook;
}

interface ContextMapAnchorHookDeps {
  config?: Partial<PluginConfig>;
}

type MarkableContext = ContextMapAnchorContext & {
  [CONTEXT_MAP_ANCHOR_STATE]?: ContextMapAnchorState;
};

const normalizeOptionalString = (value?: string): string | undefined => {
  if (!value) {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
};

const getState = (ctx: ContextMapAnchorContext): ContextMapAnchorState => {
  if (!ctx || typeof ctx !== 'object') {
    return { pendingTrigger: false };
  }

  if (CONTEXT_MAP_ANCHOR_STATE_CACHE.has(ctx)) {
    return CONTEXT_MAP_ANCHOR_STATE_CACHE.get(ctx)!;
  }

  let state: ContextMapAnchorState | undefined;
  try {
    state = (ctx as MarkableContext)[CONTEXT_MAP_ANCHOR_STATE];
  } catch {
    state = undefined;
  }

  if (!state) {
    state = { pendingTrigger: false };
    try {
      (ctx as MarkableContext)[CONTEXT_MAP_ANCHOR_STATE] = state;
    } catch {
      // Some runtimes freeze/lock context objects.
    }
  }

  CONTEXT_MAP_ANCHOR_STATE_CACHE.set(ctx, state);
  return state;
};

const markTriggered = (ctx: ContextMapAnchorContext): void => {
  const state = getState(ctx);
  state.pendingTrigger = true;
};

const clearState = (ctx: ContextMapAnchorContext): void => {
  if (!ctx || typeof ctx !== 'object') {
    return;
  }
  const state = getState(ctx);
  state.pendingTrigger = false;
  state.lastFingerprint = undefined;
};

const resolveConfiguredPath = (filePath?: string): string | undefined => {
  const normalized = normalizeOptionalString(filePath);
  if (!normalized) {
    return undefined;
  }
  return path.resolve(normalized);
};

const computeContextMapFingerprint = (config: PluginConfig): string | undefined => {
  const mapPath = resolveConfiguredPath(config.contextMapPath);
  const metaPath = resolveConfiguredPath(config.contextMapMetaPath);

  const hasher = createHash('sha256');
  let consumed = false;

  const maybeAdd = (label: 'map' | 'meta', filePath?: string): void => {
    if (!filePath) {
      return;
    }
    try {
      const content = fs.readFileSync(filePath);
      hasher.update(`${label}\0${filePath}\0`, 'utf8');
      hasher.update(content);
      consumed = true;
    } catch {
      // Missing/unreadable files are treated as absent in this scaffold.
    }
  };

  maybeAdd('map', mapPath);
  maybeAdd('meta', metaPath);

  if (!consumed) {
    return undefined;
  }

  return hasher.digest('hex');
};

const escapeXml = (text: string): string =>
  text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

const renderAnchorContext = (config: PluginConfig): string => {
  const mapPath = normalizeOptionalString(config.contextMapPath);
  const metaPath = normalizeOptionalString(config.contextMapMetaPath);
  const anchorText = normalizeOptionalString(config.contextMapAnchorText) ?? DEFAULT_ANCHOR_TEXT;

  const lines: string[] = ['<context-map-anchor>', escapeXml(anchorText)];
  if (mapPath) {
    lines.push(`map_path: ${escapeXml(mapPath)}`);
  }
  if (metaPath) {
    lines.push(`meta_path: ${escapeXml(metaPath)}`);
  }
  lines.push('</context-map-anchor>');
  return lines.join('\n');
};

export const createContextMapAnchorHooks = (
  deps?: ContextMapAnchorHookDeps,
): ContextMapAnchorHooks => {
  const config = normalizeConfig(deps?.config);

  return {
    session_start: async (_event, ctx) => {
      if (!config.enableContextMapAnchor) {
        return {};
      }
      markTriggered(ctx);
      return {};
    },
    after_compaction: async (_event, ctx) => {
      if (!config.enableContextMapAnchor) {
        return {};
      }
      markTriggered(ctx);
      return {};
    },
    before_reset: async (_event, ctx) => {
      clearState(ctx);
      return {};
    },
    before_prompt_build: async (_event, ctx) => {
      if (!config.enableContextMapAnchor) {
        return {};
      }

      const state = getState(ctx);
      if (!state.pendingTrigger) {
        return {};
      }
      state.pendingTrigger = false;

      const fingerprint = computeContextMapFingerprint(config);
      if (!fingerprint) {
        return {};
      }
      if (state.lastFingerprint === fingerprint) {
        return {};
      }

      state.lastFingerprint = fingerprint;
      return { prependContext: renderAnchorContext(config) };
    },
  };
};
