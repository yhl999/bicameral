import { spawn } from 'node:child_process';
import path from 'node:path';

import type { PluginConfig } from '../config.ts';
import { toCanonicalPath } from '../path-utils.ts';

export interface CapabilityEntry {
  id: string;
  name: string;
  kind: string;
  description: string;
  path_or_command?: string;
  risk_level?: string;
  source?: string;
  score: number;
}

export interface CapabilityInjectorInput {
  prompt: string;
  intentId?: string | null;
}

const MAX_SELECTOR_OUTPUT_BYTES = 512_000;

const containsNullByte = (value: string): boolean => value.includes('\u0000');

const escapeXmlAttr = (value: string): string =>
  value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

const escapeXmlText = (value: string): string =>
  value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

const splitCommand = (command: string | string[]): string[] => {
  if (Array.isArray(command)) {
    return command.map((p) => p.trim()).filter((p) => p.length > 0);
  }
  const parts: string[] = [];
  let current = '';
  let quote: 'single' | 'double' | null = null;
  const trimmed = command.trim();

  for (let i = 0; i < trimmed.length; i += 1) {
    const char = trimmed[i];
    if (char === '\\' && quote !== 'single') {
      const next = trimmed[i + 1];
      if (next && /[\s"'\\]/.test(next)) {
        current += next;
        i += 1;
        continue;
      }
    }
    if (char === "'" && quote !== 'double') {
      quote = quote === 'single' ? null : 'single';
      continue;
    }
    if (char === '"' && quote !== 'single') {
      quote = quote === 'double' ? null : 'double';
      continue;
    }
    if (!quote && /\s/.test(char)) {
      if (current) {
        parts.push(current);
        current = '';
      }
      continue;
    }
    current += char;
  }
  if (current) parts.push(current);
  return parts;
};

const runSelector = (
  command: string | string[],
  args: string[],
  timeoutMs: number,
  cwd: string,
): Promise<string> => {
  return new Promise((resolve, reject) => {
    let settled = false;
    const resolveOnce = (v: string) => {
      if (settled) return;
      settled = true;
      resolve(v);
    };
    const rejectOnce = (e: Error) => {
      if (settled) return;
      settled = true;
      reject(e);
    };

    const parts = splitCommand(command);
    if (parts.length === 0) {
      rejectOnce(new Error('Capability selector command is empty'));
      return;
    }
    if (parts.some(containsNullByte) || args.some(containsNullByte)) {
      rejectOnce(new Error('Capability selector args contain null bytes'));
      return;
    }

    const [cmd, ...baseArgs] = parts;
    const child = spawn(cmd, [...baseArgs, ...args], {
      stdio: ['ignore', 'pipe', 'pipe'],
      shell: false,
      cwd,
    });

    let stdout = '';
    let stderr = '';
    let stdoutBytes = 0;

    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      rejectOnce(new Error('Capability selector timed out'));
    }, timeoutMs);

    child.stdout.on('data', (chunk: Buffer) => {
      stdoutBytes += chunk.length;
      if (stdoutBytes > MAX_SELECTOR_OUTPUT_BYTES) {
        child.kill('SIGKILL');
        rejectOnce(new Error('Capability selector exceeded output limit'));
        return;
      }
      stdout += chunk.toString();
    });
    child.stderr.on('data', (chunk: Buffer) => {
      stderr += chunk.toString();
    });
    child.on('error', (err) => {
      clearTimeout(timer);
      rejectOnce(err as Error);
    });
    child.on('close', (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        const detail = stderr.trim() || `exit code ${code}`;
        rejectOnce(new Error(`Capability selector failed: ${detail}`));
        return;
      }
      resolveOnce(stdout.trim());
    });
  });
};

const parseSelected = (raw: string): CapabilityEntry[] => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error('Capability selector returned invalid JSON');
  }
  if (typeof parsed !== 'object' || parsed === null || !('selected' in parsed)) {
    throw new Error('Capability selector returned unexpected shape');
  }
  const selected = (parsed as { selected: unknown }).selected;
  if (!Array.isArray(selected)) {
    throw new Error('Capability selector .selected is not an array');
  }
  return selected.map((entry) => ({
    id: String(entry.id ?? ''),
    name: String(entry.name ?? ''),
    kind: String(entry.kind ?? ''),
    description: String(entry.description ?? ''),
    path_or_command: entry.path_or_command ? String(entry.path_or_command) : undefined,
    risk_level: entry.risk_level ? String(entry.risk_level) : undefined,
    source: entry.source ? String(entry.source) : undefined,
    score: typeof entry.score === 'number' ? entry.score : 0,
  }));
};

const formatCapabilityContext = (
  entries: CapabilityEntry[],
  intentId?: string | null,
): string => {
  if (entries.length === 0) return '';

  const lines: string[] = [];
  const intentAttr = intentId ? ` intent="${escapeXmlAttr(intentId)}"` : '';
  lines.push(`<capability-context${intentAttr} count="${entries.length}">`);
  lines.push('## Relevant Capabilities');

  for (const entry of entries) {
    const risk = entry.risk_level ? ` [${escapeXmlText(entry.risk_level)}]` : '';
    lines.push(
      `- ${escapeXmlText(entry.name)} (${escapeXmlText(entry.kind)}): ${escapeXmlText(entry.description)}${risk}`,
    );
  }

  lines.push('</capability-context>');
  return lines.join('\n');
};

export interface CapabilityInjectorDeps {
  config: PluginConfig;
}

export const createCapabilityInjector = (deps: CapabilityInjectorDeps) => {
  const config = deps.config;
  const logger = config.debug ? (msg: string) => console.log(msg) : () => undefined;

  return async (input: CapabilityInjectorInput): Promise<string | null> => {
    if (!config.enableCapabilityInjection || !config.capabilitySelectorCommand) {
      return null;
    }

    const repoRoot = config.packRouterRepoRoot ?? process.cwd();
    const resolvedRoot = toCanonicalPath(path.resolve(repoRoot), 'capability repo root');

    const args: string[] = [
      '--query',
      input.prompt,
      '--top-n',
      String(config.capabilityTopN ?? 8),
      '--json',
    ];

    if (input.intentId) {
      args.push('--intent-id', input.intentId);
    }

    if (config.capabilityIndexPath) {
      args.push('--index', config.capabilityIndexPath);
    }

    if (config.capabilityOverridesPath) {
      args.push('--overrides', config.capabilityOverridesPath);
    }

    try {
      const raw = await runSelector(
        config.capabilitySelectorCommand,
        args,
        config.capabilitySelectorTimeoutMs ?? 2000,
        resolvedRoot,
      );
      const entries = parseSelected(raw);
      if (entries.length === 0) {
        logger('Capability selector returned 0 entries');
        return null;
      }
      const context = formatCapabilityContext(entries, input.intentId);
      logger(`Capability injection: ${entries.length} entries for intent=${input.intentId ?? 'none'}`);
      return context;
    } catch (error) {
      logger(`Capability injection failed: ${(error as Error).message}`);
      return null;
    }
  };
};
