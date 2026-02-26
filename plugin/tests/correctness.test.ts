import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { createPackInjector } from '../hooks/pack-injector.ts';
import { createCaptureHook, stripInjectedContext } from '../hooks/capture.ts';
import { createLegacyBeforeAgentStartHook } from '../hooks/legacy-before-agent-start.ts';
import { createModelResolveHook } from '../hooks/model-resolve.ts';
import { createRecallHook } from '../hooks/recall.ts';
import { deriveGroupLane } from '../lane-utils.ts';
import { detectIntent } from '../intent/detector.ts';
import type { IntentRuleSet } from '../intent/types.ts';
import { loadIntentRules, normalizeConfig } from '../config.ts';
import type { PackRegistry } from '../config.ts';

const rules: IntentRuleSet = {
  schema_version: 1,
  rules: [
    {
      id: 'summary',
      consumerProfile: 'main_session_example_summary',
      workflowId: 'example_summary',
      stepId: 'draft',
      packType: 'example_summary_pack',
      keywords: ['summary', 'recap'],
      keywordWeight: 1,
      minConfidence: 0.3,
      scope: 'public',
      entityBoosts: [
        {
          summaryPattern: 'report',
          weight: 0.5,
        },
      ],
    },
    {
      id: 'research',
      consumerProfile: 'main_session_example_research',
      workflowId: 'example_research',
      stepId: 'synthesize',
      packType: 'example_research_pack',
      keywords: ['research', 'analysis'],
      keywordWeight: 1,
      minConfidence: 0.3,
      scope: 'private',
    },
  ],
};

const registry: PackRegistry = {
  schema_version: 1,
  packs: [
    {
      pack_id: 'example_summary_pack',
      pack_type: 'example_summary_pack',
      path: 'workflows/example_summary.pack.yaml',
      scope: 'group-safe',
    },
    {
      pack_id: 'example_research_pack',
      pack_type: 'example_research_pack',
      path: 'workflows/example_research.pack.yaml',
      scope: 'private',
    },
  ],
};

const makeTempDir = (t: { after: (fn: () => void) => void }, prefix: string): string => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  t.after(() => {
    fs.rmSync(dir, { recursive: true, force: true });
  });
  return dir;
};

test('P1: no keyword match yields no pack', () => {
  const { decision } = detectIntent(rules, { prompt: 'hello there', defaultMinConfidence: 0.3 });
  assert.equal(decision.matched, false);
});

test('P2: deterministic selection for same input', () => {
  const input = { prompt: 'summary please', defaultMinConfidence: 0.3 };
  const first = detectIntent(rules, input).decision;
  const second = detectIntent(rules, input).decision;
  assert.deepEqual(first, second);
});

test('P3: tie yields no pack', () => {
  const tieRules: IntentRuleSet = {
    schema_version: 1,
    rules: [
      { id: 'a', consumerProfile: 'a', keywords: ['hello'], keywordWeight: 1 },
      { id: 'b', consumerProfile: 'b', keywords: ['hello'], keywordWeight: 1 },
    ],
  };
  const { decision } = detectIntent(tieRules, { prompt: 'hello', defaultMinConfidence: 0.3 });
  assert.equal(decision.matched, false);
});

test('P4: entity boost increases score', () => {
  const base = detectIntent(rules, { prompt: 'summary', defaultMinConfidence: 0.3 });
  const boosted = detectIntent(rules, {
    prompt: 'summary',
    graphitiResults: { facts: [{ fact: 'report mentions key finding' }] },
    defaultMinConfidence: 0.3,
  });
  assert.ok(boosted.decision.score >= base.decision.score);
});

test('P5: group chat blocks private packs', async () => {
  const injector = createPackInjector({ intentRules: rules, packRegistry: registry });
  const result = await injector({
    prompt: 'research analysis',
    ctx: { messageProvider: { chatType: 'group' } },
    graphitiResults: null,
  });
  assert.equal(result, null);
});

test('P6: injector errors fallback to null', async () => {
  const injector = createPackInjector({
    intentRules: {
      schema_version: 1,
      rules: [
        {
          id: 'missing',
          consumerProfile: 'missing',
          keywords: ['missing'],
          packType: 'does_not_exist',
        },
      ],
    },
    packRegistry: registry,
  });
  const result = await injector({
    prompt: 'missing',
    ctx: {},
    graphitiResults: null,
  });
  assert.equal(result, null);
});

test('capture strips injected context blocks', () => {
  const raw = `hello\n<graphiti-context>facts</graphiti-context>\n<pack-context intent="x">pack</pack-context>`;
  assert.equal(stripInjectedContext(raw), 'hello');
});

test('invalid regex patterns log debug output', () => {
  const logs: string[] = [];
  detectIntent(
    {
      schema_version: 1,
      rules: [
        {
          id: 'bad_regex',
          consumerProfile: 'bad_regex',
          keywords: ['summary'],
          entityBoosts: [{ summaryPattern: '[', weight: 0.5 }],
        },
      ],
    },
    {
      prompt: 'summary',
      defaultMinConfidence: 0.3,
      graphitiResults: { facts: [], entities: [] },
      logger: (message) => logs.push(message),
    },
  );
  assert.ok(logs.some((entry) => entry.includes('Invalid regex pattern')));
});

test('pack context escapes XML attributes', async (t: any) => {
  const tempDir = makeTempDir(t, 'graphiti-xml-attr-');
  const packSubDir = path.join(tempDir, 'workflows');
  fs.mkdirSync(packSubDir, { recursive: true });
  fs.writeFileSync(path.join(packSubDir, 'example_summary.pack.yaml'), 'pack content', 'utf8');

  const intentId = 'intent "alpha" & <beta>';
  const packId = 'pack "alpha" & <beta>';
  const injector = createPackInjector({
    intentRules: {
      schema_version: 1,
      rules: [
        {
          id: intentId,
          consumerProfile: 'main_session_example_summary',
          workflowId: 'example_summary',
          stepId: 'draft',
          packType: packId,
          keywords: ['summary'],
          keywordWeight: 1,
          minConfidence: 0.3,
          scope: 'public',
        },
      ],
    },
    packRegistry: {
      schema_version: 1,
      packs: [
        {
          pack_id: packId,
          pack_type: packId,
          path: 'workflows/example_summary.pack.yaml',
          scope: 'public',
        },
      ],
    },
    config: {
      packRouterRepoRoot: tempDir,
    },
  });

  const result = await injector({
    prompt: 'summary',
    ctx: {},
    graphitiResults: null,
  });

  assert.ok(result);
  assert.ok(
    result.context.includes(
      'intent="intent &quot;alpha&quot; &amp; &lt;beta&gt;"',
    ),
  );
  assert.ok(
    result.context.includes(
      'primary-pack="pack &quot;alpha&quot; &amp; &lt;beta&gt;"',
    ),
  );
});

test('pack router command supports quoted paths with spaces', async (t) => {
  const tempDir = makeTempDir(t, 'graphiti pack router ');
  const packFile = path.join(tempDir, 'pack.yaml');
  fs.writeFileSync(packFile, 'router pack content', 'utf8');

  const plan = {
    consumer: 'main_session_example_summary',
    workflow_id: 'example_summary',
    step_id: 'draft',
    scope: 'public',
    task: '',
    injection_text: '',
    packs: [{ pack_id: 'router_pack', query: 'pack.yaml' }],
  };

  const scriptPath = path.join(tempDir, 'pack router.js');
  fs.writeFileSync(
    scriptPath,
    `process.stdout.write(${JSON.stringify(JSON.stringify(plan))});`,
    'utf8',
  );

  const injector = createPackInjector({
    intentRules: {
      schema_version: 1,
      rules: [
        {
          id: 'summary',
          consumerProfile: 'main_session_example_summary',
          workflowId: 'example_summary',
          stepId: 'draft',
          keywords: ['summary'],
        },
      ],
    },
    config: {
      packRouterCommand: `node "${scriptPath}"`,
      packRouterRepoRoot: tempDir,
    },
  });

  const result = await injector({
    prompt: 'summary',
    ctx: {},
    graphitiResults: null,
  });

  assert.ok(result);
  assert.ok(result.context.includes('router pack content'));
});

test('pack router legacy plan without inline content still loads YAML from query', async (t) => {
  const tempDir = makeTempDir(t, 'graphiti pack router legacy yaml ');
  fs.writeFileSync(path.join(tempDir, 'primary.yaml'), 'primary from yaml', 'utf8');
  fs.writeFileSync(path.join(tempDir, 'secondary.yaml'), 'secondary from yaml', 'utf8');

  const plan = {
    consumer: 'main_session_example_summary',
    workflow_id: 'example_summary',
    step_id: 'draft',
    scope: 'public',
    task: '',
    injection_text: '',
    packs: [
      { pack_id: 'primary_pack', query: 'primary.yaml' },
      { pack_id: 'secondary_pack', query: 'secondary.yaml' },
    ],
  };

  const scriptPath = path.join(tempDir, 'pack-router.js');
  fs.writeFileSync(
    scriptPath,
    `process.stdout.write(${JSON.stringify(JSON.stringify(plan))});`,
    'utf8',
  );

  const injector = createPackInjector({
    intentRules: {
      schema_version: 1,
      rules: [
        {
          id: 'summary',
          consumerProfile: 'main_session_example_summary',
          workflowId: 'example_summary',
          stepId: 'draft',
          keywords: ['summary'],
        },
      ],
    },
    config: {
      packRouterCommand: ['node', scriptPath],
      packRouterRepoRoot: tempDir,
    },
  });

  const result = await injector({
    prompt: 'summary',
    ctx: {},
    graphitiResults: null,
  });

  assert.ok(result);
  assert.ok(result.context.includes('primary from yaml'));
  assert.ok(result.context.includes('### Composition: secondary_pack'));
  assert.ok(result.context.includes('secondary from yaml'));
});

test('pack router inline content is preferred and does not require pack file fallback', async (t) => {
  const tempDir = makeTempDir(t, 'graphiti pack router inline content ');
  const plan = {
    consumer: 'main_session_example_summary',
    workflow_id: 'example_summary',
    step_id: 'draft',
    scope: 'public',
    task: '',
    injection_text: '',
    packs: [
      {
        pack_id: 'primary_pack',
        query: 'missing-primary.yaml',
        content: 'primary inline content',
      },
      {
        pack_id: 'secondary_pack',
        query: 'missing-secondary.yaml',
        content: 'secondary inline content',
      },
    ],
  };

  const scriptPath = path.join(tempDir, 'pack-router.js');
  fs.writeFileSync(
    scriptPath,
    `process.stdout.write(${JSON.stringify(JSON.stringify(plan))});`,
    'utf8',
  );

  const injector = createPackInjector({
    intentRules: {
      schema_version: 1,
      rules: [
        {
          id: 'summary',
          consumerProfile: 'main_session_example_summary',
          workflowId: 'example_summary',
          stepId: 'draft',
          keywords: ['summary'],
        },
      ],
    },
    config: {
      packRouterCommand: ['node', scriptPath],
      packRouterRepoRoot: tempDir,
    },
  });

  const result = await injector({
    prompt: 'summary',
    ctx: {},
    graphitiResults: null,
  });

  assert.ok(result);
  assert.ok(result.context.includes('primary inline content'));
  assert.ok(result.context.includes('### Composition: secondary_pack'));
  assert.ok(result.context.includes('secondary inline content'));
});

test('pack router rejects non-string inline content', async (t) => {
  const tempDir = makeTempDir(t, 'graphiti pack router invalid inline ');
  fs.writeFileSync(path.join(tempDir, 'pack.yaml'), 'fallback', 'utf8');

  const plan = {
    consumer: 'main_session_example_summary',
    workflow_id: 'example_summary',
    step_id: 'draft',
    scope: 'public',
    task: '',
    injection_text: '',
    packs: [{ pack_id: 'router_pack', query: 'pack.yaml', content: 123 }],
  };

  const scriptPath = path.join(tempDir, 'pack-router.js');
  fs.writeFileSync(
    scriptPath,
    `process.stdout.write(${JSON.stringify(JSON.stringify(plan))});`,
    'utf8',
  );

  const injector = createPackInjector({
    intentRules: {
      schema_version: 1,
      rules: [
        {
          id: 'summary',
          consumerProfile: 'main_session_example_summary',
          workflowId: 'example_summary',
          stepId: 'draft',
          keywords: ['summary'],
        },
      ],
    },
    config: {
      packRouterCommand: ['node', scriptPath],
      packRouterRepoRoot: tempDir,
    },
  });

  const result = await injector({
    prompt: 'summary',
    ctx: {},
    graphitiResults: null,
  });

  assert.equal(result, null);
});

test('invalid pack router output falls back to null', async (t) => {
  const tempDir = makeTempDir(t, 'graphiti pack router invalid ');
  const scriptPath = path.join(tempDir, 'pack router.js');
  fs.writeFileSync(scriptPath, 'process.stdout.write("{\\"packs\\": []}");', 'utf8');

  const injector = createPackInjector({
    intentRules: {
      schema_version: 1,
      rules: [
        {
          id: 'summary',
          consumerProfile: 'main_session_example_summary',
          workflowId: 'example_summary',
          stepId: 'draft',
          keywords: ['summary'],
        },
      ],
    },
    config: {
      packRouterCommand: `node "${scriptPath}"`,
      packRouterRepoRoot: tempDir,
    },
  });

  const result = await injector({
    prompt: 'summary',
    ctx: {},
    graphitiResults: null,
  });

  assert.equal(result, null);
});

test('pack router plan cannot escape repo root through symlink', async (t) => {
  const tempDir = makeTempDir(t, 'graphiti-pack-router-symlink-');
  const repoRoot = path.join(tempDir, 'repo');
  fs.mkdirSync(repoRoot, { recursive: true });

  const externalDir = makeTempDir(t, 'graphiti-pack-router-external-');
  const externalPackPath = path.join(externalDir, 'outside-pack.yaml');
  fs.writeFileSync(externalPackPath, 'outside content', 'utf8');

  const symlinkedPackPath = path.join(repoRoot, 'linked-pack.yaml');
  fs.symlinkSync(externalPackPath, symlinkedPackPath);

  const plan = {
    consumer: 'main_session_example_summary',
    workflow_id: 'example_summary',
    step_id: 'draft',
    scope: 'public',
    task: '',
    injection_text: '',
    packs: [{ pack_id: 'router_pack', query: 'linked-pack.yaml' }],
  };

  const scriptPath = path.join(tempDir, 'pack-router.js');
  fs.writeFileSync(
    scriptPath,
    `process.stdout.write(${JSON.stringify(JSON.stringify(plan))});`,
    'utf8',
  );

  const injector = createPackInjector({
    intentRules: {
      schema_version: 1,
      rules: [
        {
          id: 'summary',
          consumerProfile: 'main_session_example_summary',
          workflowId: 'example_summary',
          stepId: 'draft',
          keywords: ['summary'],
        },
      ],
    },
    config: {
      packRouterCommand: ['node', scriptPath],
      packRouterRepoRoot: repoRoot,
    },
  });

  const result = await injector({
    prompt: 'summary',
    ctx: {},
    graphitiResults: null,
  });

  assert.equal(result, null);
});

test('config path allowlist rejects outside roots', (t) => {
  const tempDir = makeTempDir(t, 'graphiti-config-');
  const rulesPath = path.join(tempDir, 'intent_rules.json');
  fs.writeFileSync(rulesPath, JSON.stringify({ schema_version: 1, rules: [] }), 'utf8');

  const allowedRoot = makeTempDir(t, 'graphiti-allowed-');
  assert.throws(
    () => loadIntentRules(rulesPath, [allowedRoot]),
    /outside allowed roots/,
  );
});

test('config path allowlist rejects symlink escapes', (t) => {
  const allowedRoot = makeTempDir(t, 'graphiti-allowed-root-');
  const externalRoot = makeTempDir(t, 'graphiti-external-root-');

  const externalFile = path.join(externalRoot, 'intent_rules.json');
  fs.writeFileSync(externalFile, JSON.stringify({ schema_version: 1, rules: [] }), 'utf8');

  const linkedPath = path.join(allowedRoot, 'intent_rules_link.json');
  fs.symlinkSync(externalFile, linkedPath);

  assert.throws(
    () => loadIntentRules(linkedPath, [allowedRoot]),
    /outside allowed roots/,
  );
});

test('pack context escapes XML text for workflow metadata', async (t) => {
  const tempDir = makeTempDir(t, 'graphiti-pack-router-xml-text-');
  const packFile = path.join(tempDir, 'pack.yaml');
  fs.writeFileSync(packFile, 'router pack content', 'utf8');

  const plan = {
    consumer: 'main_session_example_summary',
    workflow_id: 'wf <alpha> & beta',
    step_id: 'draft',
    scope: 'public',
    task: 'task <x> & y',
    injection_text: 'inject <tag> & z',
    packs: [{ pack_id: 'router_pack', query: 'pack.yaml' }],
  };

  const scriptPath = path.join(tempDir, 'pack-router.js');
  fs.writeFileSync(
    scriptPath,
    `process.stdout.write(${JSON.stringify(JSON.stringify(plan))});`,
    'utf8',
  );

  const injector = createPackInjector({
    intentRules: {
      schema_version: 1,
      rules: [
        {
          id: 'summary',
          consumerProfile: 'main_session_example_summary',
          workflowId: 'example_summary',
          stepId: 'draft',
          keywords: ['summary'],
        },
      ],
    },
    config: {
      packRouterCommand: ['node', scriptPath],
      packRouterRepoRoot: tempDir,
    },
  });

  const result = await injector({
    prompt: 'summary',
    ctx: {},
    graphitiResults: null,
  });

  assert.ok(result);
  assert.ok(result.context.includes('## Active Workflow: wf &lt;alpha&gt; &amp; beta'));
  assert.ok(result.context.includes('Task: task &lt;x&gt; &amp; y'));
  assert.ok(result.context.includes('inject &lt;tag&gt; &amp; z'));
});

test('pack router command path with spaces must be quoted or array form', async (t) => {
  const tempDir = makeTempDir(t, 'graphiti-router-unquoted-path-');
  const scriptPath = path.join(tempDir, 'pack router.js');
  fs.writeFileSync(scriptPath, 'process.stdout.write("{}")', 'utf8');

  const injector = createPackInjector({
    intentRules: {
      schema_version: 1,
      rules: [
        {
          id: 'summary',
          consumerProfile: 'main_session_example_summary',
          workflowId: 'example_summary',
          stepId: 'draft',
          keywords: ['summary'],
        },
      ],
    },
    config: {
      packRouterCommand: scriptPath,
      packRouterRepoRoot: tempDir,
    },
  });

  const result = await injector({
    prompt: 'summary',
    ctx: {},
    graphitiResults: null,
  });

  assert.equal(result, null);
});

test('config path allowlist rejects non-existent roots', (t) => {
  const tempDir = makeTempDir(t, 'graphiti-config-root-missing-');
  const rulesPath = path.join(tempDir, 'intent_rules.json');
  fs.writeFileSync(rulesPath, JSON.stringify({ schema_version: 1, rules: [] }), 'utf8');

  const missingRoot = path.join(tempDir, 'does-not-exist');
  assert.throws(
    () => loadIntentRules(rulesPath, [missingRoot]),
    /Unable to resolve config root/,
  );
});

test('legacy before_agent_start shim skips when messages are absent', async () => {
  let delegated = false;
  const shim = createLegacyBeforeAgentStartHook(async () => {
    delegated = true;
    return { prependContext: 'should-not-run' };
  });

  const result = await shim({ prompt: 'hello' }, {});
  assert.deepEqual(result, {});
  assert.equal(delegated, false);
});

test('legacy before_agent_start shim delegates when messages are present', async () => {
  const shim = createLegacyBeforeAgentStartHook(async () => ({
    prependContext: '<graphiti-context>ok</graphiti-context>',
  }));

  const result = await shim(
    {
      prompt: 'hello',
      messages: [{ role: 'user', content: 'hello' }],
    },
    {},
  );

  assert.equal(result.prependContext, '<graphiti-context>ok</graphiti-context>');
});

test('legacy before_agent_start shim marks context after first delegation', async () => {
  let delegatedCount = 0;
  const shim = createLegacyBeforeAgentStartHook(async () => {
    delegatedCount += 1;
    return { prependContext: '<graphiti-context>ok</graphiti-context>' };
  });

  const ctx: Record<string, unknown> = {};
  await shim(
    {
      prompt: 'hello',
      messages: [{ role: 'user', content: 'hello' }],
    },
    ctx,
  );
  const second = await shim(
    {
      prompt: 'hello again',
      messages: [{ role: 'user', content: 'hello again' }],
    },
    ctx,
  );

  assert.equal(delegatedCount, 1);
  assert.deepEqual(second, {});
});

test('legacy before_agent_start shim remains safe for frozen context objects', async () => {
  let delegatedCount = 0;
  const shim = createLegacyBeforeAgentStartHook(async () => {
    delegatedCount += 1;
    return { prependContext: '<graphiti-context>ok</graphiti-context>' };
  });

  const frozenCtx = Object.freeze({}) as Record<string, unknown>;
  await shim(
    {
      prompt: 'hello',
      messages: [{ role: 'user', content: 'hello' }],
    },
    frozenCtx,
  );
  const second = await shim(
    {
      prompt: 'hello again',
      messages: [{ role: 'user', content: 'hello again' }],
    },
    frozenCtx,
  );

  assert.equal(delegatedCount, 1);
  assert.deepEqual(second, {});
});

test('legacy before_agent_start shim skips when messages list is empty', async () => {
  let delegated = false;
  const shim = createLegacyBeforeAgentStartHook(async () => {
    delegated = true;
    return { prependContext: 'should-not-run' };
  });

  const result = await shim({ prompt: 'hello', messages: [] }, {});
  assert.deepEqual(result, {});
  assert.equal(delegated, false);
});

test('legacy before_agent_start shim skips if prompt build already executed', async () => {
  let delegated = false;
  const shim = createLegacyBeforeAgentStartHook(async () => {
    delegated = true;
    return { prependContext: 'should-not-run' };
  });

  const ctx: Record<string, unknown> = {};
  const marker = Symbol.for('graphiti.plugin.prompt-build-ran');
  ctx[marker] = true;

  const result = await shim(
    {
      prompt: 'hello',
      messages: [{ role: 'user', content: 'hello' }],
    },
    ctx,
  );

  assert.deepEqual(result, {});
  assert.equal(delegated, false);
});

test('before_model_resolve hook with no config returns no overrides', async () => {
  const hook = createModelResolveHook();
  const result = await hook({ prompt: 'route this' }, {});
  assert.equal(result.providerOverride, undefined);
  assert.equal(result.modelOverride, undefined);
});

test('before_model_resolve hook requires explicit opt-in + allowlist', async () => {
  const hook = createModelResolveHook({
    config: {
      allowModelRoutingOverride: true,
      providerOverride: ' openai ',
      modelOverride: ' gpt-5.2 ',
      allowedProviderOverrides: ['openai'],
      allowedModelOverrides: ['gpt-5.2'],
    },
  });

  const result = await hook({ prompt: 'route this' }, {});
  assert.equal(result.providerOverride, 'openai');
  assert.equal(result.modelOverride, 'gpt-5.2');
});

test('before_model_resolve hook blocks non-allowlisted overrides', async () => {
  const hook = createModelResolveHook({
    config: {
      allowModelRoutingOverride: true,
      providerOverride: 'openai',
      modelOverride: 'gpt-5.2',
      allowedProviderOverrides: ['anthropic'],
      allowedModelOverrides: ['claude-sonnet-4-6'],
    },
  });

  const result = await hook({ prompt: 'route this' }, {});
  assert.equal(result.providerOverride, undefined);
  assert.equal(result.modelOverride, undefined);
});

test('before_model_resolve hook blocks invalid override tokens', async () => {
  const hook = createModelResolveHook({
    config: {
      allowModelRoutingOverride: true,
      providerOverride: 'openai;rm -rf /',
      modelOverride: 'gpt-5.2\nmalicious',
      allowedProviderOverrides: ['openai;rm -rf /'],
      allowedModelOverrides: ['gpt-5.2\nmalicious'],
    },
  });

  const result = await hook({ prompt: 'route this' }, {});
  assert.equal(result.providerOverride, undefined);
  assert.equal(result.modelOverride, undefined);
});

test('before_model_resolve hook blocks path traversal token shapes', async () => {
  const hook = createModelResolveHook({
    config: {
      allowModelRoutingOverride: true,
      providerOverride: '../../openai',
      modelOverride: '/unsafe/model',
      allowedProviderOverrides: ['../../openai'],
      allowedModelOverrides: ['/unsafe/model'],
    },
  });

  const result = await hook({ prompt: 'route this' }, {});
  assert.equal(result.providerOverride, undefined);
  assert.equal(result.modelOverride, undefined);
});

test('recall hook emits explicit fallback error block when Graphiti fails', async () => {
  const hook = createRecallHook({
    client: {
      search: async () => {
        throw new Error('Graphiti API error 503');
      },
      ingestMessages: async () => undefined,
    },
    packInjector: async () => null,
    config: {
      memoryGroupId: 's1_sessions_main',
    },
  });

  const result = await hook(
    { prompt: 'test fallback emission' },
    {
      sessionKey: 'agent:main:telegram:group:-1003893734334',
      messageProvider: { groupId: 'telegram:-1003893734334', chatType: 'group' },
    },
  );

  const context = result.prependContext ?? '';
  assert.ok(context.includes('<graphiti-fallback>'));
  assert.ok(context.includes('ERROR_CODE: GRAPHITI_QMD_FAILOVER'));
  assert.ok(context.includes('This turn is using QMD fallback'));
});

test('normalizeConfig drops empty memoryGroupId values', () => {
  const normalized = normalizeConfig({ memoryGroupId: '   ' });
  assert.equal(normalized.memoryGroupId, undefined);
});

test('recall hook prefers provider group over session key when memoryGroupId is unset', async () => {
  let capturedGroupIds: string[] | undefined;

  const hook = createRecallHook({
    client: {
      search: async (_query, groupIds) => {
        capturedGroupIds = groupIds;
        return { facts: [] };
      },
      ingestMessages: async () => undefined,
    },
    packInjector: async () => null,
    config: {},
  });

  await hook(
    { prompt: 'provider precedence check' },
    {
      sessionKey: 'session-lane',
      messageProvider: { groupId: 'provider-lane', chatType: 'group' },
    },
  );

  assert.deepEqual(capturedGroupIds, ['provider-lane']);
});

test('recall hook prefers configured memoryGroupId over provider/session lanes', async () => {
  let capturedGroupIds: string[] | undefined;

  const hook = createRecallHook({
    client: {
      search: async (_query, groupIds) => {
        capturedGroupIds = groupIds;
        return { facts: [] };
      },
      ingestMessages: async () => undefined,
    },
    packInjector: async () => null,
    config: {
      // singleTenant must be true to allow memoryGroupId override (tenant isolation fix).
      singleTenant: true,
      memoryGroupId: 'canonical-lane',
    },
  });

  await hook(
    { prompt: 'memoryGroup override check' },
    {
      sessionKey: 'session-lane',
      messageProvider: { groupId: 'provider-lane', chatType: 'group' },
    },
  );

  assert.deepEqual(capturedGroupIds, ['canonical-lane']);
});

test('capture hook prefers provider group over session key when memoryGroupId is unset', async () => {
  let capturedGroupId: string | undefined;

  const hook = createCaptureHook({
    client: {
      search: async () => ({ facts: [] }),
      ingestMessages: async (groupId: string) => {
        capturedGroupId = groupId;
      },
    },
    config: {},
  });

  await hook(
    {
      success: true,
      messages: [
        { role: 'user', content: 'hello' },
        { role: 'assistant', content: 'world' },
      ],
    },
    {
      sessionKey: 'session-lane',
      messageProvider: { groupId: 'provider-lane', chatType: 'group' },
    },
  );

  assert.equal(capturedGroupId, 'provider-lane');
});

test('capture hook prefers configured memoryGroupId over provider/session lanes', async () => {
  let capturedGroupId: string | undefined;

  const hook = createCaptureHook({
    client: {
      search: async () => ({ facts: [] }),
      ingestMessages: async (groupId: string) => {
        capturedGroupId = groupId;
      },
    },
    config: {
      // singleTenant must be true to allow memoryGroupId override (tenant isolation fix).
      singleTenant: true,
      memoryGroupId: 'canonical-lane',
    },
  });

  await hook(
    {
      success: true,
      messages: [
        { role: 'user', content: 'hello' },
        { role: 'assistant', content: 'world' },
      ],
    },
    {
      sessionKey: 'session-lane',
      messageProvider: { groupId: 'provider-lane', chatType: 'group' },
    },
  );

  assert.equal(capturedGroupId, 'canonical-lane');
});

test('capture hook forwards user+assistant turn to fast-write runner', async () => {
  const fastWritePayloads: Array<{ source_session_id: string; role: string; content: string }> = [];
  let capturedGroupId: string | undefined;

  const hook = createCaptureHook({
    client: {
      search: async () => ({ facts: [] }),
      ingestMessages: async (groupId: string) => {
        capturedGroupId = groupId;
      },
    },
    fastWriteRunner: async (payload) => {
      fastWritePayloads.push({
        source_session_id: payload.source_session_id,
        role: payload.role,
        content: payload.content,
      });
    },
    config: {},
  });

  await hook(
    {
      success: true,
      messages: [
        { role: 'user', content: ' user says hi ' },
        { role: 'assistant', content: ' assistant replies ' },
      ],
    },
    {
      sessionKey: 'session-lane',
      messageProvider: { groupId: 'provider-lane', chatType: 'group' },
    },
  );

  assert.equal(capturedGroupId, 'provider-lane');
  assert.deepEqual(fastWritePayloads, [
    {
      source_session_id: 'provider-lane',
      role: 'user',
      content: 'user says hi',
    },
    {
      source_session_id: 'provider-lane',
      role: 'assistant',
      content: 'assistant replies',
    },
  ]);
});

test('capture hook still ingests Graphiti when fast-write runner fails', async () => {
  let capturedGroupId: string | undefined;

  const hook = createCaptureHook({
    client: {
      search: async () => ({ facts: [] }),
      ingestMessages: async (groupId: string) => {
        capturedGroupId = groupId;
      },
    },
    fastWriteRunner: async () => {
      throw new Error('boom');
    },
    config: {
      debug: true,
    },
  });

  await hook(
    {
      success: true,
      messages: [
        { role: 'user', content: 'hello' },
        { role: 'assistant', content: 'world' },
      ],
    },
    {
      sessionKey: 'session-lane',
      messageProvider: { groupId: 'provider-lane', chatType: 'group' },
    },
  );

  assert.equal(capturedGroupId, 'provider-lane');
});

// ── Tenant isolation tests ─────────────────────────────────────────────────

test('recall hook ignores memoryGroupId when singleTenant is not set', async () => {
  let capturedGroupIds: string[] | undefined;

  const hook = createRecallHook({
    client: {
      search: async (_query, groupIds) => {
        capturedGroupIds = groupIds;
        return { facts: [] };
      },
      ingestMessages: async () => undefined,
    },
    packInjector: async () => null,
    // memoryGroupId without singleTenant: true — must NOT take effect.
    config: { memoryGroupId: 'pinned-lane' },
  });

  await hook(
    { prompt: 'tenant isolation check' },
    {
      sessionKey: 'session-lane',
      messageProvider: { groupId: 'provider-lane', chatType: 'group' },
    },
  );

  // Must fall through to provider-group lane, not the pinned override.
  assert.deepEqual(capturedGroupIds, ['provider-lane']);
});

test('recall hook sanitizes long session key in fallback warn log', async () => {
  const warnLines: string[] = [];
  const originalWarn = console.warn;
  console.warn = (...args: unknown[]) => warnLines.push(args.join(' '));
  try {
    const hook = createRecallHook({
      client: {
        search: async () => {
          throw new Error('service down');
        },
        ingestMessages: async () => undefined,
      },
      packInjector: async () => null,
      config: {},
    });

    await hook(
      { prompt: 'identifier sanitization check' },
      // Long session key that should be truncated in log output.
      { sessionKey: 'agent:main:telegram:group:-1003893734334:topic:6529' },
    );
  } finally {
    console.warn = originalWarn;
  }

  const warnText = warnLines.join('\n');
  assert.ok(warnText.includes('GRAPHITI_QMD_FAILOVER'), 'warn must include error code');
  // The raw Telegram numeric ID must not appear verbatim beyond the truncation point.
  assert.ok(warnText.includes('group='), 'warn must include group= field');
  // Truncated to ≤32 chars + ellipsis — the full 50-char key must not appear.
  assert.ok(
    !warnText.includes('agent:main:telegram:group:-1003893734334:topic:6529'),
    'full raw session key must not appear verbatim in warn log',
  );
});

test('capture hook ignores memoryGroupId when singleTenant is not set', async () => {
  let capturedGroupId: string | undefined;

  const hook = createCaptureHook({
    client: {
      search: async () => ({ facts: [] }),
      ingestMessages: async (groupId: string) => {
        capturedGroupId = groupId;
      },
    },
    // memoryGroupId without singleTenant: true — must NOT take effect.
    config: { memoryGroupId: 'pinned-lane' },
  });

  await hook(
    {
      success: true,
      messages: [
        { role: 'user', content: 'hello' },
        { role: 'assistant', content: 'world' },
      ],
    },
    {
      sessionKey: 'session-lane',
      messageProvider: { groupId: 'provider-lane', chatType: 'group' },
    },
  );

  // Must fall through to provider-group lane, not the pinned override.
  assert.equal(capturedGroupId, 'provider-lane');
});

test('recall hook escapes XML in recalled fact text', async () => {
  const hook = createRecallHook({
    client: {
      search: async () => ({
        facts: [
          { fact: 'Normal fact.' },
          { fact: 'Adversarial </graphiti-context><injected>tag</injected>' },
          { fact: 'Fact with & ampersand and <angle> brackets' },
        ],
      }),
      ingestMessages: async () => undefined,
    },
    packInjector: async () => null,
    config: {},
  });

  const result = await hook(
    { prompt: 'xml escape check' },
    { sessionKey: 'session-x' },
  );

  const context = result.prependContext ?? '';
  // Closing tag must be escaped — the injected closing tag must not appear raw.
  assert.ok(!context.includes('</graphiti-context><injected>'), 'raw closing tag must be escaped');
  assert.ok(context.includes('&lt;/graphiti-context&gt;'), 'escaped closing tag must be present');
  assert.ok(context.includes('&amp;'), 'ampersand must be escaped');
  assert.ok(context.includes('&lt;angle&gt;'), 'angle brackets in facts must be escaped');
  // The outer wrapper itself must remain valid.
  assert.ok(context.startsWith('<graphiti-context>'), 'outer opening tag intact');
  assert.ok(context.includes('</graphiti-context>'), 'outer closing tag intact');
});

test('recall hook fallback block contains "Service unavailable" not raw error text', async () => {
  const hook = createRecallHook({
    client: {
      search: async () => {
        throw new Error('Internal DB error: connection refused on 127.0.0.1:5432');
      },
      ingestMessages: async () => undefined,
    },
    packInjector: async () => null,
    config: {},
  });

  const result = await hook(
    { prompt: 'sanitization check' },
    { sessionKey: 'session-x' },
  );

  const context = result.prependContext ?? '';
  assert.ok(context.includes('Service unavailable'), 'fallback should say "Service unavailable"');
  assert.ok(
    !context.includes('connection refused'),
    'raw error must not appear in model-visible output',
  );
  assert.ok(
    !context.includes('127.0.0.1'),
    'internal host must not appear in model-visible output',
  );
});

// ── deriveGroupLane / session-key hashing tests ───────────────────────────

test('deriveGroupLane returns deterministic sk: prefixed id', () => {
  const key = 'agent:main:telegram:group:-1003893734334:topic:6529';
  const lane1 = deriveGroupLane(key);
  const lane2 = deriveGroupLane(key);
  assert.equal(lane1, lane2, 'must be deterministic');
  assert.ok(lane1.startsWith('sk:'), 'must have sk: prefix');
  // prefix is 16 hex chars → total length = 19
  assert.equal(lane1.length, 19);
  // must not contain any fragment of the original key
  assert.ok(!lane1.includes('telegram'), 'must not embed platform name');
  assert.ok(!lane1.includes('1003893734334'), 'must not embed numeric chat id');
});

test('deriveGroupLane produces different ids for different keys', () => {
  const a = deriveGroupLane('session-alpha');
  const b = deriveGroupLane('session-beta');
  assert.notEqual(a, b);
});

test('recall hook uses hashed lane when only sessionKey is available', async () => {
  let capturedGroupIds: string[] | undefined;
  const hook = createRecallHook({
    client: {
      search: async (_query, groupIds) => {
        capturedGroupIds = groupIds;
        return { facts: [] };
      },
      ingestMessages: async () => undefined,
    },
    packInjector: async () => null,
    config: {},
  });

  const sessionKey = 'agent:main:telegram:group:-1003893734334:topic:6529';
  await hook(
    { prompt: 'lane hashing check for recall' },
    { sessionKey },
  );

  const expected = deriveGroupLane(sessionKey);
  assert.deepEqual(capturedGroupIds, [expected], 'recall must use hashed lane');
  assert.ok(!capturedGroupIds![0].includes('1003893734334'), 'raw key must not appear in lane');
});

test('capture hook uses hashed lane when only sessionKey is available', async () => {
  let capturedGroupId: string | undefined;
  const hook = createCaptureHook({
    client: {
      search: async () => ({ facts: [] }),
      ingestMessages: async (groupId: string) => {
        capturedGroupId = groupId;
      },
    },
    config: {},
  });

  const sessionKey = 'agent:main:telegram:group:-1003893734334:topic:6529';
  await hook(
    {
      success: true,
      messages: [
        { role: 'user', content: 'hello' },
        { role: 'assistant', content: 'world' },
      ],
    },
    { sessionKey },
  );

  const expected = deriveGroupLane(sessionKey);
  assert.equal(capturedGroupId, expected, 'capture must use hashed lane');
  assert.ok(!capturedGroupId!.includes('1003893734334'), 'raw key must not appear in lane');
});

test('recall and capture use the same hashed lane for the same sessionKey', async () => {
  const sessionKey = 'agent:main:some:session:key';
  let recallLane: string | undefined;
  let captureLane: string | undefined;

  const recallHook = createRecallHook({
    client: {
      search: async (_query, groupIds) => {
        recallLane = groupIds?.[0];
        return { facts: [] };
      },
      ingestMessages: async () => undefined,
    },
    packInjector: async () => null,
    config: {},
  });

  const captureHook = createCaptureHook({
    client: {
      search: async () => ({ facts: [] }),
      ingestMessages: async (groupId: string) => {
        captureLane = groupId;
      },
    },
    config: {},
  });

  await recallHook({ prompt: 'consistency check' }, { sessionKey });
  await captureHook(
    {
      success: true,
      messages: [
        { role: 'user', content: 'hi' },
        { role: 'assistant', content: 'hello' },
      ],
    },
    { sessionKey },
  );

  assert.equal(recallLane, captureLane, 'recall and capture must share the same hashed lane');
});

test('recall hook still uses raw provider groupId (not hashed) when available', async () => {
  let capturedGroupIds: string[] | undefined;
  const hook = createRecallHook({
    client: {
      search: async (_query, groupIds) => {
        capturedGroupIds = groupIds;
        return { facts: [] };
      },
      ingestMessages: async () => undefined,
    },
    packInjector: async () => null,
    config: {},
  });

  await hook(
    { prompt: 'provider group should not be hashed' },
    {
      sessionKey: 'agent:main:telegram:group:-1003893734334',
      messageProvider: { groupId: 'telegram:-1003893734334', chatType: 'group' },
    },
  );

  // Provider groupId is passed as-is — only sessionKey fallback is hashed.
  assert.deepEqual(capturedGroupIds, ['telegram:-1003893734334']);
});

test('capture hook still uses raw provider groupId (not hashed) when available', async () => {
  let capturedGroupId: string | undefined;
  const hook = createCaptureHook({
    client: {
      search: async () => ({ facts: [] }),
      ingestMessages: async (groupId: string) => {
        capturedGroupId = groupId;
      },
    },
    config: {},
  });

  await hook(
    {
      success: true,
      messages: [
        { role: 'user', content: 'hello' },
        { role: 'assistant', content: 'world' },
      ],
    },
    {
      sessionKey: 'agent:main:telegram:group:-1003893734334',
      messageProvider: { groupId: 'telegram:-1003893734334', chatType: 'group' },
    },
  );

  assert.equal(capturedGroupId, 'telegram:-1003893734334');
});

test('pack router resolves relative script path from packRouterRepoRoot, not process.cwd()', async (t) => {
  // Create a temp dir that is NOT process.cwd(). The relative script path
  // "./test-router.js" must be resolved against packRouterRepoRoot so that
  // spawn can find and execute it without an ENOENT error.
  const repoRoot = makeTempDir(t, 'graphiti-relative-cwd-');

  const packFile = path.join(repoRoot, 'pack.yaml');
  fs.writeFileSync(packFile, 'relative cwd pack content', 'utf8');

  const plan = {
    consumer: 'main_session_example_summary',
    workflow_id: 'example_summary',
    step_id: 'draft',
    scope: 'public',
    task: '',
    injection_text: '',
    packs: [{ pack_id: 'router_pack', query: 'pack.yaml' }],
  };

  // Write the router script directly inside repoRoot.
  const scriptName = 'test-router.js';
  fs.writeFileSync(
    path.join(repoRoot, scriptName),
    `process.stdout.write(${JSON.stringify(JSON.stringify(plan))});`,
    'utf8',
  );

  const injector = createPackInjector({
    intentRules: {
      schema_version: 1,
      rules: [
        {
          id: 'summary',
          consumerProfile: 'main_session_example_summary',
          workflowId: 'example_summary',
          stepId: 'draft',
          keywords: ['summary'],
        },
      ],
    },
    config: {
      // Relative path — spawn must use repoRoot as cwd so the file is found.
      packRouterCommand: ['node', `./${scriptName}`],
      packRouterRepoRoot: repoRoot,
    },
  });

  const result = await injector({
    prompt: 'summary',
    ctx: {},
    graphitiResults: null,
  });

  // If spawn used process.cwd() instead of repoRoot, the script would not be
  // found and the injector would return null (ENOENT). A non-null result proves
  // that spawn received the correct cwd.
  assert.ok(result, 'injector must succeed when relative script path is resolved from repoRoot');
  assert.ok(result.context.includes('relative cwd pack content'));
});
