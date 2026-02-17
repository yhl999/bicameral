import assert from 'node:assert/strict';
import test from 'node:test';

import { createPackInjector } from '../hooks/pack-injector.ts';
import { stripInjectedContext } from '../hooks/capture.ts';
import { detectIntent } from '../intent/detector.ts';
import type { IntentRuleSet } from '../intent/types.ts';
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
