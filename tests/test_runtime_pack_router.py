from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'runtime_pack_router.py'
REPO_ROOT = Path(__file__).resolve().parents[1]


class _GraphitiStubHandler(BaseHTTPRequestHandler):
    facts_by_query_keyword: dict[str, list[str]] = {}
    facts_by_group_id: dict[str, list[str]] = {}

    def do_POST(self) -> None:  # noqa: N802
        raw = self.rfile.read(int(self.headers.get('Content-Length', '0') or '0'))
        payload = json.loads(raw.decode('utf-8') or '{}')
        query = str(payload.get('query') or '')
        group_ids = payload.get('group_ids') if isinstance(payload.get('group_ids'), list) else []

        facts: list[dict[str, str]] = []
        for group_id in group_ids:
            if not isinstance(group_id, str):
                continue
            values = self.facts_by_group_id.get(group_id)
            if values:
                facts = [{'fact': value} for value in values]
                break

        if not facts:
            for keyword, values in self.facts_by_query_keyword.items():
                if keyword in query:
                    facts = [{'fact': value} for value in values]
                    break

        body = json.dumps({'facts': facts}).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class GraphitiStubServer:
    def __init__(
        self,
        *,
        facts_by_query_keyword: dict[str, list[str]] | None = None,
        facts_by_group_id: dict[str, list[str]] | None = None,
    ):
        self._facts = facts_by_query_keyword or {}
        self._facts_by_group = facts_by_group_id or {}
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> str:
        _GraphitiStubHandler.facts_by_query_keyword = self._facts
        _GraphitiStubHandler.facts_by_group_id = self._facts_by_group
        self._server = ThreadingHTTPServer(('127.0.0.1', 0), _GraphitiStubHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, port = self._server.server_address
        return f'http://{host}:{port}'

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)


class RuntimePackRouterTests(unittest.TestCase):
    def _seed_repo(self, repo: Path) -> None:
        config_dir = repo / 'config'
        workflows_dir = repo / 'workflows'
        config_dir.mkdir(parents=True, exist_ok=True)
        workflows_dir.mkdir(parents=True, exist_ok=True)

        for filename in (
            'runtime_pack_registry.yaml',
            'runtime_consumer_profiles.yaml',
        ):
            source = REPO_ROOT / 'config' / filename
            destination = config_dir / filename
            destination.write_text(source.read_text(encoding='utf-8'), encoding='utf-8')

        for filename in (
            'example_summary.pack.yaml',
            'example_research.pack.yaml',
        ):
            source = REPO_ROOT / 'workflows' / filename
            destination = workflows_dir / filename
            destination.write_text(source.read_text(encoding='utf-8'), encoding='utf-8')

    def _route(
        self,
        repo: Path,
        *,
        consumer: str,
        workflow_id: str,
        step_id: str,
        task: str,
        materialize: bool = False,
        scope: str | None = None,
        env: dict[str, str] | None = None,
    ) -> dict:
        out = repo / 'plan.json'
        cmd = [
            sys.executable,
            str(SCRIPT),
            '--consumer',
            consumer,
            '--workflow-id',
            workflow_id,
            '--step-id',
            step_id,
            '--repo',
            str(repo),
            '--task',
            task,
            '--validate',
            '--out',
            str(out),
        ]
        if materialize:
            cmd.append('--materialize')
        if scope:
            cmd.extend(['--scope', scope])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, **(env or {})},
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        return json.loads(out.read_text(encoding='utf-8'))

    def test_router_routes_example_consumers_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True)
            self._seed_repo(repo)

            summary_plan = self._route(
                repo,
                consumer='main_session_example_summary',
                workflow_id='example_summary',
                step_id='draft',
                task='Draft summary',
            )
            research_plan = self._route(
                repo,
                consumer='main_session_example_research',
                workflow_id='example_research',
                step_id='synthesize',
                task='Synthesize notes',
            )

            self.assertEqual(summary_plan['packs'][0]['pack_id'], 'example_summary_pack')
            self.assertEqual(summary_plan['packs'][0]['query'], 'workflows/example_summary.pack.yaml')
            self.assertEqual(research_plan['packs'][0]['pack_id'], 'example_research_pack')
            self.assertEqual(research_plan['packs'][0]['query'], 'workflows/example_research.pack.yaml')

            for plan in (summary_plan, research_plan):
                self.assertNotIn('repo_path', plan)
                self.assertTrue(plan['consumer'].startswith('main_session_example_'))
                self.assertIn('scope', plan)
                self.assertIn('packs', plan)
                self.assertIn('selected_packs', plan)
                self.assertIn('dropped_packs', plan)
                self.assertIn('decision_path', plan)
                self.assertIn('budget_summary', plan)
                self.assertIn('normalized_query', plan)
                self.assertIn('query_hash', plan)
                self.assertEqual(len(plan['query_hash']), 64)
                self.assertIn('index_health', plan)
                self.assertIn('vector_errors', plan)
                for selected in plan['selected_packs']:
                    self.assertIn('rank_bm25', selected)
                    self.assertIn('rank_vector', selected)

            replay = self._route(
                repo,
                consumer='main_session_example_summary',
                workflow_id='example_summary',
                step_id='draft',
                task='Draft summary',
            )
            self.assertEqual(summary_plan, replay)

    def test_router_accepts_scope_and_materialize_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True)
            self._seed_repo(repo)

            plan = self._route(
                repo,
                consumer='main_session_example_research',
                workflow_id='example_research',
                step_id='synthesize',
                task='Synthesize notes',
                materialize=True,
                scope='private',
            )

            self.assertEqual(plan['scope'], 'private')
            self.assertEqual(len(plan['selected_packs']), 1)
            selected = plan['selected_packs'][0]
            self.assertEqual(selected['pack_id'], 'example_research_pack')
            self.assertIn('materialized_excerpt', selected)
            self.assertEqual(plan['budget_summary']['selected_count'], 1)

    def test_budget_summary_defaults_tier_c_when_profile_field_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True)
            self._seed_repo(repo)

            plan = self._route(
                repo,
                consumer='main_session_example_summary',
                workflow_id='example_summary',
                step_id='draft',
                task='Draft summary',
            )

            self.assertEqual(plan['budget_summary']['tier_c_fixed_tokens'], 3000)
            events = [w.get('event') for w in plan['budget_summary']['warnings']]
            self.assertIn('TIER_C_DEFAULT_FALLBACK_USED', events)

    def test_budget_summary_forces_3000_when_pinned_profile_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True)
            (repo / 'config').mkdir(parents=True)
            (repo / 'workflows').mkdir(parents=True)

            (repo / 'workflows' / 'vc_memo.pack.yaml').write_text('memo pack', encoding='utf-8')

            registry = {
                'schema_version': 1,
                'packs': [
                    {
                        'pack_id': 'vc_memo_pack',
                        'path': 'workflows/vc_memo.pack.yaml',
                        'scope': 'private',
                        'query_template': '${path}',
                    }
                ],
            }
            profiles = {
                'schema_version': 1,
                'profiles': [
                    {
                        'consumer': 'main_session_vc_memo',
                        'workflow_id': 'vc_memo_drafting',
                        'step_id': 'draft',
                        'scope': 'private',
                        'schema_version': 1,
                        'task': 'Draft memo',
                        'injection_text': 'Memo',
                        'pack_ids': ['vc_memo_pack'],
                        'chatgpt_mode': 'scoped',
                        'tier_c_fixed_tokens': 3200,
                        'model_context_limit': 6000,
                    }
                ],
            }

            (repo / 'config' / 'runtime_pack_registry.json').write_text(
                json.dumps(registry, indent=2),
                encoding='utf-8',
            )
            (repo / 'config' / 'runtime_consumer_profiles.json').write_text(
                json.dumps(profiles, indent=2),
                encoding='utf-8',
            )

            plan = self._route(
                repo,
                consumer='main_session_vc_memo',
                workflow_id='vc_memo_drafting',
                step_id='draft',
                task='Draft memo',
            )

            self.assertEqual(plan['budget_summary']['tier_c_fixed_tokens'], 3000)
            warnings = plan['budget_summary']['warnings']
            self.assertTrue(
                any(
                    w.get('event') == 'TIER_C_PROFILE_MISMATCH'
                    and w.get('consumer') == 'main_session_vc_memo'
                    for w in warnings
                )
            )
            self.assertTrue(
                any(
                    w.get('event') == 'TIER_C_OVERSIZED'
                    and w.get('consumer') == 'main_session_vc_memo'
                    for w in warnings
                )
            )


    def test_query_normalization_and_hash_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True)
            self._seed_repo(repo)

            task = '  Ｄraft\tSUMMARY   '
            plan = self._route(
                repo,
                consumer='main_session_example_summary',
                workflow_id='example_summary',
                step_id='draft',
                task=task,
            )

            self.assertEqual(plan['normalized_query'], 'draft summary')
            self.assertEqual(
                plan['query_hash'],
                'a528f240ef53e68ca0de136406f816cd21591e44c389ace8ccd1637809cb1dc6',
            )

            replay = self._route(
                repo,
                consumer='main_session_example_summary',
                workflow_id='example_summary',
                step_id='draft',
                task=task,
            )
            self.assertEqual(plan['normalized_query'], replay['normalized_query'])
            self.assertEqual(plan['query_hash'], replay['query_hash'])

    def test_embedding_failure_emits_structured_vector_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True)
            self._seed_repo(repo)

            plan = self._route(
                repo,
                consumer='main_session_example_summary',
                workflow_id='example_summary',
                step_id='draft',
                task='Draft summary',
                env={'OM_VECTOR_EMBEDDING_FORCE_FAIL': '1'},
            )

            self.assertGreaterEqual(len(plan['vector_errors']), 1)
            event = plan['vector_errors'][0]
            self.assertEqual(event['event'], 'OM_VECTOR_QUERY_EMBEDDING_FAILED')
            self.assertEqual(event['query_hash'], plan['query_hash'])
            self.assertIn('error_message', event)
            self.assertRegex(event['timestamp'], r'^\d{4}-\d{2}-\d{2}T')
            self.assertTrue(event['timestamp'].endswith('Z'))

    def test_rank_tie_breaks_by_pack_id_and_includes_rank_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True)
            (repo / 'config').mkdir(parents=True)
            (repo / 'workflows').mkdir(parents=True)

            (repo / 'workflows' / 'a.pack.yaml').write_text('A', encoding='utf-8')
            (repo / 'workflows' / 'b.pack.yaml').write_text('B', encoding='utf-8')

            registry = {
                'schema_version': 1,
                'packs': [
                    {
                        'pack_id': 'z_pack',
                        'path': 'workflows/a.pack.yaml',
                        'scope': 'private',
                        'query_template': '${path}',
                    },
                    {
                        'pack_id': 'a_pack',
                        'path': 'workflows/b.pack.yaml',
                        'scope': 'private',
                        'query_template': '${path}',
                    },
                ],
            }
            profiles = {
                'schema_version': 1,
                'profiles': [
                    {
                        'consumer': 'main_session_rank_test',
                        'workflow_id': 'rank_test',
                        'step_id': 'draft',
                        'scope': 'private',
                        'schema_version': 1,
                        'task': 'Rank test',
                        'injection_text': 'Rank test',
                        'pack_ids': ['z_pack', 'a_pack'],
                        'chatgpt_mode': 'off',
                    }
                ],
            }

            (repo / 'config' / 'runtime_pack_registry.json').write_text(json.dumps(registry, indent=2), encoding='utf-8')
            (repo / 'config' / 'runtime_consumer_profiles.json').write_text(
                json.dumps(profiles, indent=2),
                encoding='utf-8',
            )

            plan = self._route(
                repo,
                consumer='main_session_rank_test',
                workflow_id='rank_test',
                step_id='draft',
                task='zzz',
                env={'OM_VECTOR_EMBEDDING_FORCE_FAIL': '1'},
            )

            ordered_ids = [item['pack_id'] for item in plan['selected_packs']]
            self.assertEqual(ordered_ids, ['a_pack', 'z_pack'])
            self.assertEqual(plan['selected_packs'][0]['rank_bm25'], 1)
            self.assertEqual(plan['selected_packs'][0]['rank_vector'], 1)
            self.assertEqual(plan['selected_packs'][1]['rank_bm25'], 2)
            self.assertEqual(plan['selected_packs'][1]['rank_vector'], 2)

    def test_index_schema_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True)
            (repo / 'config').mkdir(parents=True)
            (repo / 'workflows').mkdir(parents=True)

            (repo / 'workflows' / 'mismatch.pack.yaml').write_text('Mismatch', encoding='utf-8')

            registry = {
                'schema_version': 1,
                'packs': [
                    {
                        'pack_id': 'mismatch_pack',
                        'path': 'workflows/mismatch.pack.yaml',
                        'scope': 'private',
                        'query_template': '${path}',
                        'retrieval': {
                            'index': {
                                'schema_version': 1,
                                'bm25_schema_version': 1,
                                'vector_schema_version': 1,
                                'vector_dim': 123,
                            }
                        },
                    }
                ],
            }
            profiles = {
                'schema_version': 1,
                'profiles': [
                    {
                        'consumer': 'main_session_mismatch_test',
                        'workflow_id': 'mismatch_test',
                        'step_id': 'draft',
                        'scope': 'private',
                        'schema_version': 1,
                        'task': 'Mismatch test',
                        'injection_text': 'Mismatch test',
                        'pack_ids': ['mismatch_pack'],
                        'chatgpt_mode': 'off',
                    }
                ],
            }

            (repo / 'config' / 'runtime_pack_registry.json').write_text(json.dumps(registry, indent=2), encoding='utf-8')
            (repo / 'config' / 'runtime_consumer_profiles.json').write_text(
                json.dumps(profiles, indent=2),
                encoding='utf-8',
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    '--consumer',
                    'main_session_mismatch_test',
                    '--workflow-id',
                    'mismatch_test',
                    '--step-id',
                    'draft',
                    '--repo',
                    str(repo),
                    '--task',
                    'Mismatch test',
                    '--validate',
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn('OMIndexMismatchError', result.stderr)
            self.assertIn('vector_dim mismatch', result.stderr)


    def test_materialize_content_packs_uses_live_graph_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True)
            (repo / 'config').mkdir(parents=True)
            (repo / 'workflows').mkdir(parents=True)

            (repo / 'workflows' / 'content_long_form.pack.yaml').write_text(
                'domain_context: |\n  deterministic long-form scaffold',
                encoding='utf-8',
            )
            (repo / 'workflows' / 'content_voice_style.pack.yaml').write_text(
                'domain_context: |\n  STATIC VOICE FALLBACK',
                encoding='utf-8',
            )
            (repo / 'workflows' / 'content_writing_samples.pack.yaml').write_text(
                'domain_context: |\n  STATIC WRITING FALLBACK',
                encoding='utf-8',
            )

            registry = {
                'schema_version': 1,
                'packs': [
                    {
                        'pack_id': 'content_long_form',
                        'path': 'workflows/content_long_form.pack.yaml',
                        'scope': 'private',
                        'query_template': '${path}',
                    },
                    {
                        'pack_id': 'content_voice_style',
                        'path': 'workflows/content_voice_style.pack.yaml',
                        'scope': 'private',
                        'query_template': '${path}',
                        'retrieval': {
                            'group_ids_by_mode': {'default': ['s1_content_strategy']},
                        },
                        'materialization': {
                            'source': 'graphiti_content_voice_style',
                            'min_coverage_items': 2,
                            'max_items': 4,
                        },
                    },
                    {
                        'pack_id': 'content_writing_samples',
                        'path': 'workflows/content_writing_samples.pack.yaml',
                        'scope': 'private',
                        'query_template': '${path}',
                        'retrieval': {
                            'group_ids_by_mode': {'default': ['s1_writing_samples']},
                        },
                        'materialization': {
                            'source': 'graphiti_content_writing_samples',
                            'min_coverage_items': 2,
                            'max_items': 4,
                        },
                    },
                ],
            }
            profiles = {
                'schema_version': 1,
                'profiles': [
                    {
                        'consumer': 'main_session_content_long_form',
                        'workflow_id': 'content_long_form',
                        'step_id': 'draft',
                        'scope': 'private',
                        'schema_version': 1,
                        'task': 'Draft long-form content',
                        'injection_text': 'Long-form writing workflow',
                        'pack_ids': [
                            'content_long_form',
                            'content_voice_style',
                            'content_writing_samples',
                        ],
                        'chatgpt_mode': 'off',
                        'pack_modes': {
                            'content_long_form': 'long',
                            'content_voice_style': 'formal',
                            'content_writing_samples': 'formal',
                        },
                    }
                ],
            }

            (repo / 'config' / 'runtime_pack_registry.json').write_text(
                json.dumps(registry, indent=2),
                encoding='utf-8',
            )
            (repo / 'config' / 'runtime_consumer_profiles.json').write_text(
                json.dumps(profiles, indent=2),
                encoding='utf-8',
            )

            with GraphitiStubServer(
                facts_by_group_id={
                    's1_content_strategy': [
                        'Lead with a specific observation and immediate thesis.',
                        'Use concrete verbs and avoid ornamental metaphors.',
                    ],
                    's1_writing_samples': [
                        'Growth quality matters more than top-line speed in underwriting.',
                        'A strong memo pairs every claim with auditable evidence.',
                    ],
                }
            ) as base_url:
                plan = self._route(
                    repo,
                    consumer='main_session_content_long_form',
                    workflow_id='content_long_form',
                    step_id='draft',
                    task='Draft long-form content',
                    materialize=True,
                    scope='private',
                    env={
                        'GRAPHITI_BASE_URL': base_url,
                    },
                )

            selected = {pack['pack_id']: pack for pack in plan['selected_packs']}
            self.assertNotIn('content', selected['content_long_form'])
            self.assertIn('content', selected['content_voice_style'])
            self.assertIn('content', selected['content_writing_samples'])

            voice_content = selected['content_voice_style']['content']
            writing_content = selected['content_writing_samples']['content']
            self.assertIn('Live voice-style signals', voice_content)
            self.assertIn('Live writing-sample signals', writing_content)
            self.assertNotIn('STATIC VOICE FALLBACK', voice_content)
            self.assertNotIn('STATIC WRITING FALLBACK', writing_content)

    def test_materialize_content_pack_low_coverage_falls_back_to_static_domain_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True)
            (repo / 'config').mkdir(parents=True)
            (repo / 'workflows').mkdir(parents=True)

            (repo / 'workflows' / 'content_voice_style.pack.yaml').write_text(
                'domain_context: |\n  STATIC VOICE FALLBACK',
                encoding='utf-8',
            )

            registry = {
                'schema_version': 1,
                'packs': [
                    {
                        'pack_id': 'content_voice_style',
                        'path': 'workflows/content_voice_style.pack.yaml',
                        'scope': 'private',
                        'query_template': '${path}',
                        'retrieval': {
                            'group_ids_by_mode': {'default': ['s1_content_strategy']},
                        },
                        'materialization': {
                            'source': 'graphiti_content_voice_style',
                            'min_coverage_items': 2,
                            'max_items': 4,
                        },
                    }
                ],
            }
            profiles = {
                'schema_version': 1,
                'profiles': [
                    {
                        'consumer': 'main_session_voice_only',
                        'workflow_id': 'voice_only',
                        'step_id': 'draft',
                        'scope': 'private',
                        'schema_version': 1,
                        'task': 'Voice-only test',
                        'injection_text': 'Voice-only workflow',
                        'pack_ids': ['content_voice_style'],
                        'chatgpt_mode': 'off',
                    }
                ],
            }

            (repo / 'config' / 'runtime_pack_registry.json').write_text(
                json.dumps(registry, indent=2),
                encoding='utf-8',
            )
            (repo / 'config' / 'runtime_consumer_profiles.json').write_text(
                json.dumps(profiles, indent=2),
                encoding='utf-8',
            )

            with GraphitiStubServer(
                facts_by_group_id={
                    's1_content_strategy': ['Only one fact, below coverage threshold.'],
                }
            ) as base_url:
                plan = self._route(
                    repo,
                    consumer='main_session_voice_only',
                    workflow_id='voice_only',
                    step_id='draft',
                    task='Voice-only test',
                    materialize=True,
                    scope='private',
                    env={
                        'GRAPHITI_BASE_URL': base_url,
                    },
                )

            selected = plan['selected_packs'][0]
            self.assertEqual(selected['pack_id'], 'content_voice_style')
            self.assertEqual(selected.get('content'), 'STATIC VOICE FALLBACK')


class RuntimePackRouterFixturesTests(unittest.TestCase):
    def test_misconfigured_profile_type_rejects_non_string_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_repo = REPO_ROOT
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True)
            (repo / 'config').mkdir(parents=True)

            profiles = json.loads((source_repo / 'config/runtime_consumer_profiles.yaml').read_text(encoding='utf-8'))
            profiles['profiles'][0]['task'] = 123
            (repo / 'config' / 'runtime_consumer_profiles.yaml').write_text(
                json.dumps(profiles, indent=2),
                encoding='utf-8',
            )

            shutil.copyfile(
                source_repo / 'config/runtime_pack_registry.yaml',
                repo / 'config/runtime_pack_registry.yaml',
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    '--consumer',
                    'main_session_example_summary',
                    '--workflow-id',
                    'example_summary',
                    '--step-id',
                    'draft',
                    '--repo',
                    str(repo),
                    '--task',
                    'Draft summary',
                    '--validate',
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn('.profiles[0].task', result.stderr)
            self.assertIn('must be a string', result.stderr)

    def test_path_traversal_in_registry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_repo = REPO_ROOT
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True)
            (repo / 'config').mkdir(parents=True)
            (repo / 'workflows').mkdir(parents=True)

            registry = json.loads((source_repo / 'config/runtime_pack_registry.yaml').read_text(encoding='utf-8'))
            registry['packs'][0]['path'] = '../../../etc/passwd'
            (repo / 'config' / 'runtime_pack_registry.yaml').write_text(
                json.dumps(registry, indent=2),
                encoding='utf-8',
            )
            shutil.copyfile(
                source_repo / 'config/runtime_consumer_profiles.yaml',
                repo / 'config/runtime_consumer_profiles.yaml',
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    '--consumer',
                    'main_session_example_summary',
                    '--workflow-id',
                    'example_summary',
                    '--step-id',
                    'draft',
                    '--repo',
                    str(repo),
                    '--task',
                    'Draft summary',
                    '--validate',
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn('escapes repo root', result.stderr)
