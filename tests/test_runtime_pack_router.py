from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'runtime_pack_router.py'
REPO_ROOT = Path(__file__).resolve().parents[1]


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
