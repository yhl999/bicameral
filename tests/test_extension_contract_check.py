from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'extension_contract_check.py'


class ExtensionContractCheckTests(unittest.TestCase):
    def _write_manifest(self, root: Path, folder: str, payload: dict) -> None:
        manifest = root / 'extensions' / folder / 'manifest.json'
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(f'{json.dumps(payload, indent=2)}\n', encoding='utf-8')

    def _run(self, repo: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), '--extensions-dir', 'extensions', '--strict'],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_passes_with_valid_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            tool = repo / 'scripts' / 'tool.py'
            tool.parent.mkdir(parents=True, exist_ok=True)
            tool.write_text('print("ok")\n', encoding='utf-8')

            self._write_manifest(
                repo,
                'valid',
                {
                    'name': 'valid-extension',
                    'version': '0.1.0',
                    'capabilities': ['sync'],
                    'entrypoints': {'doctor': 'scripts/tool.py'},
                },
            )

            result = self._run(repo)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn('OK', result.stdout)

    def test_fails_when_entrypoint_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_manifest(
                repo,
                'invalid',
                {
                    'name': 'broken-extension',
                    'version': '0.1.0',
                    'capabilities': ['sync'],
                    'entrypoints': {'doctor': 'scripts/missing.py'},
                },
            )

            result = self._run(repo)
            self.assertEqual(result.returncode, 1)
            self.assertIn('entrypoint path missing', result.stderr)

    def test_fails_on_unsafe_entrypoint_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            outside = repo.parent / 'outside.py'
            outside.write_text('print("bad")\n', encoding='utf-8')
            self._write_manifest(
                repo,
                'invalid-path',
                {
                    'name': 'unsafe-extension',
                    'version': '0.1.0',
                    'capabilities': ['sync', 'sync'],
                    'entrypoints': {'doctor': '../outside.py'},
                },
            )

            result = self._run(repo)
            self.assertEqual(result.returncode, 1)
            self.assertIn('Unsafe package entry path', result.stderr)
            self.assertIn('capabilities` contains duplicates', result.stderr)


if __name__ == '__main__':
    unittest.main()
