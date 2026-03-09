from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_script(script_name: str, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    merged_env.pop('NEO4J_PASSWORD', None)
    if env:
        merged_env.update(env)
    return subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / script_name), *args],
        cwd=ROOT,
        env=merged_env,
        capture_output=True,
        text=True,
    )


def test_dedupe_nodes_help_no_longer_crashes_on_missing_private_graph_driver() -> None:
    result = _run_script('dedupe_nodes.py', '--help')

    assert result.returncode == 0
    assert 'group-id' in result.stdout
    assert 'backend' in result.stdout
    assert 'ModuleNotFoundError' not in result.stderr


def test_repair_timeline_help_no_longer_crashes_on_missing_private_graph_driver() -> None:
    result = _run_script('repair_timeline.py', '--help')

    assert result.returncode == 0
    assert 'group-id' in result.stdout
    assert 'backend' in result.stdout
    assert 'ModuleNotFoundError' not in result.stderr


def test_dedupe_nodes_reports_operator_friendly_neo4j_setup_error() -> None:
    result = _run_script('dedupe_nodes.py', '--group-id', 'g1', '--dry-run', '--backend', 'neo4j')

    assert result.returncode == 2
    combined = result.stdout + result.stderr
    assert 'NEO4J_PASSWORD' in combined
    assert 'ModuleNotFoundError' not in combined
    assert 'Traceback' not in combined
