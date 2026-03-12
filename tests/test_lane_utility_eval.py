from pathlib import Path

from scripts import lane_utility_eval


def test_run_phase3_probe_uses_current_python_interpreter(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        out_path = Path(cmd[cmd.index('--out') + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text('{}', encoding='utf-8')
        return None

    monkeypatch.setattr(lane_utility_eval.subprocess, 'run', fake_run)

    probe = lane_utility_eval.run_phase3_probe(
        private_root=tmp_path / 'private',
        runtime_repo=tmp_path / 'runtime',
        probe_cfg={'run_canary_smoke': False},
    )

    assert len(calls) == 2
    assert all(cmd[0] == lane_utility_eval.sys.executable for cmd in calls)
    assert 'samples_lines' in probe
    assert 'tweet_samples_lines' in probe
