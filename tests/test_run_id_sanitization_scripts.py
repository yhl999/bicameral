from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_om_closeout_report import (
    _validate_run_id as validate_closeout_run_id,
)
from scripts.build_om_closeout_report import (
    build_report,
)
from scripts.owned_paths_preflight import _validate_run_id as validate_preflight_run_id


@pytest.mark.parametrize('run_id', ['alpha-01', 'alpha_01', 'A1B2'])
def test_validate_run_id_accepts_allowed_characters(run_id: str) -> None:
    assert validate_preflight_run_id(run_id) == run_id
    assert validate_closeout_run_id(run_id) == run_id


@pytest.mark.parametrize(
    'run_id, expected_message',
    [
        ('alpha/beta', "run-id contains '/'"),
        ('alpha\\beta', "run-id contains '\\'"),
        ('alpha..beta', "run-id contains '..'"),
    ],
)
def test_validate_run_id_rejects_path_traversal(
    run_id: str,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        validate_preflight_run_id(run_id)
    assert expected_message in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        validate_closeout_run_id(run_id)
    assert expected_message in str(exc_info.value)


def test_validate_run_id_rejects_other_characters() -> None:
    with pytest.raises(ValueError, match=r'run-id must match only \[A-Za-z0-9_\-\]'):
        validate_preflight_run_id('alpha.b')

    with pytest.raises(ValueError, match=r'run-id must match only \[A-Za-z0-9_\-\]'):
        validate_closeout_run_id('alpha.b')


def test_build_report_rejects_invalid_run_id(tmp_path: Path) -> None:
    benchmark = tmp_path / 'benchmark.json'
    utility = tmp_path / 'utility.json'
    lane_hygiene = tmp_path / 'lane_hygiene.json'

    benchmark.write_text(json.dumps({'bicameral_aggregate': {}}), encoding='utf-8')
    utility.write_text(json.dumps({'aggregate': {}}), encoding='utf-8')
    lane_hygiene.write_text(json.dumps({'unresolved_lanes': []}), encoding='utf-8')

    with pytest.raises(ValueError, match="run-id contains '/'"):
        build_report(
            run_id='abc/def',
            benchmark_path=benchmark,
            utility_path=utility,
            lane_hygiene_path=lane_hygiene,
            pr_a_sha='abc',
            overlay_manifest_ref='overlay-manifest.json',
        )


def test_build_report_accepts_valid_run_id(tmp_path: Path) -> None:
    benchmark = tmp_path / 'benchmark.json'
    utility = tmp_path / 'utility.json'
    lane_hygiene = tmp_path / 'lane_hygiene.json'

    benchmark.write_text(json.dumps({'bicameral_aggregate': {}}), encoding='utf-8')
    utility.write_text(json.dumps({'aggregate': {}}), encoding='utf-8')
    lane_hygiene.write_text(json.dumps({'unresolved_lanes': []}), encoding='utf-8')

    run_id = 'run_01'
    report = build_report(
        run_id=run_id,
        benchmark_path=benchmark,
        utility_path=utility,
        lane_hygiene_path=lane_hygiene,
        pr_a_sha='abc',
        overlay_manifest_ref='overlay-manifest.json',
    )

    assert f'OM Closeout Report ({run_id})' in report
