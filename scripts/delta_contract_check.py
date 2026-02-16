#!/usr/bin/env python3
"""Validate delta-layer config and extension contracts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from delta_contracts import (
    inspect_extensions,
    validate_migration_sync_policy,
    validate_state_migration_manifest,
)
from migration_sync_lib import load_json, resolve_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Validate delta-layer contract files.')
    parser.add_argument('--repo', type=Path, default=Path('.'), help='Repository root or subdirectory')
    parser.add_argument(
        '--policy',
        type=Path,
        default=Path('config/migration_sync_policy.json'),
        help='Migration/sync policy JSON path',
    )
    parser.add_argument(
        '--state-manifest',
        type=Path,
        default=Path('config/state_migration_manifest.json'),
        help='State migration manifest JSON path',
    )
    parser.add_argument('--extensions-dir', type=Path, default=Path('extensions'))
    parser.add_argument('--strict', action='store_true', help='Exit non-zero when issues are found')
    return parser.parse_args()


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def main() -> int:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo.resolve())

    policy_path = _resolve(repo_root, args.policy)
    manifest_path = _resolve(repo_root, args.state_manifest)
    extensions_dir = _resolve(repo_root, args.extensions_dir)

    issues: list[str] = []

    try:
        validate_migration_sync_policy(
            load_json(policy_path),
            context=str(policy_path),
            strict=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        issues.append(str(exc))

    try:
        validate_state_migration_manifest(load_json(manifest_path), context=str(manifest_path))
    except (FileNotFoundError, ValueError) as exc:
        issues.append(str(exc))

    extension_report = inspect_extensions(repo_root=repo_root, extensions_dir=extensions_dir)
    issues.extend(extension_report.issues)

    if issues:
        print('Delta contract check: issues found', file=sys.stderr)
        for issue in issues:
            print(f'- {issue}', file=sys.stderr)
        return 1 if args.strict else 0

    print(
        'Delta contract check OK '
        f'(policy={policy_path.name}, state_manifest={manifest_path.name}, '
        f'extensions={len(extension_report.names)}, extension_commands={len(extension_report.command_registry)})',
    )
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
