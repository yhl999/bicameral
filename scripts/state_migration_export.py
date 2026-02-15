#!/usr/bin/env python3
"""Create deterministic migration packages for public delta-layer state."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from migration_sync_lib import (
    collect_file_entries,
    collect_manifest_files,
    copy_entry,
    dump_json,
    load_json,
    now_utc_iso,
    repo_relative,
    resolve_repo_root,
    run_git,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export migration package from configured manifest rules.')
    parser.add_argument('--repo', type=Path, default=Path('.'), help='Repository root or subdirectory')
    parser.add_argument(
        '--manifest',
        type=Path,
        default=Path('config/state_migration_manifest.json'),
        help='Manifest JSON describing required/optional files',
    )
    parser.add_argument('--out', type=Path, required=True, help='Package output directory')
    parser.add_argument('--dry-run', action='store_true', help='Write manifest preview only (skip payload copies)')
    parser.add_argument('--force', action='store_true', help='Replace output directory if it already exists')
    return parser.parse_args()


def _validate_manifest(manifest: dict) -> tuple[list[str], list[str], list[str], int, str]:
    required_files = manifest.get('required_files')
    optional_globs = manifest.get('optional_globs')
    exclude_globs = manifest.get('exclude_globs')
    version = manifest.get('version')
    package_name = manifest.get('package_name', 'graphiti-openclaw-state-migration')

    if not isinstance(required_files, list) or not all(isinstance(item, str) for item in required_files):
        raise ValueError('Manifest field `required_files` must be a list of strings')
    if not isinstance(optional_globs, list) or not all(isinstance(item, str) for item in optional_globs):
        raise ValueError('Manifest field `optional_globs` must be a list of strings')
    if not isinstance(exclude_globs, list) or not all(isinstance(item, str) for item in exclude_globs):
        raise ValueError('Manifest field `exclude_globs` must be a list of strings')
    if not isinstance(version, int):
        raise ValueError('Manifest field `version` must be an integer')
    if not isinstance(package_name, str) or not package_name.strip():
        raise ValueError('Manifest field `package_name` must be a non-empty string')

    return required_files, optional_globs, exclude_globs, version, package_name.strip()


def main() -> int:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo.resolve())

    manifest_path = args.manifest if args.manifest.is_absolute() else (repo_root / args.manifest).resolve()
    manifest = load_json(manifest_path)
    required_files, optional_globs, exclude_globs, manifest_version, package_name = _validate_manifest(manifest)

    files = collect_manifest_files(repo_root, required_files, optional_globs, exclude_globs)
    entries = collect_file_entries(files, repo_root)

    head = run_git(repo_root, 'rev-parse', 'HEAD')
    source_commit = head.stdout.strip() if head.returncode == 0 else ''

    package_root = args.out if args.out.is_absolute() else (Path.cwd() / args.out).resolve()
    payload_root = package_root / 'payload'

    if package_root.exists() and any(package_root.iterdir()):
        if not args.force:
            raise ValueError(f'Output directory already exists and is not empty: {package_root}')
        shutil.rmtree(package_root)

    package_root.mkdir(parents=True, exist_ok=True)

    package_manifest = {
        'package_version': 1,
        'manifest_version': manifest_version,
        'package_name': package_name,
        'created_at': now_utc_iso(),
        'source_repo': str(repo_root),
        'source_commit': source_commit,
        'dry_run_preview': bool(args.dry_run),
        'entry_count': len(entries),
        'entries': entries,
    }

    dump_json(package_root / 'package_manifest.json', package_manifest)

    if not args.dry_run:
        for entry in entries:
            rel = entry['path']
            copy_entry(repo_root, payload_root, rel)

    print(f'Package manifest written: {package_root / "package_manifest.json"}')
    if args.dry_run:
        print('Dry-run preview mode: payload files were not copied.')
    else:
        print(f'Payload files copied: {len(entries)}')

    by_top_level: dict[str, int] = {}
    for path in (entry['path'] for entry in entries):
        top = path.split('/', 1)[0]
        by_top_level[top] = by_top_level.get(top, 0) + 1
    summary = ', '.join(f'{key}={value}' for key, value in sorted(by_top_level.items()))
    print(f'Included files ({len(entries)}): {summary}')
    print(f'Repo root: {repo_root}')
    print(f'Manifest: {repo_relative(manifest_path, repo_root)}')

    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
