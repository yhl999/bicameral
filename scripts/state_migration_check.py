#!/usr/bin/env python3
"""Validate migration package integrity and compatibility."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from migration_sync_lib import ensure_safe_relative, load_json, sha256_file

REQUIRED_KEYS = {
    'package_version',
    'manifest_version',
    'package_name',
    'created_at',
    'source_repo',
    'source_commit',
    'dry_run_preview',
    'entry_count',
    'entries',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Validate migration package structure and checksums.')
    parser.add_argument('--package', type=Path, required=True, help='Package directory path')
    parser.add_argument('--dry-run', action='store_true', help='Validate manifest/schema only (skip payload checks)')
    return parser.parse_args()


def _validate_manifest_shape(manifest: dict) -> list[str]:
    errors: list[str] = []

    missing = sorted(REQUIRED_KEYS - set(manifest))
    if missing:
        errors.append(f'Missing required manifest keys: {", ".join(missing)}')

    entries = manifest.get('entries')
    if not isinstance(entries, list):
        errors.append('Manifest field `entries` must be a list')
        return errors

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f'Entry #{index} must be an object')
            continue

        for key in ('path', 'sha256', 'size_bytes'):
            if key not in entry:
                errors.append(f'Entry #{index} missing key `{key}`')

        path = entry.get('path')
        digest = entry.get('sha256')
        size_bytes = entry.get('size_bytes')

        if isinstance(path, str):
            try:
                ensure_safe_relative(path)
            except ValueError as exc:
                errors.append(str(exc))
        else:
            errors.append(f'Entry #{index} path must be a string')

        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in '0123456789abcdef' for ch in digest.lower()):
            errors.append(f'Entry #{index} sha256 must be a 64-char hex string')

        if not isinstance(size_bytes, int) or size_bytes < 0:
            errors.append(f'Entry #{index} size_bytes must be a non-negative integer')

    expected_count = manifest.get('entry_count')
    if isinstance(expected_count, int) and expected_count != len(entries):
        errors.append(
            f'entry_count mismatch: manifest={expected_count}, computed={len(entries)}',
        )

    return errors


def main() -> int:
    args = parse_args()
    package_root = args.package if args.package.is_absolute() else (Path.cwd() / args.package).resolve()
    manifest_path = package_root / 'package_manifest.json'

    manifest = load_json(manifest_path)
    errors = _validate_manifest_shape(manifest)

    entries = manifest.get('entries', [])
    dry_run_preview = bool(manifest.get('dry_run_preview'))
    payload_root = package_root / 'payload'

    should_verify_payload = not args.dry_run and not dry_run_preview
    if should_verify_payload:
        for entry in entries:
            rel = entry['path']
            expected_hash = entry['sha256']
            expected_size = entry['size_bytes']
            payload_file = payload_root / rel
            if not payload_file.exists() or not payload_file.is_file():
                errors.append(f'Missing payload file: {rel}')
                continue
            actual_hash = sha256_file(payload_file)
            if actual_hash != expected_hash:
                errors.append(f'Checksum mismatch for {rel}')
            actual_size = payload_file.stat().st_size
            if actual_size != expected_size:
                errors.append(f'Size mismatch for {rel}: expected {expected_size}, got {actual_size}')

    if errors:
        print('Migration package check FAILED:', file=sys.stderr)
        for issue in errors:
            print(f'- {issue}', file=sys.stderr)
        return 1

    mode = 'manifest-only' if args.dry_run or dry_run_preview else 'full payload'
    print(f'Migration package check OK ({mode}).')
    print(f'Package: {package_root}')
    print(f'Entries: {len(entries)}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
