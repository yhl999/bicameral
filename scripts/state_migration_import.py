#!/usr/bin/env python3
"""Import migration package payload into a target repository tree."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from delta_contracts import validate_package_manifest
from migration_sync_lib import evaluate_payload_entries, load_json, resolve_safe_child


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Import migration package payload into target directory.')
    parser.add_argument('--in', dest='package', type=Path, required=True, help='Package directory path')
    parser.add_argument('--target', type=Path, default=Path('.'), help='Target repository root')
    parser.add_argument('--dry-run', action='store_true', help='Show planned writes without mutating target')
    parser.add_argument('--allow-overwrite', action='store_true', help='Allow overwriting existing files')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_root = args.package if args.package.is_absolute() else (Path.cwd() / args.package).resolve()
    target_root = args.target if args.target.is_absolute() else (Path.cwd() / args.target).resolve()

    manifest = validate_package_manifest(
        load_json(package_root / 'package_manifest.json'),
        context=str(package_root / 'package_manifest.json'),
    )

    entries = manifest.get('entries', [])
    if not isinstance(entries, list):
        raise ValueError('Manifest field `entries` must be a list')

    payload_root = package_root / 'payload'
    dry_run_preview = bool(manifest.get('dry_run_preview'))

    planned_entries, missing_payload, integrity_errors = evaluate_payload_entries(
        entries=entries,
        payload_root=payload_root,
        context='migration payload entry',
    )

    planned_writes: list[tuple[str, Path, Path]] = []
    conflicts: list[str] = []
    for rel, src in planned_entries:
        dst = resolve_safe_child(target_root, rel, context='migration import target entry')
        if dst.exists() and not args.allow_overwrite:
            conflicts.append(rel)
        planned_writes.append((rel, src, dst))

    if conflicts and not args.dry_run:
        print('Import blocked: existing files would be overwritten.', file=sys.stderr)
        for rel in conflicts:
            print(f'- {rel}', file=sys.stderr)
        print('Use --allow-overwrite to apply import anyway.', file=sys.stderr)
        return 1

    if missing_payload and not args.dry_run and not dry_run_preview:
        print('Import blocked: package payload is incomplete.', file=sys.stderr)
        for rel in missing_payload:
            print(f'- {rel}', file=sys.stderr)
        return 1

    if integrity_errors and not args.dry_run:
        print('Import blocked: payload integrity check failed.', file=sys.stderr)
        for issue in integrity_errors:
            print(f'- {issue}', file=sys.stderr)
        return 1

    if args.dry_run:
        print(f'DRY RUN import plan ({len(planned_writes)} files):')
        for _, src, dst in planned_writes:
            note = ' (payload missing in dry-run preview)' if not src.exists() else ''
            print(f'- {src} -> {dst}{note}')

        if integrity_errors:
            print('Dry-run integrity warnings:')
            for issue in integrity_errors:
                print(f'- {issue}')
        return 0

    if dry_run_preview:
        raise ValueError(
            'Cannot execute non-dry-run import from dry-run preview package. '
            'Re-export without --dry-run to include payload files.',
        )

    for _, src, dst in planned_writes:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())

    print(f'Imported {len(planned_writes)} files into {target_root}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
