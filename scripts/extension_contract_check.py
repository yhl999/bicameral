#!/usr/bin/env python3
"""Validate extension manifests for public delta-layer tooling."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from migration_sync_lib import ensure_safe_relative, load_json, resolve_safe_child


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Check extension manifest contracts.')
    parser.add_argument('--extensions-dir', type=Path, default=Path('extensions'))
    parser.add_argument('--strict', action='store_true', help='Exit non-zero when issues are found')
    return parser.parse_args()


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    extensions_dir = args.extensions_dir if args.extensions_dir.is_absolute() else (repo_root / args.extensions_dir).resolve()

    if not extensions_dir.exists():
        print(f'Extensions directory missing: {extensions_dir}')
        return 1 if args.strict else 0

    issues: list[str] = []
    discovered: list[str] = []
    names: set[str] = set()

    for extension_dir in sorted(path for path in extensions_dir.iterdir() if path.is_dir()):
        manifest_path = extension_dir / 'manifest.json'
        if not manifest_path.exists():
            issues.append(f'{extension_dir.name}: missing manifest.json')
            continue

        manifest = load_json(manifest_path)
        name = manifest.get('name')
        version = manifest.get('version')
        capabilities = manifest.get('capabilities')
        entrypoints = manifest.get('entrypoints')

        if not _is_non_empty_string(name):
            issues.append(f'{extension_dir.name}: `name` must be a non-empty string')
            continue

        normalized_name = name.strip()
        if normalized_name in names:
            issues.append(f'{extension_dir.name}: duplicate extension name `{normalized_name}`')
        names.add(normalized_name)

        if not _is_non_empty_string(version):
            issues.append(f'{normalized_name}: `version` must be a non-empty string')

        if not isinstance(capabilities, list) or not all(_is_non_empty_string(item) for item in capabilities):
            issues.append(f'{normalized_name}: `capabilities` must be a list of non-empty strings')
        elif len({item.strip() for item in capabilities}) != len(capabilities):
            issues.append(f'{normalized_name}: `capabilities` contains duplicates')

        if not isinstance(entrypoints, dict) or not entrypoints:
            issues.append(f'{normalized_name}: `entrypoints` must be a non-empty object')
        else:
            for key, rel in entrypoints.items():
                if not _is_non_empty_string(key) or not _is_non_empty_string(rel):
                    issues.append(f'{normalized_name}: invalid entrypoint pair `{key}` -> `{rel}`')
                    continue

                rel_path = str(rel).strip()
                try:
                    ensure_safe_relative(rel_path)
                    candidate = resolve_safe_child(
                        repo_root,
                        rel_path,
                        context=f'extension `{normalized_name}` entrypoint',
                    )
                except ValueError as exc:
                    issues.append(f'{normalized_name}: {exc}')
                    continue

                if not candidate.exists() or not candidate.is_file():
                    issues.append(f'{normalized_name}: entrypoint path missing `{rel_path}`')

        discovered.append(normalized_name)

    if issues:
        print('Extension contract issues found:', file=sys.stderr)
        for issue in issues:
            print(f'- {issue}', file=sys.stderr)
        return 1 if args.strict else 0

    print(f'Extension contract check OK ({len(discovered)} extension(s)): {", ".join(discovered)}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
