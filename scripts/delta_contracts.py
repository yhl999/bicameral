#!/usr/bin/env python3
"""Contract validators for delta-layer config and manifest files."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from migration_sync_lib import ensure_safe_relative, load_json, resolve_safe_child

METRIC_KEYS = ('privacy_risk', 'simplicity', 'merge_conflict_risk', 'auditability')
_COMMAND_NAME_RE = re.compile(r'^[a-z0-9][a-z0-9-]*$')

_FILTERED_HISTORY_FIELDS = {
    'privacy_risk': {'base', 'block_penalty', 'ambiguous_penalty'},
    'simplicity': {'base', 'commit_divisor', 'commit_cap', 'ambiguous_penalty'},
    'merge_conflict_risk': {'base', 'commit_divisor', 'commit_cap', 'ambiguous_penalty'},
    'auditability': {'base', 'block_penalty', 'ambiguous_penalty'},
}

_CLEAN_FOUNDATION_FIELDS = {
    'privacy_risk': {'base'},
    'simplicity': {'base', 'commit_bonus_divisor', 'commit_bonus_cap'},
    'merge_conflict_risk': {'base'},
    'auditability': {'base'},
}


@dataclass(frozen=True)
class ExtensionInspection:
    """Result of inspecting extension contracts under an extension directory."""

    names: list[str]
    command_registry: dict[str, Path]
    issues: list[str]



def _expect_dict(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f'{context} must be an object')
    return value


def _expect_str(value: object, *, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f'{context} must be a string')
    return value


def _expect_non_empty_str(value: object, *, context: str) -> str:
    text = _expect_str(value, context=context).strip()
    if not text:
        raise ValueError(f'{context} must be a non-empty string')
    return text


def _expect_bool(value: object, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f'{context} must be a boolean')
    return value


def _expect_number(value: object, *, context: str, min_value: float | None = None) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f'{context} must be a number')
    number = float(value)
    if min_value is not None and number < min_value:
        raise ValueError(f'{context} must be >= {min_value}')
    return number


def _expect_int(value: object, *, context: str, min_value: int | None = None) -> int:
    if not isinstance(value, int):
        raise ValueError(f'{context} must be an integer')
    if min_value is not None and value < min_value:
        raise ValueError(f'{context} must be >= {min_value}')
    return value


def _expect_string_list(
    value: object,
    *,
    context: str,
    allow_empty: bool = True,
    unique: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f'{context} must be a list of strings')

    parsed: list[str] = []
    for index, item in enumerate(value):
        parsed.append(_expect_non_empty_str(item, context=f'{context}[{index}]'))

    if not allow_empty and not parsed:
        raise ValueError(f'{context} must not be empty')

    if unique and len(set(parsed)) != len(parsed):
        raise ValueError(f'{context} must not contain duplicates')

    return parsed


def _validate_glob_patterns(patterns: Iterable[str], *, context: str) -> list[str]:
    validated: list[str] = []
    for index, pattern in enumerate(patterns):
        stripped = _expect_non_empty_str(pattern, context=f'{context}[{index}]')
        if stripped.startswith('/'):
            raise ValueError(f'{context}[{index}] must be relative (no absolute paths)')
        if '..' in stripped.split('/'):
            raise ValueError(f'{context}[{index}] must not contain path traversal (`..`)')
        validated.append(stripped)
    return validated


def _validate_command_name(command_name: str, *, context: str) -> str:
    normalized = _expect_non_empty_str(command_name, context=context)
    if not _COMMAND_NAME_RE.fullmatch(normalized):
        raise ValueError(
            f'{context} must match pattern `{_COMMAND_NAME_RE.pattern}`',
        )
    return normalized


def validate_migration_sync_policy(
    payload: object,
    *,
    context: str = 'migration_sync_policy',
    strict: bool = False,
) -> dict[str, Any]:
    """Validate migration/sync policy schema.

    In strict mode, all first-class sections are required.
    In non-strict mode, sections are validated when present.
    """

    policy = _expect_dict(payload, context=context)

    _expect_int(policy.get('version'), context=f'{context}.version', min_value=1)

    for block_name in ('origin', 'upstream'):
        block = policy.get(block_name)
        if block is None:
            if strict:
                raise ValueError(f'{context}.{block_name} is required in strict mode')
            continue

        block_dict = _expect_dict(block, context=f'{context}.{block_name}')
        _expect_non_empty_str(block_dict.get('remote'), context=f'{context}.{block_name}.remote')
        _expect_non_empty_str(block_dict.get('branch'), context=f'{context}.{block_name}.branch')
        if block_name == 'upstream' and 'url' in block_dict:
            _expect_str(block_dict.get('url'), context=f'{context}.upstream.url')
        elif block_name == 'upstream' and strict:
            raise ValueError(f'{context}.upstream.url is required in strict mode')

    sync_policy = policy.get('sync_button_policy')
    if sync_policy is None:
        if strict:
            raise ValueError(f'{context}.sync_button_policy is required in strict mode')
    else:
        sync_policy_dict = _expect_dict(sync_policy, context=f'{context}.sync_button_policy')
        _expect_bool(
            sync_policy_dict.get('require_clean_worktree'),
            context=f'{context}.sync_button_policy.require_clean_worktree',
        )
        _expect_int(
            sync_policy_dict.get('max_origin_only_commits'),
            context=f'{context}.sync_button_policy.max_origin_only_commits',
            min_value=0,
        )
        _expect_bool(
            sync_policy_dict.get('require_upstream_only_commits'),
            context=f'{context}.sync_button_policy.require_upstream_only_commits',
        )

    scorecard = policy.get('scorecard')
    if scorecard is None:
        if strict:
            raise ValueError(f'{context}.scorecard is required in strict mode')
    else:
        scorecard_dict = _expect_dict(scorecard, context=f'{context}.scorecard')
        _expect_number(
            scorecard_dict.get('clean_foundation_threshold'),
            context=f'{context}.scorecard.clean_foundation_threshold',
            min_value=0,
        )
        weights = _expect_dict(scorecard_dict.get('weights'), context=f'{context}.scorecard.weights')
        total_weight = 0.0
        for metric in METRIC_KEYS:
            total_weight += _expect_number(
                weights.get(metric),
                context=f'{context}.scorecard.weights.{metric}',
                min_value=0,
            )
        if total_weight <= 0:
            raise ValueError(f'{context}.scorecard.weights must sum to > 0')

    schedule = policy.get('schedule')
    if schedule is None:
        if strict:
            raise ValueError(f'{context}.schedule is required in strict mode')
    else:
        schedule_dict = _expect_dict(schedule, context=f'{context}.schedule')
        _expect_non_empty_str(schedule_dict.get('timezone'), context=f'{context}.schedule.timezone')
        _expect_non_empty_str(schedule_dict.get('weekly_day'), context=f'{context}.schedule.weekly_day')
        _expect_non_empty_str(schedule_dict.get('cron_utc'), context=f'{context}.schedule.cron_utc')

    history_metrics = policy.get('history_metrics')
    if history_metrics is not None:
        history_metrics_dict = _expect_dict(history_metrics, context=f'{context}.history_metrics')

        for candidate_name, allowed_fields in (
            ('filtered_history', _FILTERED_HISTORY_FIELDS),
            ('clean_foundation', _CLEAN_FOUNDATION_FIELDS),
        ):
            candidate_cfg = history_metrics_dict.get(candidate_name)
            if candidate_cfg is None:
                continue
            candidate_dict = _expect_dict(
                candidate_cfg,
                context=f'{context}.history_metrics.{candidate_name}',
            )

            extra_metrics = set(candidate_dict) - set(allowed_fields)
            if extra_metrics:
                extras = ', '.join(sorted(extra_metrics))
                raise ValueError(
                    f'{context}.history_metrics.{candidate_name} has unsupported metrics: {extras}',
                )

            for metric_name, metric_cfg in candidate_dict.items():
                metric_dict = _expect_dict(
                    metric_cfg,
                    context=f'{context}.history_metrics.{candidate_name}.{metric_name}',
                )
                allowed = allowed_fields[metric_name]
                extra_fields = set(metric_dict) - allowed
                if extra_fields:
                    extras = ', '.join(sorted(extra_fields))
                    raise ValueError(
                        f'{context}.history_metrics.{candidate_name}.{metric_name} '
                        f'has unsupported fields: {extras}',
                    )
                for key, value in metric_dict.items():
                    _expect_number(
                        value,
                        context=f'{context}.history_metrics.{candidate_name}.{metric_name}.{key}',
                        min_value=0,
                    )

    return policy


def validate_state_migration_manifest(payload: object, *, context: str = 'state_migration_manifest') -> dict[str, Any]:
    """Validate state migration manifest schema."""

    manifest = _expect_dict(payload, context=context)
    _expect_int(manifest.get('version'), context=f'{context}.version', min_value=1)
    _expect_non_empty_str(manifest.get('package_name'), context=f'{context}.package_name')

    required_files = _expect_string_list(
        manifest.get('required_files'),
        context=f'{context}.required_files',
        allow_empty=False,
        unique=True,
    )
    for index, rel in enumerate(required_files):
        try:
            ensure_safe_relative(rel)
        except ValueError as exc:
            raise ValueError(f'{context}.required_files[{index}] invalid: {exc}') from exc

    optional_globs = _expect_string_list(
        manifest.get('optional_globs'),
        context=f'{context}.optional_globs',
        allow_empty=True,
        unique=True,
    )
    _validate_glob_patterns(optional_globs, context=f'{context}.optional_globs')

    exclude_globs = _expect_string_list(
        manifest.get('exclude_globs'),
        context=f'{context}.exclude_globs',
        allow_empty=True,
        unique=True,
    )
    _validate_glob_patterns(exclude_globs, context=f'{context}.exclude_globs')

    return manifest


def validate_package_manifest(payload: object, *, context: str = 'state_package_manifest') -> dict[str, Any]:
    """Validate exported package manifest schema."""

    manifest = _expect_dict(payload, context=context)

    required_keys = {
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
    missing = sorted(required_keys - set(manifest))
    if missing:
        raise ValueError(f'{context} missing required keys: {", ".join(missing)}')

    _expect_int(manifest.get('package_version'), context=f'{context}.package_version', min_value=1)
    _expect_int(manifest.get('manifest_version'), context=f'{context}.manifest_version', min_value=1)
    _expect_non_empty_str(manifest.get('package_name'), context=f'{context}.package_name')
    _expect_non_empty_str(manifest.get('created_at'), context=f'{context}.created_at')
    _expect_non_empty_str(manifest.get('source_repo'), context=f'{context}.source_repo')

    source_commit = manifest.get('source_commit')
    if not isinstance(source_commit, str):
        raise ValueError(f'{context}.source_commit must be a string')

    _expect_bool(manifest.get('dry_run_preview'), context=f'{context}.dry_run_preview')

    entries = manifest.get('entries')
    if not isinstance(entries, list):
        raise ValueError(f'{context}.entries must be a list')

    expected_entry_count = _expect_int(
        manifest.get('entry_count'),
        context=f'{context}.entry_count',
        min_value=0,
    )

    if expected_entry_count != len(entries):
        raise ValueError(
            f'{context}.entry_count mismatch: expected {expected_entry_count}, found {len(entries)}',
        )

    for index, entry in enumerate(entries):
        entry_context = f'{context}.entries[{index}]'
        entry_dict = _expect_dict(entry, context=entry_context)

        rel_path = _expect_non_empty_str(entry_dict.get('path'), context=f'{entry_context}.path')
        try:
            ensure_safe_relative(rel_path)
        except ValueError as exc:
            raise ValueError(f'{entry_context}.path invalid: {exc}') from exc

        digest = _expect_non_empty_str(entry_dict.get('sha256'), context=f'{entry_context}.sha256')
        if len(digest) != 64 or any(ch not in '0123456789abcdef' for ch in digest.lower()):
            raise ValueError(f'{entry_context}.sha256 must be a 64-char hex string')

        _expect_int(entry_dict.get('size_bytes'), context=f'{entry_context}.size_bytes', min_value=0)

    return manifest


def validate_extension_manifest(payload: object, *, context: str = 'extension_manifest') -> dict[str, Any]:
    """Validate extension manifest schema."""

    manifest = _expect_dict(payload, context=context)

    _expect_non_empty_str(manifest.get('name'), context=f'{context}.name')
    _expect_non_empty_str(manifest.get('version'), context=f'{context}.version')

    if 'description' in manifest and manifest.get('description') is not None:
        _expect_non_empty_str(manifest.get('description'), context=f'{context}.description')

    _expect_string_list(
        manifest.get('capabilities'),
        context=f'{context}.capabilities',
        allow_empty=False,
        unique=True,
    )

    entrypoints = _expect_dict(manifest.get('entrypoints'), context=f'{context}.entrypoints')
    if not entrypoints:
        raise ValueError(f'{context}.entrypoints must not be empty')

    for key, value in entrypoints.items():
        _expect_non_empty_str(key, context=f'{context}.entrypoints key')
        rel_path = _expect_non_empty_str(value, context=f'{context}.entrypoints.{key}')
        try:
            ensure_safe_relative(rel_path)
        except ValueError as exc:
            raise ValueError(f'{context}.entrypoints.{key} invalid: {exc}') from exc

    commands = manifest.get('commands')
    if commands is not None:
        commands_dict = _expect_dict(commands, context=f'{context}.commands')
        for command_name, rel_path in commands_dict.items():
            normalized_command = _validate_command_name(
                command_name,
                context=f'{context}.commands key',
            )
            rel = _expect_non_empty_str(rel_path, context=f'{context}.commands.{normalized_command}')
            try:
                ensure_safe_relative(rel)
            except ValueError as exc:
                raise ValueError(f'{context}.commands.{normalized_command} invalid: {exc}') from exc

    return manifest


def inspect_extensions(repo_root: Path, extensions_dir: Path) -> ExtensionInspection:
    """Inspect extension manifests, return discovered names, commands, and issues."""

    issues: list[str] = []
    names: list[str] = []
    command_registry: dict[str, Path] = {}
    seen_names: set[str] = set()

    if not extensions_dir.exists() or not extensions_dir.is_dir():
        issues.append(f'Extensions directory missing: {extensions_dir}')
        return ExtensionInspection(names=names, command_registry=command_registry, issues=issues)

    for extension_dir in sorted(path for path in extensions_dir.iterdir() if path.is_dir()):
        manifest_path = extension_dir / 'manifest.json'
        if not manifest_path.exists():
            issues.append(f'{extension_dir.name}: missing manifest.json')
            continue

        try:
            manifest = validate_extension_manifest(load_json(manifest_path), context=str(manifest_path))
        except (FileNotFoundError, ValueError) as exc:
            issues.append(str(exc))
            continue

        extension_name = str(manifest['name']).strip()
        names.append(extension_name)
        if extension_name in seen_names:
            issues.append(f'{extension_dir.name}: duplicate extension name `{extension_name}`')
        seen_names.add(extension_name)

        entrypoints = manifest.get('entrypoints', {})
        if isinstance(entrypoints, dict):
            for entry_name, rel_path in entrypoints.items():
                if not isinstance(entry_name, str) or not isinstance(rel_path, str):
                    continue
                try:
                    resolved = resolve_safe_child(
                        repo_root,
                        rel_path,
                        context=f'extension `{extension_name}` entrypoint `{entry_name}`',
                    )
                except ValueError as exc:
                    issues.append(str(exc))
                    continue

                if not resolved.exists() or not resolved.is_file():
                    issues.append(
                        f'{manifest_path}: entrypoint path missing `{rel_path}`',
                    )

        commands = manifest.get('commands')
        if isinstance(commands, dict):
            for command_name, rel_path in commands.items():
                if not isinstance(command_name, str) or not isinstance(rel_path, str):
                    continue

                normalized_command = command_name.strip()
                if normalized_command in command_registry:
                    issues.append(
                        f'{manifest_path}: duplicate command `{normalized_command}` across extensions',
                    )
                    continue

                try:
                    resolved_command = resolve_safe_child(
                        repo_root,
                        rel_path,
                        context=f'extension `{extension_name}` command `{normalized_command}`',
                    )
                except ValueError as exc:
                    issues.append(str(exc))
                    continue

                if not resolved_command.exists() or not resolved_command.is_file():
                    issues.append(
                        f'{manifest_path}: command path missing `{rel_path}`',
                    )
                    continue

                command_registry[normalized_command] = resolved_command

    return ExtensionInspection(names=names, command_registry=command_registry, issues=issues)
