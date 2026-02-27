#!/usr/bin/env python3
"""check_config_safety.py — lightweight ontology config injection scanner.

Scans string values inside YAML config files for obvious prompt-injection
markers (jailbreak phrases, delimiter-breakout tokens, system-role hijack
attempts).  Designed to run in CI with no external dependencies — only stdlib
+ PyYAML (already a project dep).

Exit codes:
    0   — no suspicious content found (or only warnings in non-strict mode)
    1   — suspicious content detected (always, even in non-strict mode)

Usage (CI):
    python3 scripts/ci/check_config_safety.py [--strict] [config_glob ...]

    --strict    Fail on warnings too (reserved for future policy tightening).
    config_glob One or more glob patterns; defaults to
                mcp_server/config/*.yaml and mcp_server/config/*.yml
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Injection marker patterns
# ---------------------------------------------------------------------------
# These are deliberately high-precision (few false positives).  The goal is
# to catch obvious copy-paste jailbreaks and delimiter-escape attempts — not
# to be a comprehensive WAF.  Keep the list short and maintainable.

# Phrase-level markers (case-insensitive substring match in any string value).
_PHRASE_MARKERS: list[re.Pattern[str]] = [
    re.compile(r'ignore\s+(all\s+)?(previous|prior|above)\s+instructions', re.IGNORECASE),
    re.compile(r'disregard\s+(the\s+)?(above|previous|prior)', re.IGNORECASE),
    re.compile(r'you\s+are\s+now\s+(a|an)\s+\w', re.IGNORECASE),
    re.compile(r'forget\s+(everything|all\s+(previous|prior))', re.IGNORECASE),
    re.compile(r'new\s+instructions\s*:', re.IGNORECASE),
    re.compile(r'system\s*:\s*you\s+are', re.IGNORECASE),
    re.compile(r'\[SYSTEM\]', re.IGNORECASE),
    re.compile(r'jailbreak', re.IGNORECASE),
    re.compile(r'DAN\s+mode', re.IGNORECASE),
    re.compile(r'act\s+as\s+(if\s+you\s+(were|are)|a\s+)\w', re.IGNORECASE),
]

# Token-level markers — delimiter-breakout sequences that could corrupt prompt
# structure (XML/markdown delimiters used in our prompt templates).
_TOKEN_MARKERS: list[re.Pattern[str]] = [
    re.compile(r'<\s*/?\s*(SYSTEM|TASK|LANE_INTENT|LANE_GUIDANCE|FACT_TYPES|ENTITIES)\s*>', re.IGNORECASE),
    re.compile(r'<\|im_start\|>', re.IGNORECASE),   # ChatML delimiter
    re.compile(r'<\|im_end\|>', re.IGNORECASE),
    re.compile(r'<<SYS>>', re.IGNORECASE),           # Llama system prompt
    re.compile(r'\[INST\]', re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# YAML value walker
# ---------------------------------------------------------------------------

def _iter_strings(value: Any, path: str = '') -> list[tuple[str, str]]:
    """Recursively yield (dot-path, string-value) pairs from a YAML structure."""
    results = []
    if isinstance(value, str):
        results.append((path, value))
    elif isinstance(value, dict):
        for k, v in value.items():
            child_path = f'{path}.{k}' if path else str(k)
            results.extend(_iter_strings(v, child_path))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            results.extend(_iter_strings(item, f'{path}[{i}]'))
    return results


# ---------------------------------------------------------------------------
# Scan logic
# ---------------------------------------------------------------------------

def scan_file(path: Path) -> list[tuple[str, str, str]] | None:
    """Scan a single YAML file for injection markers.

    Returns:
        List of (yaml_path, matched_pattern_description, matched_text_snippet),
        or None if the file could not be parsed (parse errors are treated as
        failures, not clean — fail-closed policy).
    """
    try:
        content = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        print(f'❌  PARSE ERROR: {path}: {exc}', file=sys.stderr)
        return None

    if not isinstance(content, dict):
        return []

    findings = []
    for yaml_path, string_value in _iter_strings(content):
        for pattern in _PHRASE_MARKERS:
            m = pattern.search(string_value)
            if m:
                snippet = m.group(0)[:80]
                findings.append((yaml_path, f'phrase marker: {pattern.pattern!r}', snippet))
        for pattern in _TOKEN_MARKERS:
            m = pattern.search(string_value)
            if m:
                snippet = m.group(0)[:80]
                findings.append((yaml_path, f'delimiter token: {pattern.pattern!r}', snippet))

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict', action='store_true',
                        help='Future use: fail on warnings too.')
    parser.add_argument('configs', nargs='*',
                        default=['mcp_server/config/*.yaml', 'mcp_server/config/*.yml'],
                        help='Glob patterns for config files to scan.')
    args = parser.parse_args()

    # Resolve files from globs, relative to repo root.
    repo_root = Path(__file__).parent.parent.parent
    files_to_scan: list[Path] = []
    for pattern in args.configs:
        matched = glob.glob(str(repo_root / pattern))
        files_to_scan.extend(Path(p) for p in matched)

    if not files_to_scan:
        print('check_config_safety: no config files found — nothing to scan.')
        return 0

    total_findings = 0
    parse_errors = 0
    for config_path in sorted(files_to_scan):
        rel = config_path.relative_to(repo_root)
        findings = scan_file(config_path)
        if findings is None:
            # Parse error — fail-closed: treat as hard failure, not clean.
            print(f'❌  PARSE ERROR: {rel} — malformed YAML treated as failure')
            parse_errors += 1
        elif findings:
            print(f'\n❌  SUSPICIOUS CONTENT in {rel}:')
            for yaml_path, description, snippet in findings:
                print(f'    [{yaml_path}]  {description}')
                print(f'        snippet: {snippet!r}')
            total_findings += len(findings)
        else:
            print(f'✅  {rel} — clean')

    print()
    if total_findings or parse_errors:
        msg_parts = []
        if total_findings:
            msg_parts.append(f'{total_findings} suspicious value(s) found')
        if parse_errors:
            msg_parts.append(f'{parse_errors} parse error(s)')
        print(f'check_config_safety: FAILED — {", ".join(msg_parts)}.')
        if total_findings:
            print('Review the flagged YAML fields and remove prompt-injection content.')
        if parse_errors:
            print('Fix YAML parse errors — malformed config files are treated as failures.')
        return 1

    print('check_config_safety: PASS — no injection markers found.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
