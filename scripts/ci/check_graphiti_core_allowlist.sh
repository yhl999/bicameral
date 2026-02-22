#!/usr/bin/env bash
set -euo pipefail

ALLOWLIST="config/graphiti_core_allowlist.txt"
if [[ ! -f "$ALLOWLIST" ]]; then
  echo "ERROR: missing allowlist: $ALLOWLIST" >&2
  exit 1
fi

head_ref="${GITHUB_HEAD_REF:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')}"
if [[ "$head_ref" == sync/upstream-* || "$head_ref" == upstream-sync/* ]]; then
  echo "SKIP: upstream sync branch ($head_ref)"
  exit 0
fi

if [[ -n "${GITHUB_BASE_REF:-}" ]]; then
  BASE_REF="origin/${GITHUB_BASE_REF}"
else
  BASE_REF="${1:-origin/main}"
fi

# Best effort fetch for CI/locals.
if [[ "$BASE_REF" == origin/* ]]; then
  git fetch origin "${BASE_REF#origin/}" --quiet || true
fi

if ! git rev-parse --verify "$BASE_REF" >/dev/null 2>&1; then
  echo "WARN: base ref '$BASE_REF' not found; falling back to HEAD~1" >&2
  BASE_REF="HEAD~1"
fi

changed_raw="$(git diff --name-only "$BASE_REF...HEAD" -- 'graphiti_core/**' | sort -u || true)"
changed=()
while IFS= read -r line; do
  [[ -n "$line" ]] && changed+=("$line")
done <<< "$changed_raw"

if [[ ${#changed[@]} -eq 0 ]]; then
  echo "PASS: no graphiti_core changes"
  exit 0
fi

violations=()
for f in "${changed[@]}"; do
  if ! grep -Fxq "$f" "$ALLOWLIST"; then
    violations+=("$f")
  fi
done

if [[ ${#violations[@]} -gt 0 ]]; then
  echo "FAIL: graphiti_core drift outside allowlist"
  echo "Base ref: $BASE_REF"
  echo "Allowlist: $ALLOWLIST"
  for f in "${violations[@]}"; do
    echo " - $f"
  done
  exit 1
fi

echo "PASS: graphiti_core changes constrained to allowlist"
for f in "${changed[@]}"; do
  echo " - $f"
done
