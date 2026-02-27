#!/usr/bin/env bash
# check_patch_integrity.sh — verify all patches/graphiti_core/* apply cleanly to upstream/main.
#
# Usage (local):  bash scripts/ci/check_patch_integrity.sh
# CI:             invoked automatically by the patch-integrity job in .github/workflows/ci.yml
#
# What it does:
#   1. Fetches upstream getzep/graphiti (adds remote if missing).
#   2. Creates a throwaway worktree at upstream/main.
#   3. Runs `git apply --check` for every .patch file under patches/graphiti_core/.
#   4. Reports pass/fail per patch; exits 1 if any fail.
set -euo pipefail

UPSTREAM_URL="${GRAPHITI_UPSTREAM_URL:-https://github.com/getzep/graphiti.git}"
PATCHES_DIR="patches/graphiti_core"
WORKTREE_DIR="$(mktemp -d /tmp/patch-integrity-check-XXXX)"

cleanup() { git worktree remove --force "$WORKTREE_DIR" 2>/dev/null || true; }
trap cleanup EXIT

cd "$(git rev-parse --show-toplevel)"

# Ensure upstream remote exists.
if ! git remote | grep -q '^upstream$'; then
  echo "Adding upstream remote: $UPSTREAM_URL"
  git remote add upstream "$UPSTREAM_URL"
fi

echo "Fetching upstream/main..."
git fetch upstream main --quiet

echo "Creating worktree at upstream/main..."
git worktree add --detach "$WORKTREE_DIR" upstream/main --quiet

if [[ ! -d "$PATCHES_DIR" ]]; then
  echo "WARN: $PATCHES_DIR not found; nothing to check."
  exit 0
fi

FAILED=0
PASSED=0

while IFS= read -r -d '' patchfile; do
  rel="${patchfile#./}"
  result=$(cd "$WORKTREE_DIR" && git apply --check "$OLDPWD/$patchfile" 2>&1) && rc=0 || rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "✅  $rel"
    ((PASSED++)) || true
  else
    echo "❌  FAIL: $rel"
    echo "    $result"
    ((FAILED++)) || true
  fi
done < <(find "$PATCHES_DIR" -name '*.patch' -print0 | sort -z)

echo ""
echo "Patch integrity: $PASSED passed, $FAILED failed."
if [[ $FAILED -gt 0 ]]; then
  echo "ERROR: run 'bash scripts/export-core-patches.sh' to regenerate patches, then commit." >&2
  exit 1
fi
echo "PASS: all patches apply cleanly to upstream/main."
