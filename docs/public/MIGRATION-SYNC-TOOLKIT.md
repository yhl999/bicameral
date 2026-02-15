# Public Migration + Upstream Sync Toolkit

This repository ships a **delta-layer migration/sync toolkit** designed for:

- repeatable upstream sync preflight checks,
- deterministic history candidate comparison,
- state package export/check/import,
- extension contract validation.

`graphiti_core/**` is intentionally out of scope for this toolkit.

## Policy config

- `config/migration_sync_policy.json`
  - upstream/origin remotes + branch defaults,
  - sync-button safety policy,
  - history export metric coefficients,
  - history scorecard threshold + weights,
  - weekly cadence metadata.
- `config/state_migration_manifest.json`
  - required files,
  - optional globs,
  - exclusion patterns for migration package generation.

## 1) Upstream sync doctor

```bash
python3 scripts/upstream_sync_doctor.py \
  --repo . \
  --policy config/migration_sync_policy.json \
  --dry-run \
  --check-sync-button-safety
```

Use `--ensure-upstream` to add missing upstream remote from policy (when not in dry-run).
Use `--allow-dirty` only for local diagnostics when you intentionally run checks on a dirty working tree.

## 2) History candidate reports + scorecard

```bash
python3 scripts/public_history_export.py \
  --repo . \
  --mode filtered-history \
  --dry-run \
  --report reports/publicization/filtered-history.md \
  --summary-json reports/publicization/filtered-history.json

python3 scripts/public_history_export.py \
  --repo . \
  --mode clean-foundation \
  --dry-run \
  --report reports/publicization/clean-foundation.md \
  --summary-json reports/publicization/clean-foundation.json

python3 scripts/public_history_scorecard.py \
  --filtered-summary reports/publicization/filtered-history.json \
  --clean-summary reports/publicization/clean-foundation.json \
  --policy config/migration_sync_policy.json \
  --out reports/publicization/history-scorecard.md
```

Policy fallback rule is encoded in scorecard output:
- choose clean-foundation automatically if filtered-history score is below threshold,
- or if unresolved HIGH risk remains.

## 3) State migration kit

```bash
python3 scripts/state_migration_export.py \
  --manifest config/state_migration_manifest.json \
  --out /tmp/graphiti-state-export \
  --dry-run

python3 scripts/state_migration_check.py \
  --package /tmp/graphiti-state-export \
  --dry-run

python3 scripts/state_migration_import.py \
  --in /tmp/graphiti-state-export \
  --dry-run
```

Notes:
- dry-run export writes package manifest preview (no payload files copied),
- non-dry-run export writes payload files and checksums for deterministic imports,
- import performs path-safety + checksum + file-size validation before writing payload files.

## 4) Extension contract check

```bash
python3 scripts/extension_contract_check.py --strict
```

Checks `extensions/*/manifest.json` for:
- required fields (`name`, `version`, `capabilities`, `entrypoints`),
- duplicate extension names/capabilities,
- traversal-safe relative entrypoint paths,
- missing entrypoint files.
