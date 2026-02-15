#!/usr/bin/env bash
set -euo pipefail

uvx ruff check --output-format=github
python3 scripts/public_boundary_policy_lint.py \
  --manifest config/public_export_allowlist.yaml \
  --denylist config/public_export_denylist.yaml
