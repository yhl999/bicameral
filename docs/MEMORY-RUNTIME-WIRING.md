# Memory Runtime Wiring

This document defines the runtime backend switch contract for memory retrieval.

## Runtime model

Two backend profiles are supported:

- `qmd_primary` (production default)
- `graphiti_primary` (operator opt-in)

Profiles are declared in:

- `config/runtime_memory_backend_profiles.json`

Current active state is stored in:

- `config/.runtime_memory_backend_state.json`

## Operator commands

Run from repository root.

```bash
python3 scripts/runtime_memory_backend_status.py
python3 scripts/runtime_memory_backend_switch.py --target graphiti_primary --dry-run
python3 scripts/runtime_memory_backend_switch.py --target qmd_primary --dry-run
python3 scripts/runtime_memory_backend_switch.py --target graphiti_primary --execute
python3 scripts/runtime_memory_backend_switch.py --revert --execute
```

## Guardrails

- group-safe gating must stay enabled in all active profiles
- shadow compare should remain enabled during cutover
- one-command revert must always be available after a switch

## Public/private split

Public repo contains generic switch/status framework and example/default profile config.

Private operational overlays may replace profile values at deploy time (for environment-specific behavior), but should not change the switch contract surface.

## Canonical runtime checkout

Operational runtime should execute from the canonical runtime checkout linked by:

- `tools/graphiti -> ../projects/graphiti-openclaw-runtime`

Apply private overlay before operations:

```bash
/Users/archibald/clawd/projects/graphiti-openclaw-private/scripts/apply-overlay.sh \
  /Users/archibald/clawd/projects/graphiti-openclaw-runtime
```
