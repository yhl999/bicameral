# EXEC-TYPED-CONTENT-LANE-CONFORMANCE-FIX-v0

## Goal
Restore ontology-conformant ingestion for `s1_writing_samples`, `s1_inspiration_short_form`, and `s1_inspiration_long_form` in the active Graphiti MCP path so Phase 3C craft filters stop dropping every candidate and re-extracted data can persist as typed entities + semantic relations.

## Owned Paths
- `prd/EXEC-TYPED-CONTENT-LANE-CONFORMANCE-FIX-v0.md`
- `mcp_server/config/extraction_ontologies.yaml`
- `config/extraction_ontologies.yaml`

## Problem
- Active MCP loads `mcp_server/config/extraction_ontologies.yaml`.
- For the 3 content craft lanes above, the active config is stale: no `extraction_mode: constrained_soft`, no anchor entity types, no craft-required fields, and weaker lane intent.
- Public `graphiti_core/graphiti.py` Phase 3C hard-gates those same lanes by requiring:
  - anchor types (`WritingSample`, `ShortFormPiece`, `LongFormPiece`)
  - craft fields (`evidence_span`, `craft_type`, `pattern_template`, `when_to_use`)
- Result: hydrated candidates are dropped before graph write, so episodes exist but typed entities/edges do not persist.

## Deliverable
Sync the active runtime ontology config for the 3 affected lanes to the newer constrained-soft schema already captured in the private design copy.

## DoD
- [ ] Active runtime config for the 3 lanes includes `extraction_mode: constrained_soft`.
- [ ] Active runtime config for the 3 lanes includes anchor entity types matching Phase 3C.
- [ ] Active runtime config for the 3 lanes includes craft-required field schema matching Phase 3C.
- [ ] Writing-samples lane includes `SIGNALS_STYLE` relation type and updated intent guidance.
- [ ] Top-level `config/extraction_ontologies.yaml` stays in sync with `mcp_server/config/extraction_ontologies.yaml`.
- [ ] YAML parses successfully.
- [ ] Ontology registry tests still pass.

## Validation
```bash
cd /Users/archibald/clawd/projects/bicameral-runtime
python3 - <<'PY'
import yaml
from mcp_server.src.services.ontology_registry import OntologyRegistry

for path in ['mcp_server/config/extraction_ontologies.yaml', 'config/extraction_ontologies.yaml']:
    with open(path) as f:
        yaml.safe_load(f)
    print('ok', path)

reg = OntologyRegistry.load('mcp_server/config/extraction_ontologies.yaml')
for lane, anchor, typed in [
    ('s1_writing_samples', 'WritingSample', 'VoiceFingerprint'),
    ('s1_inspiration_short_form', 'ShortFormPiece', 'RhetoricalMove'),
    ('s1_inspiration_long_form', 'LongFormPiece', 'RhetoricalMove'),
]:
    profile = reg.get(lane)
    print(lane, profile.extraction_mode, list(profile.entity_types[anchor].model_fields), list(profile.entity_types[typed].model_fields))
PY
```

Optional, once local test deps are repaired (`numpy`, async pytest support, and stale invalid-mode expectations):
```bash
pytest -q mcp_server/tests/test_ontology_registry_wiring.py mcp_server/tests/test_constrained_soft_extraction.py
```

## Post-merge / Operator Validation
```bash
# restart MCP so the one-time ontology load picks up config changes
# then re-extract 1-3 sample episodes from each lane and verify:
python3 scripts/evaluate_ontology_conformance.py --group-id s1_writing_samples
python3 scripts/evaluate_ontology_conformance.py --group-id s1_inspiration_short_form
python3 scripts/evaluate_ontology_conformance.py --group-id s1_inspiration_long_form
```

## Success Criteria
- Phase 3C drop logs for these lanes are no longer dominated by missing craft fields caused by schema drift.
- Re-extracted sample episodes persist non-zero typed entities for all 3 lanes.
- Semantic relation names are evaluated from `r.name`; operators no longer treat `type(r)=RELATES_TO` as semantic collapse.
- After re-extraction, lane conformance reports show non-zero `typed_entity_rate` and non-zero semantic allowed-relation coverage for the 3 lanes.
