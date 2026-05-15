# Phase IE6 result: learned-OT teacher/student/OOD residual tests

## Outcome

- Master-program exit label: `ie_phase_deferred_with_recorded_reason`
- Local exit label: `ie6_learned_ot_residual_deferred_no_artifact`
- Phase status: `deferred`

## Reason

No approved pre-existing teacher/student artifact with provenance was available. IE6 therefore emitted a schema-valid deferred `learned_map_residual` row and did not execute residual tests.

## Decision table

| Diagnostic | Status | Primary criterion | Promotion veto | Continuation veto | Repair trigger |
| --- | --- | --- | --- | --- | --- |
| `learned_map_residual` | `deferred` | `not_triggered` | `not_triggered` | `fail` | `missing artifact provenance` |

## Artifacts

- `experiments/dpf_monograph_evidence/reports/outputs/learned_ot_residual.json`
- `experiments/dpf_monograph_evidence/reports/learned-ot-residual-result.md`

## Non-implication

IE6 learned-OT diagnostics were deferred because no approved pre-existing teacher/student artifact with provenance was available. This deferral does not validate or invalidate learned OT, neural OT, surrogate-map quality, posterior quality, production bayesfilter code, banking use, model-risk use, or production readiness.

## Next-phase justification

Proceeding to IE7 is justified only because IE7 tests an independent fixed scalar value-gradient contract. IE6 remains an explicit learned-OT evidence gap.
