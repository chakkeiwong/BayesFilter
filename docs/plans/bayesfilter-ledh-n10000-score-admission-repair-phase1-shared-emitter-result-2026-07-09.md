# Phase 1 Result: Shared Admitted Artifact Emitter

Date: 2026-07-09

Status: `PASSED_SHARED_EMITTER_GATE_PENDING_REVIEW`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Add a shared validator-backed score-artifact builder before full score runs. | Passed: `build_ledh_score_artifact` assembles schema fields from an admitted value artifact and immediately calls `validate_ledh_score_artifact`. | Historical `manual_total_vjp*`, raw legacy JSON, tiny-as-full, missing memory pass, row mismatch, parameter mismatch, and source target tampering fail focused tests. | Existing per-model runners still contain local builders; Phase 2 should either rerun through validated output or migrate/wrap only where needed. | Execute Phase 2 LGSSM with rerun-preferred schema-valid artifact generation. | No full `N=10000` score run, no score admission, no leaderboard completion, no HMC readiness, no posterior correctness, no runtime ranking. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Is there now a shared, tested way to emit full-admission score artifacts so future full runs cannot accidentally produce raw legacy evidence? |
| Baseline/comparator | Existing per-model artifact builders, shared score validator, Phase 0 inventory, and July 8 Phase 8 blocker. |
| Primary criterion | Passed for a shared builder: artifacts only pass full admission when all full-admission inputs are present, compact, same-target, and memory-valid. |
| Veto diagnostics | Passed: tests cover historical route full rejection, raw JSON rejection, missing memory gate, row mismatch, parameter mismatch, target tampering, and tiny-not-admitted behavior. |
| Explanatory diagnostics | The helper is intentionally small and delegates final authority to the existing validator. |
| Not concluded | Per-model full run success is not concluded. |

## Implementation Summary

Added:

- `bayesfilter/highdim/ledh_score_artifact.py`
- `tests/highdim/test_ledh_score_artifact_emitter_phase1.py`

The helper:

- validates the source value artifact with `require_admitted=True`;
- copies target scalar, output field, observation policy, theta coordinate
  system, row id, and parameter order from the value artifact;
- requires score length and parameter order to match the admitted value row;
- sets no-autodiff and same-route score fields explicitly;
- preserves model-specific extra fields when supplied, such as KSC exact-native
  nonclaims;
- calls `validate_ledh_score_artifact` before returning.

## Per-Row Certification Ledger

| Row | Value artifact | Current score artifact status | Phase 1 certification status | Next action |
| --- | --- | --- | --- | --- |
| `benchmark_lgssm_exact_oracle_m3_T50` | `docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json` | Raw compact `N=10000` memory JSON only; not admitted. | Shared emitter can emit a valid LGSSM full artifact from complete full-run fields; existing raw JSON is incomplete for clean normalization because it lacks full schema identity fields. | Phase 2 rerun preferred through LGSSM runner with schema-valid output. |
| `zhao_cui_spatial_sir_austria_j9_T20` | `docs/plans/ledh-phase3-fixed-sir-forward-scalar-artifact-2026-07-07.json` | Raw full evidence is historical `manual_total_vjp*`; tiny compact artifact not admitted. | Shared emitter rejects historical full admission and tiny-as-full. | Phase 3 compact full rerun after LGSSM. |
| `zhao_cui_sv_actual_nongaussian_T1000` | `docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json` | Tiny compact only. | Shared emitter pattern is compatible with full compact artifact once full memory diagnostics exist. | Phase 4 compact full run. |
| `zhao_cui_predator_prey_T20` | `docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json` | Tiny compact only. | Shared emitter pattern is compatible with full compact artifact once full memory diagnostics exist. | Phase 5 compact full run. |
| `zhao_cui_generalized_sv_synthetic_from_estimated_values` | `docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json` | Tiny compact only. | Shared emitter preserves value target identity from the source artifact, preventing target substitution. | Phase 6 compact full run. |
| `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000` | `docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json` | Tiny compact only. | Shared emitter can preserve `claims_exact_native_actual_sv_likelihood=false` through `extra_fields`. | Phase 7 compact full run. |

## Local Checks

Passed:

```bash
python -m py_compile \
  bayesfilter/highdim/ledh_score_artifact.py \
  bayesfilter/highdim/ledh_score_contract.py \
  docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py \
  docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py \
  docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py \
  docs/benchmarks/benchmark_ledh_same_target_generalized_sv_score.py \
  docs/benchmarks/benchmark_ledh_same_target_ksc_sv_score.py
```

Passed:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_score_artifact_emitter_phase1.py \
  tests/highdim/test_ledh_score_contract_phase1.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py \
  tests/highdim/test_ledh_generalized_sv_score_phase6_contract.py \
  tests/highdim/test_ledh_ksc_sv_score_phase7_contract.py \
  tests/test_two_lane_highdim_ledh_leaderboard.py -q
```

Result:

```text
104 passed, 2 warnings
```

## Boundary Notes

- No full `N=10000` score command was run in Phase 1.
- The helper does not compute scores.
- The helper does not admit a row independently of
  `validate_ledh_score_artifact`.
- This phase does not modify leaderboard integration paths.

## Handoff To Phase 2

Phase 2 should produce the first schema-valid admitted score artifact for
LGSSM. Because the July 6 raw score-memory JSON lacks the full Phase 1 schema
identity fields, the preferred route is to rerun the LGSSM compact score runner
with full `N=10000,T=50` settings and validate the emitted artifact. Only if a
precheck proves the runner output already contains complete schema evidence
should Phase 2 normalize without rerun.
