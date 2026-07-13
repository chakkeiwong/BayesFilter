# Phase 8 Result: Compact Score Integration

Date: 2026-07-08

Status: `PASSED_INTEGRATION_POLICY_GATE_FULL_SCORE_ROWS_BLOCKED`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| LEDH inclusive score integration now requires Phase 1 validated compact score artifacts for admission. | Passed for integration policy and tests. | Historical `manual_total_vjp*` candidate is blocked; legacy raw score-memory JSONs are not admitted; blocked rows do not expose admitted score provenance. | Full `N=10000` compact score artifacts are still missing for leaderboard score admission. | Run reviewed full-row compact score gates per model, starting with schema-valid LGSSM and fixed-SIR compact artifacts. | Full LEDH score leaderboard completion, HMC readiness, posterior correctness, runtime ranking, public benchmark readiness, or scientific superiority. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Does the leaderboard score workflow default to compact forward-sensitivity score routes and block historical `manual_total_vjp*` full admission? |
| Baseline/comparator | Shared score contract, compact per-model score ports, existing inclusive leaderboard merger, and score-memory candidate artifacts. |
| Primary criterion | Passed: `benchmark_two_lane_highdim_ledh_inclusive_results.py` now admits a score only through `validate_ledh_score_artifact(..., require_admitted=True)` and requires compact provenance for admitted rows. |
| Veto diagnostics | No historical route is admitted. Fixed-SIR's legacy `manual_total_vjp` memory candidate is explicitly blocked. LGSSM's legacy raw memory JSON is blocked because it lacks the Phase 1 score artifact schema. |
| Explanatory diagnostics | The generated candidate artifact records score candidates separately from admitted score fields. |
| Not concluded | No full score row is admitted by this phase. |
| Artifact | `docs/plans/bayesfilter-ledh-compact-score-default-phase8-integration-candidate-2026-07-08.json` |

## Implementation Summary

Changed files:

- `docs/benchmarks/benchmark_two_lane_highdim_ledh_inclusive_results.py`
- `tests/test_two_lane_highdim_ledh_leaderboard.py`

Key changes:

- Added Phase 1 score artifact validation to the inclusive results merger.
- Added score candidate fields for legacy or non-admitted evidence:
  - `score_candidate_artifact`;
  - `score_candidate_derivative_provenance`;
  - `score_candidate_admission_status`.
- Kept admitted score fields empty for blocked LEDH rows:
  - `score = None`;
  - `score_l2_norm = None`;
  - `score_derivative_provenance = None`;
  - `score_evidence_artifact = None`.
- Replaced the old implicit `primary_pass` raw-memory promotion with the
  manifest policy:

```text
phase1_validated_compact_score_artifact_required_for_admission
```

## Candidate Artifact Summary

Artifact:

- `docs/plans/bayesfilter-ledh-compact-score-default-phase8-integration-candidate-2026-07-08.json`

LEDH row statuses:

| Row | Integration status | Candidate provenance | Candidate status |
| --- | --- | --- | --- |
| `benchmark_lgssm_exact_oracle_m3_T50` | `executed_value_only_score_blocked` | `compact_forward_sensitivity_no_autodiff_same_scalar_lgssm_ledh_pfpf_ot` | `legacy_raw_score_memory_not_admitted` |
| `zhao_cui_spatial_sir_austria_j9_T20` | `executed_value_only_score_blocked` | `manual_total_vjp_no_autodiff_same_scalar_fixed_sir_logscale_ledh_pfpf_ot` | `historical_diagnostic_not_admitted` |
| `zhao_cui_sv_actual_nongaussian_T1000` | `blocked` | N/A | N/A |
| `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000` | `blocked` | N/A | N/A |
| `zhao_cui_predator_prey_T20` | `blocked` | N/A | N/A |
| `zhao_cui_generalized_sv_synthetic_from_estimated_values` | `blocked` | N/A | N/A |

The parameterized SIR diagnostic row remains scoped and is not a main observed-data score row.

## Local Checks

Passed:

```bash
python -m py_compile docs/benchmarks/benchmark_two_lane_highdim_ledh_inclusive_results.py
```

Passed:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/test_two_lane_highdim_ledh_leaderboard.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result:

```text
34 passed, 2 warnings
```

Passed:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_score_contract_phase1.py \
  tests/highdim/test_ledh_generalized_sv_score_phase6_contract.py \
  tests/highdim/test_ledh_ksc_sv_score_phase7_contract.py \
  tests/test_two_lane_highdim_ledh_leaderboard.py -q
```

Result:

```text
51 passed, 2 warnings
```

Candidate generation command:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python \
  docs/benchmarks/benchmark_two_lane_highdim_ledh_inclusive_results.py \
  --output docs/plans/bayesfilter-ledh-compact-score-default-phase8-integration-candidate-2026-07-08.json \
  --markdown-output docs/plans/bayesfilter-ledh-compact-score-default-phase8-integration-candidate-2026-07-08.md
```

Result:

- Command exited 0.
- TensorFlow attempted CUDA initialization despite CPU hiding and emitted
  sandbox/device warnings; this was a CPU-only integration artifact, not GPU
  evidence.

## Review

- Claude review gate was attempted for Phase 7/8 and rejected by the local
  policy layer as external data disclosure risk.
- A fresh Codex read-only substitute review returned `VERDICT: AGREE` for the
  Phase 7 result and Phase 8 subplan.
- Phase 8 final review bundle:
  `docs/reviews/bayesfilter-ledh-compact-score-default-phase8-integration-review-bundle-2026-07-08.md`.

## Boundary Notes

- Old `manual_total_vjp*` score routes remain historical and diagnostic only.
- Tiny compact artifacts are not promoted to full score rows.
- Raw score-memory JSONs are not enough for leaderboard score admission.
- No full `N=10000` score row was launched in Phase 8.
- Runtime cross-ranking remains forbidden because non-LEDH rows are frozen baselines.

## Handoff

The compact score default master program has completed its integration-policy
phase, but the full score leaderboard is still not complete.

Next reviewed phases should generate schema-valid compact full-row score
artifacts with `score_admission_status =
n10000_same_target_no_tape_score_admitted` and validated memory diagnostics.
Recommended order:

1. LGSSM compact full-row score artifact normalization or rerun.
2. Fixed-SIR compact full-row score memory gate.
3. Actual-SV full-row compact score memory gate.
4. Predator-prey full-row compact score memory gate.
5. Generalized-SV full-row compact score memory gate.
6. KSC-SV full-row compact score memory gate.
7. Re-run inclusive integration with admitted compact score artifacts.
