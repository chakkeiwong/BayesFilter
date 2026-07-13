# Phase 8 Result: Value-Only Integration From Admitted Forward Scalars

metadata_date: 2026-07-07
status: `PASSED_VALUE_ONLY_INTEGRATION_PENDING_REVIEW`
master_program: `docs/plans/bayesfilter-ledh-forward-scalar-per-model-master-program-2026-07-07.md`
phase: 8

## Phase Objective

Build a value-only LEDH integration artifact from admitted same-target
forward-scalar artifacts for the main observed-data model rows.

Target scalar: `observed_data_log_likelihood_estimator`, reported as
`log_likelihood`.

This phase was forward-scalar-only. It did not implement, run, import, admit,
or merge scores.

## Decision Table

| Field | Result |
| --- | --- |
| Decision | Phase 8 value-only integration artifact is locally admitted. |
| Primary criterion status | Passed locally: the JSON artifact has exactly six main LEDH rows, each sourced from an admitted Phase 2-7 forward-scalar artifact and replay-validated. |
| Veto diagnostic status | No missing source artifact, stale row id, tiny artifact admission, score fields, old score artifacts, unchanged score-inclusive builder reuse, runtime cross-ranking, all-algorithm claim, diagnostic SIR promotion, or KSC exact native actual-SV claim. |
| Main uncertainty | This integrates value rows only; it does not establish score correctness, all-algorithm comparison readiness, HMC readiness, posterior correctness, scientific superiority, or fair runtime ranking. |
| Next justified action | Send this result, the integration artifact, and replay test for bounded read-only review; if review agrees, close the forward-scalar value-only runbook. |
| What is not concluded | No score admission, score correctness, all-algorithm leaderboard readiness, HMC readiness, posterior correctness, scientific superiority, Zhao-Cui source-faithfulness, or runtime ranking. |

## Entry Conditions

- Phase 1 executable schema guard passed.
- Phase 2 LGSSM artifact validates locally with `require_admitted=True`.
- Phase 3 fixed SIR artifact validates locally under `sir_log_scale_theta`.
- Phase 4 predator-prey artifact validates locally with `require_admitted=True`.
- Phase 5 actual-SV artifact validates locally with `require_admitted=True`.
- Phase 6 generalized-SV artifact validates locally with `require_admitted=True`.
- Phase 7 KSC-SV artifact validates locally with `require_admitted=True`.
- Phase 7 result and Phase 8 subplan passed bounded read-only review.

## Skeptical Plan Audit

| Risk checked | Result |
| --- | --- |
| Wrong baseline | The builder reads only the six Phase 2-7 admitted forward-scalar artifacts and validates each with the Phase 1 schema. |
| Proxy metric promoted | Runtime, compile time, GPU device, and memory are copied only as explanatory diagnostics. |
| Missing stop conditions | The subplan stopped on missing artifacts, schema failure, stale row id, target mismatch, score artifacts, diagnostic SIR promotion, and runtime ranking. |
| Hidden assumptions | Each row preserves its `target_observation_policy`; the KSC row explicitly stays finite-mixture surrogate evidence, not exact native actual-SV likelihood. |
| Artifact mismatch | The result is a value-only integration artifact, not a score-inclusive or all-algorithm leaderboard. |

Audit status: passed for Phase 8 local integration.

## Implementation

Added:

- `docs/benchmarks/benchmark_ledh_forward_scalar_value_integration.py`
- `tests/highdim/test_ledh_phase8_value_integration_artifact.py`

The builder:

- reads the six required Phase 2-7 artifacts;
- validates each with
  `validate_ledh_forward_scalar_artifact(..., require_admitted=True)`;
- computes total and per-time-step log likelihood means, sample standard
  deviations, and Monte Carlo standard errors across the five seeds;
- copies timing/device fields as diagnostic-only fields;
- preserves row-specific target-policy labels;
- marks score integration as `blocked_out_of_scope_forward_scalar_only`;
- disables runtime cross-ranking and all-algorithm comparison;
- records the parameterized SIR diagnostic row as excluded from the main value
  table.

## Integration Artifact

Wrote:

- `docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.json`
- `docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.md`

Top-level fields:

- `schema_version =
  bayesfilter.highdim.ledh_forward_scalar_value_integration.v1`;
- `main_row_count = 6`;
- `score_integration_status = blocked_out_of_scope_forward_scalar_only`;
- `runtime_cross_ranking_allowed = false`;
- `all_algorithm_comparison_allowed = false`.

Main rows:

| Row | Mean Log Likelihood | MCSE | Target Policy |
| --- | ---: | ---: | --- |
| `benchmark_lgssm_exact_oracle_m3_T50` | `-135.96007385253907` | `0.005933406369186942` | `lgssm_gaussian_observation_density` |
| `zhao_cui_spatial_sir_austria_j9_T20` | `-902.8301513671875` | `0.20822211173623006` | `fixed_sir_infectious_components_gaussian_observation_density` |
| `zhao_cui_predator_prey_T20` | `-169.8675048828125` | `0.4235013715066106` | `additive_gaussian_predator_prey` |
| `zhao_cui_sv_actual_nongaussian_T1000` | `-2289.953466796875` | `0.15099991276087987` | `transformed_actual_sv_log_y_square` |
| `zhao_cui_generalized_sv_synthetic_from_estimated_values` | `-1438.9241943359375` | `0.02997747151555195` | `source_route_prior_mean_generalized_sv` |
| `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000` | `-2287.953759765625` | `0.1402306133817723` | `ksc_log_chi_square_gaussian_mixture_surrogate` |

The KSC row preserves:

- `target_family = ksc_finite_gaussian_mixture_surrogate`;
- `target_observation_density =
  finite_ksc_log_chi_square_gaussian_mixture_log_density`;
- `target_transform = log_y_square_plus_offset`;
- `transform_offset = 1e-8`;
- `ksc_mixture_is_target_likelihood = true`;
- `actual_sv_exact_log_chi_square_target_used = false`.

The diagnostic row:

```text
zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale
```

is recorded as `excluded_from_main_value_leaderboard`.

## Execution Evidence

Builder command:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python \
  docs/benchmarks/benchmark_ledh_forward_scalar_value_integration.py \
  --output docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.json \
  --markdown-output docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.md
```

Result:

- passed;
- wrote JSON and markdown artifacts;
- did not run model simulations or GPU benchmarks;
- TensorFlow emitted CUDA initialization noise under `CUDA_VISIBLE_DEVICES=-1`,
  but this command was CPU-hidden artifact integration and not GPU evidence.

## Local Checks

Compile check:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_forward_scalar_value_integration.py \
  tests/highdim/test_ledh_phase8_value_integration_artifact.py
```

Result: passed.

Focused Phase 8 replay:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase8_value_integration_artifact.py -q
```

Result:

```text
3 passed, 2 warnings in 2.62s
```

Through-Phase-8 replay check:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase3_forward_admission.py \
  tests/highdim/test_ledh_forward_contract_phase2.py \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py \
  tests/highdim/test_ledh_phase2_lgssm_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase3_fixed_sir_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase8_value_integration_artifact.py -q
```

Result:

```text
48 passed, 2 warnings in 2.92s
```

Diff hygiene:

```text
git diff --check -- \
  docs/benchmarks/benchmark_ledh_forward_scalar_value_integration.py \
  tests/highdim/test_ledh_phase8_value_integration_artifact.py \
  docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.json \
  docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.md
```

Result: passed.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can the six main LEDH high-dimensional model rows be assembled into one value-only integration artifact using only admitted same-target forward-scalar artifacts? |
| Answer | Yes, locally. The integration artifact contains exactly six main rows, each replay-validated from an admitted Phase 2-7 artifact. |
| Baseline/comparator | Phase 1 schema validator and Phase 2-7 admitted artifacts. Old score-inclusive builders were not reused unchanged. |
| Primary criterion | Passed locally: exactly six main LEDH value rows; each row validates with `require_admitted=True`; each row has finite `log_likelihood_by_seed`; target scalar and output field are correct; score fields are absent and score integration is blocked; runtime cross-ranking is disabled; parameterized SIR diagnostic row is excluded from main rows. |
| Veto diagnostics | No missing source artifact, stale row id, target-policy mismatch, tiny artifact admission, score field merge, old score artifact use, runtime ranking, diagnostic SIR promotion, KSC exact native actual-SV claim, or metadata-only admission. |
| Explanatory diagnostics | Row mean log likelihood, row MCSE, per-time-step summaries, source artifact paths, timing/device fields copied as diagnostic-only. |
| Not concluded | No score admission, score correctness, all-algorithm comparison, HMC readiness, posterior correctness, scientific superiority, or runtime ranking. |

## Nonclaims

- No score route is implemented or admitted.
- This is not an all-algorithm leaderboard.
- This is not a runtime ranking.
- This is not HMC readiness, posterior correctness, scientific superiority, or
  Zhao-Cui source-faithfulness evidence.
- The KSC row is finite-mixture surrogate target evidence, not exact native
  actual-SV likelihood evidence.
