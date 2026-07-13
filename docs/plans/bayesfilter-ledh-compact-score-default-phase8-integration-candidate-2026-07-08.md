# Two-Lane Highdim LEDH-Inclusive Leaderboard Results

Authoritative JSON artifact: `docs/plans/bayesfilter-ledh-compact-score-default-phase8-integration-candidate-2026-07-08.json`.

Comparator mode: `frozen_non_ledh_baseline_plus_fresh_ledh`.

Runtime cross-ranking allowed: `False`.

## Rows

| Row | Algorithm | Status | Avg loglik | MCSE | Score status | Runtime rankable | Reason |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| benchmark_lgssm_exact_oracle_m3_T50 | fixed_sgqf | executed_value_score | -2.72152 |  | analytical_score_emitted | False | SGQF score vector emitted by reviewed analytical fixed-branch score path |
| benchmark_lgssm_exact_oracle_m3_T50 | ukf | executed_value_score | -2.72152 |  |  | False |  |
| benchmark_lgssm_exact_oracle_m3_T50 | zhao_cui_scalar_or_multistate | executed_value_score | -2.72152 |  |  | False |  |
| benchmark_lgssm_exact_oracle_m3_T50 | ledh_pfpf_ot | executed_value_only_score_blocked | -2.7192 | 0.000118668 | blocked_score_until_same_target_no_tape_gate | False | same-target LEDH value executed at N=10000; score remains blocked |
| zhao_cui_sv_actual_nongaussian_T1000 | fixed_sgqf | executed_value_score | -2.30091 |  | analytical_score_emitted | False | SGQF score vector emitted by reviewed analytical fixed-branch score path |
| zhao_cui_sv_actual_nongaussian_T1000 | ukf | executed_value_score | -1.4607 |  | analytical_score_emitted | False | actual-SV raw augmented-noise Gaussian-closure score emitted by reviewed factor-propagating SR-UKF manual route |
| zhao_cui_sv_actual_nongaussian_T1000 | zhao_cui_scalar_or_multistate | executed_value_score | -2.28623 |  | analytical_score_emitted | False | Zhao-Cui exact-transformed SV score emitted by scalar fixed-branch TT manual parameter-score adapter |
| zhao_cui_sv_actual_nongaussian_T1000 | ledh_pfpf_ot | blocked |  |  | blocked_score | False | No reviewed current LEDH-inclusive leaderboard adapter proves that the GPU/XLA TF32 route computes the exact requested actual-SV row target.; value_status=blocked_no_reviewed_current_gpu_xla_ledh_row_adapter; score_status=blocked_score |
| zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000 | fixed_sgqf | executed_value_score | -2.28463 |  | analytical_score_emitted | False | SGQF score vector emitted by reviewed analytical fixed-branch score path |
| zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000 | ukf | executed_value_score | -2.28463 |  | analytical_score_emitted | False | UKF score vector emitted by reviewed principal-square-root analytical component score path |
| zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000 | zhao_cui_scalar_or_multistate | executed_value_score | -2.28443 |  | analytical_score_emitted | False | Zhao-Cui KSC transformed-mixture score emitted by scalar fixed-branch TT manual parameter-score adapter |
| zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000 | ledh_pfpf_ot | blocked |  |  | blocked_score | False | No reviewed LEDH adapter exists for the KSC transformed-mixture row target.; value_status=blocked_no_ledh_ksc_row_adapter; score_status=blocked_score |
| zhao_cui_spatial_sir_austria_j9_T20 | fixed_sgqf | blocked |  |  | blocked_no_free_theta | False | no reviewed SGQF source-scope spatial SIR route is wired |
| zhao_cui_spatial_sir_austria_j9_T20 | ukf | executed_value_only | -36.832 |  |  | False |  |
| zhao_cui_spatial_sir_austria_j9_T20 | zhao_cui_scalar_or_multistate | blocked_or_status_only |  |  |  | False | P91 closes the scoped local complete-data SIR d18 component route, but the full observed-data/filtering leaderboard evaluator remains blocked by preserved source-route derivative/evaluator gaps. |
| zhao_cui_spatial_sir_austria_j9_T20 | ledh_pfpf_ot | executed_value_only_score_blocked | -45.1415 | 0.0104111 | blocked_score_until_same_target_no_tape_gate | False | fixed spatial SIR LEDH value executed at N=10000 under the amended sir_log_scale_theta row; score remains blocked until same-target no-tape evidence passes |
| zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale | fixed_sgqf | blocked |  |  | not_applicable_to_scoped_component_row | False | fixed_sgqf is not admitted for the scoped Zhao-Cui parameterized SIR local complete-data component row; the scoped row is a fixed-variant Zhao-Cui manual-score component cell. |
| zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale | ukf | blocked |  |  | not_applicable_to_scoped_component_row | False | ukf is not admitted for the scoped Zhao-Cui parameterized SIR local complete-data component row; the scoped row is a fixed-variant Zhao-Cui manual-score component cell. |
| zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale | zhao_cui_scalar_or_multistate | executed_value_score | -60.4464 |  | analytical_score_emitted | False | Zhao-Cui parameterized SIR T20 local complete-data score emitted by manual parameter-score methods |
| zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale | ledh_pfpf_ot | scoped_component_status_only |  |  | scoped_score_diagnostic_not_full_observed_data_score | False | The parameterized SIR row is not a full observed-data/filtering row in the current baseline.; value_status=scoped_or_diagnostic_only; score_status=scoped_score_diagnostic_not_full_observed_data_score |
| zhao_cui_predator_prey_T20 | fixed_sgqf | blocked |  |  | blocked_target_alignment | False | blocked_target_alignment: no reviewed fixed-SGQF evaluator is wired for the source-scope T20 predator-prey observations; the available P47 two-observation lower-rung value is diagnostic-only and is not reported as this T20 row |
| zhao_cui_predator_prey_T20 | ukf | executed_value_only | -8.56828 |  | blocked_autodiff_not_admitted | False | autodiff score provenance is diagnostic only; analytical gradient accuracy is the leaderboard benchmark |
| zhao_cui_predator_prey_T20 | zhao_cui_scalar_or_multistate | blocked_or_status_only |  |  |  | False | P8D_MODEL_SPECIFIC_NUMERIC_EVALUATOR_ADAPTER_REQUIRED |
| zhao_cui_predator_prey_T20 | ledh_pfpf_ot | blocked |  |  | blocked_score | False | No reviewed current LEDH-inclusive runner adapter proves same-target predator-prey T20 execution on the GPU/XLA TF32 route.; value_status=blocked_no_reviewed_current_gpu_xla_ledh_row_adapter; score_status=blocked_score |
| zhao_cui_generalized_sv_synthetic_from_estimated_values | fixed_sgqf | blocked |  |  | blocked_exact_source_row_evaluator_missing | False | blocked_source_row_evaluator_missing: no reviewed fixed-SGQF exact-row evaluator is wired for zhao_cui_generalized_sv_synthetic_from_estimated_values; native-oracle, precursor, auxiliary, actual-SV, and KSC evidence are not source-row admission evidence |
| zhao_cui_generalized_sv_synthetic_from_estimated_values | ukf | executed_value_only | -1.42893 |  | blocked_autodiff_not_admitted | False | autodiff score provenance is diagnostic only; analytical gradient accuracy is the leaderboard benchmark |
| zhao_cui_generalized_sv_synthetic_from_estimated_values | zhao_cui_scalar_or_multistate | blocked_or_status_only |  |  |  | False | blocked_source_row_evaluator_missing: no reviewed Zhao-Cui exact-row evaluator is wired for zhao_cui_generalized_sv_synthetic_from_estimated_values; native-oracle, precursor, auxiliary, actual-SV, and KSC evidence are not source-row admission evidence. |
| zhao_cui_generalized_sv_synthetic_from_estimated_values | ledh_pfpf_ot | blocked |  |  | blocked_score | False | No reviewed LEDH adapter proves the requested generalized-SV source-row target.; value_status=blocked_no_reviewed_same_target_ledh_row_adapter; score_status=blocked_score |

## Row Summary

| Row | LEDH status | LEDH value status | LEDH score status | Four-way value ready | Four-way score ready |
| --- | --- | --- | --- | --- | --- |
| benchmark_lgssm_exact_oracle_m3_T50 | executed_value_only_score_blocked | executed_same_target_value | blocked_score_until_same_target_no_tape_gate | True | False |
| zhao_cui_sv_actual_nongaussian_T1000 | blocked | blocked_no_reviewed_current_gpu_xla_ledh_row_adapter | blocked_score | False | False |
| zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000 | blocked | blocked_no_ledh_ksc_row_adapter | blocked_score | False | False |
| zhao_cui_spatial_sir_austria_j9_T20 | executed_value_only_score_blocked | executed_fixed_sir_value_only | blocked_score_until_same_target_no_tape_gate | False | False |
| zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale | scoped_component_status_only | scoped_or_diagnostic_only | scoped_score_diagnostic_not_full_observed_data_score | False | False |
| zhao_cui_predator_prey_T20 | blocked | blocked_no_reviewed_current_gpu_xla_ledh_row_adapter | blocked_score | False | False |
| zhao_cui_generalized_sv_synthetic_from_estimated_values | blocked | blocked_no_reviewed_same_target_ledh_row_adapter | blocked_score | False | False |

## Nonclaims

- Frozen non-LEDH rows are copied from the July 3 baseline and were not rerun.
- Runtime cross-ranking between frozen non-LEDH rows and fresh LEDH rows is forbidden.
- Raw score-memory artifacts are not admitted unless they pass the Phase 1 shared compact score artifact contract.
- Historical manual_total_vjp score-memory artifacts are diagnostic-only and blocked from leaderboard score admission.
- HMC readiness, posterior correctness, and scientific superiority are not claimed.
