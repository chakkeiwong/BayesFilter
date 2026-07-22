# Two-Lane Highdim Leaderboard Result

Authoritative JSON artifact: `docs/benchmarks/artifacts/sgqf_whole_highdim_leaderboard_repair_20260722/attempt05/sgqf-column/result.json`.

## Executed / status cells

| Row | Algorithm | Status | Score status | Batch status | GPU/XLA status | Timing rank status | Avg loglik | Runtime s | MC SE | Reason |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| benchmark_lgssm_exact_oracle_m3_T50 | fixed_sgqf | executed_value_score | analytical_score_emitted | not_claimed_no_reviewed_batched_main_row_evaluator | not_claimed_no_trusted_row_specific_gpu_xla_manifest | not_ranked_by_phase7_timing | -2.721519 | n/a | n/a | SGQF score vector emitted by reviewed analytical fixed-branch score path |
| zhao_cui_sv_actual_nongaussian_T1000 | fixed_sgqf | executed_value_score | analytical_score_emitted | not_claimed_no_reviewed_batched_main_row_evaluator | not_claimed_no_trusted_row_specific_gpu_xla_manifest | not_ranked_by_phase7_timing | -2.303577 | n/a | n/a | SGQF score vector emitted by reviewed analytical fixed-branch score path |
| zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000 | fixed_sgqf | executed_value_score | analytical_score_emitted | not_claimed_no_reviewed_batched_main_row_evaluator | not_claimed_no_trusted_row_specific_gpu_xla_manifest | not_ranked_by_phase7_timing | -2.298854 | n/a | n/a | SGQF score vector emitted by reviewed analytical fixed-branch score path |
| zhao_cui_spatial_sir_austria_j9_T20 | fixed_sgqf | executed_value_only | not_applicable_no_free_theta | not_claimed_no_reviewed_batched_main_row_evaluator | not_claimed_no_trusted_row_specific_gpu_xla_manifest | not_ranked_by_phase7_timing | -34.568460 | 0.650148 | n/a | Fixed Zhao-Cui SIR row has parameter_dim=0; a score is not mathematically applicable. |
| zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale | fixed_sgqf | blocked | not_applicable_to_scoped_component_row | not_claimed_no_reviewed_batched_main_row_evaluator | not_claimed_no_trusted_row_specific_gpu_xla_manifest | not_ranked_by_phase7_timing | n/a | n/a | n/a | fixed_sgqf is not admitted for the scoped Zhao-Cui parameterized SIR local complete-data component row; the scoped row is a fixed-variant Zhao-Cui manual-score component cell. |
| zhao_cui_predator_prey_T20 | fixed_sgqf | executed_value_score | analytical_score_emitted | not_claimed_no_reviewed_batched_main_row_evaluator | not_claimed_no_trusted_row_specific_gpu_xla_manifest | not_ranked_by_phase7_timing | -5.131135 | n/a | n/a | SGQF score vector emitted by reviewed analytical fixed-branch score path |
| zhao_cui_generalized_sv_synthetic_from_estimated_values | fixed_sgqf | executed_value_score | analytical_score_emitted | not_claimed_no_reviewed_batched_main_row_evaluator | not_claimed_no_trusted_row_specific_gpu_xla_manifest | not_ranked_by_phase7_timing | -1.426580 | 3.739052 | n/a | SGQF score vector emitted by reviewed analytical fixed-branch score path |

## Row readiness summary

| Row | Scope | Executed algorithms | Value complete | Score complete | Comparison ready | SGQF complete | Full three-way ready (deprecated) | Scoped component ready | Blocked / missing algorithms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| benchmark_lgssm_exact_oracle_m3_T50 | main_observed_data_filtering_row | fixed_sgqf | True | True | False | True | False | False | none |
| zhao_cui_sv_actual_nongaussian_T1000 | main_observed_data_filtering_row | fixed_sgqf | True | True | False | True | False | False | none |
| zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000 | main_observed_data_filtering_row | fixed_sgqf | True | True | False | True | False | False | none |
| zhao_cui_spatial_sir_austria_j9_T20 | main_observed_data_filtering_row | fixed_sgqf | True | True | False | True | False | False | none |
| zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale | scoped_component_row | none | True | True | False | True | False | True | none |
| zhao_cui_predator_prey_T20 | main_observed_data_filtering_row | fixed_sgqf | True | True | False | True | False | False | none |
| zhao_cui_generalized_sv_synthetic_from_estimated_values | main_observed_data_filtering_row | fixed_sgqf | True | True | False | True | False | False | none |

SGQF column complete: `True`.

## Nonclaims

- This artifact executes only fixed_sgqf cells and does not rerun UKF or Zhao-Cui comparators.
- sgqf_column_complete is independent of three-way comparison readiness.
- A completed deterministic column does not establish exact nonlinear likelihoods, superiority, or statistically supported ranking.
- The parameterized SIR local complete-data component is not applicable to SGQF and is excluded from the column denominator.
