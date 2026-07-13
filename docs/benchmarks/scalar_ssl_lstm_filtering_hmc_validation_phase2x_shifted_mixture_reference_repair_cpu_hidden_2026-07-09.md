# Scalar SSL-LSTM Filtering HMC Validation Phase 2X - Shifted-Mixture Reference Repair

## Decision

- phase2x_shifted_mixture_reference_repair_passed: `False`
- reference_valid: `False`
- agreement_passed: `False`
- vetoes: `['reference_ess_below_threshold', 'reference_ess_ratio_below_threshold']`
- reference_ess: `33.4215730897076`
- reference_ess_ratio: `0.01631912748520879`
- zero_divergence_claim_made: `False`
- next_justified_action: write Phase 2X blocker or narrower reference/tuning/localization repair result

## Proposal

- component_counts: `{'standard': 512, 'shifted': 1536}`
- component_weights: `{'standard': 0.25, 'shifted': 0.75}`
- pilot_center: `[0.16900152112527375, 0.34590014590251295, 0.47216707577215133, -0.3362900480743778]`
- shifted_scale: `[1.4111540908177695, 1.7433972704992957, 2.2347453201729985, 2.2206014796667195]`
- proposal_log_prob_summary: `{'shape': [2048], 'finite_count': 2048, 'nonfinite_count': 0, 'min': -16.535147597514715, 'max': -4.903169624575911, 'mean': -7.702224466988229, 'max_abs': 16.535147597514715}`

## Reference

- valid: `False`
- vetoes: `['reference_ess_below_threshold', 'reference_ess_ratio_below_threshold']`
- mean_u_new: `[-0.043945514545527586, -0.03297944040803433, -0.2764198156051569, 0.091681845356501]`
- std_u_new: `[2.1533163913207503, 2.762122672572576, 3.1974869281363474, 2.308167189648201]`
- mean_mcse_u_new: `[0.37247266183744154, 0.4777817084017203, 0.5530893983409825, 0.3992581770887797]`
- weight_summary: `{'min': 5.48049544167258e-171, 'max': 0.11446321229118656, 'sum': 1.0, 'nonzero_count': 2048}`
- log_weight_summary: `{'shape': [2048], 'finite_count': 2048, 'nonfinite_count': 0, 'min': -416.04885854676746, 'max': -26.175504948369124, 'mean': -39.25646177923905, 'max_abs': 416.04885854676746}`

## HMC Agreement

- evaluated: `False`
- passed: `None`
- vetoes: `['phase2x_agreement_not_evaluated']`
- mean_abs_delta: `None`
- mean_threshold: `None`
- std_ratio: `None`

## Inference Status

| field | value |
| --- | --- |
| hard_veto_screen | failed |
| reference_validity | failed |
| hmc_reference_agreement | failed or not interpreted |
| native_divergence | native divergence unavailable; unavailable is not zero divergences |
| zero_divergence_claim | not made |
| statistically_supported_ranking | none; one HMC chain compared to one repaired importance reference with no ranking |
| descriptive_only_differences | reference ESS, weighted moments, HMC-reference deltas, std ratios, and runtime |
| posterior_correctness | not assessed |
| hmc_readiness | not assessed; Phase 2X is a narrow reference-repair diagnostic |
| gpu_xla_readiness | blocked |
| default_readiness | not assessed |
| next_evidence_needed | reviewed reference/tuning/localization repair |

## Run Manifest

| field | value |
| --- | --- |
| command | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 600 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.md` |
| git | `{'commit': 'f297b103303c64019302ed5d9b9aaf2c8f919b64', 'dirty': True, 'dirty_line_count': 74, 'dirty_preview': [' M bayesfilter/inference/hmc_diagnostics.py', ' M bayesfilter/linear/kalman_qr_tf.py', ' M tests/test_common_inference_runtime_contracts.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1r_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_2026_07_09.py', '?? docs/benchmarks/kalman_qr_analytic_vs_autodiff_score_scaling_cpu_hidden_2026-07-09.json', '?? docs/benchmarks/kalman_qr_analytic_vs_autodiff_score_scaling_cpu_hidden_2026-07-09.md', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.json', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.md', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.json', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.md', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_cpu_hidden_2026-07-09.json']}` |
| environment | `{'python': '3.13.13', 'tensorflow': '2.20.0', 'cuda_visible_devices': '-1', 'cpu_hidden': True, 'tf_physical_devices': [{'name': '/physical_device:CPU:0', 'device_type': 'CPU'}], 'tf_logical_gpus': []}` |
| conda_env | `tfgpu` |
| cpu_gpu_status | CPU-hidden debug/reference exception |
| jit_compile | `False` |
| tf32_mode | disabled_by_cpu_hidden_debug_contract |
| random_seeds | `[[20260709, 6601]]` |
| wall_time_seconds | `113.19665156002156` |
| output_artifacts | `['docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.json', 'docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.md']` |
| plan_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md` |
| subplan_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2x-shifted-mixture-reference-repair-subplan-2026-07-09.md` |
| result_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2x-shifted-mixture-reference-repair-result-2026-07-09.md` |

## Nonclaims

- Phase 2X shifted-mixture importance-reference repair diagnostic only
- not HMC readiness evidence
- not HMC convergence evidence
- not posterior correctness evidence
- not a zero-divergence claim when native divergence is unavailable
- not sampler superiority evidence
- not statistically supported ranking evidence
- not GPU/XLA production-readiness evidence
- not default-readiness evidence
- not Zhao-Cui source-faithfulness evidence
