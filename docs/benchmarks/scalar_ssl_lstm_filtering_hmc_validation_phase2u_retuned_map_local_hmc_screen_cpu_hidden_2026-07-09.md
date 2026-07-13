# Scalar SSL-LSTM Filtering HMC Validation Phase 2U - Retuned MAP-Local Screen

## Decision

- phase2u_retuned_map_local_hmc_screen_passed: `True`
- vetoes: `[]`
- selected_candidate: `{'candidate_index': 0, 'num_leapfrog_steps': 2, 'step_size': 0.785, 'trajectory_length_L_times_epsilon': 1.57, 'acceptance_rate': 0.34375, 'selection_policy': 'first_passing_candidate_in_predeclared_order'}`
- passed_candidate_count: `4` / `4`
- zero_divergence_claim_made: `False`
- next_justified_action: write Phase 2U result and draft/review longer selected-kernel MAP-local screen subplan

## Candidate Gate

- acceptance rates: `[0.34375, 0.546875, 0.96875, 0.984375]`
- acceptance envelope: `{'lower_exclusive': 0.05, 'upper_exclusive': 0.99}`
- selection policy: first_passing_candidate_in_predeclared_order

## Candidate Rows

| candidate | L | step | trajectory | seed | status | acceptance | hard vetoes | native divergence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2 | 0.785 | 1.57 | [20260709, 6301] | passed_hard_vetoes | 0.34375 | none | {'available': False, 'status': 'not_exposed_by_kernel', 'nonclaim': 'unavailable native divergence telemetry is not zero divergences'} |
| 1 | 4 | 0.3925 | 1.57 | [20260709, 6302] | passed_hard_vetoes | 0.546875 | none | {'available': False, 'status': 'not_exposed_by_kernel', 'nonclaim': 'unavailable native divergence telemetry is not zero divergences'} |
| 2 | 8 | 0.19625 | 1.57 | [20260709, 6303] | passed_hard_vetoes | 0.96875 | none | {'available': False, 'status': 'not_exposed_by_kernel', 'nonclaim': 'unavailable native divergence telemetry is not zero divergences'} |
| 3 | 16 | 0.098125 | 1.57 | [20260709, 6304] | passed_hard_vetoes | 0.984375 | none | {'available': False, 'status': 'not_exposed_by_kernel', 'nonclaim': 'unavailable native divergence telemetry is not zero divergences'} |

## Inference Status

| field | value |
| --- | --- |
| hard_veto_screen | passed |
| native_divergence | native divergence unavailable for at least one candidate; unavailable is not zero divergences |
| zero_divergence_claim | not made |
| statistically_supported_ranking | none; fixed short grid with no uncertainty analysis |
| descriptive_only_differences | per-candidate acceptance, target-log-prob range, log-accept range, sample range, and runtime |
| posterior_correctness | not assessed |
| hmc_readiness | not assessed; Phase 2U finite/acceptance screen only |
| gpu_xla_readiness | blocked |
| default_readiness | not assessed |
| next_evidence_needed | reviewed longer selected-kernel MAP-local screen |

## Run Manifest

| field | value |
| --- | --- |
| command | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 720 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_cpu_hidden_2026-07-09.md` |
| git | `{'commit': 'f297b103303c64019302ed5d9b9aaf2c8f919b64', 'dirty': True, 'dirty_line_count': 56, 'dirty_preview': [' M bayesfilter/inference/hmc_diagnostics.py', ' M bayesfilter/linear/kalman_qr_tf.py', ' M tests/test_common_inference_runtime_contracts.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1r_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_2026_07_09.py', '?? docs/benchmarks/kalman_qr_analytic_vs_autodiff_score_scaling_cpu_hidden_2026-07-09.json', '?? docs/benchmarks/kalman_qr_analytic_vs_autodiff_score_scaling_cpu_hidden_2026-07-09.md', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.json', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.md', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.json', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.md', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_cpu_hidden_2026-07-09.json', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_cpu_hidden_2026-07-09.md', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_cpu_hidden_2026-07-09.json', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_cpu_hidden_2026-07-09.md']}` |
| environment | `{'python': '3.13.13', 'tensorflow': '2.20.0', 'cuda_visible_devices': '-1', 'cpu_hidden': True, 'tf_physical_devices': [{'name': '/physical_device:CPU:0', 'device_type': 'CPU'}], 'tf_logical_gpus': []}` |
| conda_env | `tfgpu` |
| cpu_gpu_status | CPU-hidden debug/reference exception |
| jit_compile | `False` |
| tf32_mode | disabled_by_cpu_hidden_debug_contract |
| random_seeds | `[[20260709, 6301], [20260709, 6302], [20260709, 6303], [20260709, 6304]]` |
| wall_time_seconds | `610.7140235070256` |
| output_artifacts | `['docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_cpu_hidden_2026-07-09.json', 'docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_cpu_hidden_2026-07-09.md']` |
| plan_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md` |
| subplan_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2u-retuned-map-local-hmc-screen-subplan-2026-07-09.md` |
| result_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2u-retuned-map-local-hmc-screen-result-2026-07-09.md` |

## Nonclaims

- Phase 2U finite/acceptance MAP-local HMC screen only
- not HMC readiness evidence
- not HMC convergence evidence
- not posterior correctness evidence
- not a zero-divergence claim when native divergence is unavailable
- not sampler superiority evidence
- not statistically supported ranking evidence
- not GPU/XLA production-readiness evidence
- not default-readiness evidence
- not Zhao-Cui source-faithfulness evidence
