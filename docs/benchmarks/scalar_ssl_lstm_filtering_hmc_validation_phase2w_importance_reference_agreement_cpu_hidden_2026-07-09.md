# Scalar SSL-LSTM Filtering HMC Validation Phase 2W - Importance Reference Agreement

## Decision

- phase2w_importance_reference_agreement_passed: `False`
- reference_valid: `False`
- agreement_passed: `False`
- vetoes: `['reference_ess_below_threshold', 'reference_ess_ratio_below_threshold']`
- reference_ess: `22.894679726459746`
- reference_ess_ratio: `0.022358085670370845`
- zero_divergence_claim_made: `False`
- next_justified_action: write Phase 2W blocker or narrower reference/tuning/localization repair result

## Reference

- valid: `False`
- vetoes: `['reference_ess_below_threshold', 'reference_ess_ratio_below_threshold']`
- mean_u_new: `[0.16900152112527375, 0.34590014590251295, 0.47216707577215133, -0.3362900480743778]`
- std_u_new: `[1.1289232726542155, 1.3947178163994365, 1.7877962561383989, 1.7764811837333756]`
- mean_mcse_u_new: `[0.23593759044489337, 0.2914869140558837, 0.37363774057733806, 0.3712729643488579]`
- weight_summary: `{'min': 9.592932218905643e-35, 'max': 0.17637097762827186, 'sum': 1.0000000000000002, 'nonzero_count': 1024}`
- log_weight_summary: `{'shape': [1024], 'finite_count': 1024, 'nonfinite_count': 0, 'min': -104.0199706824082, 'max': -27.425684702606866, 'mean': -34.95266416562103, 'max_abs': 104.0199706824082}`

## HMC Agreement

- evaluated: `False`
- passed: `None`
- vetoes: `['phase2w_agreement_not_evaluated']`
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
| statistically_supported_ranking | none; one HMC chain compared to one fixed importance reference with no ranking |
| descriptive_only_differences | reference ESS, weighted moments, HMC-reference deltas, std ratios, and runtime |
| posterior_correctness | not assessed |
| hmc_readiness | not assessed; Phase 2W is a narrow reference-agreement diagnostic |
| gpu_xla_readiness | blocked until a later reviewed GPU/XLA reproduction phase |
| default_readiness | not assessed |
| next_evidence_needed | reviewed reference/tuning/localization repair |

## Run Manifest

| field | value |
| --- | --- |
| command | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 420 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.md` |
| git | `{'commit': 'f297b103303c64019302ed5d9b9aaf2c8f919b64', 'dirty': True, 'dirty_line_count': 68, 'dirty_preview': [' M bayesfilter/inference/hmc_diagnostics.py', ' M bayesfilter/linear/kalman_qr_tf.py', ' M tests/test_common_inference_runtime_contracts.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1r_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_2026_07_09.py', '?? docs/benchmarks/kalman_qr_analytic_vs_autodiff_score_scaling_cpu_hidden_2026-07-09.json', '?? docs/benchmarks/kalman_qr_analytic_vs_autodiff_score_scaling_cpu_hidden_2026-07-09.md', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.json', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.md', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.json', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.md', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_cpu_hidden_2026-07-09.json', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_cpu_hidden_2026-07-09.md']}` |
| environment | `{'python': '3.13.13', 'tensorflow': '2.20.0', 'cuda_visible_devices': '-1', 'cpu_hidden': True, 'tf_physical_devices': [{'name': '/physical_device:CPU:0', 'device_type': 'CPU'}], 'tf_logical_gpus': []}` |
| conda_env | `tfgpu` |
| cpu_gpu_status | CPU-hidden debug/reference exception |
| jit_compile | `False` |
| tf32_mode | disabled_by_cpu_hidden_debug_contract |
| random_seeds | `[[20260709, 6501]]` |
| wall_time_seconds | `81.64405142096803` |
| output_artifacts | `['docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.json', 'docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.md']` |
| plan_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md` |
| subplan_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2w-importance-reference-agreement-subplan-2026-07-09.md` |
| result_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2w-importance-reference-agreement-result-2026-07-09.md` |

## Review Record

- reviewer: Codex substitute reviewer
- review_strength: weaker_than_full_claude_material_review
- claude_status: unavailable_for_repo_context_material_review_per_prior_handoff

## Nonclaims

- Phase 2W MAP-local self-normalized importance-reference agreement diagnostic only
- not HMC readiness evidence
- not HMC convergence evidence
- not posterior correctness evidence
- not a zero-divergence claim when native divergence is unavailable
- not sampler superiority evidence
- not statistically supported ranking evidence
- not GPU/XLA production-readiness evidence
- not default-readiness evidence
- not Zhao-Cui source-faithfulness evidence
