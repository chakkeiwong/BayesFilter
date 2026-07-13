# Scalar SSL-LSTM Filtering HMC Validation Phase 2Y - Target Geometry Localization

## Decision

- phase2y_target_geometry_localization_passed: `True`
- vetoes: `[]`
- artifact_bug_indicated: `False`
- proposal_family_mismatch_indicated: `True`
- zero_divergence_claim_made: `False`
- next_justified_action: write Phase 2Y result and draft/review proposal redesign or local-reference-abandonment subplan

## Anchor Summary

- anchor_count: `33`
- summary: `{'norm_summary': {'shape': [33], 'finite_count': 33, 'nonfinite_count': 0, 'min': 0.0, 'max': 9.082845253254979, 'mean': 5.015353348214025, 'max_abs': 9.082845253254979}, 'max_abs_summary': {'shape': [33], 'finite_count': 33, 'nonfinite_count': 0, 'min': 0.0, 'max': 7.771575910104696, 'mean': 4.02121060125833, 'max_abs': 7.771575910104696}, 'top_weight_count': 16, 'max_top_source_weight': 0.17637097762827186, 'source_counts': {'phase2s_center': 1, 'phase2w': 16, 'phase2x': 16}}`
- anchor_evaluation_summary: `{'target_log_prob_summary': {'shape': [33], 'finite_count': 33, 'nonfinite_count': 0, 'min': -123.07826202468586, 'max': -37.77528495512359, 'mean': -45.734027345235724, 'max_abs': 123.07826202468586}, 'target_delta_from_center_summary': {'shape': [33], 'finite_count': 33, 'nonfinite_count': 0, 'min': -85.30297706956227, 'max': 0.0, 'mean': -7.958742390112133, 'max_abs': 85.30297706956227}, 'score_norm_summary': {'shape': [33], 'finite_count': 33, 'nonfinite_count': 0, 'min': 2.4416858704074592e-11, 'max': 63.79285939037887, 'mean': 8.947955218322887, 'max_abs': 63.79285939037887}, 'norm_u_new_summary': {'shape': [33], 'finite_count': 33, 'nonfinite_count': 0, 'min': 0.0, 'max': 9.082845253254979, 'mean': 5.015353348214025, 'max_abs': 9.082845253254979}, 'target_minus_quadratic_summary': {'shape': [33], 'finite_count': 33, 'nonfinite_count': 0, 'min': -49.76041336791768, 'max': 33.577938956960324, 'mean': 6.657232764269001, 'max_abs': 49.76041336791768}}`

## Ray Summary

- profile_count: `32`
- summary: `{'target_log_prob_summary': {'shape': [192], 'finite_count': 192, 'nonfinite_count': 0, 'min': -157.35819178059398, 'max': -37.77528495512359, 'mean': -42.81871884612821, 'max_abs': 157.35819178059398}, 'target_minus_quadratic_summary': {'shape': [192], 'finite_count': 192, 'nonfinite_count': 0, 'min': -104.96420467633178, 'max': 55.366496093385635, 'mean': 3.3305450647719437, 'max_abs': 104.96420467633178}, 'radial_score_component_summary': {'shape': [160], 'finite_count': 160, 'nonfinite_count': 0, 'min': -69.3359196323118, 'max': 1.8361219790363434, 'mean': -2.7194268682863174, 'max_abs': 69.3359196323118}, 'endpoint_delta_from_center_summary': {'shape': [32], 'finite_count': 32, 'nonfinite_count': 0, 'min': -119.58290682547039, 'max': -0.5537529678243587, 'mean': -15.120139903652397, 'max_abs': 119.58290682547039}}`

## Orientation Diagnostic

- summary: `{'adapter_vs_row_formula_max_abs': 4.440892098500626e-16, 'wrong_column_vs_adapter_min_abs': 0.0, 'wrong_column_vs_adapter_max_abs': 8.881784197001252e-16, 'artifact_bug_indicated': False, 'display_string_ambiguous': True}`
- recorded_phase2s_coordinate_formula: `free = center_free_parameter_values + scale * (factor_z @ u_new)`
- phase2u_adapter_coordinate_formula: `free = center_free_parameter_values + u_new @ (diag(scale) @ factor_z).T`

## Proposal Log-Density Replay

- summary: `{'source_saved_replay_max_abs_delta': 0.0, 'artifact_bug_indicated': False, 'standard_log_prob_summary': {'shape': [33], 'finite_count': 33, 'nonfinite_count': 0, 'min': -44.92479308010693, 'max': -3.6757541328186907, 'mean': -18.291729287199825, 'max_abs': 44.92479308010693}, 'shifted_mixture_log_prob_summary': {'shape': [33], 'finite_count': 33, 'nonfinite_count': 0, 'min': -16.535147597514715, 'max': -4.854003982109466, 'mean': -10.372334212231099, 'max_abs': 16.535147597514715}}`

## Hypothesis Assessment

| hypothesis | status | evidence role |
| --- | --- | --- |
| H1_local_quadratic_trust_region_exceeded | supported_descriptively | explanatory_only |
| H2_tail_or_ridge_undercoverage | supported_descriptively | explanatory_only |
| H3_orientation_or_scaling_mismatch | not_supported | bug_localization_diagnostic |
| H4_proposal_log_density_correct_family_poor | proposal_density_replay_passed_family_mismatch_plausible | bug_localization_plus_explanatory |
| H5_local_not_global_map_locator | plausible_not_certified | explanatory_only |
| H6_quadratic_extrapolation_failure | supported_descriptively | explanatory_only |

- summary: `{'artifact_bug_indicated': False, 'proposal_family_mismatch_indicated': True, 'next_repair_hint': 'consider non-diagonal, heavy-tail, ridge/local mixture, transport proposal, or abandon SNIS reference branch'}`

## Inference Status

| field | value |
| --- | --- |
| hard_veto_screen | passed |
| reference_validity | not assessed; Phase 2Y does not build a reference |
| hmc_reference_agreement | not assessed |
| native_divergence | not assessed; no HMC run |
| zero_divergence_claim | not made |
| statistically_supported_ranking | none |
| descriptive_only_differences | anchor norms, target values, scores, proposal log densities, ray profiles, and quadratic residuals |
| posterior_correctness | not assessed |
| hmc_readiness | not assessed |
| gpu_xla_readiness | blocked |
| default_readiness | not assessed |
| next_evidence_needed | reviewed proposal redesign, transport/local mixture, or local-reference-abandonment subplan |

## Run Manifest

| field | value |
| --- | --- |
| command | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 300 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.md` |
| git | `{'commit': '52ee244498988e046a6356f926003b581103083b', 'dirty': True, 'dirty_line_count': 76, 'dirty_preview': [' M bayesfilter/inference/hmc_diagnostics.py', ' M tests/test_common_inference_runtime_contracts.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1r_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_2026_07_09.py', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.json', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.md', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.json', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.md', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_cpu_hidden_2026-07-09.json', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_cpu_hidden_2026-07-09.md', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_cpu_hidden_2026-07-09.json']}` |
| environment | `{'python': '3.13.13', 'tensorflow': '2.20.0', 'cuda_visible_devices': '-1', 'cpu_hidden': True, 'tf_physical_devices': [{'name': '/physical_device:CPU:0', 'device_type': 'CPU'}], 'tf_logical_gpus': []}` |
| conda_env | `tfgpu` |
| cpu_gpu_status | CPU-hidden debug/reference exception |
| jit_compile | `False` |
| tf32_mode | disabled_by_cpu_hidden_debug_contract |
| random_seeds | `N/A; replays deterministic saved proposal artifacts only` |
| wall_time_seconds | `54.949073967000004` |
| output_artifacts | `['docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.json', 'docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.md']` |
| plan_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md` |
| subplan_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2y-target-geometry-localization-subplan-2026-07-09.md` |
| result_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2y-target-geometry-localization-result-2026-07-09.md` |

## Review Record

- reviewer: Codex local substitute review
- review_strength: weaker_than_full_claude_material_review
- claude_status: blocked_by_approval_layer_external_data_transfer_risk

## Nonclaims

- Phase 2Y target-geometry localization diagnostic only
- not a new valid importance reference
- not HMC-vs-reference agreement evidence
- not an HMC run
- not HMC readiness evidence
- not HMC convergence evidence
- not posterior correctness evidence
- not a zero-divergence claim when native divergence is unavailable
- not sampler superiority evidence
- not statistically supported ranking evidence
- not GPU/XLA production-readiness evidence
- not default-readiness evidence
- not Zhao-Cui source-faithfulness evidence
