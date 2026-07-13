# Scalar SSL-LSTM Filtering HMC Validation Phase 2AD - Diversity-Preserving Sequential Repair

## Decision

- phase2ad_diversity_preserving_sequential_repair_passed: `False`
- candidate_nominated_for_phase2ae_replication: `False`
- vetoes: `['temperature_increment_stalled', 'temperature_schedule_did_not_reach_beta_one', 'terminal_beta_not_one']`
- terminal_beta: `0.9712250668187553`
- stage_count: `6`
- terminal_pre_final_resampling_ess_ratio: `0.5000016429467714`
- terminal_pre_final_resampling_max_weight: `0.03922774202134582`
- unique_ancestor_fraction: `0.4140625`
- aggregate_rejuvenation_acceptance: `0.6458333333333334`
- next_justified_action: write Phase 2AD result and draft a reference-method blocker or expansion decision

## Gate

- gate: `{'phase2ad_candidate_nominated': False, 'vetoes': ['temperature_increment_stalled', 'temperature_schedule_did_not_reach_beta_one', 'terminal_beta_not_one'], 'terminal_pre_final_resampling_ess_ratio': 0.5000016429467714, 'terminal_pre_final_resampling_max_weight': 0.03922774202134582, 'minimum_adaptive_post_temperature_ess_ratio': 0.5000010206048837, 'unique_ancestor_fraction': 0.4140625, 'aggregate_rejuvenation_acceptance': 0.6458333333333334, 'interpretation': 'no valid diversity-preserving sequential-reference nomination'}`

## Sequential Reference

- computed: `True`
- vetoes: `['temperature_increment_stalled', 'temperature_schedule_did_not_reach_beta_one']`
- terminal_pre_final_resampling_summary: `{'finite': True, 'ess': 64.00021029718674, 'ess_ratio': 0.5000016429467714, 'max': 0.03922774202134582, 'sum': 1.0, 'nonzero_count': 128}`
- final_weight_summary: `{'finite': True, 'ess': 64.00021029718674, 'ess_ratio': 0.5000016429467714, 'max': 0.03922774202134582, 'sum': 1.0, 'nonzero_count': 128}`
- terminal_weighted_moments: `{'computed': True, 'vetoes': [], 'mean_u_new': [1.8167175591586942, -0.4233498742882595, -0.9040788365665817, 2.235251426163586], 'std_u_new': [1.2676282087606368, 0.9618188077158364, 1.1829434340658822, 1.5049031843330305], 'second_moment_variance_u_new': [1.6068812756457005, 0.925095418875913, 1.3993551681995822, 2.264733594215695]}`

## Inference Status

| field | value |
| --- | --- |
| hard_veto_screen | failed |
| reference_validity | not established; Phase 2AD can only nominate Phase 2AE replication |
| hmc_reference_agreement | not assessed |
| statistically_supported_ranking | none |
| descriptive_only_differences | temperature schedule, ESS trajectory, ancestor diversity, rejuvenation acceptance, terminal moments, and runtime |
| posterior_correctness | not assessed |
| hmc_readiness | not assessed |
| gpu_xla_readiness | blocked |
| default_readiness | not assessed |
| zero_divergence_claim | not made |
| next_evidence_needed | reference-method blocker or expansion decision |

## Run Manifest

| field | value |
| --- | --- |
| command | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 600 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair_cpu_hidden_2026-07-09.md` |
| git | `{'commit': '52ee244498988e046a6356f926003b581103083b', 'dirty': True, 'dirty_line_count': 117, 'dirty_preview': [' M bayesfilter/inference/hmc_diagnostics.py', ' M tests/test_common_inference_runtime_contracts.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1r_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ab_transport_or_sequential_reference_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_2026_07_09.py', '?? docs/benchmarks/kalman_qr_analytic_vs_autodiff_score_scaling_cpu_xla_2026-07-09.json', '?? docs/benchmarks/kalman_qr_analytic_vs_autodiff_score_scaling_cpu_xla_2026-07-09.md', '?? docs/benchmarks/kalman_qr_parameter_count_scaling_cpu_xla_2026-07-09.json']}` |
| environment | `{'python': '3.13.13', 'tensorflow': '2.20.0', 'cuda_visible_devices': '-1', 'cpu_hidden': True, 'tf_physical_devices': [{'name': '/physical_device:CPU:0', 'device_type': 'CPU'}], 'tf_logical_gpus': []}` |
| conda_env | `tfgpu` |
| cpu_gpu_status | CPU-hidden debug/reference exception |
| jit_compile | `False` |
| tf32_mode | disabled_by_cpu_hidden_debug_contract |
| random_seeds | `[[20260709, 6801]]` |
| wall_time_seconds | `75.56550153600983` |
| output_artifacts | `['docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair_cpu_hidden_2026-07-09.json', 'docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair_cpu_hidden_2026-07-09.md']` |
| plan_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md` |
| subplan_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ad-diversity-preserving-sequential-repair-subplan-2026-07-09.md` |
| result_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ad-diversity-preserving-sequential-repair-result-2026-07-09.md` |

## Nonclaims

- Phase 2AD diversity-preserving sequential repair pilot only
- not a valid replicated reference by itself
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
