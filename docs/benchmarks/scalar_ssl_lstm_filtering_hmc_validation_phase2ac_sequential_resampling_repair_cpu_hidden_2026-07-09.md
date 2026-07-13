# Scalar SSL-LSTM Filtering HMC Validation Phase 2AC - Sequential Resampling Repair

## Decision

- phase2ac_sequential_resampling_repair_passed: `False`
- candidate_nominated_for_phase2ad_replication: `False`
- vetoes: `['unique_ancestor_fraction_below_threshold']`
- terminal_beta: `1.0`
- stage_count: `7`
- terminal_pre_final_resampling_ess_ratio: `0.9912539055044092`
- terminal_pre_final_resampling_max_weight: `0.010002188339361427`
- unique_ancestor_fraction: `0.21875`
- aggregate_rejuvenation_acceptance: `0.6529017857142857`
- next_justified_action: write Phase 2AC result and draft/reference-method blocker or focused repair subplan

## Gate

- gate: `{'phase2ac_candidate_nominated': False, 'vetoes': ['unique_ancestor_fraction_below_threshold'], 'terminal_pre_final_resampling_ess_ratio': 0.9912539055044092, 'terminal_pre_final_resampling_max_weight': 0.010002188339361427, 'minimum_adaptive_post_temperature_ess_ratio': 0.5000010206048837, 'unique_ancestor_fraction': 0.21875, 'aggregate_rejuvenation_acceptance': 0.6529017857142857, 'interpretation': 'no valid patched sequential-reference nomination'}`

## Sequential Reference

- computed: `True`
- vetoes: `[]`
- terminal_pre_final_resampling_summary: `{'finite': True, 'ess': 126.88049990456437, 'ess_ratio': 0.9912539055044092, 'max': 0.010002188339361427, 'sum': 0.9999999999999999, 'nonzero_count': 128}`
- final_weight_summary: `{'finite': True, 'ess': 126.88049990456437, 'ess_ratio': 0.9912539055044092, 'max': 0.010002188339361427, 'sum': 0.9999999999999999, 'nonzero_count': 128}`
- terminal_weighted_moments: `{'computed': True, 'vetoes': [], 'mean_u_new': [1.9326217084098882, -0.4307894645725953, -1.0886874903436403, 2.2970667840407772], 'std_u_new': [1.3300160079645396, 1.066639919530583, 1.2114330483776337, 1.4598969694429587], 'second_moment_variance_u_new': [1.7689425814419302, 1.1377207179362085, 1.4675700307015263, 2.1312991613887347]}`

## Inference Status

| field | value |
| --- | --- |
| hard_veto_screen | passed |
| reference_validity | not established; Phase 2AC can only nominate Phase 2AD replication |
| hmc_reference_agreement | not assessed |
| statistically_supported_ranking | none |
| descriptive_only_differences | temperature schedule, ESS trajectory, ancestor diversity, rejuvenation acceptance, terminal moments, and runtime |
| posterior_correctness | not assessed |
| hmc_readiness | not assessed |
| gpu_xla_readiness | blocked |
| default_readiness | not assessed |
| zero_divergence_claim | not made |
| next_evidence_needed | reviewed focused repair or reference-method blocker |

## Run Manifest

| field | value |
| --- | --- |
| command | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 600 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_cpu_hidden_2026-07-09.md` |
| git | `{'commit': '52ee244498988e046a6356f926003b581103083b', 'dirty': True, 'dirty_line_count': 109, 'dirty_preview': [' M bayesfilter/inference/hmc_diagnostics.py', ' M tests/test_common_inference_runtime_contracts.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1r_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ab_transport_or_sequential_reference_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_2026_07_09.py', '?? docs/benchmarks/kalman_qr_analytic_vs_autodiff_score_scaling_cpu_xla_2026-07-09.json', '?? docs/benchmarks/kalman_qr_analytic_vs_autodiff_score_scaling_cpu_xla_2026-07-09.md', '?? docs/benchmarks/kalman_qr_parameter_count_scaling_cpu_xla_2026-07-09.json', '?? docs/benchmarks/kalman_qr_parameter_count_scaling_cpu_xla_2026-07-09.md']}` |
| environment | `{'python': '3.13.13', 'tensorflow': '2.20.0', 'cuda_visible_devices': '-1', 'cpu_hidden': True, 'tf_physical_devices': [{'name': '/physical_device:CPU:0', 'device_type': 'CPU'}], 'tf_logical_gpus': []}` |
| conda_env | `tfgpu` |
| cpu_gpu_status | CPU-hidden debug/reference exception |
| jit_compile | `False` |
| tf32_mode | disabled_by_cpu_hidden_debug_contract |
| random_seeds | `[[20260709, 6801]]` |
| wall_time_seconds | `80.28042639500927` |
| output_artifacts | `['docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_cpu_hidden_2026-07-09.json', 'docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_cpu_hidden_2026-07-09.md']` |
| plan_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md` |
| subplan_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ac-sequential-resampling-repair-subplan-2026-07-09.md` |
| result_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ac-sequential-resampling-repair-result-2026-07-09.md` |

## Nonclaims

- Phase 2AC sequential-resampling repair pilot only
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
