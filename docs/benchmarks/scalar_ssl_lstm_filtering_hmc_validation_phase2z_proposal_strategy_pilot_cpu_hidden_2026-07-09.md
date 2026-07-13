# Scalar SSL-LSTM Filtering HMC Validation Phase 2Z - Proposal Strategy Pilot

## Decision

- phase2z_proposal_strategy_pilot_passed: `True`
- pilot_artifact_valid: `True`
- candidate_nominated: `False`
- nominated_candidates: `[]`
- vetoes: `[]`
- zero_divergence_claim_made: `False`
- next_justified_action: write Phase 2Z result and draft/review SNIS-abandonment or transport/sequential-reference decision subplan

## Candidate Gate

- summary: `{'candidate_count': 4, 'nominated_candidate_count': 0, 'nominated_candidates': [], 'candidate_names': ['student_t_centered', 'student_t_shifted', 'anchor_mixture_student_t', 'ridge_line_student_t'], 'interpretation': 'no candidate passed the pilot nomination screen'}`

## Candidate Rows

| candidate | nominated | ESS | ESS ratio | max weight | failures | hard vetoes |
| --- | --- | --- | --- | --- | --- | --- |
| student_t_centered | `False` | `21.01706760314462` | `0.02052448008119592` | `0.10525616575524867` | `['ess_below_nomination_screen', 'ess_ratio_below_nomination_screen', 'max_weight_above_nomination_screen']` | `[]` |
| student_t_shifted | `False` | `14.656246107251146` | `0.014312740339112447` | `0.2428262178992962` | `['ess_below_nomination_screen', 'ess_ratio_below_nomination_screen', 'max_weight_above_nomination_screen']` | `[]` |
| anchor_mixture_student_t | `False` | `26.071556547207543` | `0.025460504440632366` | `0.1112365984074613` | `['ess_below_nomination_screen', 'ess_ratio_below_nomination_screen', 'max_weight_above_nomination_screen']` | `[]` |
| ridge_line_student_t | `False` | `1.535816743965212` | `0.0014998210390285273` | `0.7997380494257067` | `['ess_below_nomination_screen', 'ess_ratio_below_nomination_screen', 'max_weight_above_nomination_screen']` | `[]` |

## Inference Status

| field | value |
| --- | --- |
| hard_veto_screen | passed |
| reference_validity | not assessed; Phase 2Z is pilot nomination only |
| hmc_reference_agreement | not assessed |
| native_divergence | not assessed; no HMC run |
| zero_divergence_claim | not made |
| statistically_supported_ranking | none |
| descriptive_only_differences | per-candidate ESS, ESS ratio, max weight, weighted moments, top weights, and runtime |
| posterior_correctness | not assessed |
| hmc_readiness | not assessed |
| gpu_xla_readiness | blocked |
| default_readiness | not assessed |
| next_evidence_needed | reviewed decision to abandon SNIS branch or move to transport/sequential reference |

## Run Manifest

| field | value |
| --- | --- |
| command | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 420 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_cpu_hidden_2026-07-09.md` |
| git | `{'commit': '52ee244498988e046a6356f926003b581103083b', 'dirty': True, 'dirty_line_count': 90, 'dirty_preview': [' M bayesfilter/inference/hmc_diagnostics.py', ' M tests/test_common_inference_runtime_contracts.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1r_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_2026_07_09.py', '?? docs/benchmarks/kalman_qr_analytic_vs_autodiff_score_scaling_cpu_xla_2026-07-09.json', '?? docs/benchmarks/kalman_qr_analytic_vs_autodiff_score_scaling_cpu_xla_2026-07-09.md', '?? docs/benchmarks/kalman_qr_parameter_count_scaling_cpu_xla_2026-07-09.json', '?? docs/benchmarks/kalman_qr_parameter_count_scaling_cpu_xla_2026-07-09.md', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.json', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.md']}` |
| environment | `{'python': '3.13.13', 'tensorflow': '2.20.0', 'cuda_visible_devices': '-1', 'cpu_hidden': True, 'tf_physical_devices': [{'name': '/physical_device:CPU:0', 'device_type': 'CPU'}], 'tf_logical_gpus': []}` |
| conda_env | `tfgpu` |
| cpu_gpu_status | CPU-hidden debug/reference exception |
| jit_compile | `False` |
| tf32_mode | disabled_by_cpu_hidden_debug_contract |
| random_seeds | `[[20260709, 6701], [20260709, 6702], [20260709, 6703], [20260709, 6704]]` |
| wall_time_seconds | `188.37495324999327` |
| output_artifacts | `['docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_cpu_hidden_2026-07-09.json', 'docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_cpu_hidden_2026-07-09.md']` |
| plan_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md` |
| subplan_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2z-proposal-strategy-pilot-subplan-2026-07-09.md` |
| result_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2z-proposal-strategy-pilot-result-2026-07-09.md` |

## Nonclaims

- Phase 2Z proposal-strategy pilot only
- candidate nomination is not an independent valid reference
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
