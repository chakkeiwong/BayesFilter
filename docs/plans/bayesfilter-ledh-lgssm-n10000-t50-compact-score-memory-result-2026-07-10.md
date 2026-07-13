# Same-Target LEDH LGSSM m3 T50 Value Runner

- JSON artifact: `docs/plans/ledh-lgssm-n10000-t50-compact-score-memory-2500-2026-07-10.json`
- Row: `benchmark_lgssm_exact_oracle_m3_T50`
- Target identity: `{'row_id': 'benchmark_lgssm_exact_oracle_m3_T50', 'row_scope': 'main_observed_data_filtering_row', 'forward_contract': {'schema_version': 'bayesfilter.highdim.ledh_forward_contract.v1', 'row_id': 'benchmark_lgssm_exact_oracle_m3_T50', 'row_scope': 'main_observed_data_filtering_row', 'target_scalar': 'observed_data_log_likelihood_estimator', 'output_tensor_field': 'log_likelihood', 'target_density_fields': ['transition_log_density', 'observation_log_density'], 'proposal_flow_fields': ['pre_flow_log_density', 'forward_log_det', 'proposal_observation_surface'], 'correction_formula': 'transition_log_density + observation_log_density - pre_flow_log_density + forward_log_det', 'estimator_kind': 'finite_N_fixed_randomness_ledh_log_likelihood_estimator', 'value_status': 'same_target_value_metadata_present', 'score_status': 'score_requires_no_tape_same_scalar_gate', 'full_leaderboard_row': False, 'theta_contract': {'row_id': 'benchmark_lgssm_exact_oracle_m3_T50', 'theta_coordinate_system': 'physical_benchmark_exact_oracle', 'theta_dimension': 5, 'parameter_order': ['phi1', 'phi2', 'phi3', 'q_scale', 'r_scale'], 'truth_theta': [0.72, 0.55, 0.35, 0.35, 0.45], 'theta_status': 'phase1_frozen', 'source_classification': 'benchmark_exact_oracle_row', 'nonclaims': []}, 'metadata': {'time_steps': 50, 'num_particles': 10000, 'batch_seeds': [81120]}, 'nonclaims': ['metadata alone is not row admission', 'metadata alone is not score evidence']}, 'target_scalar': 'observed_data_log_likelihood_estimator', 'target_output_tensor_field': 'log_likelihood', 'target_density_fields': ['transition_log_density', 'observation_log_density'], 'proposal_flow_fields': ['pre_flow_log_density', 'forward_log_det', 'proposal_observation_surface'], 'correction_formula': 'transition_log_density + observation_log_density - pre_flow_log_density + forward_log_det', 'dataset_seed': 81100, 'truth_theta': [0.72, 0.55, 0.35, 0.35, 0.45], 'time_steps': 50, 'num_particles': 10000, 'batch_seeds': [81120], 'transport_policy': 'active-all', 'sinkhorn_iterations': 10, 'sinkhorn_epsilon': 0.5, 'full_row_expected': {'batch_seeds': [81120, 81121, 81122, 81123, 81124], 'num_particles': 10000, 'time_steps': 50, 'transport_policy': 'active-all', 'sinkhorn_iterations': 10, 'sinkhorn_epsilon': 0.5}, 'state_dim': 3, 'obs_dim': 3, 'full_leaderboard_row': False, 'same_target_status': 'prefix_diagnostic_not_full_leaderboard_row', 'dataset_source': 'scripts/filtering_value_gradient_benchmark_generate_p8_datasets.py:_lgssm_dataset', 'model_source': 'scripts/filtering_value_gradient_benchmark_generate_p8_datasets.py:_lgssm_benchmark_model', 'exact_value_comparator': 'tf_kalman_log_likelihood on same observations/model', 'exact_total_log_likelihood': -136.0759748579247, 'exact_average_log_likelihood': -2.721519497158494, 'score_status': 'blocked_score_only_diagnostic_not_admitted', 'score_admission_status': 'blocked_score_diagnostic_stage_not_admitted'}`
- Comparison status: `executed_prefix_value_diagnostic_score_blocked`
- Value status: `executed_prefix_value_not_full_row`
- Score status: `blocked_score_only_diagnostic_not_admitted`
- Shape: `{'batch_size': 1, 'batch_seed_count': 1, 'time_steps': 50, 'num_particles': 10000, 'state_dim': 3, 'obs_dim': 3}`
- Output devices: `['/job:localhost/replica:0/task:0/device:GPU:0']`
- Precision: `{'precision_default_policy': 'production_ledh_pfpf_ot_gpu_tf32', 'default_execution_target': 'gpu', 'default_algorithm_target': 'ledh_pfpf_ot_tf32', 'default_target_status': 'production_default_by_owner_directive', 'default_route_acceptance': 'accepted_default_use_whenever_possible', 'default_route_guidance': 'use for BayesFilter DPF LEDH-PFPF-OT work whenever GPU execution and the streaming fixed-branch contract are applicable', 'default_route_rationale': 'streaming GPU TF32 LEDH-PFPF-OT avoids dense transport/history storage for large-particle DPF transport while preserving explicit reference and fallback arms', 'default_dtype': 'float32', 'active_dtype': 'float32', 'default_tf32_mode': 'enabled', 'fp64_reference_requires_explicit_dtype': True, 'scope': 'production_ledh_pfpf_ot_gpu_tf32_default_lane', 'historical_module_path': 'experiments/dpf_implementation', 'public_api_exposure': 'separately_gated', 'dtype': 'float32', 'tf_dtype': 'float32', 'tf32_mode': 'enabled', 'tf32_execution_enabled': True}`
- Compile plus first call seconds: `13.226236717076972`
- Warm-call timing summary seconds: `{'min': 3.7999129958916456, 'median': 3.7999129958916456, 'mean': 3.7999129958916456, 'max': 3.7999129958916456}`
- Total log likelihood mean: `-135.96009826660156`
- Total log likelihood SD: `None`
- Total log likelihood MCSE: `None`
- Average log likelihood mean: `-2.719201965332031`
- Average log likelihood MCSE: `None`
- Exact Kalman total log likelihood: `-136.0759748579247`
- Exact Kalman average log likelihood: `-2.721519497158494`
- Average delta to exact: `0.002317531826462993`
- Average relative error: `0.0008515580464820113`
- Finite output: `True`

## Nonclaims

- not exact Kalman score evidence
- not HMC/NUTS readiness evidence
- not posterior correctness evidence
- not runtime-rankable against frozen non-LEDH rows
- not evidence for nonlinear rows
- score-only diagnostic stage is not score admission; score and same-scalar FD must be combined before validation

## Score Memory And Time Result

Plan:
`docs/plans/bayesfilter-ledh-lgssm-n10000-t50-compact-score-memory-plan-2026-07-10.md`

JSON artifact:
`docs/plans/ledh-lgssm-n10000-t50-compact-score-memory-2500-2026-07-10.json`

Command scope:

- `N=10000`;
- `T=50`;
- one seed, `81120`;
- `row_chunk_size=2500`;
- `col_chunk_size=2500`;
- `particle_chunk_size=2500`;
- `score_mode=compact-sensitivity`;
- `score_diagnostic_stage=score-only`;
- `dtype=float32`;
- TF32 enabled;
- trusted GPU execution on `/GPU:0`.

Route checks:

- `score_route`: `compact_forward_sensitivity_no_autodiff_same_scalar_lgssm_ledh_pfpf_ot`;
- `manual_score_diagnostic.uses_full_history_reverse_route`: `false`;
- `score_output_devices`: `/job:localhost/replica:0/task:0/device:GPU:0`;
- `transport.plan_mode`: `streaming`;
- `transport.dense_transport_matrix_materialized`: `false`.

Measured memory and time:

| Setting | Score peak bytes | Score peak MiB | Value peak MiB | Elapsed seconds | Compile plus first call seconds | Warm call seconds |
|---|---:|---:|---:|---:|---:|---:|
| `N=10000,T=10,2500/2500/2500` | `724787456` | `691.211` | `83.865` | `208.010` | `11.514` | `0.682` |
| `N=10000,T=50,2500/2500/2500` | `754629632` | `719.671` | `83.865` | `674.335` | `13.226` | `3.800` |

Score vector:

`[9.913432121276855, -3.191894292831421, 0.3056414723396301, 11.38205337524414, 12.649811744689941]`

Value comparison:

- LEDH total log likelihood estimate: `-135.96009826660156`;
- exact Kalman total log likelihood: `-136.0759748579247`;
- total absolute delta: `0.11587659132314343`;
- average relative error: `0.0008515580464819656`.

During the run, an external trusted `nvidia-smi` status check observed process
reservation near `15776 MiB`. This is not the score-phase TensorFlow
reset-memory-stats peak. The durable score-phase peak recorded by the artifact
after TensorFlow memory-stat reset is `754629632` bytes, or about `719.671 MiB`.

## Decision Table

| Decision item | Status |
|---|---|
| Terminal artifact emitted | Passed |
| Score-only computation completed | Passed |
| Compact route used | Passed |
| Historical full-history reverse route avoided | Passed |
| GPU execution used | Passed |
| Float32 plus TF32 policy used | Passed |
| Score memory peak recorded | Passed |
| Score admission | Not claimed; blocked by score-only diagnostic stage |
| Same-scalar FD correctness | Not claimed; not run in this diagnostic |

## Interpretation

The `N=10000,T=50` compact LGSSM score computation completed under the same
streaming route as the `T=10` diagnostic. The TensorFlow score-phase peak grew
only modestly from about `691 MiB` at `T=10` to about `720 MiB` at `T=50`, while
wall time increased from about `208 s` to about `674 s`.
