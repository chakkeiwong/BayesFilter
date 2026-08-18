# Generated BayesFilter Tuning Definition Inventory

Generated 2026-08-16 from checked-out source with Python `ast`. This is the completeness and behavioral-fingerprint appendix to `bayesfilter-tuning-function-audit-2026-08-16.md`; it does not replace semantic review of numerical formulas.

| Path | Line | Kind | Name | Description/fingerprint |
|---|---:|---|---|---|
| `bayesfilter/highdim/capacity_tuning.py` | 18 | class | `SignificantPlacePolicy` | doc: Significant-digit-prefix comparison policy. |
| `bayesfilter/highdim/capacity_tuning.py` | 30 | def | `__post_init__` | fingerprint: calls int, ValueError, math.isfinite, float |
| `bayesfilter/highdim/capacity_tuning.py` | 55 | def | `manifest_payload` | fingerprint: returns Dict |
| `bayesfilter/highdim/capacity_tuning.py` | 65 | def | `significant_place` | doc: Return the absolute place value of the first omitted digit. |
| `bayesfilter/highdim/capacity_tuning.py` | 77 | def | `_significant_prefix` | doc: Return a sign/exponent/leading-digit prefix, without rounding. |
| `bayesfilter/highdim/capacity_tuning.py` | 93 | def | `_scalar_comparison` | fingerprint: calls float, math.isfinite, str; returns Dict |
| `bayesfilter/highdim/capacity_tuning.py` | 143 | def | `compare_likelihood_values` | doc: Compare adjacent likelihood values and optionally their increments. |
| `bayesfilter/highdim/capacity_tuning.py` | 200 | def | `assert_frozen_scope_equal` | doc: Reject non-capacity drift using stable JSON equality. |
| `bayesfilter/highdim/capacity_tuning.py` | 212 | def | `nominate_capacity` | doc: Nominate the least-cost cell stable to degree and rank refinement. |
| `bayesfilter/highdim/ledh_tuning_registry.py` | 14 | class | `LEDHTuningRoute` | fingerprint: body AnnAssign, AnnAssign, AnnAssign, AnnAssign |
| `bayesfilter/highdim/ledh_tuning_registry.py` | 131 | def | `route_for_model` | fingerprint: calls len, KeyError; returns Subscript |
| `bayesfilter/highdim/ledh_tuning_registry.py` | 138 | def | `require_active_route_tuning` | fingerprint: calls route_for_model, ValueError; returns route |
| `bayesfilter/highdim/ledh_tuning_scope.py` | 12 | class | `LEDHTuningScope` | doc: All inputs that make a tuned LEDH finite program scope-specific. |
| `bayesfilter/highdim/ledh_tuning_scope.py` | 34 | def | `__post_init__` | fingerprint: calls ValueError, min |
| `bayesfilter/highdim/ledh_tuning_scope.py` | 57 | def | `as_dict` | fingerprint: calls asdict; returns asdict(...) |
| `bayesfilter/highdim/ledh_tuning_scope.py` | 61 | def | `scope_sha256` | fingerprint: calls json.dumps(...).encode, json.dumps, self.as_dict, hashlib.sha256(...).hexdigest, hashlib.sha256; returns hashlib.sha256(...).hexdigest(...) |
| `bayesfilter/highdim/ledh_tuning_scope.py` | 68 | def | `scope_from_mapping` | doc: Build an exact scope from a serialized selection or claim record. |
| `bayesfilter/highdim/ledh_tuning_scope.py` | 74 | def | `require_scope_match` | fingerprint: calls scope_from_mapping, ValueError |
| `bayesfilter/highdim/rank_budget.py` | 55 | class | `RankBudgetConfig` | doc: Inputs for a deterministic fixed-rank memory preflight. |
| `bayesfilter/highdim/rank_budget.py` | 74 | def | `__post_init__` | fingerprint: calls positive_fields.items, int, ValueError, tuple, any; loops For |
| `bayesfilter/highdim/rank_budget.py` | 121 | class | `RankBudgetForecast` | doc: Forecast for one rank candidate under a fixed budget config. |
| `bayesfilter/highdim/rank_budget.py` | 129 | def | `manifest_payload` | fingerprint: returns Dict |
| `bayesfilter/highdim/rank_budget.py` | 139 | class | `RankBudgetPreflight` | doc: Deterministic M2 rank-budget decision for one dimension. |
| `bayesfilter/highdim/rank_budget.py` | 150 | def | `__post_init__` | fingerprint: calls ValueError, self.status.startswith |
| `bayesfilter/highdim/rank_budget.py` | 169 | def | `manifest_payload` | fingerprint: calls dict, row.manifest_payload; returns Dict |
| `bayesfilter/highdim/rank_budget.py` | 195 | class | `P57RankComparatorEvidence` | doc: Comparator evidence for one fixed TT/SIRT source-route rank. |
| `bayesfilter/highdim/rank_budget.py` | 218 | def | `__post_init__` | fingerprint: calls int, ValueError, str(...).strip |
| `bayesfilter/highdim/rank_budget.py` | 270 | def | `has_comparator` | fingerprint: returns Compare |
| `bayesfilter/highdim/rank_budget.py` | 274 | def | `passes_value_tolerances` | fingerprint: calls float; returns BoolOp |
| `bayesfilter/highdim/rank_budget.py` | 289 | def | `passes_gradient_tolerances_or_not_required` | fingerprint: calls float; returns BoolOp |
| `bayesfilter/highdim/rank_budget.py` | 300 | def | `passes_promotion_rule` | fingerprint: returns BoolOp |
| `bayesfilter/highdim/rank_budget.py` | 306 | def | `manifest_payload` | fingerprint: returns Dict |
| `bayesfilter/highdim/rank_budget.py` | 329 | class | `P57SourceFaithfulRankSelectionResult` | doc: P57 source-route rank-policy result. |
| `bayesfilter/highdim/rank_budget.py` | 355 | def | `__post_init__` | fingerprint: calls ValueError, tuple, int; loops For |
| `bayesfilter/highdim/rank_budget.py` | 405 | def | `manifest_payload` | fingerprint: calls p57_rank_promotion_tolerances, row.manifest_payload; returns Dict |
| `bayesfilter/highdim/rank_budget.py` | 423 | def | `p57_rank_promotion_tolerances` | doc: Return the source-route rank-promotion tolerances from P57-M7. |
| `bayesfilter/highdim/rank_budget.py` | 442 | def | `p57_select_source_faithful_rank` | doc: Select a source-faithful fixed TT/SIRT rank from comparator evidence. |
| `bayesfilter/highdim/rank_budget.py` | 484 | def | `p57_fixed_ttsirt_memory_terms` | doc: Return memory terms that P57 rank budgets must account for. |
| `bayesfilter/highdim/rank_budget.py` | 490 | def | `state_memory_bytes` | doc: Return ``bytes * d * n * r^2`` for TT state-core storage. |
| `bayesfilter/highdim/rank_budget.py` | 497 | def | `step_memory_bytes` | doc: Return ``bytes * d * n * (R_eff * r)^2 * omega`` for one step. |
| `bayesfilter/highdim/rank_budget.py` | 511 | def | `rank_ceiling` | doc: Return the hard rank ceiling implied by the step memory cap. |
| `bayesfilter/highdim/rank_budget.py` | 524 | def | `evaluate_rank_budget` | doc: Evaluate candidate ranks and classify the memory preflight. |
| `bayesfilter/highdim/rank_budget.py` | 560 | def | `p52_spatial_sir_rank_budget_manifest` | doc: Build the P52-M2 manifest payload for spatial SIR rank preflight. |
| `bayesfilter/highdim/rank_budget.py` | 604 | class | `P53RankSelectionResult` | doc: Fixed-rank selection outcome for the admitted P53 scaling route. |
| `bayesfilter/highdim/rank_budget.py` | 629 | def | `__post_init__` | fingerprint: calls ValueError, self.status.startswith |
| `bayesfilter/highdim/rank_budget.py` | 668 | def | `manifest_payload` | fingerprint: calls int, dict, self.preflight.manifest_payload; returns Dict |
| `bayesfilter/highdim/rank_budget.py` | 690 | def | `p53_select_fixed_rank_from_admitted_route` | doc: Select a fixed rank from admitted P53 route metadata. |
| `bayesfilter/highdim/rank_budget.py` | 766 | def | `_require_positive_rank` | fingerprint: calls int, ValueError |
| `bayesfilter/highdim/rank_budget.py` | 771 | def | `_require_finite_nonnegative` | fingerprint: calls ValueError, float |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 33 | class | `AlgebraicCoordinateMap` | doc: Vectorized source-style algebraic map for unbounded physical states. |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 38 | def | `__post_init__` | fingerprint: calls tf.reshape, tf.convert_to_tensor, ValueError, int, bool |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 49 | def | `dimension` | fingerprint: calls int; returns int(...) |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 52 | def | `forward` | fingerprint: calls tf.convert_to_tensor, ValueError, tf.clip_by_value, tf.square, tf.math.rsqrt, tf.reduce_sum; returns Tuple |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 66 | def | `inverse` | fingerprint: calls tf.convert_to_tensor, ValueError, tf.math.rsqrt, tf.square, tf.reduce_sum, tf.math.log1p; returns Tuple |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 79 | def | `manifest_payload` | fingerprint: returns Dict |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 89 | class | `FrozenProposalAPFModel` | doc: Model contract required by the fixed-branch analytical evaluator. |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 92 | def | `parameter_dim` | fingerprint: body Expr |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 94 | def | `state_dim` | fingerprint: body Expr |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 96 | def | `observation_dim` | fingerprint: body Expr |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 98 | def | `frozen_apf_measure_id` | fingerprint: body Expr |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 100 | def | `frozen_apf_score_backend_id` | fingerprint: body Expr |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 102 | def | `initial_log_density` | fingerprint: body Expr |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 104 | def | `transition_log_density` | fingerprint: body Expr |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 112 | def | `observation_log_density` | fingerprint: body Expr |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 120 | def | `initial_log_density_parameter_score` | fingerprint: body Expr |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 124 | def | `transition_log_density_parameter_score` | fingerprint: body Expr |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 132 | def | `observation_log_density_parameter_score` | fingerprint: body Expr |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 140 | def | `manifest_payload` | fingerprint: body Expr |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 144 | class | `PreparedFrozenProposalBranch` | doc: Parameter-independent particles, genealogy, and proposal densities. |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 155 | def | `__post_init__` | fingerprint: calls tf.executing_eagerly, RuntimeError, tf.convert_to_tensor, TypeError; loops For |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 231 | def | `dtype` | fingerprint: returns self.states.dtype |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 235 | def | `time_steps` | fingerprint: calls int; returns int(...) |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 239 | def | `particle_count` | fingerprint: calls int; returns int(...) |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 243 | def | `state_dimension` | fingerprint: calls int; returns int(...) |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 247 | def | `observation_dimension` | fingerprint: calls int; returns int(...) |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 250 | def | `manifest_payload` | fingerprint: returns Dict |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 270 | class | `FrozenTTSIRTProposalCompilation` | doc: Offline TTSIRT mechanics compiled into a parameter-independent branch. |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 277 | def | `__post_init__` | fingerprint: calls isinstance, TypeError, len, ValueError, str, object.__setattr__ |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 285 | def | `compile_fixed_ttsirt_proposal_branch` | doc: Compile fixed TTSIRT maps using `(previous, current)` axis ordering. |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 531 | def | `combine_fixed_ttsirt_block_compilations` | doc: Compose independent block proposals under one shared ancestor law. |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 619 | def | `prepare_frozen_proposal_branch` | doc: Issue a repository-computed identity for one realized proposal branch. |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 641 | class | `FrozenProposalAPFProgram` | doc: A model and prepared branch bound to one finite value/score program. |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 648 | def | `__post_init__` | fingerprint: calls callable, TypeError, getattr, int, ValueError, self.model.parameter_dim; loops For |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 681 | def | `evaluate` | doc: Evaluate eagerly with the same TensorFlow core used by XLA. |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 687 | def | `compiled` | doc: Build the default XLA evaluator; non-JIT is an explicit debug exception. |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 700 | def | `evaluate` | fingerprint: calls _evaluate_core; returns _evaluate_core(...) |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 705 | def | `manifest_payload` | fingerprint: calls self.branch.manifest_payload, self.model.manifest_payload; returns Dict |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 717 | def | `prepare_frozen_proposal_apf_program` | doc: Bind the actual model implementation and branch into a program identity. |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 726 | def | `_evaluate_core` | fingerprint: calls int, model.parameter_dim, tf.math.log, tf.cast, _vector, model.initial_log_density; returns Dict; loops For |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 889 | def | `_theta_vector` | fingerprint: calls tf.convert_to_tensor, ValueError, int; returns parameters |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 898 | def | `_vector` | fingerprint: calls tf.ensure_shape, tf.convert_to_tensor; returns tf.ensure_shape(...) |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 904 | def | `_score_matrix` | fingerprint: calls tf.ensure_shape, tf.convert_to_tensor; returns tf.ensure_shape(...) |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 915 | def | `_all_finite` | fingerprint: calls tf.reduce_all, tf.math.is_finite, tf.stack; returns tf.reduce_all(...) |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 920 | def | `_require_all_finite` | fingerprint: calls bool, ValueError, tf.reduce_all(...).numpy, tf.reduce_all, tf.math.is_finite |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 925 | def | `_branch_fingerprint` | fingerprint: calls hashlib.sha256, _update_hash; returns digest.hexdigest(...); loops For |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 944 | def | `_program_fingerprint` | fingerprint: calls hashlib.sha256, _update_hash, type; returns digest.hexdigest(...) |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 959 | def | `_source_digest` | fingerprint: calls hashlib.sha256(...).hexdigest, hashlib.sha256, path.read_bytes; returns hashlib.sha256(...).hexdigest(...) |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | 963 | def | `_update_hash` | fingerprint: calls isinstance, digest.update, value.dtype.name.encode; raises TypeError(...) |
| `bayesfilter/inference/fixed_kernel_arm.py` | 25 | class | `FixedKernelArmConfig` | doc: Immutable arm controls; adaptation is intentionally not representable. |
| `bayesfilter/inference/fixed_kernel_arm.py` | 39 | def | `__post_init__` | fingerprint: calls str(...).strip, str, ValueError, int, float |
| `bayesfilter/inference/fixed_kernel_arm.py` | 83 | def | `payload` | fingerprint: calls bool; returns payload |
| `bayesfilter/inference/fixed_kernel_arm.py` | 103 | class | `FixedKernelArmResult` | doc: Public-safe arm result; samples and terminal state are discarded. |
| `bayesfilter/inference/fixed_kernel_arm.py` | 110 | def | `payload` | fingerprint: calls self.config.payload, dict; returns Dict |
| `bayesfilter/inference/fixed_kernel_arm.py` | 127 | def | `_valid_samples` | fingerprint: calls tf.cast, tf.convert_to_tensor, any, ValueError; returns valid |
| `bayesfilter/inference/fixed_kernel_arm.py` | 147 | def | `minimum_latent_ess` | doc: Compute a BayesFilter-owned coordinate ESS summary. |
| `bayesfilter/inference/fixed_kernel_arm.py` | 220 | def | `minimum_latent_ess_checkpoints` | doc: Summarize same-chain ESS prefixes without exposing coordinate vectors. |
| `bayesfilter/inference/fixed_kernel_arm.py` | 275 | def | `run_fixed_kernel_arm` | doc: Run one explicit-state, no-adaptation arm and discard its draws. |
| `bayesfilter/inference/fixed_trajectory_hmc_tuning_v2.py` | 29 | class | `FixedTrajectoryHMCV2CandidateResult` | doc: One finite-grid fixed-trajectory HMC v2 tuning candidate. |
| `bayesfilter/inference/fixed_trajectory_hmc_tuning_v2.py` | 43 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/fixed_trajectory_hmc_tuning_v2.py` | 59 | class | `FixedTrajectoryHMCV2TuningResult` | doc: Structured first-slice v2 tuning artifact for a tiny Gaussian fixture. |
| `bayesfilter/inference/fixed_trajectory_hmc_tuning_v2.py` | 74 | def | `selected_candidate` | fingerprint: returns Constant; loops For |
| `bayesfilter/inference/fixed_trajectory_hmc_tuning_v2.py` | 86 | def | `passed` | fingerprint: returns BoolOp |
| `bayesfilter/inference/fixed_trajectory_hmc_tuning_v2.py` | 89 | def | `payload` | fingerprint: calls candidate.payload; returns Dict |
| `bayesfilter/inference/fixed_trajectory_hmc_tuning_v2.py` | 109 | def | `run_tiny_gaussian_fixed_trajectory_hmc_tuning_v2` | doc: Select an explicit v2 fixed-trajectory HMC candidate. |
| `bayesfilter/inference/fixed_trajectory_hmc_tuning_v2.py` | 162 | def | `target_log_prob` | fingerprint: calls tf.convert_to_tensor, tf.reduce_sum, tf.square; returns BinOp |
| `bayesfilter/inference/fixed_trajectory_hmc_tuning_v2.py` | 176 | def | `trace_fn` | fingerprint: returns Dict |
| `bayesfilter/inference/fixed_trajectory_hmc_tuning_v2.py` | 278 | def | `_validate_acceptance_band` | fingerprint: calls len, ValueError, tuple, float; returns Tuple |
| `bayesfilter/inference/fixed_trajectory_hmc_tuning_v2.py` | 289 | def | `_validate_positive_float_grid` | fingerprint: calls tuple, float, ValueError, any, np.isfinite; returns result |
| `bayesfilter/inference/fixed_trajectory_hmc_tuning_v2.py` | 298 | def | `_validate_positive_int_grid` | fingerprint: calls tuple, int, ValueError, any; returns result |
| `bayesfilter/inference/fixed_trajectory_hmc_tuning_v2.py` | 307 | def | `_select_fixed_trajectory_candidate` | fingerprint: calls min, abs, float; returns min(...) |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 65 | def | `_finite_positive` | fingerprint: calls float, ValueError, math.isfinite; returns number |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 72 | def | `_finite_probability` | fingerprint: calls float, math.isfinite; returns number |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 81 | def | `_rounded` | fingerprint: calls round, float; returns round(...) |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 86 | class | `FixedTransportHMCGridPolicyConfig` | doc: Configuration for a generic fixed-transport HMC candidate policy. |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 104 | def | `__post_init__` | fingerprint: calls tuple, _finite_positive, ValueError, int, any |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 161 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 180 | class | `FixedTransportHMCGridCandidateSpec` | doc: One executable HMC candidate emitted by a policy spec. |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 191 | def | `__post_init__` | fingerprint: calls int, ValueError, _finite_positive, str |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 219 | def | `identity` | fingerprint: returns Tuple |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 222 | def | `payload` | fingerprint: calls _rounded; returns Dict |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 236 | class | `FixedTransportHMCGridPolicySpec` | doc: Stable BayesFilter-owned policy artifact for model-specific executors. |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 246 | def | `__post_init__` | fingerprint: calls tuple, ValueError, candidate.identity, len |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 269 | def | `payload` | fingerprint: calls self.config.payload, tuple, len, candidate.payload, stable_config_hash; returns Dict |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 289 | def | `policy_hash` | fingerprint: calls str, self.payload; returns str(...) |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 294 | class | `FixedTransportHMCPreparedGridConfig` | doc: Configuration for BayesFilter-owned fixed-transport grid preparation. |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 307 | def | `__post_init__` | fingerprint: calls tuple, _finite_positive, ValueError, sorted |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 356 | def | `attempted_scale_candidates` | fingerprint: calls len; returns Subscript |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 359 | def | `payload` | fingerprint: calls self.grid_policy_config.payload; returns Dict |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 375 | class | `FixedTransportHMCPreparedGrid` | doc: Unified BayesFilter artifact for fixed-NeuTra HMC launch preparation. |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 385 | def | `__post_init__` | fingerprint: calls str, ValueError, tuple |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 400 | def | `launch_ready` | fingerprint: returns BoolOp |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 409 | def | `prepared_policy_hash` | fingerprint: returns IfExp |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 412 | def | `payload` | fingerprint: calls self.policy_spec.payload, self.config.payload, self.scale_selection.payload, len, stable_config_hash; returns Dict |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 444 | def | `prepared_grid_hash` | fingerprint: calls str, self.payload; returns str(...) |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 449 | class | `FixedTransportHMCJointPilotRow` | doc: One real pilot HMC diagnostic over a concrete step-size/L tuple. |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 465 | def | `__post_init__` | fingerprint: calls int, ValueError, _finite_positive, _finite_probability |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 509 | def | `identity` | fingerprint: returns Tuple |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 512 | def | `payload` | fingerprint: calls _rounded, bool; returns Dict |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 532 | class | `FixedTransportHMCJointPreparedGridConfig` | doc: Configuration for BayesFilter-owned joint finite-path preparation. |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 541 | def | `__post_init__` | fingerprint: calls float, ValueError, math.isfinite |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 562 | def | `payload` | fingerprint: calls self.grid_policy_config.payload; returns Dict |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 573 | class | `FixedTransportHMCJointPreparedGrid` | doc: Unified launch-prep artifact selected from joint pilot HMC rows. |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 594 | def | `__post_init__` | fingerprint: calls tuple, ValueError, str |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 652 | def | `launch_ready` | fingerprint: calls bool; returns BoolOp |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 662 | def | `prepared_policy_hash` | fingerprint: returns IfExp |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 665 | def | `payload` | fingerprint: calls self.policy_spec.payload, self.selected_joint_pilot_row.payload, self.config.payload, tuple, len, bool; returns Dict |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 739 | def | `prepared_grid_hash` | fingerprint: calls str, self.payload; returns str(...) |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 743 | def | `build_fixed_transport_hmc_grid_policy_spec` | doc: Build a candidate-grid policy spec with optional boundary refinement. |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 756 | def | `add_candidate` | fingerprint: calls int, float, FixedTransportHMCGridCandidateSpec, len |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 843 | def | `prepare_fixed_transport_hmc_grid_policy` | doc: Build a launch-safe fixed-transport HMC grid from real pilot attempts. |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 935 | def | `prepare_fixed_transport_hmc_joint_grid_policy` | doc: Build a launch-safe fixed-transport grid from joint HMC pilot rows. |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 1082 | def | `prepare_fixed_transport_hmc_adaptive_joint_grid_policy` | doc: Build or request a bounded adaptive fixed-transport HMC joint grid. |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 1274 | def | `_boundary_refinement_request` | fingerprint: calls row.get, _finite_probability, _rounded, int, by_step.setdefault(...).append; returns Constant; loops For |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 1318 | def | `_finite_domain_vetoes` | fingerprint: returns Set |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 1326 | def | `_finite_domain_ceiling_refinement_requests` | fingerprint: calls _finite_domain_vetoes, row.get, _joint_launch_eligibility_reasons, _row_hard_vetoes, finite_domain_vetoes.intersection; returns tuple(...); loops For |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 1409 | def | `_joint_pilot_row_from_mapping` | fingerprint: calls row.get, ValueError; returns FixedTransportHMCJointPilotRow(...) |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 1470 | def | `_joint_launch_eligibility_reasons` | fingerprint: calls int, reasons.append, float; returns tuple(...) |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 1487 | def | `_adaptive_ready_joint_grid` | fingerprint: calls float, min, abs, tuple, sorted; returns FixedTransportHMCJointPreparedGrid(...) |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 1576 | def | `_adaptive_round_count` | fingerprint: calls int, max; returns BinOp |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 1589 | def | `_seen_joint_pilot_tuples` | fingerprint: calls tuple; returns tuple(...) |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 1602 | def | `_adaptive_next_scale_request` | fingerprint: calls _latest_adaptive_round_rows, _latest_adaptive_scale, _rounded, max, float, min; returns Tuple |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 1672 | def | `_latest_has_finite_domain_ceiling` | fingerprint: calls _finite_domain_vetoes, finite_domain_vetoes.intersection; returns Constant; loops For |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 1682 | def | `_latest_adaptive_round_rows` | fingerprint: calls max, int, tuple; returns tuple(...) |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 1695 | def | `_latest_adaptive_scale` | fingerprint: calls float, math.isfinite, max; returns max(...) |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 1708 | def | `_adaptive_request_reason` | fingerprint: calls sorted, Constant.join; returns BinOp |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 1717 | def | `_adaptive_next_joint_pilot_tuples` | fingerprint: calls row.identity, _rounded, _joint_launch_eligibility_reasons, out.append, float; returns tuple(...); loops For |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 1751 | def | `_adaptive_local_high_warning_tuples` | fingerprint: calls _latest_adaptive_round_rows, _latest_has_finite_domain_ceiling, float, min, abs; returns tuple(...); loops For |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 1843 | def | `_adaptive_local_too_high_tuples` | fingerprint: calls _latest_adaptive_round_rows, float, len, _latest_has_finite_domain_ceiling, all; returns tuple(...); loops For |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 1920 | def | `_adaptive_round_summaries` | fingerprint: calls grouped.setdefault(...).append, int, grouped.setdefault, sorted, summaries.append, float; returns tuple(...); loops For |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 1957 | def | `_row_hard_vetoes` | fingerprint: calls row.get, isinstance, tuple, str; returns tuple(...) |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 1966 | def | `_classify_joint_acceptance` | fingerprint: calls float; returns Constant |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 1983 | def | `_infer_base_step_size` | fingerprint: calls row.get, _finite_positive, float, sorted, abs; returns Constant; loops For |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 2004 | def | `_joint_prep_failure_status` | fingerprint: calls any; returns Constant |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 2021 | def | `_joint_prep_failure_vetoes` | fingerprint: calls _joint_prep_failure_status; returns Tuple |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 2038 | def | `_refined_step_sizes` | fingerprint: calls tuple, _rounded, range; returns tuple(...) |
| `bayesfilter/inference/fixed_transport_hmc_grid_policy.py` | 2048 | def | `_refined_leapfrogs` | fingerprint: calls list, range, int, values.append; returns tuple(...) |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 54 | class | `FixedTransportHMCKernelTuningConfig` | doc: Policy for fixed-NeuTra HMC kernel tuning in transport coordinates. |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 93 | def | `__post_init__` | fingerprint: calls _positive_float, object.__setattr__, tuple, dict.fromkeys, int, any; loops For |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 199 | def | `payload` | fingerprint: calls asdict; returns payload |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 208 | class | `FixedTransportHMCCandidateResult` | fingerprint: calls object.__setattr__, int |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 220 | def | `__post_init__` | fingerprint: calls int, ValueError, object.__setattr__ |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 238 | def | `passed` | fingerprint: returns Compare |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 242 | def | `selected_step_size` | fingerprint: calls self.ladder_result.get, float; returns IfExp |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 251 | def | `selected_acceptance_rate` | fingerprint: calls _scalar_or_none, self.verification_diagnostics.get; returns _scalar_or_none(...) |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 255 | def | `artifact_hash` | fingerprint: calls _stable_hash, self.payload; returns _stable_hash(...) |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 258 | def | `payload` | fingerprint: calls _stable_hash; returns Dict |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 285 | class | `FixedTransportHMCKernelTuningResult` | fingerprint: calls int, _stable_hash, self.payload, self.config.payload, tuple |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 305 | def | `passed` | fingerprint: returns BoolOp |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 309 | def | `selected_candidate` | fingerprint: calls int; returns Subscript |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 315 | def | `final_kernel_hash` | fingerprint: calls _stable_hash; returns IfExp |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 319 | def | `artifact_hash` | fingerprint: calls _stable_hash, self.payload; returns _stable_hash(...) |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 322 | def | `payload` | fingerprint: calls self.config.payload, tuple, candidate.payload; returns Dict |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 350 | def | `tune_fixed_transport_hmc_kernel` | doc: Tune fixed-length TFP HMC and verify a frozen identity-`z` kernel. |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 455 | def | `_candidate_attempts` | fingerprint: calls _fixed_grid_attempts, _dual_averaging_candidate, enumerate; returns Tuple |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 479 | def | `_dual_averaging_candidate` | fingerprint: calls _initial_state, enumerate, _chain_config, _basic_hard_vetoes, rounds.append, repair_triggers.extend; returns FixedTransportHMCCandidateResult(...); loops For |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 612 | def | `_fixed_grid_attempts` | fingerprint: calls max, enumerate, _run_verification, _scalar_or_none, _acceptance_class; returns Tuple; loops For |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 692 | def | `_run_verification` | fingerprint: calls _chain_config, _offset_seed, _initial_state, run_full_chain, _verification_diagnostics, _error_diagnostics; returns Dict |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 764 | def | `_verification_diagnostics` | fingerprint: calls _tensor_diagnostics, dict, run_diagnostics.get, str, _int_or_none; returns payload |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 815 | def | `_classify_verification` | fingerprint: calls hard.append, _scalar_or_none, diagnostics.get, math.isfinite; returns Tuple |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 859 | def | `_basic_hard_vetoes` | fingerprint: calls diagnostics.get, hard.append; returns hard; loops For |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 875 | def | `_chain_config` | fingerprint: calls _TuningPolicy.fixed, _TuningPolicy.dual_averaging, _FullChainHMCConfig, int, float; returns _FullChainHMCConfig(...) |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 909 | def | `_initial_state` | fingerprint: calls tf.convert_to_tensor, ValueError, int, tf.zeros; returns tf.zeros(...) |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 922 | def | `_validate_initial_position` | fingerprint: calls tf.cast, tf.convert_to_tensor, ValueError, int, bool; returns tensor |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 931 | def | `_identity_mass_payload` | fingerprint: calls int, tf.eye, _json_ready; returns Dict |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 956 | def | `_map_samples` | fingerprint: calls any, ValueError, int, tf.reshape, tf.cast, adapter.latent_to_position; returns tf.reshape(...) |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 967 | def | `_select_candidate` | fingerprint: calls enumerate, min, abs, float; returns Subscript |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 986 | def | `_acceptance_class` | fingerprint: calls math.isfinite; returns Constant |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 998 | def | `_diagnostic_roles` | fingerprint: returns Dict |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 1014 | def | `_error_diagnostics` | fingerprint: calls type; returns Dict |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 1029 | def | `_base_adapter_signature` | fingerprint: calls getattr, str, callable, explicit, _stable_hash, int; returns _stable_hash(...) |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 1042 | def | `_stable_hash` | fingerprint: calls json.dumps(...).encode, json.dumps, _json_ready, hashlib.sha256(...).hexdigest, hashlib.sha256; returns hashlib.sha256(...).hexdigest(...) |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 1049 | def | `_json_ready` | fingerprint: calls tf.is_tensor, _json_ready, value.numpy, isinstance, str; returns value |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 1063 | def | `_scalar_or_none` | fingerprint: calls tf.is_tensor, tf.reshape, float, tf.convert_to_tensor, int, Subscript.numpy |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 1077 | def | `_int_or_none` | fingerprint: calls _scalar_or_none, int; returns IfExp |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 1082 | def | `_positive_float` | fingerprint: calls float, ValueError, math.isfinite; returns result |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 1089 | def | `_validate_band` | fingerprint: calls tuple, float, len, ValueError, math.isfinite; returns Tuple |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 1101 | def | `_validate_seed` | fingerprint: calls tuple, int, len, ValueError; returns seed |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 1108 | def | `_offset_seed` | fingerprint: calls int; returns Tuple |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | 1112 | def | `_string_tuple` | fingerprint: calls isinstance, tuple, str; returns tuple(...) |
| `bayesfilter/inference/frozen_kernel_validation.py` | 31 | def | `_text` | fingerprint: calls str(...).strip, str, ValueError; returns result |
| `bayesfilter/inference/frozen_kernel_validation.py` | 38 | def | `_text_tuple` | fingerprint: calls tuple, _text, len, ValueError, set; returns result |
| `bayesfilter/inference/frozen_kernel_validation.py` | 45 | def | `_json_ready` | fingerprint: calls isinstance, str, _json_ready, sorted, value.items; raises TypeError(...) |
| `bayesfilter/inference/frozen_kernel_validation.py` | 59 | def | `_freeze_mapping` | fingerprint: calls isinstance, TypeError, _json_ready, MappingProxyType; returns MappingProxyType(...) |
| `bayesfilter/inference/frozen_kernel_validation.py` | 66 | def | `_hash_payload` | fingerprint: calls json.dumps(...).encode, json.dumps, _json_ready, hashlib.sha256(...).hexdigest, hashlib.sha256; returns hashlib.sha256(...).hexdigest(...) |
| `bayesfilter/inference/frozen_kernel_validation.py` | 74 | class | `FrozenValidationCandidate` | fingerprint: calls object.__setattr__ |
| `bayesfilter/inference/frozen_kernel_validation.py` | 85 | def | `__post_init__` | fingerprint: calls object.__setattr__, _text |
| `bayesfilter/inference/frozen_kernel_validation.py` | 104 | def | `payload` | fingerprint: calls dict; returns Dict |
| `bayesfilter/inference/frozen_kernel_validation.py` | 119 | class | `FrozenTuningArtifactBinding` | fingerprint: calls object.__setattr__, _text, getattr |
| `bayesfilter/inference/frozen_kernel_validation.py` | 125 | def | `__post_init__` | fingerprint: calls object.__setattr__, _text, getattr; loops For |
| `bayesfilter/inference/frozen_kernel_validation.py` | 129 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/frozen_kernel_validation.py` | 139 | class | `FrozenValidationScope` | fingerprint: calls object.__setattr__, ValueError, _text, getattr |
| `bayesfilter/inference/frozen_kernel_validation.py` | 149 | def | `__post_init__` | fingerprint: calls object.__setattr__, _text, getattr, ValueError; loops For |
| `bayesfilter/inference/frozen_kernel_validation.py` | 159 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/frozen_kernel_validation.py` | 173 | class | `FrozenValidationPolicy` | fingerprint: calls object.__setattr__, int, _text_tuple, ValueError |
| `bayesfilter/inference/frozen_kernel_validation.py` | 178 | def | `__post_init__` | fingerprint: calls object.__setattr__, _text_tuple, int, ValueError |
| `bayesfilter/inference/frozen_kernel_validation.py` | 186 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/frozen_kernel_validation.py` | 195 | class | `FrozenValidationObservation` | fingerprint: calls object.__setattr__, float |
| `bayesfilter/inference/frozen_kernel_validation.py` | 204 | def | `__post_init__` | fingerprint: calls isinstance, TypeError, object.__setattr__, _text, _freeze_mapping |
| `bayesfilter/inference/frozen_kernel_validation.py` | 218 | def | `viable` | fingerprint: returns BoolOp |
| `bayesfilter/inference/frozen_kernel_validation.py` | 221 | def | `payload` | fingerprint: calls self.candidate.payload, dict; returns Dict |
| `bayesfilter/inference/frozen_kernel_validation.py` | 235 | class | `FrozenKernelValidationResult` | fingerprint: calls tuple, object.__setattr__, len, ValueError |
| `bayesfilter/inference/frozen_kernel_validation.py` | 242 | def | `__post_init__` | fingerprint: calls tuple, len, ValueError, set, object.__setattr__ |
| `bayesfilter/inference/frozen_kernel_validation.py` | 251 | def | `viable_observations` | fingerprint: calls tuple; returns tuple(...) |
| `bayesfilter/inference/frozen_kernel_validation.py` | 255 | def | `next_round_candidates` | fingerprint: calls tuple; returns tuple(...) |
| `bayesfilter/inference/frozen_kernel_validation.py` | 258 | def | `payload` | fingerprint: calls self.scope.payload, self.tuning_artifact.payload, self.policy.payload, tuple, item.payload; returns Dict |
| `bayesfilter/inference/frozen_kernel_validation.py` | 276 | def | `_contract_vetoes` | fingerprint: calls vetoes.append, len; returns tuple(...); loops For |
| `bayesfilter/inference/frozen_kernel_validation.py` | 307 | def | `run_frozen_kernel_validation` | doc: Run model adapters under one immutable, unranked validation contract. |
| `bayesfilter/inference/generic_hmc_tuning.py` | 38 | class | `GenericHMCTuningConfig` | doc: Client-facing fixed-kernel tuning orchestration settings. |
| `bayesfilter/inference/generic_hmc_tuning.py` | 49 | def | `__post_init__` | fingerprint: calls tuple, float, ValueError, any, np.isfinite |
| `bayesfilter/inference/generic_hmc_tuning.py` | 83 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/generic_hmc_tuning.py` | 96 | class | `GenericHMCFixedGridScaleConfig` | doc: BayesFilter-owned policy for scaling a fixed HMC step-size grid. |
| `bayesfilter/inference/generic_hmc_tuning.py` | 114 | def | `__post_init__` | fingerprint: calls tuple, float, ValueError, any, np.isfinite |
| `bayesfilter/inference/generic_hmc_tuning.py` | 168 | def | `scaled_step_size_candidates` | fingerprint: calls float, ValueError, np.isfinite, tuple; returns tuple(...) |
| `bayesfilter/inference/generic_hmc_tuning.py` | 174 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/generic_hmc_tuning.py` | 189 | class | `GenericHMCFixedGridScaleProbe` | doc: One scale-probe diagnostic for a fixed HMC grid. |
| `bayesfilter/inference/generic_hmc_tuning.py` | 201 | def | `__post_init__` | fingerprint: calls int, ValueError, float |
| `bayesfilter/inference/generic_hmc_tuning.py` | 232 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/generic_hmc_tuning.py` | 246 | class | `GenericHMCFixedGridScaleSelection` | doc: Stable result for scaling a fixed HMC grid before a full run. |
| `bayesfilter/inference/generic_hmc_tuning.py` | 257 | def | `__post_init__` | fingerprint: calls tuple, ValueError, float, np.isfinite |
| `bayesfilter/inference/generic_hmc_tuning.py` | 278 | def | `passed` | fingerprint: returns BoolOp |
| `bayesfilter/inference/generic_hmc_tuning.py` | 281 | def | `payload` | fingerprint: calls self.config.payload, tuple, item.payload; returns Dict |
| `bayesfilter/inference/generic_hmc_tuning.py` | 295 | def | `artifact_hash` | fingerprint: calls stable_config_hash, self.payload; returns stable_config_hash(...) |
| `bayesfilter/inference/generic_hmc_tuning.py` | 300 | class | `GenericHMCCandidateResult` | doc: One client-target candidate row with checkpoint-ready payload. |
| `bayesfilter/inference/generic_hmc_tuning.py` | 317 | def | `__post_init__` | fingerprint: calls int, ValueError, float, np.isfinite |
| `bayesfilter/inference/generic_hmc_tuning.py` | 358 | def | `passed_screen` | fingerprint: returns BoolOp |
| `bayesfilter/inference/generic_hmc_tuning.py` | 361 | def | `payload` | fingerprint: calls stable_config_hash; returns Dict |
| `bayesfilter/inference/generic_hmc_tuning.py` | 391 | class | `GenericHMCTuningResult` | doc: Stable generic HMC tuning result for downstream clients. |
| `bayesfilter/inference/generic_hmc_tuning.py` | 407 | def | `__post_init__` | fingerprint: calls str, ValueError, int, tuple |
| `bayesfilter/inference/generic_hmc_tuning.py` | 440 | def | `selected_candidate` | fingerprint: returns Constant; loops For |
| `bayesfilter/inference/generic_hmc_tuning.py` | 449 | def | `passed` | fingerprint: returns BoolOp |
| `bayesfilter/inference/generic_hmc_tuning.py` | 457 | def | `payload` | fingerprint: calls self.heldout_candidate.payload, stable_config_hash, tuple, item.payload, self.policy.payload, self.config.payload; returns payload |
| `bayesfilter/inference/generic_hmc_tuning.py` | 576 | def | `artifact_hash` | fingerprint: calls stable_config_hash, self.payload; returns stable_config_hash(...) |
| `bayesfilter/inference/generic_hmc_tuning.py` | 580 | def | `run_generic_hmc_tuning_orchestration` | doc: Run a small generic HMC tuning orchestration on supplied diagnostics. |
| `bayesfilter/inference/generic_hmc_tuning.py` | 700 | def | `_build_candidate_result` | fingerprint: calls float, np.isfinite, checkpoint_root.rstrip, int, GenericHMCCandidateResult; returns GenericHMCCandidateResult(...) |
| `bayesfilter/inference/generic_hmc_tuning.py` | 771 | def | `_select_generic_candidate` | fingerprint: calls CandidateResult, abs, float, select_first_tie_candidate; returns selected.payload |
| `bayesfilter/inference/generic_hmc_tuning.py` | 798 | def | `classify_hmc_fixed_grid_acceptance` | doc: Classify fixed-grid pilot acceptance without moving the target band. |
| `bayesfilter/inference/generic_hmc_tuning.py` | 822 | def | `select_hmc_fixed_grid_scale` | doc: Select ``X`` for ``X * base_step_size_candidates`` from pilot diagnostics. |
| `bayesfilter/inference/generic_hmc_tuning.py` | 935 | def | `_has_ordered_finite_domain_ceiling` | doc: Detect high-acceptance finite probes followed by larger invalid probes. |
| `bayesfilter/inference/generic_hmc_tuning.py` | 952 | class | `GenericHMCCandidateEvaluation` | doc: One generic candidate evaluation supplied by a BayesFilter worker. |
| `bayesfilter/inference/generic_hmc_tuning.py` | 970 | def | `__post_init__` | fingerprint: calls int, ValueError, float, np.isfinite |
| `bayesfilter/inference/generic_hmc_tuning.py` | 1001 | def | `trajectory_length` | fingerprint: calls float; returns float(...) |
| `bayesfilter/inference/generic_hmc_tuning.py` | 1004 | def | `stable_payload` | fingerprint: returns Dict |
| `bayesfilter/inference/generic_hmc_tuning.py` | 1019 | class | `GenericHMCTuningArtifact` | doc: Stable exported artifact returned by generic HMC tuning orchestration. |
| `bayesfilter/inference/generic_hmc_tuning.py` | 1036 | def | `__post_init__` | fingerprint: calls str, ValueError, int |
| `bayesfilter/inference/generic_hmc_tuning.py` | 1073 | def | `selected_candidate` | fingerprint: returns Constant; loops For |
| `bayesfilter/inference/generic_hmc_tuning.py` | 1082 | def | `passed` | fingerprint: returns BoolOp |
| `bayesfilter/inference/generic_hmc_tuning.py` | 1086 | def | `stable_json` | fingerprint: calls json.dumps, self.payload; returns json.dumps(...) |
| `bayesfilter/inference/generic_hmc_tuning.py` | 1090 | def | `artifact_hash` | fingerprint: calls stable_config_hash, self.payload; returns stable_config_hash(...) |
| `bayesfilter/inference/generic_hmc_tuning.py` | 1093 | def | `payload` | fingerprint: calls self.mass_artifact.signature_payload, tuple, item.stable_payload, self.policy.payload, stable_config_hash, selected.stable_payload; returns Dict |
| `bayesfilter/inference/generic_hmc_tuning.py` | 1145 | def | `orchestrate_generic_hmc_tuning` | doc: Select a generic HMC candidate and return a stable client artifact. |
| `bayesfilter/inference/hmc_budget_ladder.py` | 65 | class | `FixedMassHMCTuningBudgetCallbackResult` | doc: Role-separated client callback diagnostics for one screen round. |
| `bayesfilter/inference/hmc_budget_ladder.py` | 74 | def | `__post_init__` | fingerprint: calls object.__setattr__, _string_tuple |
| `bayesfilter/inference/hmc_budget_ladder.py` | 92 | def | `has_stop_veto` | fingerprint: calls bool; returns bool(...) |
| `bayesfilter/inference/hmc_budget_ladder.py` | 95 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_budget_ladder.py` | 106 | class | `FixedMassHMCTuningBudgetLadderConfig` | doc: Configuration for one fixed-mass, fixed-leapfrog budget ladder. |
| `bayesfilter/inference/hmc_budget_ladder.py` | 143 | def | `__post_init__` | fingerprint: calls tuple, int, ValueError, any, object.__setattr__; loops For |
| `bayesfilter/inference/hmc_budget_ladder.py` | 337 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_budget_ladder.py` | 387 | class | `FixedMassHMCTuningBudgetRound` | doc: One tune/screen round in the budget ladder. |
| `bayesfilter/inference/hmc_budget_ladder.py` | 408 | def | `__post_init__` | fingerprint: calls object.__setattr__, int, _validate_seed |
| `bayesfilter/inference/hmc_budget_ladder.py` | 453 | def | `passed` | fingerprint: returns Compare |
| `bayesfilter/inference/hmc_budget_ladder.py` | 457 | def | `repair_compatible` | fingerprint: returns Compare |
| `bayesfilter/inference/hmc_budget_ladder.py` | 460 | def | `payload` | fingerprint: calls self.callback_result.payload; returns Dict |
| `bayesfilter/inference/hmc_budget_ladder.py` | 483 | class | `FixedMassHMCTuningBudgetLadderResult` | doc: Complete fixed-mass HMC tuning-budget ladder artifact. |
| `bayesfilter/inference/hmc_budget_ladder.py` | 499 | def | `__post_init__` | fingerprint: calls str, ValueError, object.__setattr__ |
| `bayesfilter/inference/hmc_budget_ladder.py` | 535 | def | `passed` | fingerprint: returns Compare |
| `bayesfilter/inference/hmc_budget_ladder.py` | 539 | def | `selected_round` | fingerprint: returns Subscript |
| `bayesfilter/inference/hmc_budget_ladder.py` | 545 | def | `selected_config_hash` | fingerprint: calls stable_config_hash; returns IfExp |
| `bayesfilter/inference/hmc_budget_ladder.py` | 550 | def | `selected_config_payload` | fingerprint: calls self._selected_target_scope; returns Dict |
| `bayesfilter/inference/hmc_budget_ladder.py` | 575 | def | `last_finite_tuned_round` | fingerprint: calls reversed; returns Constant; loops For |
| `bayesfilter/inference/hmc_budget_ladder.py` | 582 | def | `last_repair_compatible_round` | fingerprint: calls reversed; returns Constant; loops For |
| `bayesfilter/inference/hmc_budget_ladder.py` | 589 | def | `repair_config_payload` | fingerprint: calls repair_round.screen_diagnostics.get, isinstance, _positive_finite_or_none, directional.get, _fixed_mass_bracket_state_payload; returns Dict |
| `bayesfilter/inference/hmc_budget_ladder.py` | 644 | def | `repair_config_hash` | fingerprint: calls stable_config_hash; returns IfExp |
| `bayesfilter/inference/hmc_budget_ladder.py` | 649 | def | `artifact_hash` | fingerprint: calls stable_config_hash, self.payload; returns stable_config_hash(...) |
| `bayesfilter/inference/hmc_budget_ladder.py` | 652 | def | `payload` | fingerprint: calls self.config.payload, tuple, stable_config_hash, round_result.payload; returns Dict |
| `bayesfilter/inference/hmc_budget_ladder.py` | 686 | def | `_selected_target_scope` | fingerprint: calls self._round_target_scope; returns self._round_target_scope(...) |
| `bayesfilter/inference/hmc_budget_ladder.py` | 692 | def | `_round_target_scope` | fingerprint: calls round_result.screen_config_payload.get, str; returns IfExp |
| `bayesfilter/inference/hmc_budget_ladder.py` | 702 | def | `run_fixed_mass_hmc_tuning_budget_ladder` | doc: Run a finite fixed-mass HMC tuning-budget ladder. |
| `bayesfilter/inference/hmc_budget_ladder.py` | 1416 | def | `_validate_mass_artifact_for_ladder` | fingerprint: calls mass_artifact.validate_for_adapter |
| `bayesfilter/inference/hmc_budget_ladder.py` | 1423 | class | `_FixedMassLatentValueScoreAdapter` | doc: Latent fixed-mass target with a mass-bound stable adapter signature. |
| `bayesfilter/inference/hmc_budget_ladder.py` | 1426 | def | `__init__` | fingerprint: calls super(...).__init__, super, str, ValueError |
| `bayesfilter/inference/hmc_budget_ladder.py` | 1448 | def | `adapter_signature` | fingerprint: returns self._adapter_signature |
| `bayesfilter/inference/hmc_budget_ladder.py` | 1451 | def | `value_score_capability` | fingerprint: calls value_score_capability, str, bool, ValueScoreCapability; returns ValueScoreCapability(...) |
| `bayesfilter/inference/hmc_budget_ladder.py` | 1475 | def | `_resolve_target_scope` | fingerprint: calls value_score_capability, str, ValueError; returns str(...) |
| `bayesfilter/inference/hmc_budget_ladder.py` | 1496 | def | `_build_fixed_mass_hmc_adapter` | fingerprint: calls mass_artifact.build_latent_transform, program_signature, stable_adapter_signature, transform.signature_payload, _FixedMassLatentValueScoreAdapter; returns _FixedMassLatentValueScoreAdapter(...) |
| `bayesfilter/inference/hmc_budget_ladder.py` | 1524 | def | `build_fixed_mass_hmc_adapter` | doc: Build BayesFilter's canonical latent adapter for a frozen mass. |
| `bayesfilter/inference/hmc_budget_ladder.py` | 1546 | def | `_tune_config` | fingerprint: calls HMCTuningPolicy.fixed_mass_dual_averaging, int, FullChainHMCConfig, float; returns FullChainHMCConfig(...) |
| `bayesfilter/inference/hmc_budget_ladder.py` | 1574 | def | `_screen_config` | fingerprint: calls FullChainHMCConfig, float; returns FullChainHMCConfig(...) |
| `bayesfilter/inference/hmc_budget_ladder.py` | 1595 | def | `_diagnostics_payload` | fingerprint: calls dict, _scalar_or_none, _int_or_none; returns payload |
| `bayesfilter/inference/hmc_budget_ladder.py` | 1748 | def | `_finite_count_summary_payload` | fingerprint: calls _int_or_none, diagnostics.get, int; returns payload |
| `bayesfilter/inference/hmc_budget_ladder.py` | 1791 | def | `_run_full_chain_with_optional_reusable_route` | doc: Run HMC through a scoped reusable runner when the static contract matches. |
| `bayesfilter/inference/hmc_budget_ladder.py` | 1920 | def | `_run_with_incall_progress_monitor` | doc: Emit opt-in aggregate work snapshots without changing the HMC call. |
| `bayesfilter/inference/hmc_budget_ladder.py` | 1941 | def | `monitor` | fingerprint: calls stop_event.wait, _public_incall_progress_snapshot, _emit_budget_ladder_boundary_progress, monitor_errors.append, len, tuple; loops While |
| `bayesfilter/inference/hmc_budget_ladder.py` | 1997 | def | `_public_incall_progress_snapshot` | doc: Return a scalar-only adapter snapshot suitable for public progress. |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2031 | def | `_reusable_static_contract_payload` | fingerprint: calls dict, config.signature_payload, payload.pop, _reusable_state_template_contract; returns Dict |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2064 | def | `_reusable_state_template_contract` | fingerprint: calls tf.cast, tf.convert_to_tensor, ValueError, template.shape.as_list, any; returns Dict |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2080 | def | `_runner_route_summary` | fingerprint: calls len, sum, tuple; returns Dict |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2126 | def | `_emit_budget_ladder_boundary_progress` | fingerprint: calls str, int, bool |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2200 | def | `_fixed_mass_public_timeout_state` | fingerprint: calls float, max, min, time.perf_counter; returns Dict |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2239 | def | `_fixed_mass_public_timeout_preflight` | fingerprint: calls dict, _fixed_mass_public_timeout_state, bool, float, str; returns Dict |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2281 | def | `_fixed_mass_public_timeout_diagnostics` | fingerprint: calls dict, str; returns Dict |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2299 | def | `_fixed_mass_public_timeout_closeout_round` | fingerprint: calls _fixed_mass_public_timeout_diagnostics, dict, FixedMassHMCTuningBudgetRound, tune_config.signature_payload, FixedMassHMCTuningBudgetCallbackResult; returns FixedMassHMCTuningBudgetRound(...) |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2350 | def | `_telemetry_payload` | fingerprint: calls _json_ready, dict(...).items, dict, bool, _bool_or_none; returns payload |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2361 | def | `_tune_hard_vetoes` | fingerprint: calls vetoes.append, diagnostics.get, _finite_number; returns vetoes |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2399 | def | `_classify_screen_round` | fingerprint: calls hard_vetoes.append, screen_diagnostics.get, _finite_number; returns Tuple |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2533 | def | `_next_initial_step_after_screen_repair` | fingerprint: calls float, ValueError, np.isfinite; returns float(...) |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2592 | def | `_call_screen_callback` | fingerprint: calls FixedMassHMCTuningBudgetCallbackResult, callback, _coerce_callback_result, str, type |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2614 | def | `_coerce_callback_result` | fingerprint: calls FixedMassHMCTuningBudgetCallbackResult, isinstance, ValueError, tuple; returns FixedMassHMCTuningBudgetCallbackResult(...) |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2630 | def | `_mass_artifact_signature` | fingerprint: calls program_signature, mass_artifact.signature_payload, np.asarray; returns program_signature(...) |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2641 | def | `_error_diagnostics` | fingerprint: calls str, type; returns Dict |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2654 | def | `_step_stability_payload` | fingerprint: calls float, max, abs, np.finfo; returns Dict |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2680 | def | `_directional_repair_tune_skip_diagnostics` | fingerprint: calls float, ValueError, np.isfinite; returns Dict |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2697 | def | `_with_directional_step_repair_diagnostics` | fingerprint: calls _scalar_or_none, screen_diagnostics.get, np.isfinite, dict, float; returns Tuple |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2812 | def | `_screen_diagnostics_have_directional_step_repair` | fingerprint: calls diagnostics.get, isinstance, _positive_finite_or_none, payload.get; returns Compare |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2822 | def | `_should_extend_fixed_mass_repair_screens` | doc: Allow bounded extra screens only for private bracket repair work. |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2855 | def | `_fixed_mass_bracket_state_payload` | doc: Extract private bracket state needed to continue fixed-screen repair. |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2887 | def | `_coerce_fixed_mass_bracket_state` | fingerprint: calls isinstance, ValueError, _positive_finite_or_none, state.get; returns Dict |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2930 | def | `_round_seed` | fingerprint: calls int; returns Tuple |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2934 | def | `_validate_seed` | fingerprint: calls tuple, int, len, ValueError; returns values |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2941 | def | `_validate_band` | fingerprint: calls len, ValueError, tuple, float; returns Tuple |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2952 | def | `_validate_step_repair_multiplier` | fingerprint: calls float, ValueError, np.isfinite; returns multiplier |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2959 | def | `_string_tuple` | fingerprint: calls isinstance, tuple, str; returns tuple(...) |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2967 | def | `_positive_finite_or_none` | fingerprint: calls _scalar_or_none, np.isfinite; returns scalar |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2974 | def | `_finite_number` | fingerprint: calls _scalar_or_none, bool, np.isfinite; returns BoolOp |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2979 | def | `_tensor_to_numpy` | fingerprint: calls hasattr, value.numpy; returns value |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2985 | def | `_scalar_or_none` | fingerprint: calls np.asarray, _tensor_to_numpy, float, array.reshape |
| `bayesfilter/inference/hmc_budget_ladder.py` | 2997 | def | `_int_or_none` | fingerprint: calls _scalar_or_none, int; returns IfExp |
| `bayesfilter/inference/hmc_budget_ladder.py` | 3002 | def | `_bool_or_none` | fingerprint: calls np.asarray, _tensor_to_numpy, bool, array.reshape; returns bool(...) |
| `bayesfilter/inference/hmc_budget_ladder.py` | 3011 | def | `_json_ready` | fingerprint: calls hasattr, _json_ready, value.numpy, isinstance, value.tolist; returns value |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 108 | def | `_strict_integer` | fingerprint: calls isinstance, ValueError, int; returns result |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 117 | def | `_strict_seed` | fingerprint: calls tuple, ValueError, len, _strict_integer; returns tuple(...) |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 129 | def | `_nonempty` | fingerprint: calls str, ValueError; returns result |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 136 | def | `_reason_tuple` | fingerprint: calls tuple, dict.fromkeys, str, ValueError, set(...).issubset, set; returns reasons |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 145 | def | `_json_ready` | fingerprint: calls isinstance, str, _json_ready, sorted, value.items; returns value |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 153 | def | `_signature` | fingerprint: calls json.dumps(...).encode, json.dumps, _json_ready, hashlib.sha256(...).hexdigest, hashlib.sha256; returns hashlib.sha256(...).hexdigest(...) |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 164 | class | `FixedMetricSearchLineage` | doc: Content signatures frozen across every tune and screen callback. |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 172 | def | `__post_init__` | fingerprint: calls object.__setattr__, _nonempty, getattr; loops For |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 186 | def | `signature` | fingerprint: calls _signature, self.payload; returns _signature(...) |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 189 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 201 | class | `FixedMetricGridSearchConfig` | doc: Reviewed broad-grid controls with optional one-round refinement. |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 211 | def | `__post_init__` | fingerprint: calls tuple, _strict_integer, len, ValueError |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 252 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 267 | class | `FixedMetricGridExecutionConfig` | doc: Execution topology for a fixed-metric candidate grid. |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 281 | def | `__post_init__` | fingerprint: calls str, ValueError, object.__setattr__ |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 351 | def | `payload` | fingerprint: calls dict, tuple, environment.get, sorted; returns Dict |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 367 | class | `FixedMetricCandidateRunners` | doc: Application-owned tune/screen callbacks constructed inside one worker. |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 373 | def | `__post_init__` | fingerprint: calls TypeError, callable |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 379 | class | `FixedMetricCandidateWorkerRequest` | doc: Complete immutable input supplied to an application worker factory. |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 390 | class | `FixedMetricCandidateWorkerOutcome` | doc: Typed cross-process envelope; raw samples and states are never included. |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 398 | def | `__post_init__` | fingerprint: calls str, ValueError, object.__setattr__, isinstance |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 432 | def | `fixed_metric_search_seed` | doc: Derive an order-independent seed from the complete candidate identity. |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 465 | class | `FixedMetricTuneRequest` | fingerprint: body AnnAssign, AnnAssign, AnnAssign, AnnAssign |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 474 | class | `FixedMetricTuneOutcome` | fingerprint: body AnnAssign, AnnAssign, AnnAssign, AnnAssign |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 482 | class | `FixedMetricScreenRequest` | fingerprint: body AnnAssign, AnnAssign, AnnAssign, AnnAssign |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 494 | class | `FixedMetricScreenOutcome` | fingerprint: body AnnAssign, AnnAssign, AnnAssign, AnnAssign |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 503 | class | `CandidateTuneRejected` | doc: Typed candidate-local tune failure; other grid candidates remain valid. |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 506 | def | `__init__` | fingerprint: calls _reason_tuple, super(...).__init__, super |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 515 | class | `CandidateScreenRejected` | doc: Typed candidate-local screen failure; the shared harness remains valid. |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 518 | def | `__init__` | fingerprint: calls _reason_tuple, super(...).__init__, super |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 527 | class | `SharedGridSearchInvalidity` | doc: Typed shared contract failure that stops the complete search barrier. |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 530 | def | `__init__` | fingerprint: calls _reason_tuple, super(...).__init__, super |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 539 | class | `GridSearchResourceCloseout` | doc: Typed resource stop that preserves completed candidate evidence. |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 542 | def | `__init__` | fingerprint: calls _nonempty, super(...).__init__, super |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 547 | class | `GridSearchTargetVeto` | doc: Typed shared target-health veto preserved for caller-owned closeout. |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 550 | def | `__init__` | fingerprint: calls _nonempty, super(...).__init__, super |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 555 | def | `_validate_lineage` | fingerprint: calls isinstance, SharedGridSearchInvalidity, tuple, getattr |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 580 | class | `FixedMetricScreenRecord` | fingerprint: calls hmc_acceptance_evidence_from_payload, object.__setattr__, evidence.payload, _signature, self.payload |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 584 | def | `__post_init__` | fingerprint: calls hmc_acceptance_evidence_from_payload, object.__setattr__, evidence.payload |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 589 | def | `evidence` | fingerprint: calls hmc_acceptance_evidence_from_payload; returns hmc_acceptance_evidence_from_payload(...) |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 593 | def | `signature` | fingerprint: calls _signature, self.payload; returns _signature(...) |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 596 | def | `payload` | fingerprint: calls self.request.lineage.payload; returns Dict |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 611 | class | `FixedMetricEvidenceExtensionRecord` | fingerprint: calls self.replacement.payload |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 616 | def | `payload` | fingerprint: calls self.replacement.payload; returns Dict |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 627 | class | `FixedMetricCandidateRecord` | fingerprint: calls all, len, _signature, self.payload, tuple |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 639 | def | `survivor` | fingerprint: calls all, len; returns BoolOp |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 648 | def | `signature` | fingerprint: calls _signature, self.payload; returns _signature(...) |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 651 | def | `payload` | fingerprint: calls tuple, item.payload; returns Dict |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 669 | class | `FixedMetricCandidateEvidencePolicy` | doc: Opt-in uncertainty-aware candidate confirmation policy. |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 681 | def | `__post_init__` | fingerprint: calls _strict_integer, ValueError, float |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 699 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 717 | class | `FixedMetricAggregateEvidence` | doc: Candidate-level working evidence across replications and chains. |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 735 | def | `provisional_viable` | fingerprint: returns BoolOp |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 738 | def | `payload` | fingerprint: calls self.policy.payload; returns Dict |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 761 | class | `FixedMetricCandidateConfirmationRecord` | doc: Fresh fixed-epsilon confirmation linked to one immutable nomination. |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 774 | def | `disposition` | fingerprint: returns self.confirmation.disposition |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 782 | def | `provisional_viable` | fingerprint: returns BoolOp |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 785 | def | `payload` | fingerprint: calls self.nomination.payload, tuple, self.confirmation.payload, item.payload; returns Dict |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 807 | def | `refinement_l_values` | doc: Return untested integer midpoints adjacent to every initial survivor. |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 843 | class | `FixedMetricGridSearchResult` | fingerprint: calls field, tuple, sorted, self.config.payload, self.lineage.payload, self.acceptance_policy.payload |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 856 | def | `candidates` | fingerprint: returns BinOp |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 860 | def | `survivors` | fingerprint: calls tuple, sorted; returns tuple(...) |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 870 | def | `payload` | fingerprint: calls self.config.payload, self.lineage.payload, self.acceptance_policy.payload, tuple; returns Dict |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 902 | def | `public_summary` | doc: Return aggregate, non-replayable status without HMC mechanics. |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 955 | def | `_callback_shared_failure` | fingerprint: calls isinstance, SharedGridSearchInvalidity; returns SharedGridSearchInvalidity(...) |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 961 | def | `_run_tune` | fingerprint: calls fixed_metric_search_seed, FixedMetricTuneRequest, tune_runner, _callback_shared_failure, isinstance, SharedGridSearchInvalidity; returns Tuple |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 1002 | def | `_run_screen` | fingerprint: calls fixed_metric_search_seed, FixedMetricScreenRequest, screen_runner, _callback_shared_failure, isinstance, SharedGridSearchInvalidity; returns FixedMetricScreenRecord(...) |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 1070 | def | `_extension_eligible` | fingerprint: calls tuple, all; returns BoolOp |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 1086 | def | `aggregate_fixed_metric_candidate_evidence` | doc: Summarize one complete evidence phase without changing screen decisions. |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 1194 | def | `confirm_fixed_metric_candidate` | doc: Run fresh confirmation at one immutable candidate's frozen epsilon. |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 1285 | def | `run_fixed_metric_confirmation_screen` | doc: Run one fresh confirmation replication through the public API. |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 1343 | def | `_run_candidate` | fingerprint: calls fixed_metric_search_seed, _run_tune, FixedMetricCandidateRecord, range, _run_screen, screens.append; returns FixedMetricCandidateRecord(...) |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 1478 | def | `run_fixed_metric_candidate` | doc: Run one complete candidate using the same semantics as grid execution. |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 1524 | def | `_resolve_worker_factory` | fingerprint: calls locator.split, importlib.import_module, attribute_path.split, getattr, callable, TypeError; returns value; loops For |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 1534 | def | `_run_candidate_process_worker` | doc: Spawn-worker entry point; environment is inherited before module import. |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 1581 | def | `_temporary_worker_environment` | fingerprint: calls os.environ.get, os.environ.update, dict, previous.items, os.environ.pop |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 1596 | def | `_run_process_candidate_barrier` | fingerprint: calls tuple, int, enumerate, before_candidate, FixedMetricCandidateWorkerRequest; returns Tuple; loops For |
| `bayesfilter/inference/hmc_fixed_metric_grid_search.py` | 1685 | def | `run_fixed_metric_grid_search` | doc: Run the complete broad barrier and the configured refinement phase. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 30 | class | `_CandidateRetuneHealthError` | doc: The nominated exact-L setting failed local numerical/target health. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 33 | def | `__init__` | fingerprint: calls tuple, dict.fromkeys, str, super(...).__init__, super |
| `bayesfilter/inference/hmc_kernel_selection.py` | 38 | class | `_SharedRetuneInvalidityError` | doc: The exact-L runner, adapter, or returned schema is invalid. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 41 | def | `__init__` | fingerprint: calls tuple, dict.fromkeys, str, super(...).__init__, super |
| `bayesfilter/inference/hmc_kernel_selection.py` | 56 | def | `deterministic_candidate_order` | doc: Order an already-qualified arbitrary candidate set without ranking evidence. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 76 | def | `key` | fingerprint: calls isinstance, item.get, getattr; returns Tuple |
| `bayesfilter/inference/hmc_kernel_selection.py` | 96 | class | `VerifiedFixedKernelHandoff` | doc: BayesFilter-owned handoff for one independently verified fixed pair. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 106 | def | `__post_init__` | fingerprint: calls _strict_integer, float, ValueError, np.isfinite, str; loops For |
| `bayesfilter/inference/hmc_kernel_selection.py` | 134 | def | `signature` | fingerprint: calls _signature, self.payload; returns _signature(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 137 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_selection.py` | 159 | def | `build_verified_fixed_kernel_handoff` | doc: Choose from hard-screen-passed candidates using policy fields only. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 211 | def | `_candidate_handoff_policy` | fingerprint: calls str, Constant.join, ValueError, sorted; returns policy |
| `bayesfilter/inference/hmc_kernel_selection.py` | 219 | def | `_candidate_handoff_disposition` | fingerprint: calls str, Constant.join, ValueError, sorted; returns disposition |
| `bayesfilter/inference/hmc_kernel_selection.py` | 229 | def | `candidate_handoff_policy_payload` | doc: Return the typed policy contract without target- or run-specific data. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 281 | def | `_strict_integer` | fingerprint: calls isinstance, ValueError, int; returns result |
| `bayesfilter/inference/hmc_kernel_selection.py` | 293 | def | `_strict_seed` | fingerprint: calls tuple, ValueError, len, _strict_integer; returns tuple(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 305 | def | `_strict_bool` | fingerprint: calls isinstance, ValueError, bool; returns bool(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 311 | def | `_target_status_policy` | fingerprint: calls str, ValueError; returns policy |
| `bayesfilter/inference/hmc_kernel_selection.py` | 320 | def | `_json_ready` | fingerprint: calls isinstance, str, _json_ready, sorted, value.items; returns value |
| `bayesfilter/inference/hmc_kernel_selection.py` | 330 | def | `_signature` | fingerprint: calls json.dumps, _json_ready, hashlib.sha256(...).hexdigest, hashlib.sha256, text.encode; returns hashlib.sha256(...).hexdigest(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 340 | def | `private_start_bank_content_signature` | fingerprint: calls np.asarray, hashlib.sha256, np.ascontiguousarray(...).tobytes, np.ascontiguousarray, digest.update, str(...).encode; returns digest.hexdigest(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 350 | def | `_candidate_execution_contract_signature` | fingerprint: calls _strict_integer, _strict_seed, _strict_bool, _target_status_policy; returns _signature(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 392 | def | `fixed_trajectory_candidate_values` | doc: Return distinct sorted ``{floor(L0/2), L0, 2*L0}`` within bounds. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 418 | def | `paired_candidate_seed` | doc: Derive an order-independent seed from candidate identity and replicate. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 449 | class | `FixedTrajectoryCandidate` | doc: One predeclared deterministic trajectory candidate. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 460 | def | `__post_init__` | fingerprint: calls _strict_integer, fixed_trajectory_candidate_values, ValueError; loops For |
| `bayesfilter/inference/hmc_kernel_selection.py` | 498 | def | `signature` | fingerprint: calls _signature, self.payload; returns _signature(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 501 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_selection.py` | 514 | def | `_exact_l_retune_signature` | doc: Bind a selected handoff to the exact final-``L`` retune config. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 567 | class | `FixedTrajectoryReplication` | doc: Sanitized evidence from one paired candidate replication. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 579 | def | `__post_init__` | fingerprint: calls isinstance, TypeError, _strict_integer, ValueError, _strict_seed, hmc_acceptance_evidence_from_payload; loops For |
| `bayesfilter/inference/hmc_kernel_selection.py` | 628 | def | `evidence` | fingerprint: calls hmc_acceptance_evidence_from_payload; returns hmc_acceptance_evidence_from_payload(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 632 | def | `signature` | fingerprint: calls _signature, self.payload; returns _signature(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 635 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_selection.py` | 650 | class | `FixedTrajectoryCandidateResult` | doc: Complete three-replication evidence for one candidate. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 658 | def | `__post_init__` | fingerprint: calls isinstance, TypeError, tuple, len, ValueError, any |
| `bayesfilter/inference/hmc_kernel_selection.py` | 687 | def | `decisions` | fingerprint: calls tuple; returns tuple(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 691 | def | `evidence_validities` | fingerprint: calls tuple; returns tuple(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 695 | def | `viable` | fingerprint: calls all; returns all(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 699 | def | `evidence_extension_eligible` | doc: Whether longer evidence may resolve this candidate at fixed epsilon. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 716 | def | `candidate_data_invalid` | fingerprint: calls any; returns any(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 723 | def | `shared_invalidity` | fingerprint: calls any; returns any(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 730 | def | `resonance_detected` | fingerprint: calls any; returns any(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 739 | def | `resonance_repair_detected` | fingerprint: calls any; returns any(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 749 | def | `trajectory_repair_detected` | fingerprint: calls any; returns any(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 757 | def | `signature` | fingerprint: calls _signature, self.payload; returns _signature(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 763 | def | `payload` | fingerprint: calls self.candidate.payload, tuple, item.payload; returns Dict |
| `bayesfilter/inference/hmc_kernel_selection.py` | 784 | class | `FixedTrajectoryCandidateRetuneFailure` | doc: One candidate-local exact-``L`` failure in deterministic nominee order. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 792 | def | `__post_init__` | fingerprint: calls str, _strict_seed, tuple, dict.fromkeys, _strict_integer |
| `bayesfilter/inference/hmc_kernel_selection.py` | 812 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_selection.py` | 823 | def | `_ordered_viable_candidate_results` | fingerprint: calls tuple, sorted, abs; returns tuple(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 840 | def | `_ordered_mixed_evidence_candidate_results` | doc: Return provisional candidates only when the complete matrix is clean. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 879 | def | `_ordered_candidate_local_mixed_evidence_results` | doc: Return locally clean nominees while preserving shared-invalidity vetoes. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 914 | def | `_mixed_evidence_nominee` | fingerprint: calls _candidate_handoff_policy, _ordered_mixed_evidence_candidate_results, _ordered_candidate_local_mixed_evidence_results, ValueError; returns IfExp |
| `bayesfilter/inference/hmc_kernel_selection.py` | 937 | class | `FixedTrajectorySelection` | doc: Permutation-invariant representative selection over completed candidates. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 953 | def | `__post_init__` | fingerprint: calls _strict_integer, _candidate_handoff_policy, _candidate_handoff_disposition, ValueError, tuple; loops For |
| `bayesfilter/inference/hmc_kernel_selection.py` | 1161 | def | `representative` | fingerprint: calls next; returns next(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 1172 | def | `signature` | fingerprint: calls _signature, self.payload; returns _signature(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 1181 | def | `payload` | fingerprint: calls tuple, len, item.payload, candidate_handoff_policy_payload; returns payload |
| `bayesfilter/inference/hmc_kernel_selection.py` | 1221 | def | `select_fixed_trajectory_representative` | doc: Aggregate a completed candidate batch without using descriptive ranking. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 1329 | class | `FixedTrajectoryEvidenceExtensionSlot` | doc: One inconclusive replication replaced by a fresh longer trace. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 1340 | def | `__post_init__` | fingerprint: calls isinstance, TypeError, _strict_integer, _strict_seed, str |
| `bayesfilter/inference/hmc_kernel_selection.py` | 1375 | def | `extended_decision` | fingerprint: returns self.extended_replication.evidence.acceptance_decision |
| `bayesfilter/inference/hmc_kernel_selection.py` | 1378 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_selection.py` | 1402 | class | `FixedTrajectoryEvidenceExtension` | doc: Complete candidate-matrix barrier for one fixed extension checkpoint. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 1420 | def | `__post_init__` | fingerprint: calls _strict_integer, _strict_seed, float, tuple; loops For |
| `bayesfilter/inference/hmc_kernel_selection.py` | 1631 | def | `source_selection_signature` | fingerprint: returns self.source_selection.signature |
| `bayesfilter/inference/hmc_kernel_selection.py` | 1635 | def | `matrix_selection_signature` | fingerprint: returns self.matrix_selection.signature |
| `bayesfilter/inference/hmc_kernel_selection.py` | 1639 | def | `finalized_selection_signature` | fingerprint: returns self.finalized_selection.signature |
| `bayesfilter/inference/hmc_kernel_selection.py` | 1643 | def | `matrix_disposition` | fingerprint: returns self.matrix_selection.disposition |
| `bayesfilter/inference/hmc_kernel_selection.py` | 1647 | def | `finalized_disposition` | fingerprint: returns self.finalized_selection.disposition |
| `bayesfilter/inference/hmc_kernel_selection.py` | 1651 | def | `seeds` | fingerprint: calls tuple; returns tuple(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 1654 | def | `payload` | fingerprint: calls tuple, len, item.payload; returns Dict |
| `bayesfilter/inference/hmc_kernel_selection.py` | 1684 | class | `FixedTrajectoryCandidateHandoffLineage` | doc: Terminal mixed-evidence nomination boundary after extension exhaustion. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 1694 | def | `__post_init__` | fingerprint: calls _candidate_handoff_policy, _strict_integer, _candidate_handoff_disposition, ValueError, TypeError |
| `bayesfilter/inference/hmc_kernel_selection.py` | 1783 | def | `signature` | fingerprint: calls _signature, self.payload; returns _signature(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 1789 | def | `payload` | fingerprint: calls candidate_handoff_policy_payload; returns Dict |
| `bayesfilter/inference/hmc_kernel_selection.py` | 1816 | class | `FixedTrajectorySelectionRepairAttempt` | doc: One complete paired-selection matrix and its optional step repair. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 1831 | def | `__post_init__` | fingerprint: calls _strict_integer, _strict_seed, float, ValueError, len; loops For |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2000 | def | `replication_seeds` | fingerprint: calls tuple; returns tuple(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2010 | def | `extension_seeds` | fingerprint: calls tuple; returns tuple(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2014 | def | `all_replication_execution_seeds` | fingerprint: returns BinOp |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2018 | def | `exact_l_retune_seed` | fingerprint: returns IfExp |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2023 | def | `exact_l_retune_seeds` | fingerprint: calls tuple, paired_candidate_seed; returns BinOp |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2048 | def | `output_step_size` | fingerprint: returns self.repair.repaired_step_size |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2054 | def | `bracket_after` | fingerprint: returns self.repair.bracket |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2060 | def | `lower_bound_source_attempt_index_after` | fingerprint: calls np.isclose, float; returns IfExp |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2083 | def | `payload` | fingerprint: calls tuple, len, bool, self.repair.payload; returns payload |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2172 | class | `BoundedFixedTrajectorySelectionResult` | doc: Terminal result of the BayesFilter-owned bounded selection repair loop. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2181 | def | `__post_init__` | fingerprint: calls tuple, _candidate_handoff_policy, _strict_integer, str, int; loops For |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2342 | def | `selection` | fingerprint: returns Subscript.selection |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2346 | def | `representative` | fingerprint: returns self.selection.representative |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2350 | def | `repair_direction_history` | fingerprint: calls tuple, str; returns tuple(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2359 | def | `repaired_step_history` | fingerprint: calls tuple, float; returns tuple(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2368 | def | `step_veto_recovery_count` | fingerprint: calls sum; returns sum(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2377 | def | `final_bracket` | fingerprint: returns Subscript.bracket_after |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2381 | def | `signature` | fingerprint: calls _signature, self.payload; returns _signature(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2389 | def | `payload` | fingerprint: calls tuple, len, sum, item.payload; returns payload |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2429 | def | `private_evidence_ledger` | doc: Serialize complete aggregate matrices without samples or bank values. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2432 | def | `replication_payload` | fingerprint: calls sanitize_health_failure_reasons; returns Dict |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2472 | def | `matrix_payload` | fingerprint: calls tuple, sorted, replication_payload; returns Dict |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2575 | def | `run_bounded_operational_fixed_trajectory_selection` | doc: Run complete paired matrices until selection or a typed bounded stop. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2920 | def | `_selection_attempt_root_seed` | fingerprint: calls _strict_integer, _strict_seed, paired_candidate_seed; returns paired_candidate_seed(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2939 | def | `_coerce_empirical_bracket` | fingerprint: calls float, ValueError, np.isfinite; returns Tuple; loops For |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2953 | def | `_validate_selection_frozen_lineage` | fingerprint: calls _strict_integer, fixed_trajectory_candidate_values, tuple, sorted, ValueError; loops For |
| `bayesfilter/inference/hmc_kernel_selection.py` | 2992 | def | `_validate_initial_selection_seed_lineage` | fingerprint: calls paired_candidate_seed, ValueError; loops For |
| `bayesfilter/inference/hmc_kernel_selection.py` | 3009 | def | `_validate_selection_execution_contract` | fingerprint: calls acceptance_policy.payload, _candidate_execution_contract_signature, evidence.policy.payload, ValueError, int; loops For |
| `bayesfilter/inference/hmc_kernel_selection.py` | 3053 | def | `_validate_selection_exact_l_retune_lineage` | fingerprint: calls ValueError, _exact_l_retune_signature |
| `bayesfilter/inference/hmc_kernel_selection.py` | 3086 | def | `_validate_returned_evidence_extension` | fingerprint: calls isinstance, TypeError, _signature, acceptance_policy.payload, ValueError, _strict_seed |
| `bayesfilter/inference/hmc_kernel_selection.py` | 3142 | def | `extend_operational_fixed_trajectory_evidence` | doc: Replace every inconclusive slot with a fresh longer fixed-kernel trace. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 3313 | def | `run_operational_fixed_trajectory_selection` | doc: Execute the repaired paired-candidate policy through TF/TFP HMC. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 3447 | def | `_validated_operational_start_bank` | fingerprint: calls np.asarray, ValueError, np.all, np.isfinite; returns bank |
| `bayesfilter/inference/hmc_kernel_selection.py` | 3463 | def | `_run_operational_candidate_replication` | fingerprint: calls FullChainHMCConfig, int, float, str, runner; returns FixedTrajectoryReplication(...); loops For |
| `bayesfilter/inference/hmc_kernel_selection.py` | 3643 | def | `_finalize_operational_selection_nomination` | fingerprint: calls _ordered_viable_candidate_results, enumerate, paired_candidate_seed, FullChainHMCConfig, _exact_l_retune_signature, tuple; returns FixedTrajectorySelection(...); loops For |
| `bayesfilter/inference/hmc_kernel_selection.py` | 3808 | def | `_validated_exact_l_retune_step` | doc: Bind an exact-L retuned step to its finite adaptive TF/TFP trace. |
| `bayesfilter/inference/hmc_kernel_selection.py` | 3900 | def | `_numpy` | fingerprint: calls hasattr, value.numpy, np.asarray; returns np.asarray(...) |
| `bayesfilter/inference/hmc_kernel_selection.py` | 3906 | def | `_optional_float` | fingerprint: calls hasattr, value.numpy, np.asarray, float, array.reshape, np.isfinite; returns IfExp |
| `bayesfilter/inference/hmc_kernel_selection.py` | 3918 | def | `_optional_int` | fingerprint: calls hasattr, value.numpy, np.asarray, array.reshape(...).item, array.reshape, isinstance; returns Constant |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 138 | def | `_validate_metric_update_requirement` | fingerprint: calls str, Constant.join, ValueError, sorted; returns requirement |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 171 | def | `_validated_operational_verification_bracket_policy` | fingerprint: calls str, Constant.join, ValueError, sorted; returns policy |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 182 | def | `_operational_verification_starts_per_outer_attempt` | fingerprint: calls _validated_operational_verification_bracket_policy; returns IfExp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 192 | def | `_validated_operational_candidate_handoff_policy` | fingerprint: calls candidate_handoff_policy_payload, contract.get, ValueError, str; returns str(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 328 | class | `HMCGeometryScaledBudgetTimingPolicy` | doc: Public-safe policy tying HMC tuning budgets to dimension and geometry. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 362 | def | `__post_init__` | fingerprint: calls str, ValueError, object.__setattr__, float, getattr; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 440 | def | `_default_stage_time_budget_multiplier` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 450 | def | `geometry_summary` | fingerprint: calls int, ValueError, dict, _geometry_policy_eigenvalues, _geometry_policy_condition_number; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 516 | def | `geometry_multiplier` | fingerprint: calls max, float, np.log10; returns float(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 560 | def | `attempt_budget_payload` | fingerprint: calls int, ValueError, self.geometry_summary; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 633 | def | `bootstrap_screen_counts` | fingerprint: calls self.geometry_summary, int, np.ceil, float, np.sqrt; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 679 | def | `stage_budgets_s` | fingerprint: calls self.geometry_summary, int, float, self.stage_time_budget_multiplier.items, min; returns stage_budgets; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 701 | def | `stage_budget_provenance` | fingerprint: returns DictComp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 707 | def | `staged_timeout_policy` | fingerprint: calls HMCStagedTimeoutPolicy, self.stage_budgets_s, self.stage_budget_provenance; returns HMCStagedTimeoutPolicy(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 717 | def | `budget_formula_parameters` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 734 | def | `payload` | fingerprint: calls self.budget_formula_parameters, dict; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 767 | class | `HMCStagedTimeoutPolicy` | doc: Opt-in public-safe staged timeout accounting policy. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 780 | def | `__post_init__` | fingerprint: calls str, ValueError, dict, _default_staged_timeout_policy_stage_budgets, set; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 834 | def | `payload` | fingerprint: calls dict; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 848 | def | `_default_staged_timeout_policy_stage_budgets` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 859 | def | `_default_staged_timeout_policy_stage_budget_provenance` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 870 | def | `_default_staged_timeout_policy` | fingerprint: calls HMCGeometryScaledBudgetTimingPolicy(...).staged_timeout_policy, HMCGeometryScaledBudgetTimingPolicy; returns HMCGeometryScaledBudgetTimingPolicy(...).staged_timeout_policy(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 874 | def | `_geometry_scaled_budget_timing_policy` | fingerprint: calls HMCGeometryScaledBudgetTimingPolicy; returns HMCGeometryScaledBudgetTimingPolicy(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 878 | def | `_geometry_policy_eigenvalues` | fingerprint: calls eigen_summary.get, np.asarray, np.all, np.maximum, tuple; returns np.ones(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 901 | def | `_geometry_policy_condition_number` | fingerprint: calls eigen_summary.get, float, np.isfinite, np.asarray; returns IfExp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 922 | def | `_geometry_policy_effective_dimension` | fingerprint: calls np.asarray, np.isfinite, float, np.sum; returns float(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 934 | def | `_geometry_policy_regularization_counts` | fingerprint: calls max, int, int_field, bool, regularization_report.get; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 937 | def | `int_field` | fingerprint: calls max, int; returns Constant; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 965 | def | `_attempt_budget_policy_from_payload` | fingerprint: calls operational_keys.intersection, Constant.join, ValueError, sorted, HMCOperationalStatisticalWorkPolicy; returns _HMCAttemptBudgetPolicy(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1175 | def | `_validate_max_leapfrog_steps` | fingerprint: calls int, ValueError; returns max_l |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1182 | def | `_validate_handoff_screen_policy` | fingerprint: calls str, Constant.join, ValueError, sorted; returns policy |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1190 | def | `_phase23_nomination_policy_active` | fingerprint: calls str; returns Compare |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1194 | def | `_trajectory_window_class_penalty` | fingerprint: calls str; returns Constant |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1203 | def | `_validate_trajectory_window_multiplier` | fingerprint: calls float, ValueError, np.isfinite; returns multiplier |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1214 | def | `_validate_step_repair_multiplier` | fingerprint: calls float, ValueError, np.isfinite; returns multiplier |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1221 | def | `_validate_staged_timeout_policy_or_none` | fingerprint: calls isinstance, TypeError; returns value |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1231 | def | `_validate_nonnegative_perf_counter_or_none` | fingerprint: calls float, ValueError, np.isfinite; returns perf_counter |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1244 | def | `_validate_staged_timeout_enlargement_rounds` | fingerprint: calls isinstance, TypeError, value.items, int, ValueError, str; returns rounds; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1260 | def | `_validate_trajectory_window_multipliers` | fingerprint: calls _validate_trajectory_window_multiplier, ValueError; returns Tuple |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1284 | def | `_trajectory_window_bounds` | fingerprint: calls float, ValueError, np.isfinite, _validate_trajectory_window_multipliers; returns Tuple |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1300 | def | `_trajectory_window_relation` | fingerprint: calls float, np.isfinite; returns Constant |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1318 | def | `_trajectory_window_payload` | fingerprint: calls float, int, ValueError, np.isfinite, _validate_max_leapfrog_steps; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1368 | def | `_validate_positive_int_or_none` | fingerprint: calls int, ValueError; returns integer |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1377 | def | `_bootstrap_selected_kernel_payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1406 | def | `_bootstrap_geometry_preflight_kernel_payload` | doc: Return a non-promoting kernel seed when bootstrap only passed preflight. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1453 | def | `_bootstrap_hard_vetoes` | fingerprint: calls tuple, dict.fromkeys, str; returns tuple(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1463 | def | `_bootstrap_repair_triggers` | fingerprint: calls tuple, dict.fromkeys, str; returns tuple(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1473 | def | `_bootstrap_preflight_passed` | fingerprint: calls _bootstrap_hard_vetoes; returns UnaryOp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1477 | def | `_active_bootstrap_handoff_kernel_payload` | fingerprint: calls _bootstrap_preflight_passed, ValueError, _bootstrap_geometry_preflight_kernel_payload; returns _bootstrap_geometry_preflight_kernel_payload(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1494 | def | `_active_bootstrap_handoff_kernel_hash` | fingerprint: calls stable_config_hash, _active_bootstrap_handoff_kernel_payload; returns stable_config_hash(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1507 | def | `_phase7_windowed_mass_seed_kernel_payload` | doc: Choose the private kernel used to collect Phase 4 mass-window draws. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1567 | class | `HMCGeometryInitializationConfig` | doc: Configuration for geometry-derived initial HMC kernel parameters. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1588 | def | `__post_init__` | fingerprint: calls float, ValueError, np.isfinite |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1645 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1663 | class | `HMCGeometryInitializationResult` | doc: Geometry-derived starting kernel, not HMC tuning evidence. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1681 | def | `__post_init__` | fingerprint: calls str, ValueError, int |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1721 | def | `payload` | fingerprint: calls self.config.payload, self.mass_artifact.to_payload; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1748 | def | `artifact_hash` | fingerprint: calls stable_config_hash, self.payload; returns stable_config_hash(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1753 | class | `HMCBootstrapScreenConfig` | doc: Policy-level config for the Phase 3 bootstrap screen. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1775 | def | `__post_init__` | fingerprint: calls float, ValueError, np.isfinite, object.__setattr__, _validate_band; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1824 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1843 | class | `HMCBootstrapRepairRound` | doc: One Phase 3 fixed-kernel screen attempt and repair decision. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1862 | def | `__post_init__` | fingerprint: calls object.__setattr__, int, _validate_seed, float, ValueError |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1899 | def | `passed` | fingerprint: returns Compare |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1902 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1923 | class | `HMCBootstrapScreenResult` | doc: Phase 3 bootstrap artifact; not final HMC tuning evidence. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1940 | def | `__post_init__` | fingerprint: calls str, object.__setattr__, getattr, ValueError, int; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1982 | def | `passed` | fingerprint: returns Compare |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1986 | def | `selected_round` | fingerprint: returns Subscript |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 1992 | def | `selected_kernel_payload` | fingerprint: calls _bootstrap_selected_kernel_payload; returns _bootstrap_selected_kernel_payload(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2007 | def | `selected_kernel_hash` | fingerprint: calls stable_config_hash; returns IfExp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2012 | def | `artifact_hash` | fingerprint: calls stable_config_hash, self.payload; returns stable_config_hash(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2015 | def | `payload` | fingerprint: calls self.config.payload, tuple, stable_config_hash, round_result.payload; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2050 | class | `HMCWindowedMassStageConfig` | doc: Policy-level config for the Phase 4 windowed mass stage. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2076 | def | `__post_init__` | fingerprint: calls str, ValueError, object.__setattr__, float, np.isfinite |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2190 | def | `payload` | fingerprint: calls self.staged_timeout_policy.payload, dict; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2222 | class | `HMCWindowedMassStageResult` | doc: Phase 4 adapted-mass handoff; not posterior or final tuning evidence. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2254 | def | `__post_init__` | fingerprint: calls str, object.__setattr__, getattr, ValueError, int; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2373 | def | `passed` | fingerprint: returns Compare |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2377 | def | `adapted_mass_artifact_payload` | fingerprint: calls self.operational_mass_artifact.signature_payload; returns self.windowed_mass_result.final_mass_artifact_payload |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2385 | def | `adapted_mass_artifact_signature` | fingerprint: calls _mass_artifact_signature; returns self.windowed_mass_result.final_mass_artifact_signature |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2393 | def | `candidate_step_size` | fingerprint: calls float; returns self.windowed_mass_result.final_step_size |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2401 | def | `artifact_hash` | fingerprint: calls stable_config_hash, self.payload; returns stable_config_hash(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2404 | def | `payload` | fingerprint: calls self.config.payload, self.windowed_mass_result.payload, self.operational_warmup_result.public_payload, self.operational_warmup_closeout.public_payload; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2454 | class | `HMCFixedMassStepStageConfig` | doc: Policy-level config for Phase 5 fixed-mass step tuning. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2488 | def | `__post_init__` | fingerprint: calls float, ValueError, np.isfinite, object.__setattr__, _validate_band |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2661 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2700 | class | `_HMCPhase5CandidateSourceContext` | doc: Immutable upstream lineage for the private Phase 5 candidate batch. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2718 | def | `__post_init__` | fingerprint: calls str, object.__setattr__, getattr, ValueError, int; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2751 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2771 | class | `_HMCPhase5CandidateRecord` | doc: One normalized Phase 5 candidate with source-complete private lineage. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2816 | def | `source_key` | fingerprint: returns Tuple |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2823 | def | `__post_init__` | fingerprint: calls int, ValueError, object.__setattr__; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2950 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 2996 | class | `_HMCPhase5CandidateSignalRecord` | fingerprint: calls str, int, object.__setattr__ |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3001 | def | `__post_init__` | fingerprint: calls str, ValueError, int |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3015 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3024 | class | `_HMCPhase5CandidateBatchHandoff` | doc: Private immutable Phase 5 batch consumed by later BayesFilter phases. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3040 | def | `__post_init__` | fingerprint: calls isinstance, TypeError, tuple, ValueError, all; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3086 | def | `candidate_count` | fingerprint: calls len; returns len(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3090 | def | `handoff_eligible_count` | fingerprint: calls sum; returns sum(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3093 | def | `flattened_codes` | fingerprint: calls tuple, dict.fromkeys; returns tuple(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3100 | def | `payload` | fingerprint: calls self.source_context.payload, tuple, signal.payload, record.payload; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3125 | class | `_HMCPhase7DirectCandidateQueuePlan` | doc: Private, pre-draw identity/seed/allocation contract for one attempt. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3140 | def | `__post_init__` | fingerprint: calls str, ValueError, object.__setattr__, getattr; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3196 | def | `candidate_count` | fingerprint: calls len; returns len(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3200 | def | `allocated_start_count` | fingerprint: calls min; returns min(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3203 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3229 | def | `__repr__` | fingerprint: returns JoinedStr |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3240 | class | `_HMCPhase7FixedKernelVerificationInput` | doc: Immutable source-normalized mechanics and lineage for one verifier call. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3272 | def | `__post_init__` | fingerprint: calls str, ValueError, object.__setattr__, int; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3422 | def | `source_identity` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3449 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3485 | def | `__repr__` | fingerprint: returns JoinedStr |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3494 | class | `_HMCPhase7FixedKernelVerificationOutcome` | doc: Private role-separated result from the shared Phase 7 verifier core. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3519 | def | `__post_init__` | fingerprint: calls isinstance, TypeError, dict, object.__setattr__ |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3594 | def | `historical_tuple` | fingerprint: returns Tuple |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3613 | def | `__repr__` | fingerprint: returns JoinedStr |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3624 | class | `_HMCPhase7DirectCandidateQueueResult` | doc: Private aggregate preserving per-candidate states without public leakage. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3649 | def | `__post_init__` | fingerprint: calls isinstance, TypeError, tuple, dict, len, ValueError; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3743 | def | `started_count` | fingerprint: calls sum, item.get; returns sum(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3747 | def | `not_run_count` | fingerprint: calls sum, item.get; returns sum(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3751 | def | `cost_stopped_count` | fingerprint: calls sum, item.get; returns sum(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3758 | def | `promotion_vetoed_count` | fingerprint: calls sum, item.get; returns sum(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3765 | def | `repair_direction_conflict` | fingerprint: calls len; returns Compare |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3768 | def | `private_diagnostics` | fingerprint: calls dict, max; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3799 | def | `__repr__` | fingerprint: returns JoinedStr |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3810 | class | `_HMCPhase7FixedKernelVerificationExecution` | doc: Execution-mode output consumed once by the shared Phase 7 finalizer. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3822 | def | `__post_init__` | fingerprint: calls object.__setattr__, dict, isinstance, TypeError |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3844 | class | `HMCFixedMassStepStageResult` | doc: Phase 5 fixed-mass step handoff; not trajectory or posterior evidence. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 3901 | def | `__post_init__` | fingerprint: calls str, object.__setattr__, getattr, ValueError, float; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4062 | def | `passed` | fingerprint: returns Compare |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4066 | def | `selected_step_size` | fingerprint: calls float; returns float(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4072 | def | `repair_step_size` | fingerprint: calls float; returns float(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4078 | def | `artifact_hash` | fingerprint: calls stable_config_hash, self.payload; returns stable_config_hash(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4081 | def | `private_evidence_ledger` | doc: Return aggregate operational evidence through a private-only API. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4088 | def | `payload` | fingerprint: calls self.config.payload, self.budget_ladder_result.payload, len, tuple; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4216 | def | `_phase5_candidate_batch_handoff` | doc: Return the validated private Phase 5 candidate batch, when one exists. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4250 | def | `_phase5_candidate_source_key` | fingerprint: calls int, str; returns Tuple |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4260 | def | `_phase5_normalize_optional_number` | fingerprint: calls source.get, float, np.isfinite; returns Tuple |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4276 | def | `_phase5_candidate_ladders_by_source` | fingerprint: calls int, str, round_payload.get, tuple, isinstance, ValueError; returns ladders_by_source; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4307 | def | `_phase5_ladder_target_scope` | fingerprint: calls isinstance, str, payload.get; returns str(...); loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4316 | def | `_phase5_candidate_record` | fingerprint: calls _phase5_candidate_source_key, _validate_handoff_screen_policy, candidate.get, ValueError, _phase5_normalize_optional_number; returns _HMCPhase5CandidateRecord(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4424 | def | `_phase5_candidate_signal_records` | fingerprint: calls set, seen.add, signals.append, _HMCPhase5CandidateSignalRecord, tuple; returns tuple(...); loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4450 | def | `_phase5_candidate_reference` | fingerprint: calls len, ValueError; returns Tuple |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4463 | def | `_phase5_verification_order_seed` | fingerprint: calls tuple, ValueError; returns Tuple |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4478 | def | `_build_phase5_candidate_batch_handoff` | fingerprint: calls tuple, _phase5_candidate_ladders_by_source, _validate_handoff_screen_policy, _phase5_candidate_record, enumerate; returns handoff |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4541 | def | `_validate_phase5_candidate_batch_handoff` | fingerprint: calls stable_config_hash, ValueError, handoff.payload, any; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4600 | def | `referenced_record` | fingerprint: calls ValueError, range, len; returns record |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4681 | class | `HMCFrozenStepTrajectoryStageConfig` | doc: Policy-level config for Phase 6 frozen-step trajectory tuning. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4710 | def | `__post_init__` | fingerprint: calls float, ValueError, np.isfinite, object.__setattr__, _validate_band |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4820 | def | `payload` | fingerprint: calls self.staged_timeout_policy.payload, dict; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4853 | class | `HMCFrozenStepTrajectoryStageResult` | doc: Phase 6 frozen-step trajectory handoff; not final verification evidence. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4888 | def | `__post_init__` | fingerprint: calls str, object.__setattr__, getattr, ValueError, float; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4980 | def | `passed` | fingerprint: returns Compare |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4984 | def | `selected_candidate` | fingerprint: returns Subscript |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4990 | def | `selected_num_leapfrog_steps` | fingerprint: calls int; returns IfExp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 4995 | def | `selected_trajectory_length` | fingerprint: calls float; returns IfExp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 5000 | def | `artifact_hash` | fingerprint: calls stable_config_hash, self.payload; returns stable_config_hash(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 5003 | def | `payload` | fingerprint: calls self.config.payload; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 5051 | class | `HMCTuneVerifyRepairLoopConfig` | doc: Policy-level config for the Phase 7 outer tuning loop. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 5099 | def | `__post_init__` | fingerprint: calls str, ValueError, object.__setattr__ |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 5366 | def | `payload` | fingerprint: calls candidate_handoff_policy_payload, self.staged_timeout_policy.payload, dict; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 5434 | class | `HMCTuneVerifyRepairAttempt` | doc: One Phase 7 tune-verify attempt and its private budget state. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 5452 | def | `__post_init__` | fingerprint: calls object.__setattr__, int, dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 5488 | def | `passed` | fingerprint: returns Compare |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 5491 | def | `payload` | fingerprint: calls self.verification_callback_result.payload, self.windowed_stage.payload, self.fixed_mass_step_stage.payload, self.frozen_step_trajectory_stage.payload; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 5518 | class | `HMCTuneVerifyRepairLoopResult` | doc: Phase 7 frozen-kernel handoff after tune/verify/repair. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 5538 | def | `__post_init__` | fingerprint: calls str, object.__setattr__, getattr, ValueError, int; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 5595 | def | `passed` | fingerprint: returns Compare |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 5599 | def | `artifact_hash` | fingerprint: calls stable_config_hash, self.payload; returns stable_config_hash(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 5602 | def | `private_evidence_ledger` | doc: Collect private Phase 5 evidence without changing serialized payloads. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 5633 | def | `payload` | fingerprint: calls _public_final_kernel_summary_from_private_payload, self.config.payload, tuple, len, attempt.payload; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 5673 | class | `HMCKernelTuningConfig` | doc: Public one-call HMC kernel tuning policy. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 5737 | def | `__post_init__` | fingerprint: calls str, ValueError, object.__setattr__ |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6104 | def | `smoke` | fingerprint: calls payload.update, cls; returns cls(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6115 | def | `standard` | fingerprint: calls payload.update, cls; returns cls(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6125 | def | `diagnostic` | fingerprint: calls payload.update, cls; returns cls(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6135 | def | `diagnostic_plus` | fingerprint: calls payload.update, cls; returns cls(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6145 | def | `serious` | fingerprint: calls payload.update, cls; returns cls(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6156 | def | `is_smoke` | fingerprint: returns Compare |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6160 | def | `uses_serious_budget_policy` | fingerprint: returns Compare |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6163 | def | `payload` | fingerprint: calls candidate_handoff_policy_payload, _geometry_scaled_budget_timing_policy(...).payload, _public_tuning_preset_role, _public_tuning_forbidden_fields, self.staged_timeout_policy.payload, dict; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6267 | class | `HMCKernelTuningResult` | doc: Public one-call HMC kernel tuning result and frozen-kernel handoff gate. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6288 | def | `__post_init__` | fingerprint: calls isinstance, TypeError, str, ValueError, object.__setattr__, int |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6381 | def | `passed` | fingerprint: returns Compare |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6385 | def | `final_frozen_kernel_handoff` | fingerprint: returns self.final_kernel_payload |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6389 | def | `artifact_hash` | fingerprint: calls stable_config_hash, self.payload; returns stable_config_hash(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6392 | def | `private_evidence_ledger` | doc: Return the private operational evidence ledger, when Phase 7 ran. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6399 | def | `payload` | fingerprint: calls self.config.payload, self.geometry.payload, self.bootstrap.payload, self.tune_verify_repair_loop.payload; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6448 | class | `RetainedFrozenKernelAdapterReplayResult` | doc: BayesFilter-owned replay of a verified retained fixed-kernel adapter. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6463 | def | `__post_init__` | fingerprint: calls isinstance, TypeError, dict, str, contract.get |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6485 | def | `payload` | fingerprint: calls self.final_kernel_payload.get; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6516 | def | `step_size` | fingerprint: calls self.final_kernel_payload.get, ValueError, float; returns float(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6523 | def | `num_leapfrog_steps` | fingerprint: calls self.final_kernel_payload.get, ValueError, int; returns int(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6530 | def | `build_retained_frozen_kernel_hmc_adapter_from_tuning_payload` | doc: Rebuild the BayesFilter adapter stack verified by one-call tuning. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6663 | def | `build_retained_frozen_kernel_hmc_adapter_from_tuning_result` | doc: Replay a passed public tuning result without exposing mechanics to callers. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6729 | def | `admitted_kernel_mechanics_payload_from_tuning_result` | doc: Extract the transition mechanics needed for durable kernel replay. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 6804 | def | `admitted_kernel_mechanics_payload_from_serialized_tuning_payload` | doc: Migrate a passed serialized private tuning result into replay mechanics. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 7038 | def | `build_retained_frozen_kernel_hmc_adapter_from_mechanics_payload` | doc: Rebuild an admitted retained HMC adapter without invoking the tuner. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 7148 | def | `_required_mapping` | fingerprint: calls payload.get, isinstance, ValueError; returns value |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 7155 | def | `_replay_final_kernel_payload` | fingerprint: calls _required_mapping, loop.get, isinstance, top_level_payload.get, ValueError; returns top_level_payload |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 7171 | def | `_geometry_config_from_payload` | fingerprint: calls HMCGeometryInitializationConfig, float, int, bool; returns HMCGeometryInitializationConfig(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 7202 | def | `_resolve_replay_target_scope` | fingerprint: calls value_score_capability, final_kernel_payload.get, config_payload.get, str, ValueError; returns scope; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 7232 | def | `_expected_phase4_adapter_signature` | fingerprint: calls bootstrap_payload.get, candidates.append, str, _replay_attempts, attempt.get, isinstance; returns _single_replay_signature(...); loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 7260 | def | `_expected_final_adapter_signature` | fingerprint: calls _replay_final_kernel_payload, final_kernel.get, _replay_attempts, attempt.get; returns _single_replay_signature(...); loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 7293 | def | `_replay_attempts` | fingerprint: calls _required_mapping, loop.get, isinstance, ValueError, tuple; returns IfExp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 7305 | def | `_single_replay_signature` | fingerprint: calls tuple, dict.fromkeys, str, ValueError, len; returns Subscript |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 7314 | def | `initialize_hmc_kernel_geometry` | doc: Build an initial mass artifact and formula-derived epsilon/L. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 7393 | def | `run_hmc_bootstrap_screen` | doc: Run a short fixed-kernel screen and bounded epsilon repair. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 7456 | def | `emit_bootstrap_progress` | fingerprint: calls int, bool, payload.get |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 7890 | def | `_operational_windowed_mass_capture` | doc: Run R3 and build a deliberately non-operational v1 compatibility view. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 7933 | def | `boundary_callback` | fingerprint: calls _windowed_mass_public_timeout_preflight, dict, sum, int, item.get; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 7954 | def | `segment_callback` | fingerprint: calls int, _windowed_mass_public_timeout_preflight, _emit_windowed_mass_progress, dict, time.perf_counter; returns Constant |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 8237 | def | `run_hmc_windowed_mass_stage` | doc: Capture retained diagnostic draws and run windowed mass adaptation. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 8892 | def | `run_hmc_fixed_mass_step_stage` | doc: Run Phase 5 fixed-mass step tuning from a passed Phase 4 handoff. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 9411 | def | `_run_operational_fixed_mass_step_stage` | doc: Run R5 from the frozen operational bank, never historical latent zero. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 9772 | def | `run_hmc_frozen_step_trajectory_stage` | doc: Run Phase 6 frozen-step trajectory tuning from a passed Phase 5 handoff. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 10396 | def | `_coerce_private_runner_cache_handoff_mapping` | fingerprint: calls isinstance, TypeError, dict; returns dict(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 10404 | def | `_coerce_private_runner_contract_handoff_mapping` | fingerprint: calls isinstance, TypeError, str, dict, value.items; returns DictComp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 10414 | def | `_callable_accepts_private_diagnostic_callback` | fingerprint: calls inspect.signature, any, parameters.values; returns any(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 10427 | def | `_callable_accepts_runner_cache_handoff` | fingerprint: calls inspect.signature, any, parameters.values; returns any(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 10440 | def | `run_hmc_tune_verify_repair_loop` | doc: Run Phase 7 tune/verify/repair with private budget escalation. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 10993 | def | `forward_windowed_mass_progress` | fingerprint: calls _emit_phase7_progress, bool, _windowed_mass_progress_extra, payload.get |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 11801 | def | `tune_hmc_kernel` | doc: Tune a frozen HMC kernel from model-facing inputs. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 11908 | def | `write_private_event` | fingerprint: calls _write_private_tuning_event, int; returns event |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 11926 | def | `write_private_mass_event` | fingerprint: calls _write_private_mass_matrix_artifact, dict, write_private_event; returns write_private_event(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 11949 | def | `write_private_tuning_diagnostic` | fingerprint: calls str, payload.get, isinstance, write_private_mass_event, payload.items |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12025 | def | `write_progress` | fingerprint: calls phase7_state.get, isinstance |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12106 | def | `write_loop_progress` | fingerprint: calls dict, int, event_payload.get, isinstance |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12141 | def | `write_result_artifact` | fingerprint: calls write_progress, _write_public_tuning_artifact_if_requested, str |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12257 | def | `write_bootstrap_progress` | fingerprint: calls write_progress, stage.endswith, dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12589 | class | `_GeometryHint` | fingerprint: body AnnAssign, AnnAssign, AnnAssign, AnnAssign |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12596 | def | `_validate_position` | fingerprint: calls np.asarray, ValueError, np.all, np.isfinite; returns array.copy(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12607 | def | `_select_geometry_hint` | fingerprint: calls _identity_hint, _hint_from_value, failures.append, tuple, str; returns _identity_hint(...); loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12655 | def | `_hint_from_value` | fingerprint: calls int, _validate_matrix, PrecomputedMassArtifact.from_negative_hessian, _GeometryHint, dict, bool; raises ValueError(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12760 | def | `_identity_hint` | fingerprint: calls int, _GeometryHint, np.eye, dict, bool; returns _GeometryHint(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12788 | def | `_validate_matrix` | fingerprint: calls np.asarray, ValueError, np.all, np.isfinite; returns BinOp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12797 | def | `_validate_spd` | fingerprint: calls np.linalg.eigvalsh, np.all, ValueError, np.isfinite, np.any |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12805 | def | `_build_mass_artifact` | fingerprint: calls str, hint.report.get, PrecomputedMassArtifact.from_covariance, dict; returns PrecomputedMassArtifact.from_covariance(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12831 | def | `_curvature_frequencies` | fingerprint: calls np.linalg.eigh, np.any, ValueError, np.all, np.isfinite, np.diag; returns np.sqrt(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12849 | def | `_target_trajectory_length` | fingerprint: calls float, np.median, ValueError, np.isfinite; returns float(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12856 | def | `_initial_step_size` | fingerprint: calls float, np.sqrt, np.mean, np.square, np.max; returns step |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12877 | def | `_curvature_report` | fingerprint: calls float, int, bool; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12889 | def | `_derive_seed` | fingerprint: calls int; returns Tuple |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12893 | def | `_mass_artifact_signature` | fingerprint: calls stable_config_hash, mass_artifact.signature_payload, np.asarray; returns stable_config_hash(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12904 | class | `_BootstrapFixedMassLatentValueScoreAdapter` | doc: Latent fixed-mass target with a mass-bound stable adapter signature. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12907 | def | `__init__` | fingerprint: calls hasattr, TypeError, bool, getattr |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12943 | def | `adapter_signature` | fingerprint: returns self._adapter_signature |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12946 | def | `value_score_capability` | fingerprint: calls value_score_capability, str, bool, ValueScoreCapability; returns ValueScoreCapability(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12969 | def | `initial_position` | fingerprint: calls tf.zeros; returns tf.zeros(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12974 | def | `latent_to_position` | fingerprint: calls self._validate_trailing_dimension, tf.convert_to_tensor, tf.tensordot; returns BinOp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12982 | def | `theta_score_to_latent_score` | fingerprint: calls self._validate_trailing_dimension, tf.convert_to_tensor, tf.tensordot; returns tf.tensordot(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12989 | def | `log_prob_and_grad` | fingerprint: calls self._log_prob_and_grad_status; returns Tuple |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 12993 | def | `_log_prob_and_grad_status` | fingerprint: calls self._validate_trailing_dimension, self.latent_to_position, self.base_adapter.log_prob_and_grad_status, self.base_adapter.log_prob_and_grad, tf.convert_to_tensor; returns Tuple |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13014 | def | `log_prob_and_grad_status` | fingerprint: calls TypeError, self._log_prob_and_grad_status, isinstance, dict; returns Tuple |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13024 | def | `target_status_telemetry` | fingerprint: calls getattr, callable, TypeError, self.latent_to_position, telemetry, isinstance; returns payload |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13034 | def | `_validate_trailing_dimension` | fingerprint: calls tf.convert_to_tensor, ValueError, int; returns tensor |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13052 | def | `_resolve_bootstrap_target_scope` | fingerprint: calls value_score_capability, str, ValueError; returns str(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13073 | def | `_resolve_windowed_stage_target_scope` | fingerprint: calls value_score_capability, str, ValueError; returns str(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13094 | def | `_resolve_fixed_mass_step_stage_target_scope` | fingerprint: calls value_score_capability, str, ValueError; returns str(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13115 | def | `_resolve_frozen_step_trajectory_stage_target_scope` | fingerprint: calls value_score_capability, str, ValueError; returns str(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13136 | def | `_validate_windowed_stage_inputs` | fingerprint: calls stable_adapter_signature, ValueError, _bootstrap_preflight_passed, geometry.mass_artifact.validate_for_adapter |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13164 | def | `_validate_fixed_mass_step_stage_inputs` | fingerprint: calls ValueError, stable_adapter_signature |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13227 | def | `_validate_frozen_step_trajectory_stage_inputs` | fingerprint: calls ValueError, stable_adapter_signature |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13354 | def | `_validate_tune_verify_loop_inputs` | fingerprint: calls stable_adapter_signature, ValueError, _bootstrap_preflight_passed |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13377 | def | `_public_tuning_forbidden_fields` | fingerprint: returns Tuple |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13398 | def | `_public_tuning_preset_role` | fingerprint: calls ValueError; raises ValueError(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13412 | def | `_public_geometry_config` | fingerprint: calls HMCGeometryInitializationConfig, _derive_seed; returns HMCGeometryInitializationConfig(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13429 | def | `_public_bootstrap_config` | fingerprint: calls _geometry_scaled_budget_timing_policy(...).bootstrap_screen_counts, int, _geometry_scaled_budget_timing_policy, HMCBootstrapScreenConfig, _derive_seed; returns HMCBootstrapScreenConfig(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13470 | def | `_public_loop_config` | fingerprint: calls HMCTuneVerifyRepairLoopConfig, _derive_seed; returns HMCTuneVerifyRepairLoopConfig(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13532 | def | `_public_budget_policy_factory` | fingerprint: calls _geometry_scaled_budget_timing_policy, HMCOperationalStatisticalWorkPolicy, _default_attempt_budget_policy, _operational_verification_starts_per_outer_attempt, ValueError; returns factory |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13572 | def | `serious_factory` | fingerprint: calls _default_attempt_budget_policy, ValueError; returns policy |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13589 | def | `factory` | fingerprint: calls int, ValueError, min, max; returns _HMCAttemptBudgetPolicy(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13697 | def | `_public_final_kernel_handoff_payload` | fingerprint: calls ValueError, _public_final_kernel_summary_from_private_payload; returns _public_final_kernel_summary_from_private_payload(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13708 | def | `_public_final_kernel_summary_from_private_payload` | doc: Return a non-replayable public summary of a private frozen HMC kernel. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13753 | def | `_phase7_private_resume_split_contract` | doc: Build a private-only Phase 7 repair handoff contract. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13828 | def | `_phase7_resume_split_entry_stage` | fingerprint: calls str, handoff_state.get, ValueError; raises ValueError(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13841 | def | `_phase7_resume_split_public_summary` | doc: Return the public-safe summary of a private resume/split contract. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13891 | def | `_public_tuning_artifact_path` | fingerprint: calls Path; returns BinOp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13900 | def | `_public_tuning_progress_path` | fingerprint: calls Path, path.with_name; returns BinOp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13909 | def | `_private_tuning_diagnostics_dir` | fingerprint: calls Path; returns BinOp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13918 | def | `_private_tuning_events_path` | fingerprint: returns BinOp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13924 | def | `_utc_now_isoformat` | fingerprint: calls datetime.now(...).isoformat(...).replace, datetime.now(...).isoformat, datetime.now; returns datetime.now(...).isoformat(...).replace(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13928 | def | `_selection_route_public_payload` | fingerprint: calls require_hmc_algorithm_route, bool, decision.payload, windowed_algorithm_for_selection_algorithm; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 13964 | def | `_mass_matrix_private_summary` | fingerprint: calls np.asarray, np.diag, np.linalg.eigvalsh, bool, np.all; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14000 | def | `_write_private_mass_matrix_artifact` | fingerprint: calls private_dir.mkdir, _mass_artifact_signature, Constant.join(...).strip, Constant.join, str, char.isalnum; returns _mass_matrix_private_summary(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14029 | def | `_private_event_hash` | fingerprint: calls stable_config_hash, event.items; returns stable_config_hash(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14033 | def | `_write_private_tuning_event` | fingerprint: calls events_path.parent.mkdir, str, _utc_now_isoformat, dict, _private_event_hash; returns event |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14063 | def | `_private_tuning_public_summary` | fingerprint: calls bool, int, state.get; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14088 | def | `_write_public_tuning_progress_if_requested` | fingerprint: calls progress_path.parent.mkdir, _selection_route_public_payload, str, _utc_now_isoformat, os.getpid |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14166 | def | `_write_public_tuning_artifact_if_requested` | fingerprint: calls artifact_path.parent.mkdir, _public_tuning_artifact_payload, artifact_path.write_text, json.dumps, _json_ready |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14180 | def | `_public_tuning_artifact_payload` | fingerprint: calls _phase7_public_summary, _phase7_early_closeout_public_summary, _selection_route_public_payload, result.config.payload, _bootstrap_public_summary; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14255 | def | `_phase7_early_closeout_public_summary` | fingerprint: calls isinstance; returns DictComp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14294 | def | `_phase7_public_summary` | fingerprint: calls _frozen_step_trajectory_public_summary, tuple, _phase7_attempt_public_summary, _phase7_loop_resume_split_public_summary, _selection_route_public_payload, len; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14352 | def | `_phase7_terminal_budget_guard_public_summary` | fingerprint: calls isinstance; returns DictComp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14411 | def | `_phase7_checkpoint_progress_extra` | doc: Return the only checkpoint payload shape allowed in public progress. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14432 | def | `_phase7_attempt_public_summary` | doc: Summarize a Phase 7 attempt without exposing HMC mechanics. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14492 | def | `_phase7_loop_resume_split_public_summary` | fingerprint: calls tuple, next, int, ValueError, _phase7_attempt_resume_split_public_summary |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14536 | def | `_phase7_attempt_resume_split_public_summary` | doc: Public-safe resume/split availability without private HMC mechanics. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14610 | def | `_stage_status_public_summary` | fingerprint: calls getattr, bool, tuple; returns summary |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14634 | def | `_windowed_mass_public_timeout_closeout_summary` | fingerprint: calls stage.diagnostics.get, isinstance; returns DictComp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14671 | def | `_fixed_mass_step_public_timeout_closeout_summary` | fingerprint: calls stage.diagnostics.get, isinstance; returns DictComp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14724 | def | `_phase7_attempt_budget_public_summary` | fingerprint: calls isinstance; returns summary |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14761 | def | `_phase7_verification_public_summary` | fingerprint: calls isinstance, config_payload.get; returns summary |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14899 | def | `_phase7_pre_windowed_timeout_public_summary` | fingerprint: calls isinstance; returns DictComp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14938 | def | `_checkpoint_reference_is_public_safe` | fingerprint: calls isinstance, assert_sequential_rhat_checkpoint_public_reference_safe; returns Constant |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14948 | def | `_verification_acceptance_log_health_passed` | fingerprint: calls diagnostics.get; returns Compare |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14954 | def | `_verification_target_value_health_passed` | fingerprint: calls diagnostics.get; returns Compare |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14960 | def | `_public_tuning_diagnostic_roles` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 14970 | def | `_bootstrap_public_summary` | doc: Summarize Phase 3 routing evidence without raw HMC mechanics. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 15095 | def | `_bootstrap_acceptance_relation` | fingerprint: calls _scalar_or_none, np.isfinite; returns Constant |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 15110 | def | `_public_bootstrap_hard_veto_category` | fingerprint: calls mapping.get, str; returns mapping.get(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 15123 | def | `_bootstrap_cap_saturation_direction` | fingerprint: calls tuple, dict.fromkeys, len, any; returns Constant |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 15134 | def | `_bootstrap_round_metadata` | fingerprint: calls diagnostics.get, isinstance; returns IfExp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 15139 | def | `_bootstrap_round_jit_metadata` | fingerprint: calls _bootstrap_round_metadata, _bool_or_none, metadata.get; returns IfExp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 15147 | def | `_summarize_bootstrap_jit_metadata` | fingerprint: calls any, all; returns Constant |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 15157 | def | `_bootstrap_round_timing_scope` | fingerprint: calls _bootstrap_round_metadata, metadata.get, str; returns IfExp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 15166 | def | `_summarize_bootstrap_timing_scope` | fingerprint: calls tuple, dict.fromkeys, len; returns IfExp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 15173 | def | `_bootstrap_round_uses_fixture_or_synthetic` | fingerprint: calls _bootstrap_round_metadata, _metadata_marks_fixture_or_synthetic; returns _metadata_marks_fixture_or_synthetic(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 15180 | def | `_resolve_tune_verify_loop_target_scope` | fingerprint: calls value_score_capability, str, ValueError; returns str(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 15202 | class | `_HMCAttemptBudgetPolicy` | fingerprint: calls tuple, object.__setattr__, operational_fields.items, any |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 15235 | def | `__post_init__` | fingerprint: calls int, object.__setattr__, getattr, ValueError, tuple; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 15427 | def | `payload` | fingerprint: calls dict; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 15498 | def | `_phase7_progress_budget_payload` | fingerprint: calls dict; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 15532 | def | `_emit_windowed_mass_progress` | fingerprint: calls str, bool, route_decision.payload, int |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 15594 | def | `_windowed_mass_progress_extra` | fingerprint: returns DictComp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 15626 | def | `_staged_timeout_round` | fingerprint: calls int, rounds.get, str; returns int(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 15635 | def | `_staged_timeout_public_state` | fingerprint: calls time.perf_counter, float, str, max; returns payload |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 15703 | def | `_staged_timeout_public_payload_for_config` | fingerprint: calls _staged_timeout_public_state; returns _staged_timeout_public_state(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 15727 | def | `_windowed_mass_public_timeout_state` | fingerprint: calls _staged_timeout_public_payload_for_config, float, max, min, time.perf_counter; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 15781 | def | `_phase7_early_global_timeout_before_loop` | fingerprint: calls _staged_timeout_public_state, _bootstrap_public_summary, bool, dict, int, float; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 15892 | def | `_windowed_mass_public_timeout_preflight` | fingerprint: calls dict, _windowed_mass_public_timeout_state, bool, float, str; returns payload |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 15934 | def | `_windowed_mass_next_segment_soft_deadline_preflight` | fingerprint: calls dict, _windowed_mass_public_timeout_state, tuple, float, min, max; returns payload |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 16015 | def | `_phase7_public_timeout_before_windowed_mass` | fingerprint: calls _windowed_mass_public_timeout_preflight, dict, int; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 16052 | def | `_budget_ladder_progress_extra` | fingerprint: returns DictComp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 16091 | def | `_trajectory_candidate_progress_extra` | fingerprint: calls str, int, stable_config_hash; returns payload |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 16149 | def | `_phase6_soft_deadline_state` | fingerprint: calls float, max, min, time.perf_counter; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 16192 | def | `_phase6_next_candidate_soft_deadline_veto` | fingerprint: calls dict, _phase6_soft_deadline_state, tuple, float, max; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 16245 | def | `_fixed_mass_step_public_timeout_state` | fingerprint: calls _staged_timeout_public_payload_for_config, float, max, min, time.perf_counter; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 16307 | def | `_fixed_mass_step_next_candidate_soft_deadline_veto` | fingerprint: calls dict, _fixed_mass_step_public_timeout_state, tuple, float, max; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 16381 | def | `_phase7_verification_acceptance_budget_blocker` | fingerprint: calls _phase7_should_run_operational_repair_verification, float, time.perf_counter, max; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 16478 | def | `_phase7_verify_only_budget_saturation_blocker` | fingerprint: calls _phase7_should_retry_verification_only, int, _phase7_verification_num_results; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 16527 | def | `_phase7_extended_attempt_stall_blocker` | fingerprint: calls int, str, tuple, _phase7_should_retry_verification_only, _phase7_verification_num_results; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 16607 | def | `_phase7_repair_handoff_attempt_slot_blocker` | fingerprint: calls int; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 16654 | def | `_phase7_terminal_phase6_repair_slot_eligible` | fingerprint: calls int, str, tuple; returns BoolOp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 16677 | def | `_phase7_terminal_phase6_repair_slot_payload` | fingerprint: calls int; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 16718 | def | `_phase7_terminal_phase6_repair_slot_exhausted_payload` | fingerprint: calls int, bool; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 16770 | def | `_phase7_should_retry_verification_only` | fingerprint: returns BoolOp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 16783 | def | `_phase7_should_run_operational_repair_verification` | fingerprint: calls attempt_state.direct_candidate_handoff.get; returns BoolOp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 16803 | def | `_phase7_should_prepare_verification_only_retry` | fingerprint: calls tuple, str, _phase7_should_retry_verification_only, _verification_acceptance_log_health_passed, _verification_target_value_health_passed; returns BoolOp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 16831 | def | `_phase7_verification_result_supports_verification_only_retry` | doc: Return true only for valid verify-only retry outcomes. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 16877 | def | `_emit_phase7_progress` | fingerprint: calls str, int, bool, _phase7_progress_budget_payload, dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 16933 | def | `_phase7_verified_endpoint_payload` | fingerprint: calls ValueError, stable_config_hash, evidence.payload, getattr; returns payload |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 16968 | def | `_coerce_phase7_verified_endpoint` | fingerprint: calls isinstance, ValueError, dict, str, result.pop, result.get; returns result; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 17011 | def | `_coerce_phase7_verified_acceptance_bracket_state` | fingerprint: calls isinstance, ValueError, dict, str, result.pop, result.get; returns result |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 17075 | def | `_phase7_verified_acceptance_bracket_payload` | fingerprint: calls stable_config_hash, _coerce_phase7_verified_acceptance_bracket_state; returns _coerce_phase7_verified_acceptance_bracket_state(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 17100 | class | `_HMCPhaseAttemptState` | fingerprint: calls dataclasses.field, object.__setattr__ |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 17131 | def | `__post_init__` | fingerprint: calls dict, object.__setattr__, str, np.asarray(...).copy, canonical_theta.setflags |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 17426 | def | `has_mass_handoff` | fingerprint: returns BoolOp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 17433 | def | `has_step_handoff` | fingerprint: returns BoolOp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 17441 | def | `has_required_repair_handoff` | fingerprint: returns self.has_stage_repair_handoff |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 17445 | def | `has_stage_repair_handoff` | fingerprint: returns BoolOp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 17469 | def | `has_final_kernel_handoff` | fingerprint: returns BoolOp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 17481 | def | `payload` | fingerprint: calls stable_config_hash, self.canonical_theta_state.tolist; returns payload |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 17543 | def | `_coerce_phase7_fixed_mass_bracket_state` | fingerprint: calls isinstance, ValueError, _phase7_positive_finite_or_none, state.get; returns result |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 17585 | def | `_phase7_positive_finite_or_none` | fingerprint: calls _scalar_or_none, np.isfinite, float; returns float(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 17592 | def | `_default_attempt_budget_policy` | fingerprint: calls _geometry_scaled_budget_timing_policy, dict, central.attempt_budget_payload, int, payload.update; returns _attempt_budget_policy_from_payload(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 17641 | def | `_phase7_attempt_seed` | fingerprint: calls _derive_seed, int; returns _derive_seed(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 17645 | def | `_phase7_direct_candidate_seed_reports` | doc: Preserve private direct seed provenance without changing legacy fields. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 17816 | def | `_phase7_direct_candidate_seed` | doc: Fold stable candidate identity into the attempt seed without Python hash(). |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 17849 | def | `_phase7_direct_candidate_queue_plan` | doc: Validate the entire direct queue and its seed map before any draw. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 17902 | def | `_phase7_uses_operational_budget_policy` | fingerprint: returns Compare |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 17908 | def | `_phase7_verification_num_results` | fingerprint: calls int, _phase7_uses_operational_budget_policy; returns int(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 17919 | def | `_phase7_verification_num_burnin_steps` | fingerprint: calls int, _phase7_uses_operational_budget_policy; returns int(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 17930 | def | `_phase7_operational_work_reconciliation` | doc: Charge every started verification at its cap against the Phase 5 manifest. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18002 | def | `_phase7_direct_candidate_queue_route` | doc: Choose direct, terminal-joint, or explicit historical compatibility. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18037 | def | `_phase7_direct_candidate_queue_timeout_closeout` | doc: Apply the existing reserve policy before every direct verifier start. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18130 | def | `_phase7_phase5_candidates_not_run_diagnostics` | doc: Preserve private candidate closeout without treating unrun work as failure. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18175 | def | `_staged_timeout_stage_budget` | fingerprint: calls float, str; returns float(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18185 | def | `_phase7_windowed_stage_config` | fingerprint: calls HMCWindowedMassStageConfig, windowed_algorithm_for_selection_algorithm, _derive_seed, _staged_timeout_stage_budget, _phase7_attempt_seed, time.perf_counter; returns HMCWindowedMassStageConfig(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18222 | def | `_phase7_fixed_step_stage_config` | fingerprint: calls HMCFixedMassStepStageConfig, _derive_seed, _staged_timeout_stage_budget, _phase7_attempt_seed, time.perf_counter; returns HMCFixedMassStepStageConfig(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18272 | def | `_phase7_trajectory_stage_config` | fingerprint: calls HMCFrozenStepTrajectoryStageConfig, _derive_seed, _staged_timeout_stage_budget, _phase7_attempt_seed, time.perf_counter; returns HMCFrozenStepTrajectoryStageConfig(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18312 | def | `_build_bootstrap_fixed_mass_adapter` | fingerprint: calls mass_artifact.build_latent_transform, program_signature, stable_adapter_signature, transform.signature_payload, _BootstrapFixedMassLatentValueScoreAdapter; returns _BootstrapFixedMassLatentValueScor... |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18342 | def | `_phase4_latent_adapter_for_step_stage` | fingerprint: calls _mass_artifact_signature, ValueError, _phase4_adapted_mass_artifact, isinstance, TypeError; returns hmc_adapter |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18369 | def | `_phase4_adapted_mass_artifact` | fingerprint: calls ValueError, isinstance, TypeError; returns artifact |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18383 | def | `_selected_bootstrap_kernel_from_windowed_stage` | fingerprint: calls _active_bootstrap_handoff_kernel_payload, _active_bootstrap_handoff_kernel_hash, ValueError; returns selected |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18402 | def | `_mass_window_seed_kernel_from_windowed_stage` | fingerprint: calls _selected_bootstrap_kernel_from_windowed_stage, ValueError, payload.get, float; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18425 | def | `_fixed_mass_step_stage_ladder_config` | fingerprint: calls FixedMassHMCTuningBudgetLadderConfig, float, int, _derive_seed; returns FixedMassHMCTuningBudgetLadderConfig(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18484 | def | `_fixed_mass_step_initial_step` | fingerprint: calls float, ValueError, np.isfinite; returns float(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18508 | def | `_fixed_mass_step_initial_state_factory` | fingerprint: calls int, ValueError, np.zeros; returns factory |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18515 | def | `factory` | fingerprint: calls np.zeros; returns np.zeros(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18527 | def | `_joint_l_epsilon_anchor_l` | fingerprint: calls int, float; returns anchor |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18568 | def | `_joint_l_epsilon_grid_values` | fingerprint: calls int, ValueError, _validate_max_leapfrog_steps, np.clip; returns tuple(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18586 | def | `_joint_l_epsilon_ladder_config` | fingerprint: calls _fixed_mass_step_stage_ladder_config, int, dataclasses.replace, _round_seed; returns dataclasses.replace(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18611 | def | `_joint_l_epsilon_ladder_candidate_payload` | fingerprint: calls _validate_handoff_screen_policy, _phase23_nomination_policy_active, _scalar_or_none, int, selected.screen_diagnostics.get, float; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18763 | def | `_joint_l_epsilon_ladder_private_diagnostic_summary` | doc: Summarize the last fixed-mass ladder round for private diagnostics only. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18794 | def | `_private_log_accept_diagnostic_summary` | fingerprint: calls diagnostics.get, isinstance, tuple, trace_summary.get, str; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18842 | def | `_joint_l_epsilon_ladder_error_candidate_payload` | fingerprint: calls _validate_handoff_screen_policy, _phase23_nomination_policy_active, int, str; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18894 | def | `_collect_ladder_hard_vetoes_for_joint_grid` | fingerprint: calls values.extend, tuple, dict.fromkeys, str; returns tuple(...); loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18903 | def | `_collect_ladder_continuation_vetoes_for_joint_grid` | fingerprint: calls values.extend, tuple, dict.fromkeys, str; returns tuple(...); loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18912 | def | `_collect_ladder_repair_triggers_for_joint_grid` | fingerprint: calls values.extend, tuple, dict.fromkeys, str; returns tuple(...); loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18921 | def | `_select_joint_l_epsilon_candidate` | fingerprint: calls _validate_handoff_screen_policy, _phase23_nomination_policy_active, min, int, _scalar_or_none; returns int(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 18930 | def | `phase23_selection_key` | fingerprint: calls _scalar_or_none, candidate.get, float; returns Tuple |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 19003 | def | `_joint_l_epsilon_selected_at_grid_edge` | fingerprint: calls tuple, sorted, dict.fromkeys, int, _validate_max_leapfrog_steps; returns Constant |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 19021 | def | `_joint_l_epsilon_round_summary` | fingerprint: calls int, str, tuple; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 19055 | def | `_run_joint_l_epsilon_grid_round` | fingerprint: calls tuple, int, enumerate, _fixed_mass_step_next_candidate_soft_deadline_veto, _joint_l_epsilon_ladder_config, time.perf_counter; returns Dict; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 19149 | def | `forward_ladder_progress` | fingerprint: calls _emit_phase7_progress, bool, payload.get, _budget_ladder_progress_extra |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 19413 | def | `_select_joint_l_epsilon_repair_ladder` | fingerprint: calls _select_joint_l_epsilon_repair_ladder_with_source; returns ladder |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 19427 | def | `_select_joint_l_epsilon_repair_ladder_with_source` | fingerprint: calls round_payload.get, tuple, isinstance, int, ladders.get; returns Tuple; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 19494 | def | `_fixed_mass_step_frozen_mass_invariant` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 19508 | def | `_fixed_mass_step_stage_diagnostics` | fingerprint: calls str, type, hard_from_ladder.extend, continuation_from_ladder.extend, repair_from_ladder.extend, len; returns Dict; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 19554 | def | `_required_selected_step_size` | fingerprint: calls ValueError, float, np.isfinite; returns value |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 19566 | def | `_frozen_step_trajectory_candidate_generation` | fingerprint: calls _validate_handoff_screen_policy, _phase23_nomination_policy_active, float, ValueError, np.isfinite; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 19694 | def | `_joint_l_epsilon_selected_pair_candidate_generation` | fingerprint: calls _validate_handoff_screen_policy, _phase23_nomination_policy_active, float, ValueError, np.isfinite, int; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 19756 | def | `_frozen_step_trajectory_order_candidates` | doc: Order Phase 6 screens without changing the private candidate set. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 19784 | def | `trajectory_length` | fingerprint: calls float, int; returns BinOp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 19787 | def | `inside_rank` | fingerprint: calls trajectory_length, float; returns IfExp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 19819 | def | `_frozen_step_trajectory_screen_config` | fingerprint: calls FullChainHMCConfig, float, int; returns FullChainHMCConfig(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 19848 | def | `_frozen_step_trajectory_diagnostics_payload` | fingerprint: calls dict, _bootstrap_diagnostics_payload, payload.get, screen.get; returns payload |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 19863 | def | `_run_kernel_stage_with_optional_reusable_route` | fingerprint: calls route_events.append, run_full_chain, dict, str, stable_config_hash, bool; returns FullChainHMCRunResult(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 19951 | def | `_kernel_stage_reusable_static_contract_payload` | fingerprint: calls dict, config.signature_payload, payload.pop, _kernel_stage_reusable_state_template_contract; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 19984 | def | `_kernel_stage_reusable_state_template_contract` | fingerprint: calls tf.cast, tf.convert_to_tensor, ValueError, template.shape.as_list, any; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 20000 | def | `_kernel_stage_runner_route_summary` | fingerprint: calls int, len, ValueError, tuple, item.get, sum; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 20086 | def | `_frozen_step_trajectory_error_diagnostics` | fingerprint: calls dict, _bootstrap_error_diagnostics, payload.get; returns payload |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 20098 | def | `_call_trajectory_screen_callback` | fingerprint: calls FixedMassHMCTuningBudgetCallbackResult, callback, _coerce_trajectory_callback_result, str, type |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 20120 | def | `_call_phase7_verification_callback` | doc: Preserve callback output while retaining private exception provenance. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 20147 | def | `_coerce_trajectory_callback_result` | fingerprint: calls FixedMassHMCTuningBudgetCallbackResult, isinstance, ValueError, tuple; returns FixedMassHMCTuningBudgetCallbackResult(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 20165 | def | `_classify_frozen_step_trajectory_candidate` | fingerprint: calls hard_vetoes.append, diagnostics.get, _finite_number; returns Tuple |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 20292 | def | `_select_frozen_step_trajectory_candidate` | fingerprint: calls _phase23_nomination_policy_active, enumerate, candidate.get, min, abs, _trajectory_window_class_penalty; returns int(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 20327 | def | `_frozen_step_trajectory_selected_payload` | fingerprint: calls float, int, str; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 20364 | def | `_frozen_step_trajectory_frozen_mass_invariant` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 20378 | def | `_frozen_step_trajectory_frozen_step_invariant` | fingerprint: calls float; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 20392 | def | `_frozen_step_trajectory_stage_diagnostics` | fingerprint: calls _validate_handoff_screen_policy, _phase23_nomination_policy_active, len, int, max, tuple; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 20466 | def | `_acceptance_relation_to_band` | fingerprint: calls _scalar_or_none, float; returns Constant |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 20481 | def | `_phase7_verification_runtime_context` | doc: Rebuild the shared Phase 4 and fixed-mass adapters for verification. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 20539 | def | `_phase7_verification_initial_state` | doc: Map the frozen canonical start bank through both active affine layers. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 20607 | def | `build_operational_fixed_mass_hmc_adapter` | doc: Return the exact frozen-mass adapter and post-warmup start bank. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 20665 | def | `_phase7_historical_verification_input` | doc: Normalize the existing Phase-6-fed verifier without changing replay. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 20822 | def | `_phase7_direct_candidate_verification_input` | doc: Normalize one eligible Phase 5 record without constructing Phase 6. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 20965 | def | `_phase7_operational_selection_verification_input` | doc: Normalize the R5 v2 representative without v1 or Phase 6 lineage. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 21071 | def | `_phase7_operational_repair_verification_input` | doc: Change only epsilon while preserving the frozen operational kernel. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 21142 | def | `_phase7_operational_evidence_extension_input` | fingerprint: calls _phase7_operational_selection_verification_input, paired_candidate_seed, _phase7_attempt_seed, int, str, dataclasses.replace; returns dataclasses.replace(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 21175 | def | `_frozen_step_trajectory_public_summary` | doc: Summarize Phase 6 without exposing candidate grids or HMC mechanics. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 21333 | def | `_run_phase7_final_verification` | fingerprint: calls _phase7_historical_verification_input, _run_phase7_fixed_kernel_verification(...).historical_tuple, _run_phase7_fixed_kernel_verification; returns _run_phase7_fixed_kernel_verification(...).historic... |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 21383 | def | `_run_phase7_direct_candidate_verification` | fingerprint: calls _phase7_direct_candidate_verification_input, _run_phase7_fixed_kernel_verification; returns _run_phase7_fixed_kernel_verification(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 21425 | def | `_run_phase7_operational_selection_verification` | fingerprint: calls _phase7_operational_selection_verification_input, _run_phase7_fixed_kernel_verification; returns _run_phase7_fixed_kernel_verification(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 21465 | def | `_run_phase7_operational_repair_verification` | fingerprint: calls _phase7_operational_repair_verification_input, _run_phase7_fixed_kernel_verification; returns _run_phase7_fixed_kernel_verification(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 21507 | def | `_run_phase7_operational_evidence_extension` | fingerprint: calls _phase7_operational_evidence_extension_input, _run_phase7_fixed_kernel_verification; returns _run_phase7_fixed_kernel_verification(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 21546 | def | `_run_phase7_direct_candidate_queue` | doc: Run the audited two-start serial reference queue. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 21882 | def | `_run_phase7_fixed_kernel_verification` | fingerprint: calls isinstance, TypeError, stable_config_hash, ValueError, verification_input.payload, int; returns _finalize_phase7_fixed_kernel_verification(...); loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 21973 | def | `_run_phase7_injected_final_verification` | fingerprint: calls FullChainHMCConfig, _phase7_verification_num_results, _phase7_verification_num_burnin_steps, int, _run_kernel_stage_with_optional_reusable_route, dict; returns _HMCPhase7FixedKernelVerificationExecu... |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 22112 | def | `_run_phase7_sequential_rhat_final_verification` | fingerprint: calls _phase7_verification_num_results, min, int, HMCAcceptancePolicy; returns _HMCPhase7FixedKernelVerificationExecution(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 22280 | def | `record_checkpoint_reference` | fingerprint: calls checkpoint_references.append, checkpoint_reference_callback |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 22441 | def | `_finalize_phase7_fixed_kernel_verification` | doc: Classify once, enforce fixed mechanics once, and assign private scope. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 22572 | def | `_classify_phase7_final_verification` | fingerprint: calls diagnostics.get, _classify_phase7_acceptance_evidence_verification, hard_vetoes.append, _finite_number; returns Tuple |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 22642 | def | `_classify_phase7_acceptance_evidence_verification` | doc: Classify operational verification from role-separated v3 evidence. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 22835 | def | `_phase7_attempt_state_from_stages` | fingerprint: calls _phase4_adapted_mass_artifact, int, fixed_mass_step_stage.repair_step_payload.get, isinstance, _phase6_retry_l_anchor_payload, _phase7_verification_repair_handoff_payload; returns _HMCPhaseAttemptSt... |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 22988 | def | `_phase7_attempt_state_from_direct_outcome` | doc: Build truthful retry/final state from a Phase 5 candidate, not Phase 6. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 23206 | def | `_phase7_operational_verified_bracket_repair_handoff_payload` | doc: Build one bounded empirical bracket from frozen-kernel verifications. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 23460 | def | `_phase7_verification_repair_handoff_payload` | fingerprint: calls dict, _scalar_or_none, diagnostics.get, isinstance, hmc_acceptance_evidence_from_payload; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 23690 | def | `_phase6_retry_l_anchor_payload` | doc: Choose a private Phase 6 retry L anchor from failed candidates. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 23724 | def | `candidate_distance` | fingerprint: calls _scalar_or_none, candidate.get; returns Tuple |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 23739 | def | `acceptance_relation` | fingerprint: calls candidate.get, isinstance, diagnostics.get, _acceptance_relation_to_band; returns _acceptance_relation_to_band(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 23765 | def | `_phase6_trajectory_feasible_step_interval` | doc: Private feasible step interval implied by ``tau = L * step``. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 23817 | def | `_phase6_clamp_repair_step_to_feasible_tau` | fingerprint: calls float, ValueError, np.isfinite, _phase6_trajectory_feasible_step_interval, np.clip; returns Tuple |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 23856 | def | `_phase6_fixed_mass_bracket_state_payload` | doc: Private handoff that makes the next fixed-mass stage screen directly. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 23895 | def | `_phase6_trajectory_repair_handoff_payload` | doc: Build a private step repair when Phase 6 fails directionally. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 24355 | def | `_phase7_final_kernel_payload` | fingerprint: calls _phase4_adapted_mass_artifact, ValueError, _mass_artifact_signature; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 24417 | def | `_phase7_direct_final_kernel_payload` | doc: Emit a replayable private kernel without fabricating Phase 6 lineage. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 24524 | def | `_windowed_mass_stage_internal_config` | fingerprint: calls int, max, min; returns WindowedMassAdaptationConfig(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 24557 | def | `_windowed_stage_initial_mass_artifact` | fingerprint: calls int, ValueError, PrecomputedMassArtifact.from_payload, _mass_artifact_signature, str; returns PrecomputedMassArtifact.from_covariance(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 24595 | def | `_windowed_stage_draw_capture_policy` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 24612 | def | `_windowed_stage_diagnostic_run_config` | fingerprint: calls FullChainHMCConfig, float, int; returns FullChainHMCConfig(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 24634 | def | `_windowed_stage_chunk_run_config` | fingerprint: calls FixedSizeHMCChunkConfig; returns FixedSizeHMCChunkConfig(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 24655 | def | `_windowed_stage_valid_rows` | fingerprint: calls np.asarray, _tensor_to_numpy; returns Subscript |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 24661 | def | `_windowed_stage_acceptance_capture` | doc: Return per-draw acceptance plus raw runtime decision counts. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 24700 | def | `_windowed_stage_per_draw_trace` | doc: Reduce scalar or per-chain draw telemetry to a finite per-draw vector. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 24719 | def | `_windowed_stage_segmented_capture_payload` | doc: Run windowed-mass diagnostic draws as small state-carrying HMC chunks. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 24980 | def | `_windowed_stage_capture_payload` | fingerprint: calls dict, np.asarray, _tensor_to_numpy, _trace_array_or_none; returns payload |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25099 | def | `_with_windowed_stage_timing_metadata` | fingerprint: calls dict, payload.get, metadata.get, timing_buckets.update; returns payload |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25140 | def | `_windowed_stage_public_timeout_capture` | fingerprint: calls dict; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25173 | def | `_windowed_stage_error_capture` | fingerprint: calls str, type; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25201 | def | `_classify_windowed_stage_capture` | fingerprint: calls capture.get, hard_vetoes.append; returns tuple(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25237 | def | `_valid_draw_matrix` | fingerprint: calls np.asarray, int, bool, np.all, np.isfinite; returns bool(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25246 | def | `_valid_trace_vector` | fingerprint: calls np.asarray, int, np.all, np.isfinite, np.any; returns Constant |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25266 | def | `_acceptance_trace_is_default_like` | fingerprint: calls np.asarray, bool, np.allclose, array.reshape; returns bool(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25275 | def | `_windowed_stage_acceptance_has_runtime_decision_support` | fingerprint: calls capture.get, _valid_trace_vector; returns bool(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25349 | def | `_windowed_stage_acceptance_policy_filled_or_default` | fingerprint: calls capture.get, _valid_trace_vector, _acceptance_trace_is_default_like, _windowed_stage_acceptance_has_runtime_decision_support; returns UnaryOp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25361 | def | `_windowed_stage_runtime_evidence` | fingerprint: calls _metadata_marks_fixture_or_synthetic, metadata.get, _int_or_none; returns Constant |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25372 | def | `_metadata_marks_fixture_or_synthetic` | fingerprint: calls metadata.get, isinstance, tuple, any; returns any(...); loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25391 | def | `_windowed_stage_diagnostics` | fingerprint: calls float, int, capture.get; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25461 | def | `_warmup_draw_provenance` | fingerprint: calls capture.get, stable_config_hash, bool; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25479 | def | `_acceptance_telemetry_provenance` | fingerprint: calls capture.get, bool; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25515 | def | `_validate_value_score_shapes` | fingerprint: calls ValueError, int; raises ValueError(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25549 | def | `_bootstrap_screen_config` | fingerprint: calls FullChainHMCConfig, float, int; returns FullChainHMCConfig(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25571 | def | `_bootstrap_leapfrog_payload` | fingerprint: calls int, np.ceil, float, ValueError, _validate_max_leapfrog_steps; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25599 | def | `_bootstrap_reusable_static_contract_payload` | doc: Return the bootstrap fields that fix the reusable runner graph. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25636 | def | `_bootstrap_diagnostics_payload` | fingerprint: calls dict, _scalar_or_none, diagnostics.get, _runtime_seconds_or_none; returns payload |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25747 | def | `_telemetry_payload` | fingerprint: calls _json_ready, dict(...).items, dict, bool, _bool_or_none; returns payload |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25758 | def | `_runtime_seconds_or_none` | fingerprint: calls _scalar_or_none, metadata.get; returns Constant; loops For |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25766 | def | `_bootstrap_error_diagnostics` | fingerprint: calls str, len, type; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25793 | def | `_public_failure_diagnostics` | fingerprint: calls str, len, type; returns Dict |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25811 | def | `_public_bootstrap_failure_diagnostics` | doc: Summarize a returned bootstrap hard veto without exposing HMC mechanics. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25844 | def | `_classify_bootstrap_screen` | fingerprint: calls hard_vetoes.append, diagnostics.get, _finite_number; returns Tuple |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25897 | def | `_bootstrap_repair_action` | fingerprint: calls np.isfinite, float, _validate_bootstrap_repair_bracket; returns Constant |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25917 | def | `_bootstrap_update_repair_bracket` | fingerprint: calls np.isfinite, float, ValueError; returns Tuple |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25942 | def | `_validate_bootstrap_repair_bracket` | fingerprint: calls float, ValueError, np.isfinite |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25960 | def | `_repair_step_size` | fingerprint: calls ValueError, np.isfinite, float, _validate_bootstrap_repair_bracket, np.exp; returns repaired |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 25986 | def | `_bootstrap_repair_makes_effective_progress` | doc: Return whether the next repair changes the bootstrap HMC kernel. |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 26009 | def | `_round_seed` | fingerprint: calls int; returns Tuple |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 26013 | def | `_validate_seed` | fingerprint: calls tuple, int, len, ValueError; returns values |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 26020 | def | `_seed_from_mapping` | fingerprint: calls mapping.get, _validate_seed; returns _validate_seed(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 26027 | def | `_validate_band` | fingerprint: calls tuple, len, ValueError, float; returns Tuple |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 26039 | def | `_ceil_div` | fingerprint: calls int, ValueError; returns UnaryOp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 26047 | def | `_string_tuple` | fingerprint: calls isinstance, tuple, str; returns tuple(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 26055 | def | `_finite_number` | fingerprint: calls _scalar_or_none, bool, np.isfinite; returns BoolOp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 26060 | def | `_tensor_to_numpy` | fingerprint: calls hasattr, value.numpy; returns value |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 26066 | def | `_trace_array_or_none` | fingerprint: calls np.asarray, _tensor_to_numpy; returns np.asarray(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 26072 | def | `_scalar_or_none` | fingerprint: calls np.asarray, _tensor_to_numpy, float, array.reshape |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 26084 | def | `_int_or_none` | fingerprint: calls _scalar_or_none, int; returns IfExp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 26089 | def | `_strict_scalar_int_or_none` | fingerprint: calls np.asarray, _tensor_to_numpy, array.reshape(...).item, array.reshape, isinstance; returns IfExp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 26101 | def | `_strict_finite_scalar_or_none` | fingerprint: calls np.asarray, _tensor_to_numpy, array.reshape(...).item, array.reshape, isinstance, float; returns IfExp |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 26117 | def | `_bool_or_none` | fingerprint: calls np.asarray, _tensor_to_numpy, bool, array.reshape; returns bool(...) |
| `bayesfilter/inference/hmc_kernel_tuning.py` | 26126 | def | `_json_ready` | fingerprint: calls hasattr, _json_ready, value.numpy, isinstance, value.tolist; returns value |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 86 | def | `_strict_integer` | fingerprint: calls isinstance, ValueError, int; returns result |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 95 | def | `_nonempty` | fingerprint: calls str, ValueError; returns result |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 102 | def | `_strict_seed` | fingerprint: calls tuple, ValueError, len, _strict_integer; returns tuple(...) |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 114 | def | `_finite_step` | fingerprint: calls float, ValueError, math.isfinite; returns result |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 121 | def | `_json_ready` | fingerprint: calls isinstance, str, _json_ready, sorted, value.items; returns value |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 129 | def | `_signature` | fingerprint: calls json.dumps(...).encode, json.dumps, _json_ready, hashlib.sha256(...).hexdigest, hashlib.sha256; returns hashlib.sha256(...).hexdigest(...) |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 139 | def | `operational_broad_seed` | doc: Derive an order-independent seed from the full role and pair identity. |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 179 | class | `OperationalBroadGridPolicy` | doc: Reviewed route controls; no directional L refinement is permitted. |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 192 | def | `__post_init__` | fingerprint: calls _strict_seed, tuple, _strict_integer, ValueError |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 247 | def | `evidence_unit_count` | fingerprint: returns BinOp |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 250 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 273 | class | `OperationalStatisticalEpsilonRepairPolicy` | doc: Prospective CCMA epsilon-repair controls over replication means. |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 290 | def | `__post_init__` | fingerprint: calls _strict_seed, ValueError, _strict_integer |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 337 | def | `evidence_unit_count` | fingerprint: returns BinOp |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 340 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 364 | class | `OperationalBroadGridExecutionConfig` | doc: Opt-in spawn topology for independent primary and guard processes. |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 375 | def | `__post_init__` | fingerprint: calls str, ValueError, _strict_integer |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 428 | def | `payload` | fingerprint: calls dict, tuple, environment.get, sorted; returns Dict |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 444 | class | `OperationalMassHandoff` | doc: Qualified dense-metric handoff from modern operational warmup. |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 460 | def | `__post_init__` | fingerprint: calls str, ValueError, object.__setattr__, _nonempty, getattr; loops For |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 500 | def | `grid_ready` | fingerprint: calls bool; returns bool(...) |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 516 | def | `signature` | fingerprint: calls _signature, self.payload; returns _signature(...) |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 519 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 544 | class | `OperationalPairEvidence` | doc: Uncertainty-aware tuning heuristic over fresh replication means. |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 558 | def | `viable` | fingerprint: returns Compare |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 561 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 588 | def | `classify_operational_pair_evidence` | doc: Classify replicated tuning evidence without promoting partial overlap. |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 666 | class | `OperationalStatisticalEpsilonEvidence` | doc: Five-replication working evidence for tuning or qualification. |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 680 | def | `admitted` | fingerprint: returns Compare |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 684 | def | `candidate_nominated` | fingerprint: returns Compare |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 687 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 713 | def | `classify_operational_statistical_epsilon_evidence` | doc: Classify five fresh replication means under the reviewed working model. |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 784 | class | `OperationalStatisticalEpsilonRepairDecision` | doc: One append-only controller transition after a complete tuning attempt. |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 796 | def | `__post_init__` | fingerprint: calls _strict_integer, _finite_step, str, ValueError, _validated_epsilon_bracket |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 822 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 837 | def | `_validated_epsilon_bracket` | fingerprint: calls tuple, ValueError, _finite_step; returns Tuple |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 851 | def | `advance_operational_statistical_epsilon_repair` | doc: Advance a bounded tuning-only epsilon bracket from interval evidence. |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 903 | class | `OperationalPrimaryRequest` | fingerprint: calls _strict_integer, object.__setattr__, ValueError, _strict_seed |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 908 | def | `__post_init__` | fingerprint: calls _strict_integer, ValueError, object.__setattr__, _strict_seed |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 925 | def | `signature` | fingerprint: calls _signature, self.payload; returns _signature(...) |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 928 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 939 | class | `OperationalPrimaryCandidate` | fingerprint: calls object.__setattr__, isinstance, TypeError, _finite_step |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 948 | def | `__post_init__` | fingerprint: calls isinstance, TypeError, object.__setattr__, _finite_step; loops For |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 963 | def | `viable` | fingerprint: returns self.evidence.viable |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 967 | def | `pair_identity` | fingerprint: calls self.tuned_step_size.hex; returns Tuple |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 979 | def | `signature` | fingerprint: calls _signature, self.payload; returns _signature(...) |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 982 | def | `payload` | fingerprint: calls self.request.payload, self.evidence.payload; returns Dict |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 997 | class | `SameEpsilonNeighborGuardRequest` | fingerprint: calls _strict_integer, tuple, object.__setattr__ |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1008 | def | `__post_init__` | fingerprint: calls _strict_integer, ValueError, tuple, any; loops For |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1050 | def | `pair_identity` | fingerprint: calls self.inherited_step_size.hex; returns Tuple |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1062 | def | `signature` | fingerprint: calls _signature, self.identity_payload; returns _signature(...) |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1068 | def | `identity_payload` | doc: The unchanged v1 request identity payload. |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1087 | def | `payload` | fingerprint: calls self.identity_payload; returns Dict |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1097 | class | `SameEpsilonNeighborGuard` | fingerprint: calls isinstance, TypeError, _signature, self.identity_payload |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1101 | def | `__post_init__` | fingerprint: calls isinstance, TypeError |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1108 | def | `viable` | fingerprint: returns self.evidence.viable |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1112 | def | `signature` | fingerprint: calls _signature, self.identity_payload; returns _signature(...) |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1118 | def | `identity_payload` | doc: The unchanged v1 guard identity payload. |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1128 | def | `payload` | fingerprint: calls self.identity_payload; returns Dict |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1136 | def | `primary_requests` | fingerprint: calls isinstance, TypeError, tuple, OperationalPrimaryRequest; returns tuple(...) |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1160 | def | `expand_same_epsilon_neighbor_guards` | doc: Expand viable primaries once into exact-epsilon coverage probes. |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1235 | class | `OperationalBarrier` | fingerprint: calls _nonempty, tuple, object.__setattr__ |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1241 | def | `__post_init__` | fingerprint: calls _nonempty, tuple |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1258 | def | `complete` | fingerprint: calls set; returns BoolOp |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1264 | def | `payload` | fingerprint: calls len; returns Dict |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1275 | class | `OperationalBroadGridResult` | fingerprint: calls OperationalBroadGridExecutionConfig, tuple, sorted, isinstance |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1286 | def | `viable_primary_candidates` | fingerprint: calls tuple; returns tuple(...) |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1290 | def | `viable_guard_candidates` | fingerprint: calls tuple; returns tuple(...) |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1294 | def | `viable_coverage_candidates` | doc: Compatible one-hop probes; failures do not veto parent primaries. |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1300 | def | `coverage_barrier` | doc: Compatibility alias naming the barrier's active scientific role. |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1306 | def | `next_round_candidates` | doc: Return the complete unranked primary-plus-coverage union. |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1337 | def | `next_round_l_values` | doc: Sorted unique ``L`` values for the unranked next-round union. |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1349 | def | `payload` | fingerprint: calls self.policy.payload, self.mass_handoff.payload, tuple, self.primary_barrier.payload; returns Dict |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1381 | def | `public_payload` | fingerprint: calls self.execution.payload, len; returns Dict |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1429 | class | `OperationalCandidateUnionSelection` | doc: Deterministic policy selection over a complete viable ``(L, epsilon)`` union. |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1445 | def | `__post_init__` | fingerprint: calls _strict_integer, tuple, dict, ValueError, object.__setattr__ |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1482 | def | `_sort_key` | fingerprint: calls abs, int, str, record.get; returns Tuple |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1489 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1503 | def | `select_operational_candidate_union` | doc: Validate and select a complete, viable candidate union by policy only. |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1580 | def | `assemble_operational_broad_grid_result` | doc: Validate serial or process-parallel records against both barriers. |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1666 | def | `run_operational_broad_grid` | doc: Run the serial reference orchestration used to verify parallel semantics. |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1735 | def | `_resolve_factory` | fingerprint: calls locator.split, importlib.import_module, attribute_path.split, getattr, callable, TypeError; returns value; loops For |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1745 | def | `_primary_process_worker` | fingerprint: calls _resolve_factory(...), _resolve_factory, isinstance, TypeError, ValueError, type |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1762 | def | `_guard_process_worker` | fingerprint: calls _resolve_factory(...), _resolve_factory, isinstance, TypeError, ValueError, type |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1780 | def | `_temporary_worker_environment` | fingerprint: calls os.environ.get, prior.items, os.environ.pop |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1794 | def | `_process_barrier` | fingerprint: calls multiprocessing.get_context, _temporary_worker_environment, concurrent.futures.ProcessPoolExecutor, future_to_request.items, executor.submit, future.result; returns Tuple; loops For |
| `bayesfilter/inference/hmc_operational_broad_grid.py` | 1837 | def | `run_operational_broad_grid_process_parallel` | doc: Run two complete spawn barriers, with guards dependent on primaries. |
| `bayesfilter/inference/hmc_robust_broad_grid.py` | 63 | def | `_seed` | fingerprint: calls hashlib.sha256(...).digest, hashlib.sha256, JoinedStr.encode, int; returns IfExp |
| `bayesfilter/inference/hmc_robust_broad_grid.py` | 75 | def | `_finite_positive` | fingerprint: calls float, ValueError, math.isfinite; returns result |
| `bayesfilter/inference/hmc_robust_broad_grid.py` | 83 | class | `RobustBroadGridConfig` | doc: Reviewed controls for the generic five-stage tuning campaign. |
| `bayesfilter/inference/hmc_robust_broad_grid.py` | 105 | def | `__post_init__` | fingerprint: calls tuple, int, any, ValueError, len; loops For |
| `bayesfilter/inference/hmc_robust_broad_grid.py` | 168 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_robust_broad_grid.py` | 194 | def | `_jsonable` | fingerprint: calls isinstance, str, _jsonable, value.items; returns value |
| `bayesfilter/inference/hmc_robust_broad_grid.py` | 210 | def | `_trace_tensors` | fingerprint: calls tf.cast, tf.convert_to_tensor, isinstance, dict, any, ValueError; returns Tuple; loops For |
| `bayesfilter/inference/hmc_robust_broad_grid.py` | 229 | def | `_evidence` | fingerprint: calls _trace_tensors, evaluate_hmc_acceptance_evidence, int, tf.reduce_sum(...).numpy, tf.reduce_sum, tf.cast; returns evaluate_hmc_acceptance_evidence(...) |
| `bayesfilter/inference/hmc_robust_broad_grid.py` | 244 | def | `_fixed_config` | fingerprint: calls FullChainHMCConfig; returns FullChainHMCConfig(...) |
| `bayesfilter/inference/hmc_robust_broad_grid.py` | 259 | def | `_tune_one_l` | fingerprint: calls FullChainHMCConfig, _seed, HMCTuningPolicy.fixed_mass_dual_averaging, build_reusable_full_chain_tfp_hmc_runner, runner.run, run.trace.get; returns Dict |
| `bayesfilter/inference/hmc_robust_broad_grid.py` | 295 | def | `_repair_one_l` | fingerprint: calls _finite_positive, build_reusable_full_chain_tfp_hmc_runner, _fixed_config, _seed, range; returns Dict; loops For |
| `bayesfilter/inference/hmc_robust_broad_grid.py` | 323 | def | `_qualification` | fingerprint: calls _fixed_config, _seed, run_full_chain_tfp_hmc, _evidence, _trace_tensors, adapter.latent_to_position; returns Dict |
| `bayesfilter/inference/hmc_robust_broad_grid.py` | 356 | def | `_suitable` | fingerprint: calls Subscript.get, bool, acceptance.get; returns bool(...) |
| `bayesfilter/inference/hmc_robust_broad_grid.py` | 372 | def | `select_robust_candidate` | doc: Select highest minimum bulk ESS only among fully suitable candidates. |
| `bayesfilter/inference/hmc_robust_broad_grid.py` | 388 | def | `tune_hmc_kernel_robust_broad_grid` | doc: Run the five-stage robust synthetic broad-grid tuning campaign. |
| `bayesfilter/inference/hmc_robust_broad_grid.py` | 406 | def | `progress` | fingerprint: calls progress_callback, len, payload.get, time.perf_counter |
| `bayesfilter/inference/hmc_tuning.py` | 39 | class | `HMCTuningPolicy` | doc: Bounded HMC tuning-policy metadata. |
| `bayesfilter/inference/hmc_tuning.py` | 65 | def | `__post_init__` | fingerprint: calls str, ValueError, Constant.join, int |
| `bayesfilter/inference/hmc_tuning.py` | 106 | def | `fixed_kernel_screen` | doc: Current fail-closed BayesFilter behavior: no adaptive kernel. |
| `bayesfilter/inference/hmc_tuning.py` | 126 | def | `dual_averaging_step_size` | doc: Reviewed diagnostic-only step-size adaptation with fixed mass. |
| `bayesfilter/inference/hmc_tuning.py` | 154 | def | `fixed_mass_dual_averaging` | doc: Alias-like reviewed policy for fixed-mass step-size adaptation. |
| `bayesfilter/inference/hmc_tuning.py` | 182 | def | `windowed_mass_adaptation_future` | doc: Named future policy that is intentionally not executable. |
| `bayesfilter/inference/hmc_tuning.py` | 202 | def | `windowed_mass_adaptation` | doc: Reviewed non-default policy for Phase 4 windowed mass diagnostics. |
| `bayesfilter/inference/hmc_tuning.py` | 231 | def | `fixed_trajectory_tuning` | doc: Reviewed non-default policy for Phase 5 fixed-trajectory diagnostics. |
| `bayesfilter/inference/hmc_tuning.py` | 258 | def | `manual_ladder_diagnostic` | doc: Metadata for externally reviewed fixed-kernel ladders. |
| `bayesfilter/inference/hmc_tuning.py` | 278 | def | `uses_dual_averaging` | fingerprint: returns Compare |
| `bayesfilter/inference/hmc_tuning.py` | 282 | def | `uses_windowed_mass_adaptation` | fingerprint: returns Compare |
| `bayesfilter/inference/hmc_tuning.py` | 286 | def | `uses_fixed_trajectory_tuning` | fingerprint: returns Compare |
| `bayesfilter/inference/hmc_tuning.py` | 289 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_tuning.py` | 304 | class | `HMCTuningDiagnosticResult` | doc: Tiny adaptation diagnostic result; not posterior evidence. |
| `bayesfilter/inference/hmc_tuning.py` | 312 | def | `payload` | fingerprint: calls self.policy.payload; returns Dict |
| `bayesfilter/inference/hmc_tuning.py` | 322 | class | `InitialStepBracketAttempt` | doc: One finite-step bracketing attempt for fixed-mass HMC tuning. |
| `bayesfilter/inference/hmc_tuning.py` | 329 | def | `__post_init__` | fingerprint: calls float, ValueError, np.isfinite, str, object.__setattr__ |
| `bayesfilter/inference/hmc_tuning.py` | 340 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_tuning.py` | 349 | class | `InitialStepBracketResult` | doc: Recorded initial-step bracket, not posterior or convergence evidence. |
| `bayesfilter/inference/hmc_tuning.py` | 362 | def | `__post_init__` | fingerprint: calls tuple, ValueError, float, np.isfinite |
| `bayesfilter/inference/hmc_tuning.py` | 381 | def | `payload` | fingerprint: calls tuple, attempt.payload; returns Dict |
| `bayesfilter/inference/hmc_tuning.py` | 391 | class | `FixedMassStepTuningResult` | doc: Fixed-mass step-size tuning evidence with frozen-artifact invariants. |
| `bayesfilter/inference/hmc_tuning.py` | 410 | def | `__post_init__` | fingerprint: calls str, ValueError, dict, bool |
| `bayesfilter/inference/hmc_tuning.py` | 431 | def | `payload` | fingerprint: calls self.policy.payload, self.initial_step_bracket.payload, dict, self.minimum_phase4_telemetry_present; returns Dict |
| `bayesfilter/inference/hmc_tuning.py` | 449 | def | `minimum_phase4_telemetry_present` | fingerprint: calls bool, diagnostics.get; returns Dict |
| `bayesfilter/inference/hmc_tuning.py` | 474 | class | `WindowedMassAdaptationConfig` | doc: Non-default warmup-window semantics for mass adaptation diagnostics. |
| `bayesfilter/inference/hmc_tuning.py` | 491 | def | `__post_init__` | fingerprint: calls int, object.__setattr__, getattr, ValueError; loops For |
| `bayesfilter/inference/hmc_tuning.py` | 561 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_tuning.py` | 580 | class | `WindowedWarmupWindow` | doc: One contiguous fast/slow/final warmup window. |
| `bayesfilter/inference/hmc_tuning.py` | 589 | def | `__post_init__` | fingerprint: calls int, str, ValueError |
| `bayesfilter/inference/hmc_tuning.py` | 607 | def | `length` | fingerprint: returns BinOp |
| `bayesfilter/inference/hmc_tuning.py` | 610 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_tuning.py` | 622 | class | `WelfordCovarianceResult` | doc: Online covariance result for one mass-update window. |
| `bayesfilter/inference/hmc_tuning.py` | 630 | def | `__post_init__` | fingerprint: calls int, ValueError, np.asarray(...).copy, np.asarray |
| `bayesfilter/inference/hmc_tuning.py` | 648 | def | `payload` | fingerprint: calls self.mean.tolist, self.covariance.tolist; returns Dict |
| `bayesfilter/inference/hmc_tuning.py` | 658 | class | `WindowedMassUpdate` | doc: One empirical covariance, shrinkage, mass rebuild, and reset event. |
| `bayesfilter/inference/hmc_tuning.py` | 668 | def | `__post_init__` | fingerprint: calls float, ValueError, np.isfinite, str, dict |
| `bayesfilter/inference/hmc_tuning.py` | 683 | def | `payload` | fingerprint: calls self.window.payload, self.welford.payload; returns Dict |
| `bayesfilter/inference/hmc_tuning.py` | 695 | class | `WindowedMassAdaptationResult` | doc: Windowed mass adaptation evidence; not posterior convergence evidence. |
| `bayesfilter/inference/hmc_tuning.py` | 722 | def | `__post_init__` | fingerprint: calls str, ValueError, tuple |
| `bayesfilter/inference/hmc_tuning.py` | 772 | def | `final_step_size` | fingerprint: returns Subscript |
| `bayesfilter/inference/hmc_tuning.py` | 775 | def | `semantic_checks` | fingerprint: calls sum, update.reset_event.get, _windows_are_contiguous, bool; returns Dict |
| `bayesfilter/inference/hmc_tuning.py` | 801 | def | `payload` | fingerprint: calls self.policy.payload, self.config.payload, tuple, bool, len; returns Dict |
| `bayesfilter/inference/hmc_tuning.py` | 831 | class | `FixedTrajectoryTuningConfig` | doc: Closed-grid fixed-trajectory tuning screen for a frozen HMC kernel. |
| `bayesfilter/inference/hmc_tuning.py` | 841 | def | `__post_init__` | fingerprint: calls tuple, int, ValueError, any, _validate_acceptance_band |
| `bayesfilter/inference/hmc_tuning.py` | 867 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_tuning.py` | 879 | class | `FixedTrajectoryCandidateResult` | doc: One fixed leapfrog-count candidate for a frozen step size and mass. |
| `bayesfilter/inference/hmc_tuning.py` | 893 | def | `__post_init__` | fingerprint: calls float, ValueError, np.isfinite, int |
| `bayesfilter/inference/hmc_tuning.py` | 924 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_tuning.py` | 940 | class | `FixedTrajectoryTuningResult` | doc: Frozen-kernel trajectory tuning evidence, not convergence evidence. |
| `bayesfilter/inference/hmc_tuning.py` | 963 | def | `__post_init__` | fingerprint: calls str, ValueError, float, np.isfinite, tuple |
| `bayesfilter/inference/hmc_tuning.py` | 1012 | def | `selected_candidate` | fingerprint: returns Constant; loops For |
| `bayesfilter/inference/hmc_tuning.py` | 1023 | def | `passed` | fingerprint: returns BoolOp |
| `bayesfilter/inference/hmc_tuning.py` | 1026 | def | `payload` | fingerprint: calls self.policy.payload, self.config.payload, tuple, self.selected_candidate.payload, len, sum; returns Dict |
| `bayesfilter/inference/hmc_tuning.py` | 1066 | def | `normalize_hmc_tuning_policy` | doc: Normalize reviewed tuning policies while keeping raw adaptation fail-closed. |
| `bayesfilter/inference/hmc_tuning.py` | 1084 | def | `require_executable_tuning_policy` | fingerprint: calls ValueError; raises ValueError(...) |
| `bayesfilter/inference/hmc_tuning.py` | 1116 | def | `classify_fixed_kernel_screen_with_tuning_policy` | doc: Classify a fixed-kernel screen under the explicit Stage 7 policy label. |
| `bayesfilter/inference/hmc_tuning.py` | 1141 | def | `classify_hmc_tuning_diagnostic` | doc: Separate tuning telemetry from target validity and convergence claims. |
| `bayesfilter/inference/hmc_tuning.py` | 1182 | def | `bracket_initial_step_size` | doc: Find a finite initial step and record every attempt. |
| `bayesfilter/inference/hmc_tuning.py` | 1237 | def | `run_fixed_mass_step_tuning_diagnostic` | doc: Run tiny fixed-mass step-size telemetry without mutating the mass artifact. |
| `bayesfilter/inference/hmc_tuning.py` | 1309 | def | `build_windowed_warmup_schedule` | doc: Build a contiguous fast/slow/final warmup schedule. |
| `bayesfilter/inference/hmc_tuning.py` | 1363 | def | `welford_covariance` | doc: Compute sample covariance with Welford's online recursion. |
| `bayesfilter/inference/hmc_tuning.py` | 1392 | def | `validate_windowed_shrinkage_target` | doc: Fail closed unless the empirical covariance and target share coordinates. |
| `bayesfilter/inference/hmc_tuning.py` | 1441 | def | `run_windowed_mass_adaptation_diagnostic` | doc: Run a bounded windowed-mass semantic diagnostic on supplied warmup draws. |
| `bayesfilter/inference/hmc_tuning.py` | 1579 | def | `production_leapfrog_count` | doc: Return production and theory leapfrog counts for a fixed trajectory. |
| `bayesfilter/inference/hmc_tuning.py` | 1610 | def | `run_fixed_trajectory_tuning_diagnostic` | doc: Select a fixed leapfrog count from a frozen Phase 4 mass/step artifact. |
| `bayesfilter/inference/hmc_tuning.py` | 1704 | def | `run_gaussian_dual_averaging_diagnostic` | doc: Run a tiny Gaussian TFP diagnostic for reviewed dual averaging. |
| `bayesfilter/inference/hmc_tuning.py` | 1743 | def | `target_log_prob` | fingerprint: calls tf.convert_to_tensor, tf.reduce_sum, tf.square; returns BinOp |
| `bayesfilter/inference/hmc_tuning.py` | 1758 | def | `trace_fn` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_tuning.py` | 1821 | def | `_mass_artifact_payload` | fingerprint: calls mass_artifact.signature_payload, tuple, ValueError, Constant.join; returns payload |
| `bayesfilter/inference/hmc_tuning.py` | 1840 | def | `_mass_artifact_signature` | fingerprint: calls _mass_artifact_payload, np.asarray, _stable_payload_signature; returns _stable_payload_signature(...) |
| `bayesfilter/inference/hmc_tuning.py` | 1850 | def | `_adapt_windowed_step_size` | fingerprint: calls np.log, float, np.clip, np.exp; returns float(...) |
| `bayesfilter/inference/hmc_tuning.py` | 1863 | def | `_validate_acceptance_band` | fingerprint: calls len, ValueError, tuple, float; returns Tuple |
| `bayesfilter/inference/hmc_tuning.py` | 1874 | def | `_shrink_covariance` | fingerprint: calls np.asarray, ValueError, np.all; returns BinOp |
| `bayesfilter/inference/hmc_tuning.py` | 1890 | def | `_run_fixed_trajectory_candidate` | fingerprint: calls tf.zeros, int, tf.convert_to_tensor, tf.reduce_sum, tf.square, tfm.HamiltonianMonteCarlo; returns FixedTrajectoryCandidateResult(...) |
| `bayesfilter/inference/hmc_tuning.py` | 1906 | def | `target_log_prob` | fingerprint: calls tf.convert_to_tensor, tf.reduce_sum, tf.square; returns BinOp |
| `bayesfilter/inference/hmc_tuning.py` | 1916 | def | `trace_fn` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_tuning.py` | 1981 | def | `_select_fixed_trajectory_candidate` | fingerprint: calls min, abs, float; returns min(...) |
| `bayesfilter/inference/hmc_tuning.py` | 2001 | def | `_rebuild_windowed_mass_artifact` | fingerprint: calls _regularize_windowed_covariance, PrecomputedMassArtifact.from_covariance, np.asarray; returns PrecomputedMassArtifact.from_covariance(...) |
| `bayesfilter/inference/hmc_tuning.py` | 2038 | def | `_regularize_windowed_covariance` | fingerprint: calls np.asarray, ValueError, np.all, np.isfinite, float; returns Tuple |
| `bayesfilter/inference/hmc_tuning.py` | 2101 | def | `_windows_are_contiguous` | fingerprint: calls all, zip; returns all(...) |
| `bayesfilter/inference/hmc_tuning.py` | 2109 | def | `_frozen_mass_invariant` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_tuning.py` | 2120 | def | `_failed_fixed_mass_diagnostic` | fingerprint: calls str, HMCTuningDiagnosticResult; returns HMCTuningDiagnosticResult(...) |
| `bayesfilter/inference/hmc_tuning.py` | 2160 | def | `_validate_target_failure_classification` | fingerprint: calls dict, tuple, ValueError, Constant.join, str; returns result |
| `bayesfilter/inference/hmc_tuning.py` | 2200 | def | `_stable_payload_signature` | fingerprint: calls json.dumps, _normalize_payload, hashlib.sha256(...).hexdigest, hashlib.sha256, blob.encode; returns hashlib.sha256(...).hexdigest(...) |
| `bayesfilter/inference/hmc_tuning.py` | 2208 | def | `_normalize_payload` | fingerprint: calls isinstance, str, _normalize_payload, value.items; returns value |
| `bayesfilter/inference/hmc_tuning_artifacts.py` | 58 | def | `_strict_scalar_integer` | fingerprint: calls isinstance, ValueError, int; returns int(...) |
| `bayesfilter/inference/hmc_tuning_artifacts.py` | 64 | def | `_strict_seed` | fingerprint: calls ValueError, isinstance, len, tuple, _strict_scalar_integer; returns tuple(...) |
| `bayesfilter/inference/hmc_tuning_artifacts.py` | 72 | def | `_json_ready` | fingerprint: calls isinstance, str, _json_ready, sorted, value.items; returns value |
| `bayesfilter/inference/hmc_tuning_artifacts.py` | 84 | def | `canonical_json_bytes` | fingerprint: calls json.dumps(...).encode, json.dumps, _json_ready; returns json.dumps(...).encode(...) |
| `bayesfilter/inference/hmc_tuning_artifacts.py` | 93 | def | `canonical_sha256` | fingerprint: calls hashlib.sha256(...).hexdigest, hashlib.sha256, canonical_json_bytes; returns hashlib.sha256(...).hexdigest(...) |
| `bayesfilter/inference/hmc_tuning_artifacts.py` | 97 | def | `file_sha256` | fingerprint: calls hashlib.sha256, Path(...).open, iter, digest.update, Path, handle.read; returns digest.hexdigest(...) |
| `bayesfilter/inference/hmc_tuning_artifacts.py` | 105 | def | `atomic_write_json` | fingerprint: calls Path, destination.parent.mkdir, json.dumps, _json_ready, destination.with_name, os.getpid |
| `bayesfilter/inference/hmc_tuning_artifacts.py` | 123 | def | `kernel_state_summary` | fingerprint: calls ValueError, float, int, tuple, str, canonical_sha256; returns Dict |
| `bayesfilter/inference/hmc_tuning_artifacts.py` | 150 | def | `private_start_bank_summary` | fingerprint: calls str, ValueError, tuple, int; returns Dict |
| `bayesfilter/inference/hmc_tuning_artifacts.py` | 171 | def | `_finite_real` | fingerprint: calls isinstance, ValueError, float, math.isfinite; returns result |
| `bayesfilter/inference/hmc_tuning_artifacts.py` | 181 | def | `_validate_reasonable_epsilon_payload` | fingerprint: calls isinstance, ValueError, payload.get, _finite_real; returns selected; loops For |
| `bayesfilter/inference/hmc_tuning_artifacts.py` | 287 | def | `_validate_operational_warmup_payload` | fingerprint: calls ValueError, warmup.get, set; loops For |
| `bayesfilter/inference/hmc_tuning_artifacts.py` | 708 | def | `transition_ledger_payload` | fingerprint: calls tuple, ValueError, len, record.payload, canonical_sha256; returns Dict |
| `bayesfilter/inference/hmc_tuning_artifacts.py` | 722 | def | `build_hmc_tuning_engineering_artifact` | fingerprint: calls str, ValueError, _strict_scalar_integer, isinstance; returns Dict |
| `bayesfilter/inference/hmc_tuning_artifacts.py` | 786 | def | `validate_hmc_tuning_engineering_artifact` | fingerprint: calls ValueError, isinstance, payload.get, expected_fields.add, set; returns Dict; loops For |
| `bayesfilter/inference/hmc_tuning_artifacts.py` | 1140 | def | `load_and_replay_hmc_tuning_artifact` | fingerprint: calls json.loads, Path(...).read_text, Path, payload.get, _legacy_hmc_tuning_artifact_migration_view, validate_hmc_tuning_engineering_artifact; returns validate_hmc_tuning_engineering_artifact(...) |
| `bayesfilter/inference/hmc_tuning_artifacts.py` | 1149 | def | `_legacy_hmc_tuning_artifact_migration_view` | doc: Read a hash-valid v2 envelope without granting v3 repair authority. |
| `bayesfilter/inference/hmc_tuning_artifacts.py` | 1193 | class | `KillableChildSpec` | fingerprint: calls tuple, float, object.__setattr__ |
| `bayesfilter/inference/hmc_tuning_artifacts.py` | 1201 | def | `__post_init__` | fingerprint: calls tuple, str, float, ValueError, object.__setattr__ |
| `bayesfilter/inference/hmc_tuning_artifacts.py` | 1217 | def | `run_killable_child` | fingerprint: calls isinstance, TypeError, os.environ.copy, environment.update, str; returns closeout |
| `bayesfilter/inference/hmc_tuning_artifacts.py` | 1289 | def | `validate_killable_child_closeout` | fingerprint: calls payload.get, ValueError |
| `bayesfilter/inference/hmc_tuning_state.py` | 67 | def | `sanitize_health_failure_reasons` | doc: Return only fixed public-safe reason codes, failing unknowns closed. |
| `bayesfilter/inference/hmc_tuning_state.py` | 111 | class | `HMCTuningTransition` | fingerprint: calls str, object.__setattr__, ValueError |
| `bayesfilter/inference/hmc_tuning_state.py` | 119 | def | `__post_init__` | fingerprint: calls str, ValueError, LEGAL_TUNING_TRANSITIONS.get, set; loops For |
| `bayesfilter/inference/hmc_tuning_state.py` | 139 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_tuning_state.py` | 151 | class | `HMCStepRepair` | fingerprint: calls str, float, object.__setattr__ |
| `bayesfilter/inference/hmc_tuning_state.py` | 163 | def | `__post_init__` | fingerprint: calls str, ValueError, float, np.isfinite |
| `bayesfilter/inference/hmc_tuning_state.py` | 242 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_tuning_state.py` | 262 | def | `directional_evidence_count` | doc: Count typed directional records in the immutable evidence batch. |
| `bayesfilter/inference/hmc_tuning_state.py` | 271 | def | `neutral_evidence_count` | doc: Count valid neutral records retained alongside directional evidence. |
| `bayesfilter/inference/hmc_tuning_state.py` | 277 | def | `one_sided_directional_support` | doc: Whether every non-neutral record supports this repair direction. |
| `bayesfilter/inference/hmc_tuning_state.py` | 299 | def | `aggregate_step_repair` | doc: Aggregate a completed batch without depending on candidate order. |
| `bayesfilter/inference/hmc_tuning_state.py` | 371 | def | `aggregate_bracketed_step_repair` | doc: Aggregate a batch and retain an empirical acceptance bracket. |
| `bayesfilter/inference/hmc_tuning_state.py` | 513 | def | `aggregate_step_veto_bracket_repair` | doc: Reject reason-only epsilon repair under the v3 evidence contract. |
| `bayesfilter/inference/hmc_tuning_state.py` | 564 | def | `_step_repair_disposition` | doc: Return fail-closed batch scope and its unique supported direction. |
| `bayesfilter/inference/hmc_tuning_state.py` | 596 | def | `_positive_bound_or_none` | fingerprint: calls float, ValueError, np.isfinite; returns bound |
| `bayesfilter/inference/hmc_tuning_state.py` | 605 | def | `_realized_directional_factor` | fingerprint: calls float; returns IfExp |
| `bayesfilter/inference/hmc_uncertainty_retuning.py` | 39 | def | `_canonical` | doc: Normalize JSON list/tuple differences for strict payload comparison. |
| `bayesfilter/inference/hmc_uncertainty_retuning.py` | 49 | def | `_signature` | fingerprint: calls json.dumps(...).encode, json.dumps, _canonical, hashlib.sha256(...).hexdigest, hashlib.sha256; returns hashlib.sha256(...).hexdigest(...) |
| `bayesfilter/inference/hmc_uncertainty_retuning.py` | 59 | def | `_strict_integer` | fingerprint: calls isinstance, TypeError, int, ValueError; returns result |
| `bayesfilter/inference/hmc_uncertainty_retuning.py` | 69 | class | `HMCUncertaintyRetuningPolicy` | doc: Explicit, opt-in nomination policy for noisy fixed-metric screens. |
| `bayesfilter/inference/hmc_uncertainty_retuning.py` | 85 | def | `__post_init__` | fingerprint: calls float, ValueError, math.isfinite, tuple |
| `bayesfilter/inference/hmc_uncertainty_retuning.py` | 110 | def | `expected_chain_run_count` | fingerprint: returns BinOp |
| `bayesfilter/inference/hmc_uncertainty_retuning.py` | 113 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/hmc_uncertainty_retuning.py` | 136 | class | `HMCUncertaintyRetuningSummary` | doc: Descriptive candidate-level spread and nomination result. |
| `bayesfilter/inference/hmc_uncertainty_retuning.py` | 150 | def | `nominated` | fingerprint: returns Compare |
| `bayesfilter/inference/hmc_uncertainty_retuning.py` | 154 | def | `signature` | fingerprint: calls _signature, self.payload; returns _signature(...) |
| `bayesfilter/inference/hmc_uncertainty_retuning.py` | 157 | def | `payload` | fingerprint: calls self.policy.payload; returns Dict |
| `bayesfilter/inference/hmc_uncertainty_retuning.py` | 180 | class | `HMCUncertaintyConfirmationAdmission` | doc: Lineage-bound admission of one uncertainty nominee to confirmation. |
| `bayesfilter/inference/hmc_uncertainty_retuning.py` | 193 | def | `__post_init__` | fingerprint: calls isinstance, TypeError; loops For |
| `bayesfilter/inference/hmc_uncertainty_retuning.py` | 232 | def | `signature` | fingerprint: calls _signature, self.payload; returns _signature(...) |
| `bayesfilter/inference/hmc_uncertainty_retuning.py` | 237 | def | `payload` | fingerprint: calls self.lineage.payload, self.nomination.payload; returns Dict |
| `bayesfilter/inference/hmc_uncertainty_retuning.py` | 260 | def | `fixed_metric_search_lineage_from_payload` | doc: Strictly reconstruct one fixed-metric lineage payload. |
| `bayesfilter/inference/hmc_uncertainty_retuning.py` | 280 | def | `fixed_metric_screen_record_from_payload` | doc: Strictly reconstruct one fixed-metric screen record payload. |
| `bayesfilter/inference/hmc_uncertainty_retuning.py` | 323 | def | `fixed_metric_candidate_record_from_payload` | doc: Strictly reconstruct one fixed-metric candidate payload. |
| `bayesfilter/inference/hmc_uncertainty_retuning.py` | 373 | def | `admit_hmc_uncertainty_nomination_for_confirmation` | doc: Admit one exact uncertainty nominee to a separate fresh confirmation. |
| `bayesfilter/inference/hmc_uncertainty_retuning.py` | 394 | def | `hmc_uncertainty_retuning_policy_from_payload` | doc: Reconstruct and validate one uncertainty-retuning policy payload. |
| `bayesfilter/inference/hmc_uncertainty_retuning.py` | 416 | def | `hmc_uncertainty_retuning_summary_from_payload` | doc: Reconstruct and validate one uncertainty-retuning summary payload. |
| `bayesfilter/inference/hmc_uncertainty_retuning.py` | 436 | def | `summarize_hmc_uncertainty_for_retuning` | doc: Compute descriptive spread and an opt-in fresh-retuning nomination. |
| `bayesfilter/inference/neutra_broad_grid.py` | 54 | def | `combined_target_health` | doc: Hard target-validity reasons for one fixed-kernel run's samples. |
| `bayesfilter/inference/neutra_broad_grid.py` | 108 | def | `evaluate_fixed_screen_run` | doc: Shared hard gates and acceptance chain means for one fixed screen run. |
| `bayesfilter/inference/neutra_broad_grid.py` | 180 | class | `NeuTraBroadGridTuningConfig` | doc: Execution controls not already frozen by the broad-grid policy. |
| `bayesfilter/inference/neutra_broad_grid.py` | 196 | def | `__post_init__` | fingerprint: calls float, ValueError, math.isfinite, tuple, int, any; loops For |
| `bayesfilter/inference/neutra_broad_grid.py` | 223 | def | `payload` | fingerprint: returns Dict |
| `bayesfilter/inference/neutra_broad_grid.py` | 240 | def | `_json_ready` | fingerprint: calls isinstance, str, _json_ready, value.items; returns value |
| `bayesfilter/inference/neutra_broad_grid.py` | 250 | def | `_signature` | fingerprint: calls json.dumps(...).encode, json.dumps, _json_ready, hashlib.sha256(...).hexdigest, hashlib.sha256; returns hashlib.sha256(...).hexdigest(...) |
| `bayesfilter/inference/neutra_broad_grid.py` | 260 | def | `build_fixed_identity_broad_grid_handoff` | doc: Issue fixed-identity metric and coordinate lineage for one transport. |
| `bayesfilter/inference/neutra_broad_grid.py` | 314 | class | `NeuTraBroadGridCallbacks` | doc: Target-agnostic dual-averaging and fixed-screen callbacks. |
| `bayesfilter/inference/neutra_broad_grid.py` | 317 | def | `__init__` | fingerprint: calls tf.constant, tf.broadcast_to, int, str, getattr, ValueError |
| `bayesfilter/inference/neutra_broad_grid.py` | 375 | def | `_combined_health` | fingerprint: calls combined_target_health; returns combined_target_health(...) |
| `bayesfilter/inference/neutra_broad_grid.py` | 382 | def | `_screen` | fingerprint: calls time.perf_counter, self._screen_runner.run, float, int, evaluate_fixed_screen_run; returns row |
| `bayesfilter/inference/neutra_broad_grid.py` | 419 | def | `primary` | fingerprint: calls ValueError, time.perf_counter, self._tune_runner.run, tf.convert_to_tensor, tune.trace.get; returns OperationalPrimaryCandidate(...) |
| `bayesfilter/inference/neutra_broad_grid.py` | 504 | def | `guard` | fingerprint: calls ValueError, tuple, self._screen, dict.fromkeys, classify_operational_pair_evidence; returns SameEpsilonNeighborGuard(...) |
| `bayesfilter/inference/neutra_broad_grid.py` | 537 | def | `run_neutra_operational_broad_grid_tuning` | doc: Run the complete tuning barriers and emit private/public artifacts. |
| `bayesfilter/inference/neutra_staged_training.py` | 12 | class | `NeuTraStagedTrainingError` | doc: Raised when a staged NeuTra run violates its finite or phase contract. |
| `bayesfilter/inference/neutra_staged_training.py` | 17 | class | `NeuTraVariablePart` | doc: One trainable tensor or a binary-masked portion of that tensor. |
| `bayesfilter/inference/neutra_staged_training.py` | 25 | class | `NeuTraVariableGroup` | doc: Named collection of trainable tensor parts activated as one unit. |
| `bayesfilter/inference/neutra_staged_training.py` | 31 | def | `__post_init__` | fingerprint: calls ValueError |
| `bayesfilter/inference/neutra_staged_training.py` | 37 | class | `NeuTraAdaptiveStagePolicy` | doc: Bounded held-out plateau scheduler for one optimizer phase. |
| `bayesfilter/inference/neutra_staged_training.py` | 46 | def | `__post_init__` | fingerprint: calls int, ValueError, math.isfinite |
| `bayesfilter/inference/neutra_staged_training.py` | 64 | class | `NeuTraStageSpec` | doc: One independently tuned optimizer phase inside stages one through four. |
| `bayesfilter/inference/neutra_staged_training.py` | 76 | def | `__post_init__` | fingerprint: calls ValueError, int, len |
| `bayesfilter/inference/neutra_staged_training.py` | 104 | class | `NeuTraFiveStageSpec` | doc: Four optimizer stages followed by one untouched validation stage. |
| `bayesfilter/inference/neutra_staged_training.py` | 112 | def | `__post_init__` | fingerprint: calls ValueError, int, any |
| `bayesfilter/inference/neutra_staged_training.py` | 125 | def | `optimizer_phases` | fingerprint: returns Tuple |
| `bayesfilter/inference/neutra_staged_training.py` | 130 | class | `NeuTraLearningRateResult` | fingerprint: body AnnAssign, AnnAssign, AnnAssign, AnnAssign |
| `bayesfilter/inference/neutra_staged_training.py` | 148 | class | `NeuTraStageResult` | fingerprint: body AnnAssign, AnnAssign, AnnAssign, AnnAssign |
| `bayesfilter/inference/neutra_staged_training.py` | 164 | class | `NeuTraFiveStageResult` | fingerprint: body AnnAssign, AnnAssign, AnnAssign, AnnAssign |
| `bayesfilter/inference/neutra_staged_training.py` | 174 | def | `_scheduled_learning_rate` | fingerprint: calls float; returns BinOp |
| `bayesfilter/inference/neutra_staged_training.py` | 189 | def | `_state` | fingerprint: calls tuple, tf.identity; returns tuple(...) |
| `bayesfilter/inference/neutra_staged_training.py` | 193 | def | `_restore` | fingerprint: calls len, NeuTraStagedTrainingError, zip, tf.convert_to_tensor, variable.assign; loops For |
| `bayesfilter/inference/neutra_staged_training.py` | 203 | def | `_finite_scalar` | fingerprint: calls tf.convert_to_tensor, NeuTraStagedTrainingError, tf.debugging.assert_all_finite, float, tensor.numpy; returns float(...) |
| `bayesfilter/inference/neutra_staged_training.py` | 211 | def | `_normalize_groups` | fingerprint: calls id, len, ValueError, tf.zeros_like; returns output; loops For |
| `bayesfilter/inference/neutra_staged_training.py` | 255 | def | `_active_variables_and_masks` | fingerprint: calls tuple, ValueError, Constant.join, Subscript.items, tf.maximum, active_masks.get; returns Tuple; loops For |
| `bayesfilter/inference/neutra_staged_training.py` | 281 | def | `_validate_joint_coverage` | fingerprint: calls id, zip, coverage.get, ValueError, bool; loops For |
| `bayesfilter/inference/neutra_staged_training.py` | 293 | def | `neutra_full_variable_masks` | doc: Return one validated mask for every transport trainable variable. |
| `bayesfilter/inference/neutra_staged_training.py` | 318 | def | `_validate_cumulative_groups` | fingerprint: calls id, tf.zeros_like, _active_variables_and_masks, tf.cast; loops For |
| `bayesfilter/inference/neutra_staged_training.py` | 344 | def | `_optimizer` | fingerprint: calls tf.keras.optimizers.Adam, float, optimizer.build; returns optimizer |
| `bayesfilter/inference/neutra_staged_training.py` | 362 | def | `train_neutra_five_stage` | doc: Tune and execute four continuation stages, then validate without updates. |
| `bayesfilter/inference/neutra_staged_training.py` | 456 | def | `train_step` | fingerprint: calls tf.GradientTape, tape.watch, transport.forward_and_logdet, tf.convert_to_tensor, tf.reduce_mean, target_log_prob_fn; returns Tuple |
| `bayesfilter/inference/neutra_staged_training.py` | 494 | def | `update` | fingerprint: calls optimizer.apply_gradients, zip, tf.cast; returns tf.cast(...) |
| `bayesfilter/inference/neutra_staged_training.py` | 654 | def | `dense_iaf_five_stage_variable_groups` | doc: Partition a composed dense IAF into generic continuation groups. |
| `bayesfilter/inference/neutra_staged_training.py` | 703 | def | `dense_iaf_five_stage_spec` | doc: Return the generic dense-IAF continuation recipe for a stage count. |
| `bayesfilter/inference/neutra_state_continuing_broad_grid.py` | 71 | def | `next_repair_epsilon` | doc: Return the next epsilon and updated monotone acceptance bracket. |
| `bayesfilter/inference/neutra_state_continuing_broad_grid.py` | 116 | def | `_state_signature` | fingerprint: calls bytes, tf.io.serialize_tensor(...).numpy, tf.io.serialize_tensor, tf.convert_to_tensor, hashlib.sha256(...).hexdigest, hashlib.sha256; returns hashlib.sha256(...).hexdigest(...) |
| `bayesfilter/inference/neutra_state_continuing_broad_grid.py` | 124 | class | `NeuTraStateContinuingBroadGridConfig` | doc: Execution controls for the state-continuing epsilon-repair route. |
| `bayesfilter/inference/neutra_state_continuing_broad_grid.py` | 143 | def | `__post_init__` | fingerprint: calls float, ValueError, math.isfinite, tuple, int, any; loops For |
| `bayesfilter/inference/neutra_state_continuing_broad_grid.py` | 200 | def | `warm_start_epsilon` | fingerprint: calls float, int; returns float(...) |
| `bayesfilter/inference/neutra_state_continuing_broad_grid.py` | 206 | def | `payload` | fingerprint: calls str, self.initial_epsilon_by_l.items; returns Dict |
| `bayesfilter/inference/neutra_state_continuing_broad_grid.py` | 234 | class | `NeuTraStateContinuingBroadGridCallbacks` | doc: Per-``L`` adaptation, calibration repair, and disjoint final screens. |
| `bayesfilter/inference/neutra_state_continuing_broad_grid.py` | 237 | def | `__init__` | fingerprint: calls tf.constant, tf.broadcast_to, int, str, getattr, ValueError |
| `bayesfilter/inference/neutra_state_continuing_broad_grid.py` | 312 | def | `_summarize_run` | fingerprint: calls evaluate_fixed_screen_run, tf.convert_to_tensor, int, float, _json_ready; returns Dict |
| `bayesfilter/inference/neutra_state_continuing_broad_grid.py` | 351 | def | `_fixed_run` | fingerprint: calls _state_signature, time.perf_counter, runner.run, float, int, self._summarize_run; returns self._summarize_run(...) |
| `bayesfilter/inference/neutra_state_continuing_broad_grid.py` | 379 | def | `calibrate_primary` | fingerprint: calls ValueError, int, self.config.warm_start_epsilon, _state_signature, time.perf_counter, self._adapt_runner.run; returns Tuple; loops For |
| `bayesfilter/inference/neutra_state_continuing_broad_grid.py` | 482 | def | `primary` | fingerprint: calls int, self.calibrate_primary, tuple, self._fixed_run, range, operational_broad_seed; returns candidate |
| `bayesfilter/inference/neutra_state_continuing_broad_grid.py` | 539 | def | `guard` | fingerprint: calls ValueError, tuple, sorted, any, len; returns SameEpsilonNeighborGuard(...) |
| `bayesfilter/inference/neutra_state_continuing_broad_grid.py` | 584 | def | `run_neutra_state_continuing_broad_grid_tuning` | doc: Run the state-continuing repair barriers and emit private/public artifacts. |
| `bayesfilter/inference/staged_fixed_kernel_hmc.py` | 42 | def | `_seed` | fingerprint: calls tuple, int, len, ValueError; returns result |
| `bayesfilter/inference/staged_fixed_kernel_hmc.py` | 50 | class | `StagedFixedKernelHMCConfig` | doc: Exact user-requested burn-in and retained-sample ladder. |
| `bayesfilter/inference/staged_fixed_kernel_hmc.py` | 68 | def | `__post_init__` | fingerprint: calls float, ValueError, math.isfinite, object.__setattr__, int; loops For |
| `bayesfilter/inference/staged_fixed_kernel_hmc.py` | 113 | def | `payload` | fingerprint: calls asdict; returns Dict |
| `bayesfilter/inference/staged_fixed_kernel_hmc.py` | 118 | class | `StagedFixedKernelHMCResult` | fingerprint: calls self.config.payload |
| `bayesfilter/inference/staged_fixed_kernel_hmc.py` | 131 | def | `payload` | fingerprint: calls self.config.payload; returns Dict |
| `bayesfilter/inference/staged_fixed_kernel_hmc.py` | 150 | def | `_json` | fingerprint: calls isinstance, str, _json, value.items; returns value |
| `bayesfilter/inference/staged_fixed_kernel_hmc.py` | 168 | def | `_hash` | fingerprint: calls hashlib.sha256(...).hexdigest, hashlib.sha256, json.dumps(...).encode, json.dumps, _json; returns hashlib.sha256(...).hexdigest(...) |
| `bayesfilter/inference/staged_fixed_kernel_hmc.py` | 174 | def | `_diag` | fingerprint: calls ValueError, tf.concat, tf.transpose, bool, tf.reduce_all(...).numpy; returns Dict |
| `bayesfilter/inference/staged_fixed_kernel_hmc.py` | 217 | def | `_validate_chunk` | doc: Apply shared hard health checks and evaluate model-owned status telemetry. |
| `bayesfilter/inference/staged_fixed_kernel_hmc.py` | 270 | def | `run_staged_fixed_kernel_hmc_estimation` | doc: Execute the exact staged policy with fixed kernel and four-chain handoff. |
| `bayesfilter/inference/staged_fixed_kernel_hmc.py` | 308 | def | `write_progress` | fingerprint: calls progress_file.parent.mkdir, progress_file.with_name, temporary.write_text, json.dumps, _json, temporary.replace |

**Total definitions: 1417 across 26 files.**
