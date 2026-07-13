# Scalar SSL-LSTM Filtering HMC Validation Phase 2S - Geometry Centering Repair

## Decision

- phase2s_geometry_centering_repair_passed: `True`
- vetoes: `[]`
- viable_for_map_local_reference_subplan: `True`
- zero_divergence_claim_made: `False`
- next_justified_action: draft and review MAP-local reference-agreement or retuned fixed-kernel HMC screen subplan

## Initializer

- accepted/status: `True` / `usable`
- map_candidate_role: `locator_position_geometry_covariance_only`
- locator diagnostics: `{'schema': 'bayesfilter.quadratic_map_covariance.locator.v1', 'method': 'tfp_lbfgs_minimize_negative_log_prob', 'optimizer_role': 'finite_neighborhood_locator_only', 'uses_optimizer_inverse_hessian': False, 'initial_log_prob': -37.786099036348105, 'initial_score_norm': 0.9312012159249712, 'config': {'enabled': True, 'max_iterations': 50, 'tolerance': 1e-08, 'log_prob_tolerance': 1e-08, 'parallel_iterations': 1}, 'status': 'tfp_lbfgs_locator_accepted', 'accepted_optimizer_position': True, 'optimizer_converged': True, 'optimizer_failed': False, 'optimizer_iterations': 9, 'optimizer_objective_value': 37.77528495512359, 'candidate_log_prob': -37.77528495512359, 'candidate_score_norm': 1.017238315038726e-10, 'candidate_evaluation_status': 'finite', 'locator_log_prob': -37.77528495512359, 'locator_score_norm': 1.017238315038726e-10, 'fallback_reason': None}`

## Geometry

- finite sample count: `90`
- required finite samples: `45`
- regression parameter count: `9`
- holdout count: `22`
- holdout passed: `True`
- holdout rmse: `0.058211774395612294`
- score rmse: `0.30615107387554297`
- precision eigen summary: `{'finite': True, 'positive': True, 'min': 1.2627877672895487, 'max': 56.83468717817434, 'condition_number': 45.0073152832043, 'eigenvalues': [1.2627877672895487, 1.5130710093917887, 30.38610927191768, 56.83468717817434]}`
- covariance eigen summary: `{'finite': True, 'positive': True, 'min': 0.017594888784467865, 'max': 0.7918987068954606, 'condition_number': 45.007315283204306, 'eigenvalues': [0.017594888784467865, 0.03290977436601861, 0.6609075144476984, 0.7918987068954606]}`

## Mass Regularization

- regularization report: `{'method': 'symmetric_eigendecomposition_floor', 'jitter': 1e-09, 'requested_eigenvalue_floor': 0.0001, 'effective_eigenvalue_floor': 0.0005683468717817434, 'max_condition_number': 100000.0, 'raw_min_eigenvalue': 1.2627877672895491, 'raw_max_eigenvalue': 56.83468717817434, 'regularized_min_eigenvalue': 1.2627877672895491, 'regularized_max_eigenvalue': 56.83468717817434, 'raw_nonpositive_eigenvalue_count': 0, 'clipped_eigenvalue_count': 0, 'symmetry_projection': 'average_with_transpose', 'input_asymmetry_max_abs': 0.0, 'input_asymmetric': False, 'diagonal_fallback_used': False, 'silent_eigenvalue_reflection': False}`

## Inference Status

| field | value |
| --- | --- |
| hard_veto_screen | passed |
| statistically_supported_ranking | none; single diagnostic initializer |
| descriptive_only_differences | locator movement, target replay values, fit residuals, eigen summaries, and distances to old reference/HMC summaries |
| posterior_correctness | not assessed |
| hmc_readiness | not assessed |
| gpu_xla_readiness | blocked until local repair handoff passes |
| default_readiness | not assessed |
| zero_divergence_claim | not made |
| next_evidence_needed | draft and review MAP-local reference-agreement or retuned fixed-kernel HMC screen subplan |

## Nonclaims

- Phase 2S MAP-local geometry centering diagnostic only
- optimizer output is a finite-neighborhood locator only
- not a certified global MAP
- not posterior covariance correctness evidence
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
