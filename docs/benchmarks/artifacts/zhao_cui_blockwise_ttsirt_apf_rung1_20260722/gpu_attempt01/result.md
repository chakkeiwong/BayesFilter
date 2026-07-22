# Zhao-Cui Blockwise TTSIRT-APF Rung-1 Result

Status: `PASS_ENGINEERING_RUNG1`

This is a synthetic independent-block mechanics result for a BayesFilter extension. It is not source-faithful Zhao-Cui, nonlinear-model evidence, Austria SIR evidence, or NAWM evidence.

## Decision

| Field | Result |
| --- | --- |
| candidate_rejected | `False` |
| research_direction_rejected | `False` |
| primary_criterion_status | `passed` |
| veto_diagnostic_status | `passed` |
| main_uncertainty | `synthetic independent blocks and diagnostic finite-grid KR only` |
| next_justified_action | `coupled nonlinear block/rank rung` |
| not_concluded | `no nonlinear, HMC, source-faithful, NAWM, or default-readiness claim` |

## Gates

| Gate | Status |
| --- | --- |
| all_candidate_fits_finite | `True` |
| reference_target_mass_error_le_0p03 | `True` |
| positive_defensive_mass | `True` |
| paired_core_conditional_backend | `True` |
| diagnostic_nonproduction_kr_classification | `True` |
| independent_block_proposal_randomness | `True` |
| conditional_formula_tieout_le_1e_10 | `True` |
| inverse_forward_roundtrip_le_1e_4 | `True` |
| candidate_finite | `True` |
| proposal_density_recomputed_at_online_state | `True` |
| all_arms_warmed_repeatability_error_le_1e_5 | `True` |
| same_scalar_score_fd_error_le_0p03 | `True` |
| candidate_minimum_ess_fraction_ge_0p5 | `True` |
| xla_enabled | `True` |
| expected_device | `True` |
| memory_growth_verified | `True` |

## Arms

| Arm | ESS fraction | Score/FD max error | Log likelihood |
| --- | ---: | ---: | ---: |
| exact_predictive_auxiliary | 1 | 0.000666142 | -89.3329468 |
| exact_uniform_auxiliary | 0.571157 | 0.00299692 | -89.2121658 |
| fitted_ttsirt_uniform_auxiliary | 0.562903 | 0.00384426 | -89.2182159 |

## Inference Status

No stochastic ranking is supported. The exact arms are mechanism references; observed cross-arm differences are descriptive only. The candidate either passes or fails the predeclared hard screen.

## Nonclaims

- no source-faithful Zhao-Cui claim
- no exact randomized-estimator or pseudo-marginal claim
- no nonlinear or cross-coordinate scalability claim
- no Austria SIR or NAWM claim
- no posterior correctness or HMC convergence claim
- no statistically supported ranking or superiority claim
- no production KR closure or default-readiness claim
