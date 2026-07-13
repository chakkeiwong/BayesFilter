# Scalar SSL-LSTM Filtering HMC Validation Phase 2T - MAP-Local Handoff

## Decision

- phase2t_map_local_reference_handoff_passed: `True`
- vetoes: `[]`
- viable_for_phase2u_retuned_map_local_hmc_screen_subplan: `True`
- zero_divergence_claim_made: `False`
- next_justified_action: draft and review Phase 2U retuned MAP-local fixed-kernel HMC screen subplan

## Matrix Checks

- precision_z covariance_z identity error: `1.1254733403169899e-15`
- factor_z covariance reconstruction error: `8.881784197001252e-16`
- precision theta scale-transform error: `1.0000285044498014e-09`
- covariance theta scale-transform error: `4.887283910903761e-10`

## Phase 2U Handoff

- candidate grid: `[{'num_leapfrog_steps': 2, 'step_size': 0.785, 'trajectory_length_L_times_epsilon': 1.57}, {'num_leapfrog_steps': 4, 'step_size': 0.3925, 'trajectory_length_L_times_epsilon': 1.57}, {'num_leapfrog_steps': 8, 'step_size': 0.19625, 'trajectory_length_L_times_epsilon': 1.57}, {'num_leapfrog_steps': 16, 'step_size': 0.098125, 'trajectory_length_L_times_epsilon': 1.57}]`
- selection policy: first candidate in listed order that passes hard vetoes and acceptance envelope
- native divergence policy: positive native divergence is a hard veto when available; unavailable native divergence is recorded as unavailable and is not zero-divergence evidence

## Inference Status

| field | value |
| --- | --- |
| hard_veto_screen | passed |
| statistically_supported_ranking | none; no sampler run and no method comparison |
| descriptive_only_differences | matrix residuals and old-geometry projection diagnostics |
| posterior_correctness | not assessed |
| hmc_readiness | not assessed |
| gpu_xla_readiness | blocked |
| default_readiness | not assessed |
| zero_divergence_claim | not made |
| next_evidence_needed | draft and review Phase 2U retuned MAP-local fixed-kernel HMC screen subplan |

## Nonclaims

- Phase 2T MAP-local reference handoff diagnostic only
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
