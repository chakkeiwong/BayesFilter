# Scalar SSL-LSTM Filtering HMC Validation Phase 2R - Localization

## Decision

- phase2r_localization_passed: `True`
- selected_outcome: `outside_geometry_trust_region`
- vetoes: `[]`
- zero_divergence_claim_made: `False`
- next_justified_action: draft geometry/centering repair or MAP-local reference subplan

## Diagnostics

- transform identity max abs error: `3.3306690738754696e-15`
- point norms in u: `{'reference_mean': 0.4278034459629271, 'pooled_hmc_mean': 3.2079965478482895, 'seed_0_mean': 1.9314939055758316, 'seed_1_mean': 3.024038476914353, 'seed_2_mean': 5.718576264956027}`
- outside trust region points: `{'pooled_hmc_mean': 3.2079965478482895, 'seed_0_mean': 1.9314939055758316, 'seed_1_mean': 3.024038476914353, 'seed_2_mean': 5.718576264956027}`
- local quadratic drops: `{'reference_mean': -0.09150789418887717, 'pooled_hmc_mean': 4.785183748624255, 'seed_0_mean': 1.8530276413241875, 'seed_1_mean': 4.01623507453999, 'seed_2_mean': 15.838221711125513}`
- large quadratic drop points: `{'seed_2_mean': 15.838221711125513}`

## Target Replay

- computed: `True`
- pooled minus center value: `-0.2815211066872152`
- role: explanatory_only

## Nonclaims

- Phase 2R localization diagnostic only
- not an exact posterior reference
- not HMC readiness evidence
- not HMC convergence evidence
- not posterior correctness evidence
- not a zero-divergence claim when native divergence is unavailable
- not sampler superiority evidence
- not statistically supported ranking evidence
- not GPU/XLA production-readiness evidence
- not default-readiness evidence
- not Zhao-Cui source-faithfulness evidence
