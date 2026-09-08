# Corrected q=20 Fresh Theta Pilot

Status: `PASS_THETA_MEASURE_PILOT`

Both arms keep target and proposal log densities in theta in R^4. The geometry artifact is a calibration warm start only.

| Arm | Status | Role |
|---|---|---|
| M0 | `PASS_THETA_MEASURE_PILOT` | fresh_theta_m0_candidate_not_smu_u_admitted |
| C0 | `PASS_THETA_MEASURE_PILOT` | fresh_theta_c0_descriptive_comparator |

## Decision

Finite measure and status gates can nominate Phase 29. They do not admit an SMC-U authority or a posterior claim.

## Nonclaims

- ESS, mass, mode occupancy, and root counts are descriptive diagnostics.
- No IID, whitening, mode-discovery, LEDH, NeuTra, HMC, or default promotion claim.
