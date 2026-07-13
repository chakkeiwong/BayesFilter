# Scalar SSL-LSTM Filtering HMC Validation Phase 2 - Local Quadratic Reference

## Decision

- phase2_local_quadratic_reference_agreement_passed: `False`
- vetoes: `['mean_abs_error_above_0p5', 'std_ratio_outside_0p5_2p0']`
- zero_divergence_claim_made: `False`
- next_justified_action: write Phase 2 result and draft reference/localization repair before GPU/XLA

## Reference

- formula: `{'local_log_density': 'c + l_z^T z - 0.5 z^T K_z z', 'coordinate_map': 'z = F u, F = chol(M_z)', 'precision_u': 'K_u = F.T @ K_z @ F', 'covariance_u': 'C_u = inv(K_u)', 'mean_u': 'm_u = C_u @ F.T @ l_z'}`
- reference mean u: `[-0.044565226160942085, 0.14339340258862682, 0.12440293581509161, 0.3807781120936724]`
- reference std u: `[0.9999999999999999, 0.9999999999999999, 1.0, 1.0000000000000016]`
- precision-u identity max abs error: `3.219646771412954e-15`

## HMC Summary

- pooled mean u: `[2.684115678520539, 0.6267063734817137, 1.5582468030301495, 0.5156267037406872]`
- pooled std u: `[3.513587386436987, 2.7850393993384595, 2.0517916307643334, 2.2664029297024846]`
- acceptance rates: `[0.921875, 0.734375, 0.578125]`
- native divergence statuses: `['not_exposed_by_kernel', 'not_exposed_by_kernel', 'not_exposed_by_kernel']`

## Agreement

- mean abs error: `[2.728680904681481, 0.48331297089308684, 1.433843867215058, 0.1348485916470148]`
- mean abs error max: `2.728680904681481`
- std ratio: `[3.5135873864369875, 2.78503939933846, 2.0517916307643334, 2.266402929702481]`
- std ratio range: `2.0517916307643334` to `3.5135873864369875`
- interpretation: local quadratic reference screen failed; localize geometry, transform, or short-chain behavior before GPU/XLA

## Inference Status

| field | value |
| --- | --- |
| hard_veto_screen | failed |
| statistically_supported_ranking | none; no method comparison and no uncertainty interval |
| descriptive_only_differences | mean errors, standard-deviation ratios, acceptance, and log-accept tails |
| posterior_correctness | not assessed; local quadratic reference only |
| default_readiness | not assessed |
| gpu_xla_readiness | not assessed; CPU-hidden artifact analysis |
| hmc_readiness | not assessed |
| zero_divergence_claim | not made |
| next_evidence_needed | localize geometry/transform/short-chain mismatch before GPU/XLA |

## Nonclaims

- local quadratic reference agreement screen only
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
