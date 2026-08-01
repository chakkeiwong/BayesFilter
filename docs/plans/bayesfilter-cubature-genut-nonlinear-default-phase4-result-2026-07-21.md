# Cubature/GenUT Nonlinear Default Program: Phase 4 Result

Date: 2026-07-21

Status: `HISTORICAL_NONDGP_FINITE_PROGRAM_VARIANCE_ONLY_NO_SV_NOMINATION`

> **Correction, 2026-07-22:** The hand-written transformed observations are not
> SV-DGP data.  The variance calculation is valid only conditional on that
> arbitrary finite-program input and cannot nominate an antithetic policy for
> SV.  All SV-method interpretation is revoked.

## Outcome

The exact transformed-SV candidate route was evaluated for 16 fixed-randomness
seeds at `T=4`, using independent clouds and sign-reflected innovation pairs.
The arithmetic mean of each pair is a candidate antithetic estimator; each
constituent cloud still uses the same finite value/score program.

| N | Coordinate | Independent SD | Antithetic SD | Ratio |
|---:|---|---:|---:|---:|
| 12 | `theta_gamma` | 0.193222 | 0.092528 | 0.4789 |
| 12 | `theta_log_beta` | 0.716331 | 0.270274 | 0.3773 |
| 24 | `theta_gamma` | 0.167085 | 0.068061 | 0.4073 |
| 24 | `theta_log_beta` | 0.312655 | 0.194522 | 0.6222 |

All four ratios are below one. The value means changed descriptively by less
than `0.04` in this small diagnostic. Maximum reset/marginal residual was
`3.13e-7`; all rows were finite and the artifact was hard-valid.

## Checks

| Check | Result |
|---|---|
| Diagnostic command | Passed as `attempt02`; `hard_valid=true` |
| Attempt01 | Failed before GPU/numerical work because direct script lacked repository import bootstrap; no artifact created |
| Runtime | `8.813 s` CPU-hidden diagnostic |
| GPU/XLA | Not claimed; GPU intentionally hidden and `jit_compile=false` |
| Artifact | `docs/benchmarks/artifacts/cubature_genut_score_variance_20260721/attempt02/` |

## Interpretation

Antithetic innovations are a viable variance-reduction nomination for this
scope. This is not yet a tuning default or promotion result because the paired
estimator needs an explicit probability/finite-target contract, more replicated
seeds, target-horizon testing, and comparison with other variance remedies.

## Decision Table

| Decision | Status |
|---|---|
| Same-scalar variance nomination | Passed this diagnostic |
| Statistical superiority | Not tested or claimed |
| Default variance policy | Not selected |
| Nonlinear full-horizon precision | Not established |
| Next justified action | Phase 5 GPU/XLA/TF32 smoke and scaling gate |

## Nonclaims

No exact filtering, unbiasedness, method superiority, default readiness,
leaderboard admission, HMC readiness, or NAWM result follows.
