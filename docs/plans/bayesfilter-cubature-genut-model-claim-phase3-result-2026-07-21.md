# Cubature/GenUT Model-Claim Phase 3 Result

Date: 2026-07-21

Status: `HISTORICAL_NONDGP_ENGINEERING_ONLY_SV_SCIENTIFIC_CLAIMS_REVOKED`

> **Correction, 2026-07-22:** The observations were generated directly as iid
> Normal transformed values, not by the SV latent transition and log-chi-square
> observation equation.  The target-horizon failure, bias interpretation,
> tuning result, and proposed scientific repair are irrelevant to SV and
> revoked.  Finiteness, placement, residual, runtime, and same-finite-scalar
> derivative evidence survive only as engineering checks.  See
> `bayesfilter-exact-sv-nondgp-fixture-demotion-correction-2026-07-22.md`.

## Outcome

The exact transformed-SV model-claim diagnostic was tuned on disjoint
calibration/validation seeds and evaluated on 16 untouched claim seeds at
`T=2,10,50`. The candidate remained finite and its recursive score was
computed through the loop-native tangent path. However, the independent dense
same-target reference showed horizon-dependent value discrepancy.

The initial tuning/claim artifact is preserved at:

`docs/benchmarks/artifacts/cubature_genut_model_claim_20260721/attempt03/`

Its selected controls were:

```text
epsilon=2.0, sinkhorn_steps=8, ridge=1.0e-4
```

The frozen-control particle ladder is:

`docs/benchmarks/artifacts/cubature_genut_model_claim_20260721/n_scaling_attempt01/result.json`

| N | T | Maximum absolute value error | Score SD (theta_gamma, theta_log_beta) | Gate |
|---:|---:|---:|---|---|
| 12 | 2 | 0.589 | (0.147, 0.626) | Fail |
| 12 | 10 | 0.709 | (0.347, 0.803) | Fail |
| 12 | 50 | 3.613 | (0.834, 3.792) | Fail |
| 48 | 2 | 0.169 | (0.056, 0.264) | Pass budget |
| 48 | 10 | 0.284 | (0.138, 0.321) | Fail |
| 48 | 50 | 1.715 | (0.422, 0.996) | Fail |
| 96 | 2 | 0.158 | (0.063, 0.221) | Pass budget |
| 96 | 10 | 0.346 | (0.116, 0.163) | Fail |
| 96 | 50 | 0.730 | (0.322, 1.051) | Fail |

The predeclared value-error budget was `0.25`. The result therefore does not
support a target-horizon exact-SV claim. Increasing `N` reduces the discrepancy
descriptively, especially at `T=50`, but the target budget is not met.

## Interpretation

This is evidence against the current finite candidate configuration for the
target-horizon exact-SV claim, not evidence against Cubature/GenUT as a general
research direction. The dominant unresolved hypothesis is accumulated finite
program bias from the staged OT/reset approximation, not merely score Monte
Carlo variance. The Contract E comparator remains blocked because its proposal,
continuation, and reset scope are not identical to this candidate finite
program.

## Decision Table

| Decision | Status |
|---|---|
| Candidate finite value/recursive score | Finite for all recorded rows |
| Scope-specific tuning | Completed as diagnostic, selected controls frozen |
| Target-horizon exact-SV accuracy | Failed at `T=10` and `T=50` under `N<=96` |
| Particle-bias repair | Descriptively favorable but insufficient |
| Score variance | Still large at `T=50`; explanatory, not a promotion gate |
| Same-target Contract E comparison | Blocked by finite-program identity mismatch |
| Default/leaderboard readiness | False |
| Next justified action | Test a target-preserving reset/transport bias repair or substantially larger N with fresh scope tuning; do not assemble leaderboard rows |

## Nonclaims

No method ranking, exact nonlinear filtering theorem, unbiasedness, HMC
readiness, leaderboard admission, default promotion, or NAWM result follows.
