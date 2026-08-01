# Phase 8 Result: Paired 16-Seed Reset Audit At N=128

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status: `MIXED_OR_INCONCLUSIVE_SHARED_UPSTREAM_ERROR_DOMINATES`

## Outcome

The paired 16-seed `T=2,N=128` diagnostic completed for Contract E and the
no-reset weighted baseline. Both independently capped CPU-hidden float64 XLA
arms passed. Prepared-input identity was exact for every tensor except the
predeclared reset mask, and the seed order was exactly `81220..81235`.

The reviewed paired absolute-loss classifier returned
`mixed_or_inconclusive`. Five of six simultaneous paired intervals contained
zero. Only `r_scale` showed statistically supported higher mean absolute error
for Contract E under the predeclared Student/Bonferroni model, and that effect
was small: mean normalized loss increase `0.0011934`, interval
`[0.0002480, 0.0021389]`.

Contract E failed every small-shape simultaneous equivalence interval. The
no-reset baseline also failed every corresponding interval. Their mean errors
and interval patterns are similar, so the dominant Kalman discrepancy is shared
upstream of Contract E reset at this shape. This does not prove a causal
mechanism, but it rules out reset tuning as the primary justified next repair.

## Paired Loss Intervals

Positive values mean larger Contract E absolute error.

| Quantity | Mean | Simultaneous interval | Direction |
| --- | ---: | ---: | --- |
| value | `-0.0000959` | `[-0.0006946, 0.0005028]` | inconclusive |
| `phi1` | `-0.0027531` | `[-0.0113361, 0.0058298]` | inconclusive |
| `phi2` | `0.0188823` | `[-0.0055970, 0.0433616]` | inconclusive |
| `phi3` | `-0.0109356` | `[-0.0558423, 0.0339710]` | inconclusive |
| `q_scale` | `-0.0047332` | `[-0.0134503, 0.0039840]` | inconclusive |
| `r_scale` | `0.0011934` | `[0.0002480, 0.0021389]` | Contract E higher error |

Overall classification is mixed because all six directions do not agree.

## Contract E Mean-Error Intervals

| Quantity | Mean signed normalized error | Simultaneous interval | Boundary | Equivalent |
| --- | ---: | ---: | ---: | --- |
| value | `0.0109112` | `[0.0024298, 0.0193925]` | `0.001` | no |
| `phi1` | `0.0046598` | `[-0.0434320, 0.0527517]` | `0.05` | no |
| `phi2` | `0.0627892` | `[-0.1514848, 0.2770632]` | `0.05` | no |
| `phi3` | `-0.0841630` | `[-1.4644867, 1.2961606]` | `0.05` | no |
| `q_scale` | `-0.0654351` | `[-0.1246444, -0.0062258]` | `0.05` | no |
| `r_scale` | `-0.0270434` | `[-0.0515149, -0.0025719]` | `0.05` | no |

The no-reset intervals likewise fail all six boundaries. In particular,
no-reset mean signed normalized errors were `0.0108434` for value,
`-0.0701283` for `q_scale`, and `-0.0260097` for `r_scale`.

## Evidence Classification

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Do not attribute dominant error to Contract E reset | Five paired effects inconclusive; total errors nearly shared | Small supported `r_scale` reset worsening | Common proposal/importance-weight finite-N error | Trace and test common no-reset weighted LEDH path | Reset is harmless or exact |
| Keep Phase 8 scientific gate closed | All Contract E equivalence intervals fail | No FD/identity/chart veto | Small shape versus primary-shape behavior | Common-path root-cause diagnostic before any primary-shape run | Full-row failure or HMC readiness |
| Do not rank routes globally | Overall paired direction mixed | Model-based intervals, no power guarantee | Six outcomes and small shape | Preserve mixed verdict | Superiority |

## Artifact

Aggregate result:
`docs/plans/logs/contract-e-canonical-gradient-migration-continuation-20260714-115526/phase8/paired-reset-audit16/attempt1/result.json`
with SHA-256
`4512c012a6668133e37e532ddafd02e5abc2fa921741ed3bf5d23a658576b592`.

Both arm processes completed in about `77.6` seconds; aggregate wall time was
`156.861` seconds. The exact Bonferroni Student critical value was
`3.0362837314605713` with 15 degrees of freedom. No power, normality,
distribution-free, primary-shape, admission, HMC, or leaderboard claim follows.

## Post-Run Red Team

Strongest alternative explanation: both arms share finite-particle log-estimator
bias and score variance from the common LEDH proposal/importance-weight path;
the reset effect is secondary at `T=2,N=128`. Another possibility is a shared
common-path implementation mismatch. The next diagnostic must distinguish these
before increasing reset complexity.

What would overturn this conclusion: a common-path exact-oracle tie-out failure,
or a larger paired design showing a stable reset-specific effect comparable to
the total error.

Weakest evidence: the Student model is assumed, audit count 16 has no power
guarantee, and the result is small-shape only.
