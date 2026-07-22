# P6 R4 Result: SIR-SGQF NeuTra Confirmation

Date: 2026-07-16

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `NEUTRA_CONFIRMED_THREE_PHYSICAL_MEANS_ONLY`

## Decision

The fresh frozen `dim3_lr1e3` dense-IAF transport passed target-bound NeuTra
HMC confirmation against the admitted same-target plain-HMC comparator. The
supported claim is limited to simultaneous agreement of the three declared
physical posterior means for this one T=20 SIR fixture:

- `kappa = 0.1 * exp(theta[0])`;
- `nu = 18 * exp(theta[1])`;
- observation-noise standard deviation `= 10 * exp(theta[2])`.

This is not a full-distribution, covariance, tail, mode, exact-filter,
calibration, forecasting, robustness, superiority, production, or default-
readiness result.

## Binding Evidence

Terminal root:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p6/SIR-SGQF/neutra-confirmation/attempt-01/`.

- target signature:
  `0e7921dbd1a2c9a943674b16fd10ccd8b68e1c889e9ae8269a06e0359a750fbc`;
- comparator result SHA-256:
  `621c3d6e748eed38433efaa02ff097a971132de89f323f12702533723e3ce9b2`;
- final training result SHA-256:
  `c69b4e4e02b68d13be74f7a87ffc0ec9b1d6a47bc8438d56c048577a78531854`;
- frozen transport hash:
  `dbd29efe786ec23c7b1098ba95ec6cad3a439b4889e04c67eeb2127965949c89`;
- R4 result SHA-256:
  `e8b6c159648ade9f2919d97674ffc50a8b55d75d591a256291c3abfdcd4dbcce`;
- recursive artifact-ledger SHA-256:
  `0cdb0bf006aa681259db524a5d3d3be5a74da3cfa7a605d543f7d7d715524a7d`.

All 29 entries in the recursive ledger were independently rehashed and
matched.

## Tuning, Warm-Up, And Retained Sampling

Six 64-burn-in/128-draw short probes only ordered candidates. They did not
admit a kernel. The ordered first candidate, step size `0.20` with eight
leapfrog steps, passed a disjoint 1,000-burn-in/1,000-draw verifier:

| Diagnostic | Result |
| --- | ---: |
| modern R-hat | `1.0005924637` |
| acceptance, explanatory only | `0.9935` |
| declared energy divergences | `0` |
| target/status/finite health | pass |
| verifier seed | `(20260716, 32102)` |

Warm-up retained two separate 1,000-draw chunks per chain. At 2,000 warm-up
draws per chain, the recent 1,000-draw window had maximum modern R-hat
`1.0014079967`, below the prospective `1.05` threshold. Warm-up archives are
preserved and excluded from posterior summaries.

Retained sampling used two separate 2,000-draw chunks per chain and stopped at
4,000 draws per chain only after the joint convergence-and-agreement rule
passed. All modern R-hat values are the maximum of rank-normalized split and
folded rank-normalized split R-hat.

| Retained diagnostic | Result |
| --- | ---: |
| maximum modern R-hat | `1.0000688996` |
| minimum bulk ESS | `16,358.48` |
| minimum tail ESS | `14,568.53` |
| declared energy divergences | `0` |
| target/status/finite health | pass |
| retained draws per chain | `4,000` |

## Simultaneous Physical-Mean Agreement

The prospective Bonferroni family-wise `0.05` rule required, for every mean,
`abs(mean_N - mean_H) + z * sqrt(MCSE_N^2 + MCSE_H^2)` to be no greater than
`0.10` times the comparator posterior standard deviation.

| Estimand | Absolute mean gap | Simultaneous upper bound | Margin | Status |
| --- | ---: | ---: | ---: | --- |
| `kappa` | `0.0000127661` | `0.0000425625` | `0.0001125664` | pass |
| `nu` | `0.0010951230` | `0.0059644463` | `0.0185300086` | pass |
| observation-noise SD | `0.0081327057` | `0.0234875259` | `0.0572063042` | pass |

No supported disagreement or unresolved-precision branch fired. Quantiles,
standard deviations, correlations, acceptance, runtime, and losses remain
explanatory only.

## Engineering And Diagnostic Checks

- GPU/XLA canary passed on the registered target; outputs were GPU-resident.
- TensorFlow memory growth was configured before logical-device initialization;
  full-device preallocation was disabled.
- Warm-up, retained chunks, and cumulative archives have distinct paths and
  the same target signature.
- Focused CPU-hidden confirmation/HMC/training/batching regression:
  `38 passed`.
- TensorFlow Probability emitted complex-to-real cast warnings from its
  FFT-based autocorrelation implementation. This is diagnostic-library
  telemetry: HMC states, target values, scores, log-accept ratios, and final
  diagnostics were finite real tensors.
- Recorded wall time was `3725.78` seconds.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| confirm `SIR-SGQF` at three-mean scope | convergence, ESS, health, and all three simultaneous mean bounds passed | clear | one fixture and mean-level agreement only | close P6 and execute P7 integrity synthesis | distributional equivalence, SGQF exactness, SIR calibration, superiority, robustness, readiness |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Clear for this target-bound run. |
| Statistically supported ranking | None; no transport or filter superiority comparison was designed. |
| Descriptive-only differences | Acceptance, runtime, training loss, quantiles, standard deviations, and correlations. |
| Default readiness | Not established. |
| Next evidence needed | Distribution-sensitive estimands and repeated fixtures for claims broader than the three means on this fixture. |

## Post-Run Red Team

The strongest alternative explanation is that NeuTra and plain HMC agree on
means while differing in tails, covariance, or modes. The present artifact
cannot rule that out and does not claim to. The weakest evidence is
generalization beyond the fixture. The strongest evidence is the exact target
and transport binding, separate archives, modern convergence diagnostics, and
prospectively frozen simultaneous mean rule.

