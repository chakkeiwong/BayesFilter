# Austria GenUT Pairwise Verdict Reassessment

Date: 2026-08-05

Prior result:
`docs/plans/bayesfilter-austria-sir-pairwise-moment-genut-score-trial-result-2026-07-30.md`

Primary artifact:
`docs/benchmarks/artifacts/austria_sir_pairwise_moment_genut_score_20260730/attempt01/result.json`

Current verdict: `PAIRWISE_IS_MATERIALLY_USEFUL_AUSTRIA_SCORE_STABILIZER`

## Revised Verdict

The pairwise higher-moment correction is materially useful for Austria SIR
GenUT. It prevented the recursive score from deteriorating to the catastrophic
scale observed under diagonal-only correction. It must be retained as the
primary stabilization candidate and as a required comparator in every further
Austria score, NeuTra-training, or HMC-facing investigation.

This revises the earlier practical interpretation, not the historical gate
outcome. The July experiment correctly recorded
`PAIRWISE_SCORE_VARIANCE_PROMOTION_FAIL` because the pairwise finite likelihood
mean moved relative to the diagonal finite approximation. That gate answered
whether pairwise preserved the diagonal approximation closely enough for
immediate promotion. It did not answer whether pairwise prevented an unusably
unstable score. On the latter question, the evidence is strongly favorable.

Pairwise is not yet an admitted score route. Absolute score accuracy, derivative
identity for the pairwise scalar, cross-process reproducibility, and downstream
NeuTra behavior remain unproved.

## Evidence Reinterpreted

At the frozen Austria `T=20`, `N=1008` scope with 16 common particle seeds, the
comparison was:

| Coordinate | Diagonal score SD | Pairwise score SD | SD reduction |
|---|---:|---:|---:|
| `log_kappa_scale` | `3435.632` | `36.512` | `94.1x` |
| `log_nu_scale` | `1272.439` | `17.762` | `71.6x` |
| `log_observation_noise_scale` | `301.967` | `20.635` | `14.6x` |

All `16/16` pairwise rows were finite and mechanically valid. The paired-seed
aggregate geometric variance ratio was `0.000468`, with the declared 95%
bootstrap interval `[0.000082, 0.063166]`. Thus the score-dispersion reduction
is statistically supported for this frozen scope; it is not merely a favorable
single seed.

The diagonal route's later NeuTra-readiness result reinforces the practical
importance of that reduction. Its score changed from approximately
`(-37,-76,7)` at `theta=(-0.002,-0.002,-0.002)` to
`(-949,341,38)` at `theta=(0.002,0.002,0.002)`. Historical diagonal runs also
contained a first-coordinate score near `-13723`. These are precisely the
large local and seedwise excursions that pairwise matching was designed to
suppress.

The pairwise value changed from a diagonal mean of `-683.364` to `-682.104`, a
shift of `1.260` log units. This remains material relative to the diagonal
seedwise Monte Carlo standard error and explains the historical promotion
failure. It is not evidence that pairwise made the likelihood less accurate:
the diagonal finite approximation is not truth, and the descriptive SGQF value
`-682.348` is closer to the pairwise mean. SGQF is not an exact nonlinear
oracle, so this observation cannot establish that the pairwise value is more
accurate either.

## Skeptical Audit

The revised verdict was checked against the main ways it could be misleading:

| Risk | Audit finding | Consequence |
|---|---|---|
| Wrong baseline | Diagonal-only is the current batch-native baseline but is demonstrably unstable; it is not an accuracy oracle | Do not treat preserving the diagonal value as the overriding scientific criterion |
| Proxy promoted to accuracy | Seedwise score SD measures precision, not score bias or same-scalar correctness | Claim stabilization only, not score accuracy |
| Value shift ignored | The `1.260` shift is real and failed the frozen gate | Preserve it as an unresolved target-change diagnostic |
| Small or selected sample | The primary result used 16 predeclared common seeds and a paired bootstrap interval | Scope-limited variance conclusion is supported; broad generalization is not |
| Particle-count dependence | At `N=4032`, pairwise remained finite for `3/3` descriptive seeds, but the sample is too small for ranking and full-filter tangent growth was not uniformly improved | Require particle-count-specific tuning and validation |
| Stale controls | The July pairwise arm used Sinkhorn/balance `16/16`; the later current-source diagonal tuning selected `8/8` | Retune pairwise for the exact current batch-native scope before a new claim |
| Missing same-scalar test | Later diagonal/antithetic routes failed endpoint or finite-difference identity; the July pairwise trial did not establish current batch-native pairwise identity | Same-scalar value/JVP parity is a hard prerequisite for training admission |
| Arithmetic mismatch | Existing pairwise evidence used FP32, TF32, GPU/XLA; no clean FP64/no-TF32 localization exists | Run arithmetic localization before attributing residual failures to the algorithm |

The audit supports the revised stabilization verdict while blocking a stronger
accuracy, default, NeuTra-readiness, or HMC-readiness claim.

## Research Decision

| Decision | Primary evidence | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain pairwise as the leading Austria GenUT stabilization candidate | `14.6x` to `94.1x` lower score SD with `16/16` valid rows | No stabilization veto | Absolute score bias and exact current-scope derivative identity | Port pairwise correction to the batch-native route and retune it for the exact `T=20,N=1008` scope | No exact-score or posterior claim |
| Reject diagonal-only as the sole Austria training candidate | Catastrophic seedwise and local score excursions | Practical score-stability veto | Whether a repaired diagonal design could become stable | Keep diagonal only as the naive baseline and failure control | No universal rejection of diagonal moment matching |
| Preserve the value-shift concern | Pairwise mean moved by `1.260` log units | Blocks immediate default promotion | Which finite approximation is closer to the intended likelihood | Compare pairwise and diagonal values against the strongest available independent reference with uncertainty | No claim that diagonal value is more accurate |
| Defer NeuTra training admission | Current batch-native diagonal endpoint identity fails; current batch-native pairwise identity is untested | Hard same-scalar engineering veto | Whether a faithful batch-native pairwise JVP passes parity and replay | Localize primal mismatch, implement pairwise value/JVP parity, then run the biased-score training experiment | No training or HMC readiness |

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto evidence | Current diagonal batch-native route fails endpoint identity; pairwise has not yet passed the equivalent current-scope gate |
| Statistically supported result | Pairwise reduces score dispersion relative to diagonal at the frozen `T=20,N=1008`, 16-seed scope |
| Descriptive-only differences | SGQF proximity, `N=4032` three-seed results, tangent-growth factors, and runtime |
| Viable candidates | Pairwise is the leading stabilization candidate; diagonal remains a failure-control baseline |
| Default readiness | Not established |
| Next evidence needed | Current-scope batch-native pairwise implementation, exact-scope tuning, primal equality, scalar parity, two-step finite differences, replay, and then heldout NeuTra training behavior |

## Updated Experiment Order

1. Locate and repair the first per-time-step divergence between tangent-free and
   tangent-carrying primal values.
2. Add pairwise value and total-JVP operations to the true batch-native route;
   do not substitute the historical scalar implementation into training.
3. Retune diagonal and pairwise controls separately for the exact current
   Austria `T=20,N=1008`, dtype, TF32, XLA, batch, and data scope.
4. Require pairwise primal identity, scalar-row parity, stable two-step central
   differences, and same-/cross-process replay. Treat failure as an engineering
   or numerical repair trigger, not evidence that pairwise stabilization was
   useless.
5. If those gates pass, run the intended experiment: train NeuTra with the
   pairwise score and judge it using heldout endpoint objectives and an
   MH-corrected proposal canary. Include diagonal-only and untrained transports
   as baselines.

## Post-Review Nonclaims

This reassessment does not show that the pairwise score is unbiased, that it is
the exact derivative of the intended physical likelihood, that its shifted
finite value is more accurate, or that it will train a useful NeuTra transport.
It establishes the narrower but important conclusion that dismissing pairwise
because of the prior value-shift gate was scientifically too strong: pairwise
is the only tested Austria correction that robustly prevented massive score
deterioration at the primary particle count.
