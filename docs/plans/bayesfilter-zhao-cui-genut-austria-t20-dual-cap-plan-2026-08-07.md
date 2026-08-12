# Austria SIR T20 Pairwise/Dual-Cap GenUT Comparison Plan

Date: 2026-08-07
Status: `AUTHORIZED_T20_FINITE_PROGRAM_COMPARISON`

## Research Intent

Determine whether the new pairwise-plus-dual-cap GenUT finite program is more
usable at the production-length Austria SIR target (`T=20`) than the prior
uncapped pairwise route and the diagonal-only route. The run tests numerical
validity, value stability, and score dispersion on the same target and seeds.

This is not a teacher-fidelity experiment. A strict sampled bounded Zhao-Cui
teacher currently exists only for `T=1/2`; no valid `T=20` teacher tensor is
available. The coordinate cap is therefore classified as `extension_or_invention`,
not source-faithful Zhao-Cui behavior.

## Evidence Contract

| Field | Frozen contract |
|---|---|
| Target | `austria_sir_T20`, `T=20`, state dimension 18, observation dimension 9, parameter dimension 3, `N=1008` |
| Data/event order | Existing sealed observations, hash recorded by `_build_targets`, `x0_then_transition_before_y1_to_y20` |
| Baseline | Diagonal-only higher-moment route: diagonal steps 4, strength 0.2; pairwise and caps disabled |
| Prior comparator | Historically tuned T20 pairwise route: diagonal steps 4, diagonal strength 0.2, pairwise steps 4, pairwise strength 0.02; caps disabled |
| New candidates | Prior T20 pairwise comparator plus smooth coordinate cap `b` in `{0.90, 0.95, 0.98}`, power 8; radial cap is crossed at `{off, 2.0}` only for `b=0.98` |
| Seeds | Tuning `98301,98302`; claim `98201..98216`; common random numbers across all arms |
| Runtime | FP32, TF32 enabled, TensorFlow GPU/XLA, memory growth configured before initialization |
| Hard gates | finite/program-valid, GPU execution, mean/row/column residuals and score-increment sum residual `<5e-4`, normalized displacement `<=2`, post-cap bounded coordinates `<1` when cap enabled |
| Primary descriptive comparison | Per-arm value mean/SD/95% t interval and each score coordinate mean/SD/95% t interval; paired candidate-minus-baseline rows |
| Secondary diagnostics | cap activity/displacement, inverse derivative, pairwise residual, score SD ratios and paired bootstrap interval, internal finite-difference score check where computationally feasible |
| Promotion rule | No default or scientific promotion. A candidate is only a viable T20 finite-program arm if all hard gates pass; lower dispersion/value shifts are descriptive with 16 seeds |
| Nonclaims | No exact nonlinear Austria score, no bias reduction, no teacher agreement, no superiority, no HMC/NeuTra/default readiness |
| Artifact | `docs/benchmarks/artifacts/zhao_cui_genut_austria_t20_dual_cap_20260807/attempt01/` |

## Skeptical Audit

| Risk | Control |
|---|---|
| Wrong scope | T20 target is built from the repository-owned target builder and its observation hash/event order are recorded |
| Invalid teacher substitution | No T2 teacher is reused; the absence of a T20 teacher is a stated limitation |
| Weak baseline | Diagonal-only and prior pairwise arms are both included with common random numbers |
| Proxy promotion | Score variance and finite-difference diagnostics are explanatory/descriptive; only hard finite-program gates can establish viability |
| Cap changes objective | Value shifts, cap activity, and nonclaims are recorded; no accuracy claim is made |
| Hidden stochastic advantage | Same particle seeds and data are used for every arm; 16 seeds remain descriptive rather than a ranking proof |
| Environment mismatch | GPU/XLA/TF32/memory-growth settings are recorded in the manifest |

Audit decision: `PASS_FOR_BOUNDED_T20_EXECUTION`.

## Stop Conditions And Budget

Run one focused campaign with six arms. Stop the campaign on target/hash
mismatch, GPU or memory-policy failure, missing required diagnostics, an invalid
baseline, or invalid artifacts. A failed candidate rejects that arm but does not
stop the remaining predeclared comparisons. Do not expand the grid after viewing
claim results.
