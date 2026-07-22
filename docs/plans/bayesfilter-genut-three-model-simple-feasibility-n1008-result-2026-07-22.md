# GenUT Three-Model N=1008 Feasibility Result

Date: 2026-07-22
Status: `aggregate_suite_status_revoked_reduced_sir_mechanics_only`
Plan: `docs/plans/bayesfilter-genut-three-model-simple-feasibility-plan-2026-07-22.md`
Artifact: `docs/benchmarks/artifacts/genut_three_model_simple_feasibility_20260722/attempt02_n1008/`

Suite correction: this aggregate is not a valid three-actual-model feasibility
result. The reduced-SIR phase is an artificial mechanics fixture, and the
existing Chapter 18b structural target was omitted. Generalized-SV and KSC
phase results remain model-specific diagnostics. See
`docs/plans/bayesfilter-genut-actual-model-suite-correction-2026-07-22.md`.

## Scope correction

The prior campaign used `N=96`, which is below the owner-required numerical
test scope. It is mechanics-only evidence. This replacement uses `N=1008`, the
smallest convenient count above 1000 divisible by six, so the positive Gaussian
GenUT weights are represented exactly for both one- and two-dimensional states.

All three models use `T=10`, one seed, FP32, TF32, GPU, XLA, verified TensorFlow
memory growth, and the same warm-start controls: epsilon `2`, eight Sinkhorn
steps, and ridge `1e-5`. This remains an untuned feasibility comparison.

## Value and score comparison

| Target and parameter order | Route | Value | Score |
|---|---|---:|---|
| reduced preclip SIR `(log_kappa, log_nu, log_obs_noise)` | GenUT | -11.5997181 | `(-0.00510440, 0.06945710, 4.08647919)` |
| reduced preclip SIR `(log_kappa, log_nu, log_obs_noise)` | dense manual-score reference | -10.8591241 | `(-0.00460581, 0.06417235, 2.21323426)` |
| generalized SV `(z_gamma, log_tau, mu_over_tau)` | GenUT | -16.0161610 | `(-0.10777602, -0.15420970, 0.02178448)` |
| generalized SV `(z_gamma, log_tau, mu_over_tau)` | fixed-branch Zhao-Cui diagnostic | -16.0198730 | `(-0.12547017, -0.15484276, 0.02226093)` |
| KSC SV `(z_gamma, log_beta)` | GenUT | -19.9688282 | `(-0.67607313, 0.50751644)` |
| KSC SV `(z_gamma, log_beta)` | fixed SGQF | -19.9509416 | `(-0.69247488, 0.60957816)` |
| KSC SV `(z_gamma, log_beta)` | principal-square-root UKF | -19.9509416 | `(-0.69247488, 0.60957816)` |

## Differences

| Target | GenUT minus comparator value | GenUT minus comparator score |
|---|---:|---|
| reduced preclip SIR vs dense | -0.740594 | `(-0.000499, 0.005285, 1.873245)` |
| generalized SV vs Zhao-Cui diagnostic | 0.003712 | `(0.017694, 0.000633, -0.000476)` |
| KSC SV vs SGQF/UKF | -0.017887 | `(0.016402, -0.102062)` |

Generalized SV has close value and score at this scope. KSC value is close but
the `log_beta` score still needs an `N`/tuning ladder. Reduced SIR improved over
the mechanics run but remains inaccurate in value and especially the
observation-noise score.

## Recursive score and numerical gates

| Target | Maximum scaled recursive-score/FD error | Maximum transport/reset residual | Gate |
|---|---:|---:|---|
| reduced preclip SIR | 0.0007553 | 1.0215e-5 | pass |
| generalized SV | 0.0003268 | 1.0848e-6 | pass |
| KSC mixture SV | 0.0001617 | 2.5192e-6 | pass |

The runtime scores are manual recursive derivatives. Finite differences are
representative-point diagnostics of the identical finite scalar only. The
checks make a missing derivative term an implausible explanation for the
remaining SIR and KSC comparator gaps; approximation and untuned finite-cloud
behavior remain the leading hypotheses.

## Execution

The campaign completed all three phases in `51.72` seconds on an RTX 4080 SUPER.
XLA compiled CUDA clusters and all GenUT results were finite. Recorded allocator
peaks were approximately `100.8 MB` for SIR and `75.5 MB` for each scalar-SV
phase. These TensorFlow allocator measurements are diagnostic and should not be
interpreted as total process or device reservation.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Continue generalized SV | strong short-prefix agreement | no hard veto | one seed, short horizon, approximate comparator | target-specific tuning and multi-seed `N` ladder | ranking or leaderboard readiness |
| Continue KSC with score-focused repair | close value; residual `log_beta` score gap | no hard veto | finite-cloud bias versus comparator approximation | tune controls and run an `N`/seed ladder | exact-SV validity or superiority |
| Do not advance SIR directly to claim run | value and observation-noise score remain discrepant | implementation/FD/transport gates pass | clipping nonlinearity, controls, and particle count | tune against dense reference and increase `N` | canonical Austria-SIR result or rejection of GenUT |

## Inference status

| Item | Status |
|---|---|
| Hard veto screen | Passed for all three corrected `N=1008` phases |
| Statistically supported ranking | None; one seed and one short prefix |
| Descriptive-only differences | All value, score, runtime, memory, and `N=96` versus `N=1008` changes |
| Default readiness | Not evaluated |
| Next evidence needed | Target-specific tuning, multiple seeds, larger-`N` ladders, uncertainty intervals, and untouched longer horizons |

The aggregate JSON is
`docs/benchmarks/artifacts/genut_three_model_simple_feasibility_20260722/attempt02_n1008/result.json`.
