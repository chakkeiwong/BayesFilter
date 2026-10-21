# M13 score measurement before and after main synchronization — 2026-10-04

The merge produces **no material improvement in M13 score accuracy at these eight
fixed probes**. Both versions agree with the numerical derivative of their own
finite LEDH likelihood. The large remaining discrepancy is between that
likelihood's score and the score of the separate likelihood used for HMC
acceptance. At the low-Q probes and the pooled mean, even the zero, prior-only,
and saved-Gaussian scores are closer to the acceptance score than the LEDH score.

This completes the requested measurement within the surrogate-HMC test campaign.
It does not complete the all-model campaign or resolve M13's mixing problem.
No HMC transitions, fresh tuning, or runtime/default changes were made.

## What was compared

M13 is the range-bearing LG state / nonlinear observation model: state dimension
4, observation dimension 2, horizon 20, data seed 1704. Parameters are
theta=(lambda_Q, lambda_R), with Q=Q0 exp(2 lambda_Q) and R=R0 exp(2 lambda_R).
The actual frozen proposal uses the analytical canonical LEDH score with 4096
particles. HMC acceptance uses the frozen defensive trajectory-importance
likelihood (8 replicates, 4096 trajectories per replicate), with its existing
seed [2026092402, 9101]. Both posterior scores include the same Gaussian prior,
with standard deviation log(2).

For score g=d log likelihood/d theta, the proposal potential gradient is
-g+theta/log(2)^2. Under theta=center+L z, the posterior score becomes L^T g_post.
The code evaluates the actual runtime force wrapper and verifies this relation
at all eight probes in both source versions. The FD reference is a derivative
of the **finite acceptance likelihood**, not an exact integrated model score.

The pre-sync snapshot reconstructs the saved working source from stash
2ca7b4f06b623cdfe694ed67c2fde89449cecf37 plus hash-matched original untracked
files. The post-sync snapshot records HEAD a55c6be7a16e9401242ee3449248fa347eb58d90
plus its local modifications. All saved-run force dependency hashes and all
scope fields matched. Main's KSC observation callback is not called by M13:
M13 uses make_range_bearing_model and polar_observation callbacks. Seven shared
dependencies changed, so the numerical comparison was still necessary.

The points were frozen before score evaluation: archived rows at process-scale
percentiles 1, 5, 25, 50, 75, 95, 99 and the pooled mean. These are states visited
by an incompletely mixed chain, not eight independent posterior draws. Exact
coordinates, source hashes, original row indices, data, controls and designs
are in [inputs.json](inputs.json) and [source-snapshots.json](source-snapshots.json).
The [plan](../../surrogate-hmc-m13-score-before-after-2026-10-04.md) records
assumptions, comparison rules, limits, and the pre-run skeptical audit.

## Derivative accuracy and precision

The largest pre/post change in an FP64 analytical score component is
2.9416e-12. The largest component discrepancy
between analytical score and its own finest central difference is
2.5673e-06 before and
2.565e-06 after. Errors decrease
with the expected central-difference refinement:

| Central-difference step | Before: maximum component error | After: maximum component error |
|---|---:|---:|
| 0.001 | 0.00025655493 | 0.00025655485 |
| 0.0003 | 2.3090337e-05 | 2.3090791e-05 |
| 0.0001 | 2.5673121e-06 | 2.5650487e-06 |

Every base value, derivative evaluation, endpoint evaluation, and finite-
difference stability check passed. The evidence supports derivative agreement
at these points and this fixed design; it is not a proof over the whole domain.
FP64 promotes the once-rounded FP32 inputs and coefficients and approximates the
smooth finite algorithm, not differentiation of floating-point rounding.

In the primary FP32 runs, maximum relative score error versus the corresponding
FP64 score was 0.904%
before and 0.955% after.
Cross-process repeat evaluations varied by at most
0.329712
and 0.225433
in a component, or 0.704%
and 0.287%
in relative vector norm. Repeat log-likelihood differences were at most
0.061972 and
0.097827.
This is observed execution variation across processes; its compiler/kernel
cause was not isolated, and it does not establish within-process randomness.

Consequently the report's primary-run FP32 pointwise improvement/worsening
labels describe those evaluations only. Their FD error allowance does not
include this execution variation. They cannot establish a reproducible merge
benefit or a statistical ranking. These small variations do not explain the
much larger acceptance-score discrepancy or justify changing the FP32 default.

## Agreement with the acceptance likelihood

Components below are likelihood derivatives with respect to (lambda_Q,lambda_R),
before adding the common prior. FP64 is shown to separate approximation error
from FP32 execution variation.

| Fixed visited-state probe | LEDH score, before FP64 | LEDH score, after FP64 | Acceptance-likelihood score | LEDH minus acceptance log likelihood |
|---|---:|---:|---:|---:|
| visited_Q_p01 | (127.35414, 63.65716) | (127.35414, 63.65716) | (5.65158, 2.23317) | -34.88056 |
| visited_Q_p05 | (84.37511, 32.18949) | (84.37511, 32.18949) | (4.79652, -0.59652) | -20.64232 |
| visited_Q_p25 | (19.92491, 6.64844) | (19.92491, 6.64844) | (2.11747, 0.05328) | -3.58492 |
| visited_Q_p50 | (18.83997, 25.89777) | (18.83997, 25.89777) | (6.74010, 20.09376) | -3.07469 |
| visited_Q_p75 | (-0.73853, -1.65849) | (-0.73853, -1.65849) | (-2.69675, -3.14192) | -0.69339 |
| visited_Q_p95 | (-4.49805, -0.10449) | (-4.49805, -0.10449) | (-5.40170, -0.82295) | -0.48395 |
| visited_Q_p99 | (-7.92160, -2.89371) | (-7.92160, -2.89371) | (-8.46874, -3.23601) | -0.33665 |
| visited_pooled_mean | (6.86423, 2.57464) | (6.86423, 2.57464) | (0.13141, -0.46258) | -1.56826 |

At the first point theta=(-0.73272, -0.01466), the **actual FP32**
likelihood score changes from (126.77988, 64.00382) to
(127.76327, 64.13024). The acceptance-likelihood score is
(5.65158, 2.23317). Its componentwise FD
refinement discrepancies are 2.14e-7 and 1.01e-6. The roughly 20-fold excess in
the Q score and 29-fold excess in the R score is far beyond reference resolution
or FP32 variation.

Acceptance-likelihood values and derivative estimates are bitwise identical
between the two source versions. The LEDH-minus-acceptance log-likelihood
difference ranges from about -34.88 to -0.34 over these points; this is not a
constant offset that disappears on differentiation.

The minimum trajectory-importance ESS ranges from 2093.12 to 2228.20
out of 4096, and the largest normalized trajectory weight is 0.002673.
These are integration diagnostics, not MCMC ESS or proof that this finite
likelihood equals the model's integrated likelihood.

## Constructed simple score comparisons

The entries are whitened posterior-score L2 error against the same finite
acceptance target. Zero ignores all curvature, prior-only ignores the data,
and the saved Gaussian uses -C^(-1)(theta-center) from the actual frozen geometry.
This is a conditional score-accuracy sanity check, not a sampler comparison.

| Probe | Updated LEDH | Zero | Prior only | Saved Gaussian |
|---|---:|---:|---:|---:|
| visited_Q_p01 | 39.43977 | 2.24069 | 1.78455 | 0.62811 |
| visited_Q_p05 | 25.17358 | 1.91331 | 1.52569 | 0.18530 |
| visited_Q_p25 | 5.61733 | 0.75631 | 0.65910 | 0.13862 |
| visited_Q_p50 | 3.87316 | 5.71208 | 5.44855 | 4.07961 |
| visited_Q_p75 | 0.68296 | 1.17985 | 1.10166 | 0.79877 |
| visited_Q_p95 | 0.30908 | 2.01139 | 1.67103 | 0.67303 |
| visited_Q_p99 | 0.18383 | 3.09226 | 2.66840 | 1.25645 |
| visited_pooled_mean | 2.14773 | 0.05763 | 0.13399 | 0.22810 |

The LEDH score loses to all three simple scores at the low-Q probes p01, p05,
p25 and the pooled mean. That vetoes a claim of uniformly adequate force
approximation over the visited region. At the other four probes it has smaller
observed score error. Neither result establishes an HMC efficiency ranking.
No controls were tuned to these comparisons.

## Decision and inference status

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| No measured M13 derivative repair from this merge | FP64 changes below FD resolution at all 8 probes | No numerical, scope or source veto | Sparse fixed design; other regions unchecked | Preserve this completed before/after result | Global equality of algorithms |
| Own LEDH derivative agrees with its finite likelihood | Maximum component error below 2.6e-6 and refining | All finite and status checks passed | FP64 reference approximates the real finite algorithm | Treat derivative debugging separately from likelihood approximation | Exact model likelihood or full-domain correctness |
| Low-Q proposal/acceptance mismatch remains | Large resolved score disagreement | Simple-score sanity check fails at 4 probes | Finite acceptance target and only 8 visited states | Within the broader surrogate-HMC campaign, investigate low-Q likelihood approximation before retuning proposals | Sole cause of sticky chains or invalid MH correction |
| FP32 merge benefit unsupported | Small, mixed primary-run changes | Cross-process repeatability limits attribution | Only one additional process per version | Keep differences descriptive | Default change, performance ranking or posterior certification |

| Inference item | Status |
|---|---|
| Hard veto screen | All numerical/reference/source checks pass; adequate-force claim fails conditional simple-score comparison |
| Statistically supported ranking | None; deterministic fixed-probe comparisons do not rank sampler performance |
| Descriptive-only differences | FP32 pre/post changes, repeat variability, runtime and conditional force errors |
| Default-readiness | Not evaluated; no runtime/default changed |
| Next evidence needed | A target-specific likelihood repair and downstream surrogate-HMC evaluation; a reproducible FP32 benefit requires process/compilation replication |

The next investigation named above belongs to the existing surrogate-HMC goal.
It was not launched here: this authorized task was a bounded score measurement,
and the previous sampling allocation had expired. A new long sampling run is
neither needed to answer this question nor implied by this result.

## Execution, review and limits

Numerical work used 32.03 aggregate minutes including CPU preparation,
all primary workers, the failed packaging attempt and both repeat workers,
within the 60-minute cap. The actual GPU was NVIDIA GeForce RTX 5080
(CUDA_VISIBLE_DEVICES=1), with memory growth configured and verified before
initialization. TensorFlow 2.20.0-dev0+selfbuilt / TFP 0.25.0, XLA batch 4,
FP32 with TF32 enabled, and FP64 diagnostic references were recorded.
Peak main-worker TensorFlow allocator usage was 274369024 bytes in FP32 and
544613376 bytes in FP64. These timings are not a performance comparison.

One post-sync snapshot import failed because the Python-only source copy lacked
an existing compiled Sylvester library. Copying that unchanged library repaired
packaging; the original failure and library SHA are retained. A postprocessing
parity assertion then exposed one FP32 ULP from reconstructing constant division
as scalar division rather than the observed XLA rounded-reciprocal operation.
The analysis was corrected and all eight actual runtime forces matched exactly
in both versions. Experimental settings and numerical source were unchanged.

Review checked the saved consumer scope, source/import closure, once-rounded
common inputs, derivative step ordering/refinement, finite/status flags, force
sign and prior, L^T coordinate transformation, absolute-coefficient propagation
of derivative-resolution bounds, and all-eight runtime force parity. The
machine-readable [final audit](final-audit.json) records the checks.

The strongest alternative explanation is that these probes miss a region where
a shared fix matters, or that the finite importance target itself differs from
the integrated model. Both remain possible. A resolved derivative failure at
additional fixed probes, or a better independent likelihood reference showing
the importance approximation is inaccurate here, would alter the interpretation.
The weakest evidence is broad coverage: only eight visited states and two FP32
processes per version were checked. No new chain, acceptance, R-hat, MCMC ESS,
or posterior comparison was measured.

Artifacts: [structured result](report.json), [run manifest](run-manifest.json),
[scope parity](saved-chain-scope-check.json), [source call-chain impact](source-impact-check.json),
[analysis](analyze.py), [FP32 repeat diagnostics](observability-01/manifest.json).
