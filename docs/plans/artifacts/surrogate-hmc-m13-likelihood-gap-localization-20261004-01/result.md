# M13 likelihood-gap localization — 2026-10-04

The frozen LEDH calculation has severe importance-weight concentration and
already gives an inaccurate likelihood and parameter score at the first
observation. At all eight fixed probes, a simple bootstrap calculation using
the same initial particles and process noises is closer to first-observation
quadrature in both log likelihood and score-vector norm. This describes these
realized calculations; it is not a statistically supported ranking over
particle designs. No method or default was changed.

At the low-process-noise probe theta=(-0.73272085,-0.01465508), the first
observation has the following results. Scores differentiate the log process
and log observation standard-deviation multipliers and exclude the prior.

| First observation | Log likelihood | Q score | R score | Particle ESS |
|---|---:|---:|---:|---:|
| Independent position-marginal quadrature | -0.13791267 | 0.01149807 | -0.10229746 | N/A |
| Same-particle bootstrap, no flow | -0.13013616 | 0.00812633 | -0.10443582 | 205.04 |
| Canonical LEDH, selected frozen controls | -2.22116858 | 3.57681741 | 0.19335105 | 6.66 |

The canonical trace exactly reproduces the saved full-consumer FP64 value and
both score components at all eight points: maximum absolute discrepancy zero.
The preceding measurement showed analytical derivatives agreeing with finite
differences of the full-T20 LEDH likelihood to about 2.6e-6. The derivative of
that finite objective is correctly computed. Its accuracy as a derivative of
the model's integrated likelihood fails at T1 in this configuration.

## Why the proposal loses useful particles

The code samples a transition conditional on each ancestor,

    u_i = A x_previous_i + chol(Q_theta) epsilon_i,

but constructs the flow using the per-particle UKF covariance

    P_predict_i = A P_previous_i A' + Q_theta.

The initial cloud represents N(m0,P0), and each local UKF covariance is also
initialized to P0. P_predict is a proposal-design covariance; Q_theta is the
conditional covariance used to draw u_i and evaluate its transition density.
At low Q, the first predicted position variance is 116.74 times the process
variance. The flow displaces particles by an RMS Mahalanobis distance of 20.61
in process-noise units (the four-coordinate norm, not 20.61 per coordinate).

A move towards the observation that is plausible under the broad UKF
covariance can be very improbable under the narrow transition from a
particular ancestor. Most moved particles then receive negligible importance
weight. At T1 only 6.66 of 4,096 particles remain effective; 40 carry 99% of the
weight. Across T20 the minimum ESS is 1.1579; at observation 20 one particle
carries 92.9146% of the weight. These are filtering-particle ESS values, not
MCMC effective sample sizes.

The code includes the transition/proposal density ratio and flow determinant:

    w_i = (1/N) g(y | F_i(u_i))
          p_Q(F_i(u_i) | A x_previous_i)
          |det D F_i(u_i)| / p_Q(u_i | A x_previous_i).

Its flow coefficients use an auxiliary trajectory initialized at the
ancestor's predicted mean. Conditional on that ancestor, the auxiliary
trajectory is independent of the current innovation u_i. The actual flow
therefore composes affine steps with recorded determinants. For an invertible
F_i, removing g gives an integral equal to one by substituting x=F_i(u).
The finite-particle calculation estimates that normalization as only
0.00344047 at T1. This exposes severely inadequate realized importance
integration even though the numerical validity flags pass.

The scale mismatch, weights, and normalization identity support the proposed
mechanism. A controlled refinement remains necessary to quantify the roles of
flow settings and particle design. These results do not prove an omitted
derivative term, a universally incorrect LEDH algorithm, or a working repair.

Errors also accumulate later. At observations 11, 12, 13 the low-Q likelihood
contributes Q scores 15.56, 25.79, 16.15 while ESS is 2.99, 3.17, 9.29. The full
Q score is 127.35 versus 5.65 for the fixed importance acceptance target, with
log likelihood lower by 34.88. This is parameter-dependent distortion, not
a harmless constant likelihood offset.

## What this establishes

The first likelihood increment and its score precede the first transport
reset. The initial failure therefore does not require reset accumulation.
Across inspected steps, the reset preserves the computed weighted mean within
1.56e-14 filtering SD and covariance within 6.07e-15 relative Frobenius norm.
That cannot restore a filtering distribution that was already poorly
estimated; it also does not rule out later higher-moment approximation error.

The discrepancy persists in FP64. The preceding measurement found no material
FP64 before/after-main score change. More arithmetic precision, whitening, or
longer chains does not by itself repair this likelihood approximation.

The new T1 reference integrates the two-dimensional Gaussian position marginal
with the wrapped observation law, using the exact rounded A,Q,R of LEDH.
The 512/1,024 grids differ by at most 1.78e-15 in log likelihood and 8.89e-12
in finite-difference score. Changing FD step 3e-4 to 1e-4 changes a component
by at most 1.94e-9. The relative omitted-Gaussian-tail bound is at most 1.29e-12.
These are refinement checks and a value-tail bound, not an analytic error
bound for every score component. Bootstrap callback scores also pass their
declared 1e-6 finite-difference check.

The full-T20 acceptance reference remains a finite importance approximation;
its score uncertainty across independent designs was not measured here.
The T1 diagnosis no longer depends on assuming that full-T20 reference is exact.
Neither this trace nor the bootstrap comparator establishes a working
replacement force or full-T20 filter.

Earlier materiality-control selection tested sensitivity to numerical
protection settings. It did not establish likelihood or score accuracy
against this first-step reference. A passed validity or control-sensitivity
screen cannot answer this accuracy question.

## Conditional comparisons

| Probe | LEDH T1 log error | Bootstrap T1 log error | LEDH T1 score error norm | Bootstrap T1 score error norm | Minimum T20 LEDH ESS |
|---|---:|---:|---:|---:|---:|
| visited_Q_p01 | -2.083256 | 0.007777 | 3.577556 | 0.003993 | 1.158 |
| visited_Q_p05 | -1.641223 | 0.007196 | 2.942086 | 0.005316 | 1.662 |
| visited_Q_p25 | -0.775181 | 0.002819 | 1.222279 | 0.019086 | 3.853 |
| visited_Q_p50 | -0.495791 | 0.003603 | 0.852868 | 0.127889 | 2.832 |
| visited_Q_p75 | -0.413193 | -0.012555 | 0.704171 | 0.078369 | 10.167 |
| visited_Q_p95 | -0.239344 | -0.050723 | 0.508432 | 0.190299 | 27.766 |
| visited_Q_p99 | -0.157822 | -0.061876 | 0.394597 | 0.186645 | 58.110 |
| visited_pooled_mean | -0.578413 | -0.002460 | 0.902257 | 0.040238 | 6.284 |

The labels are eight previously fixed visited-state probes, not an assertion
that the archived chain converged. All comparisons describe frozen designs.

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Localization completed; frozen approximation inaccurate at T1 | Exact trace/consumer parity; quadrature resolves T1 gap | Weight concentration and same-particle bootstrap defeat an accuracy claim at these probes | Effective repair and later filtering errors | Existing master force/likelihood repair: proposal/particle refinement against T1 reference, then T20 score refinement | No rejection of surrogate HMC; no repair selected or new HMC admission |

| Inference status | Finding |
|---|---|
| Hard validity screen | No source, input, finite-value, parity, or quadrature-domain failures |
| Statistically supported ranking | None; one frozen design at eight points |
| Descriptive-only differences | ESS, weight maxima, bootstrap/LEDH distances, runtime |
| Default readiness | Unchanged; FP32 remains runtime default and canonical LEDH policy is unchanged |
| Next evidence needed | Scope-specific proposal/particle refinement and independent T20 score uncertainty |

A poor surrogate force can still define valid Metropolis-corrected proposals
for the separately specified acceptance target when other sampler conditions
hold. This does not make force accuracy a new convergence requirement. It
identifies an approximation problem that can explain poor proposals; it does
not replace cumulative posterior diagnostics or prove that repairing the
force repairs mixing.

The strongest alternative explanation for the T20 comparison is reference
error. Independent T1 quadrature and the same-callback bootstrap comparison
limit that explanation for the initial failure. Reproducible refinement
restoring normalization and value/score agreement would overturn a claim that
the failure is intrinsic to the method. The weakest remaining evidence is
causal attribution among proposal covariance, discretization and particle
design; none has been isolated by changing one factor.

## Reproducibility

Call chain: M13LikelihoodTarget.evaluate -> canonical_batch_fused_value_score ->
_single_cloud_model and canonical_value_and_analytical_score. The diagnostic
invokes that same final function with trace outputs. Executable parity covers
all eight values and both parameter directions.

Source anchors in the frozen source_after snapshot and matching checkout:
bayesfilter/highdim/ledh_range_bearing_target_tf.py:103,166;
bayesfilter/highdim/ledh_canonical_batch_fused_tf.py:366;
bayesfilter/highdim/ledh_canonical_score_tf.py:431,458,594,615,649,706,1206,1403;
bayesfilter/testing/surrogate_hmc_m13_reference.py:228.

Plan: ../../surrogate-hmc-m13-likelihood-gap-localization-2026-10-04.md.
Commands, commit, environment, source hashes, data/seeds, memory policy and
timings are in run-manifest.json, first-step-baseline/run-manifest.json, and
the corresponding reports. Both workers used GPU/XLA with verified memory
growth. FP64 is a diagnostic reference exception. Trace peak TensorFlow
allocation was 605,578,752 bytes. No HMC transitions or runtime edits occurred.

The workers consumed 102.27 seconds. Total score-diagnostic compute is now
2,024.10/3,600 seconds; 1,575.90 remain. Both exited zero. summary.json preserves
comparisons; report.json preserves per-observation summaries; final-audit.json
records checks. Earlier sources and results are preserved.
