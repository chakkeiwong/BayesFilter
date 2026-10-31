# M13 independent likelihood reference — 2026-09-23

This is the existing closure master's M13 integrated-reference task. It is
not an extra posterior look for AR2 or LG1. The question is whether an
independent bootstrap particle filter gives a precise marginal likelihood for
the actual 20-observation range/bearing fixture, after the wrapped-law repair.

## Evidence contract and derivation

Use the repository fixture's A, m0, P0, observations and transition-before-
observation timing (generation seed 701). State dimension is four and
observation dimension two. The prospective parameter coordinates are
lambda_Q and lambda_R, scaling Q and R by exp(2 lambda) respectively. This
stage evaluates the nine fixed pairs in {-log(2),0,log(2)} squared. It does
not define a prior, truncate posterior support, tune a DCTR scope, or establish
a parameter-posterior reference. A prior and its tail coverage remain required
before posterior testing.

For T=1, position is exactly Gaussian with mean (A m0)[:2] and covariance
(A P0 A' + exp(2 lambda_Q) Q)[:2,:2]; velocity integrates out. Integrate that
2D Gaussian times the observation density on a Cartesian trapezoid grid,
with 128, 256 and 512 intervals per axis on +/-8 marginal SD. The omitted
Gaussian mass is at most 2 erfc(8/sqrt(2)); multiply by a valid upper bound
on the observation density to bound omitted likelihood. The two position
coordinates are independent for this fixture, but the implementation uses the
full 2x2 Gaussian covariance and verifies the diagonal-domain bound applies.
Require the last two log integrals to differ by <=1e-6, with the tail
likelihood bound relative to the reference <=1e-8.

The independent observation implementation sums angular Gaussian aliases -2
through 2 and includes the real-valued range Gaussian. The maximum bearing
SD on this grid is .08, so omitted wrapped probability is bounded by
2 exp(-.5*(5*pi/.08)^2). Record its logarithmic bound to avoid calling
floating-point underflow exact zero. Preserve the fixture's sqrt(r^2+1e-12).
Check its value against the repaired actual callback at fixed points.

The bootstrap filter propagates the Gaussian dynamics and weights by that
independent density. It uses systematic resampling, with one uniform shift
per independent replicate at each time. Conditional expected offspring counts
are N times the normalized weights. Consequently, for any f, the conditional
expectation of Zhat_(t-1) times the particle mean of g_t f after propagation
equals the preceding unnormalized empirical measure applied to M_t(g_t f).
Induction gives E Zhat_t = p(y_1:t) (up to floating-point and the stated alias
truncation). Average likelihood estimates, not log likelihood estimates.

Run 16 independent replicates at N=4096,16384,65536. Preserve individual
T=1 and T=20 log estimates, minimum ESS and finite-value status. Construct
Student-t intervals on rescaled likelihood estimates, then transform their
positive endpoints to log units. These are normal-approximation Monte Carlo
intervals, not finite-sample coverage theorems. Use alpha=.01/45 for the
nine T1 checks, nine T20 checks and all nine pairwise comparisons between
the three particle levels. Qualification needs: finite estimates, positive
interval lower endpoints, T1 quadrature inside the highest-N interval, all
T20 level intervals overlapping pairwise, and highest-N log halfwidth <=.02.
The 45-way allowance covers these prospective comparisons conservatively;
no interval is used for posterior equivalence or repeated chain assessment.

Primary promotion: a qualified fixed-point marginal-likelihood reference table
for this grid and fixture. Nonfinite values, mismatched observation/timing,
failed T1 integral checks or failed interval screens veto this qualification.
Precision failure triggers reference repair, not rejection of surrogate HMC.
Reported runtime, acceptance, likelihood offsets, ESS and lower-level
differences are explanatory only. No sampler, DCTR accuracy, posterior
equivalence, speed ranking, full parameter-domain or default-readiness claim.

Constructed sanity comparators: likelihood at the predicted mean (ignores
state uncertainty); linearized Gaussian predictive likelihood (cheap local
geometry); unresampled prior-path importance sampling (correct law but likely
weight collapse). Evaluate each of the nine small/nominal/large process and
observation-noise cases separately. T1 quadrature supplies the common oracle;
report errors and uncertainty without selecting on these comparators.
Any statistically supported reference underperformance is a promotion veto;
descriptive differences do not establish a ranking.

## Assumptions, budget and pre-mortem

| Choice | Provenance/status | Justification | Failure mode and early check |
|---|---|---|---|
| T20/data701 | Existing fixture, target | Uses the actual model sequence | Wrong timing; T1 analytical predictive distribution check |
| Q/R scaling grid | New diagnostic hypothesis | Nine interpretable halving/doubling cases | Does not cover posterior tails; do not call this a posterior reference |
| 16 replicates and N ladder | Reference accuracy hypothesis | Measure independent MC noise | Skew/heavy-tailed likelihoods; report replicate spread, interval positivity, and level agreement |
| Systematic resampling | Reference algorithm | Expected empirical measure preserves weighting | Wrong indices or missing weight normalization; deterministic known-weight regression |
| FP64/GPU/XLA | Explicit independent-reference precision | Limits numerical error while keeping bounded cost | XLA random/resampling mismatch; T1 oracle and CPU focused regressions |
| Fixed aliases | Derived bound at sigma<=.08 | Independent density implementation | Mismatched covariance or larger scope; refuse unsupported scales |
| Trapezoid +/-8SD | Reference hypothesis plus Gaussian tail bound | Smooth 2D integrand at these scales | Missed narrow likelihood; three resolutions and tail bound |

Reserve 2400 CPU-worker and 3600 GPU-wall seconds from the unchanged global
54000/226800 campaign caps. Current recorded balances before this pass are
18661.233399 CPU and 193661.497302 GPU seconds. The review allowance is 180 CPU
seconds; the worker allowance is 1800 CPU / 2400 wall seconds. A 1024-particle
pilot is repeated once for timing only; launch the ladder only if its forecast
plus compilation reserve fits the worker's remaining allowance. At most three
fresh launches, each charged; no unchanged retry after a scientific veto.
Do not alter the reference grid or tolerances after looking at its result.

Pre-mortem: successful execution could still hide a shared wrong law, so the
oracle independently integrates positions and callback parity is executable.
A narrow likelihood could fool a coarse integral, so require refinement and a
tail bound. Tiny Monte Carlo error could coexist with bias if logs are averaged,
so test the likelihood-mean transformation explicitly. Cross-level agreement
does not prove interval coverage; skewness and individual values are retained.
A precise fixed grid is not a posterior reference; enforce separate flags.

Skeptical audit: pass for this bounded reference question. No approximate UKF
is treated as truth, no HMC convergence claim follows from a likelihood probe,
and no model horizon is shortened for the full-sequence calculation. The
reference has an explicit error limitation and the expensive stage is cost
gated. Canonical DCTR consumer construction and its scoped controls remain
later master tasks.

## Execution and artifacts

Environment: tftwogpu; GPU trusted/escalated; memory growth verified before
initialization; FP64 numerical reference, stable tf.function signatures and
XLA enabled. The master sets CPU hiding for its focused tests. No NumPy.

Command: /home/chakwong/anaconda3/envs/tftwogpu/bin/python
scripts/run_surrogate_hmc_program.py --stage closure
--closure-stages m13-reference --output
docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-reference-01

Use fresh numbered directories on localized repairs. Preserve the source
snapshot, exact commands, seeds, tensor-derived JSON values, timing, allocator,
scope, qualification, accounting and run manifest. Result:
docs/plans/surrogate-hmc-m13-reference-result-2026-09-23.md.
