# M13 independent likelihood reference — 2026-09-23

This is the existing closure master's M13 integrated-reference task. It is
not an extra posterior look for AR2 or LG1. The question is whether an
independent particle-filter reference gives a precise marginal likelihood for
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

## Guided reference repair after attempt 01

Attempt 01 completed with 35 focused tests and unchanged captured sources.
All nine T1 grids refined successfully and the highest-N intervals covered
their independent integrals. T20 precision failed at all nine points;
the nominal halfwidth was .05858, and the low-observation-noise cases had
particularly skewed likelihood estimates. These are reference limitations,
not evidence that surrogate HMC fails. CPU31.712198/GPU65.571358 were charged.

The next existing-master repair is an auxiliary importance proposal using the
current observation. It changes the reference proposal, not the model, data,
success criteria, posterior test or DCTR route. Retain the bootstrap attempt
and use fresh random seed 20260924 for the guided reference. The unchanged
particle ladder, independent T1 oracle, controls, nine parameter points and
qualification thresholds are rerun. No favorable subset replaces the grid.

For parent i with normalized weight w_i, let a_i>0 be a local Gaussian
predictive approximation using the polar Jacobian at A*x_i. Set
c=sum_i w_i*a_i, resample with probabilities w_i*a_i/c, and draw x from the
local conditional Gaussian q_i(x). The incremental likelihood estimate is

    c * mean_j [ g(y|x_j) f(x_j|parent_j) / (a_parent_j q_parent_j(x_j)) ].

Conditional expected offspring counts and importance integration cancel both
a and q, giving sum_i w_i integral f(x|i)g(y|x) dx. Thus the same induction
as above preserves the model likelihood; the approximations affect variance
only. The actual numerator still uses the wrapped observation density.
A nearest-image Gaussian in a is a proposal heuristic and is not substituted
for that density.

For this fixture, the two position innovations have common variance q and
are independent of velocity innovations. At predicted position m, define
orthonormal radial/tangential directions e_r,e_a. The two Jacobian magnitudes
are h_r=|m|/sqrt(|m|^2+1e-12) and h_a=1/|m|. In those directions the predictive
variances are s_j^2+q*h_j^2, the conditional mean displacements are
q*h_j*residual_j/(s_j^2+q*h_j^2), and the proposal variances are
q*s_j^2/(s_j^2+q*h_j^2). Velocity is drawn from its transition density, which
cancels in f/q. The position density ratio, auxiliary weight, and c are all
included explicitly. Reject a reference with invalid geometry/variances;
no covariance ridge, altered noise law or density fallback is used.

Assumption audit: isotropic position Q and diagonal full Q/R are verified,
not generalized beyond this fixture. Positive variances follow directly from
q>0 and s_j^2>0; the declared grid bounds them away from zero, so zero ridge is
justified algebraically. Near-origin geometry is a fail-closed guard with no
change to accepted results. Focused no-fire tests cover known-good paths.
A local Gaussian proposal may still have high-variance tails or ancestors;
the unchanged T1 and replicated T20 screens expose that risk.

Skeptical implementation review must check the f*g/(a*q) term, the factor c,
parent-index alignment, velocity cancellation, transition-before-observation,
and preservation of nonuniform correction weights between time steps. The
same 2400 CPU / 3600 GPU pass allocation remains; no added campaign budget.
Command stage is m13-reference-guided in fresh m13-reference-02.

Skeptical review before attempt 02: the implementation includes log(c),
-log(a_parent), log(f/q), and mean normalization; every proposal field uses
the same gathered parent indices. The diagonal-Q velocity factors cancel
exactly, and the orthonormal position basis has unit absolute determinant.
Correction weights remain nonuniform for the next ancestor-selection step.
The known-limit regression covers all three reference routes; the independent
T1 integral remains the early numerical check for nontrivial proposals.
No promotion criterion, target, data, grid, or allocation is changed.

## Final reference proposal repair after attempt 02

Attempt02 passed 37 focused tests, source checks and all T1 checks. At N65536,
only point8 satisfied the T20 precision criterion. Nominal halfwidth .04047
and lowest-Q/lowest-R halfwidth2.03898 are descriptive; no superiority ranking
over attempt01 is supported. Charged CPU34.008267/GPU67.823195 seconds.

The remaining planned attempt uses a Gaussian proposal for the entire latent
position sequence, with a defensive prior mixture and exact importance
weights. This is the existing master's independent-reference task: the same
likelihood, data, parameter points, success criteria and total allocation.
It is a proposal repair within importance sampling, not a DCTR change or a
new surrogate-HMC target. The original sequential references are preserved.
No proposal or threshold is selected after inspecting the final attempt.

For Z=(p1,...,pT), linear Gaussian dynamics imply exactly
Z ~ N(mu, C0 + exp(2 lambda_Q) Cq). Construct C0 from the propagated initial
state and Cq from all transition innovations; integrate velocities out.
Transition-before-observation determines every block. T1 is the first two
coordinates of this same Gaussian. The desired marginal likelihood is

    integral p_theta(z) product_t g_theta(y_t|z_t) dz.

Fit a Gaussian proposal using deterministic Gauss-Newton iterations on the
negative log of this integrand. Use the exact wrapped density and its
alias-weighted angular score. The approximate Hessian is prior precision
plus H' R^-1 H, hence positive definite without a ridge. Backtracking must
reduce the exact objective; failed geometry or convergence rejects the
proposal. This optimizer defines a proposal only; it does not replace the
integral by a Laplace likelihood.

With Gaussian q_G(z), use q(z)=(15/16)q_G(z)+(1/16)p_theta(z), and independently
sample that mixture. Record exact weights
w(z)=p_theta(z)*product_t g_theta(y_t|z_t)/q(z).
Then E_q[w] is the desired likelihood. Since q>=p/16 and g is bounded on
the fixed grid, weights have a finite bound and finite variance. This fact
does not guarantee that finite samples reveal every important tail.
Keep 16 independent replicate likelihood estimates and the original
4096/16384/65536 ladder, T1 oracle and T20 criteria.

Assumptions and defaults: the joint Gaussian prior is derived from the actual
fixture and checked against explicit constant-velocity covariance formulas.
Gauss-Newton is a proposal-construction hypothesis, initialized at the prior
mean; 60 iterations, squared Newton decrement<=1e-8, and up to16 backtracking
halvings are bounded numerical controls, with convergence and objective
decrease checked. Proposal covariance is 1.2 times the inverse Gauss-Newton
precision; this fixed modest expansion is a tail-coverage hypothesis, not a
tuned default. The defensive mixture bounds weights independently of its
adequacy. The ratio and mixture normalization are checked in an exactly
Gaussian, uninformative-observation limit, and the nontrivial T1 integral
remains the early end-to-end falsification test. Two distinct predeclared
initializations (prior mean and observed polar positions) must agree on the
objective within1e-7, or qualification fails; this is a mode diagnostic,
not evidence that no other mode exists.

Skeptical audit: a fitted Gaussian is not itself the reference value; only
corrected integration is admissible. Preserve all nine points, both independent
sequential comparators and the heuristic table. Joint-prior construction,
normalization, velocity integration, exact density numerator, disjoint random
seeds and all captured proposal diagnostics need executable checks. Skewed
weights or small observed error bars can mislead; retain convergence,
replicate uncertainty, particle refinement and T1 vetoes. No posterior,
ranking, or default-readiness conclusion is supported by this stage.

Remaining M13 allocation CPU2334.279535/GPU3466.605447332. Attempt03 is the last
numerical reference launch in this pass; CPU1800/GPU2400 job caps remain.
Use seed20260925 and stage m13-reference-trajectory in fresh m13-reference-03.

Implementation review before attempt03: the consumer calls the joint-trajectory
reference through its own master stage and captures its source. Exact prior
blocks include every innovation at the correct date. Sampling and density use
the same defensive mixture and covariance factors; Gaussian determinants are
included. The observation helper broadcasts over the time axis and its prior
scalar use remains covered. Both proposal starts, final decrements, centers,
curvature diagonals, and maximum weights are retained. Failure status prevents
qualification; no failed proposal density is interpreted as zero model density.
All numerical kernels have fixed signatures and XLA; no pfor, runtime NumPy,
covariance ridge, adaptive target, or stochastic HMC force is introduced.
Static checks passed. Focused regressions remain the master prerequisite;
the independent T1 oracle is the runtime nontrivial importance-ratio check.
