# M10 independent posterior reference

The scientific goal is testing endpoint-corrected HMC with an imperfect full
canonical DCTR force. This stage prepares the independent comparator R for the
first nominal M10 scope. Its input is the frozen claim-nominal T2 dataset from
the original M10 reference run. Reading claim observations for independent
integration is allowed; they must not select DCTR controls or HMC settings.
Stress inference remains a separate scope. The sine/quadratic law and independent
Normal(0,log(2)^2) log-noise priors remain unchanged.

Integrate the repaired two-state likelihood times the actual prior over the
parameter square [-9s,9s]^2, s=log(2). Use tensor-product Gauss-Legendre rules
with24,40,64 nodes per axis. The initial state integration uses20 Legendre
nodes per composite interval and128 normal nodes. Reevaluate all64^2 parameter
points with32/128 and32/64 state/convolution nodes. These are numerical
hypotheses tested by separate parameter, state and convolution refinements.
Zero likelihood from FP64 underflow is allowed only in the value-only integral;
it contributes zero mass and is counted. It cannot issue a gradient or authorize
an HMC endpoint. All densities and unnormalized predictive integrals must be
finite, nonnegative, and ordered between zero and likelihood.

Freeze posterior means, standard deviations and marginal .1/.5/.9 quantiles
from the64-by64,32/128 result. Compare13 functions: two standardized log-noise
means, two centered second moments, the cross moment, six marginal CDFs, and
two posterior predictive probabilities Pr(y2_rep<=.5) and Pr(y2_rep<=2).
Here y2_rep is an independent observation conditional on the same x2 and noise
scale as the second observed datum; this is a posterior replicate, not y3.
Integrate its normal CDF with the same latent posterior weights. The thresholds
are prospective fixed scales, not chosen from claim observations.

Use Legendre interpolation of each marginal at its quadrature nodes to integrate
partial CDF cells. For coefficient c_k of P_k(x), integrate P_0 as x+1 and
P_k as (P_(k+1)-P_(k-1))/(2k+1) for k>=1. A numerical test against known
uniform and normal CDFs, a monotonicity/positivity screen and parameter
refinement must pass before those interpolants define reference cuts.

For a full-prior tail bound, Normal(y;h(x),R)<=1/(sqrt(2*pi)*sigma_R)
gives likelihood times prior bounded by
C*exp(2*s^2)*Normal(lambda_Q;0,s^2)*Normal(lambda_R;-2*s^2,s^2),
where C=1/(2*pi*exp(-1.2)). Compute the mass and centered second-moment
integrals outside the square analytically. A union bound for the two coordinate
tails supplies nonnegative upper bounds; Cauchy-Schwarz supplies absolute
first-moment bounds, and abs(ab)<=(a^2+b^2)/2 bounds the cross moment.
Bound the omitted state probability by6*Phi(-10), times the same likelihood
envelope. State truncation is assessed separately from quadrature refinement.

The denominator for these bounds needs a strictly positive lower bound, not
a posterior estimate silently called exact. Construct one by restricting both
noise scales to a finite positive-prior rectangle and x0,x1,x2 to finite
intervals. Integrate x0 prior mass exactly; lower-bound both transition and
observation normal densities using their largest possible residual and the
smallest/largest scale in that rectangle. Multiply by the x1,x2 interval
lengths and noise-prior rectangle mass. Use the largest of a finite declared
set of such valid lower bounds: noise intervals [j*s/2,(j+1)*s/2] for
j=-2,...,5, latent half-widths .05,.1,.2,.4, and x0 centered at its prior
mean; x1,x2 centers are the positive observation roots, or the quadratic
vertex for observations below its minimum. Maximizing valid lower bounds
does not select a sampler or change its target.

Primary numerical qualification: the final parameter, state and convolution
normalization changes are each<=1e-6 relatively; the last parameter function change is <=half the previous
change+1e-5. For each function, twice the sum of the maximum of the last two
parameter changes, the state refinement and convolution refinement, plus
analytical tail/state allowances, must be <=one tenth of its eventual margin
(.1 for five moments, .02 for six CDFs, .03 for two predictive probabilities).
Require finite positive normalization and scales, valid inverse residuals,
probability bounds and interpolation checks. Empirical quadrature allowances
are not rigorous error bounds; the omitted-tail bounds are conditional only
on the explicitly constructed analytical evidence lower bound.

Failure is a reference repair trigger and blocks B/R or C/R using this
reference. It does not reject the surrogate-HMC direction or the separately
defined C/D sampling question. Runtime and posterior summaries are explanatory;
there is no sampler ranking or default promotion. The later heuristic set is
direct quadrature sampling, prior independence Metropolis, a Gaussian
independence proposal fitted to this frozen reference, and zero-force HMC,
evaluated centrally and in marginal tails. All heuristic results remain NOT_RUN.

CPU-only independent-reference exception, FP64 TensorFlow/XLA, GPU hidden,
threads2/1. No NumPy, pfor, learned component or runtime DCTR change. New
value-only output omits gradient fields and reuses the same quadrature algebra.
Budget:40 CPU seconds for focused tests and700 CPU seconds/600 wall seconds
for preparation including at most one localized infrastructure retry; all
charge the existing M10 allocation. Time representative values at both state
resolutions and forecast1.5 times all evaluations plus60 CPU seconds overhead
before the full ladder. Defer if the forecast exceeds the remaining allowance.
Archive all grids, source/data hashes, lower-bound inputs, exact command,
environment, seed, timings and accounting under fresh
surrogate-hmc-all-models-20260921-m10-posterior-* roots.

Skeptical audit: probe accuracy alone was insufficient, so this stage compares
integrated posterior functions over a larger domain and bounds omitted mass.
State and parameter refinements cannot substitute for one another. The proper
prior must be included once; a truncated integral must retain its omission
bound. Posterior replicates must not be mislabeled as future observations.
The claim data is used solely for reference integration. If the positive
evidence lower bound is too weak, record the qualification failure rather than
replacing it with an unjustified empirical denominator. The audit passes the
bounded stage subject to the numerical checks above.

Pre-run implementation verification: four focused tests pass, including the
previous score/local-grid tests, value-only likelihood parity, known uniform
and normal partial CDFs/quantiles, prior normalization and moments, and a finite
positive analytical evidence lower bound. Interpolation screening on8193
points permits only1e-8 absolute CDF rounding excursions or decreases and
negative density no larger than1e-5 of its maximum. These are empirical
diagnostics, not positivity proofs. Finest parameter interpolation must pass at
each of the three state/convolution configurations. Scores are absent from the
value-only return; zero-mass values cannot be mistaken for HMC score evidence.

## Measured execution repair, before the integration ladder

The first CPU attempt stopped at its prospective cost gate:1219.74 remaining
CPU seconds/886.15 wall seconds exceeded700/600. No parameter grid was run.
Actual profiling cost7.492621479 CPU seconds. Its analytical evidence lower
bound for the fixed claim dataset is3.251838322365109e-6.

Evaluate the identical FP64/XLA value-only kernels on designated GPU1, with
verified memory growth and TF32 disabled, before spending the integration
budget. At each of the three state/convolution settings compare CPU and GPU
likelihoods and normalized predictive probabilities at the three existing
profile points: relative likelihood and absolute probability differences
must be<=1e-10. Nonfinite values or parity failure veto this backend repair.
Measure GPU timings after compilation and apply the same1.5-times forecast.
Peak TensorFlow allocator use must be below4 GiB. Summaries remain the same
TensorFlow computation; no numerical thresholds, model, prior, data, grid or
function changes. Scalar parameter evaluations are independent-reference work,
not a batch-native training claim.

This moves reference computation to the repository's designated GPU hardware
within M10's existing7200-GPU-second ceiling. It reserves at most600 GPU
seconds for this stage; the original700 CPU/600 wall ceilings are cumulative
including the CPU attempt, with the40 CPU test allowance separate. This is a
localized execution repair, not an expanded campaign or a second statistical
look. Preserve the CPU cost stop and use a fresh root for this second execution
attempt. A second cost failure is a stage deferral, not permission to drop
refinements. Invocation adds `--gpu` under trusted permissions with the same
designated GPU UUID and growth environment used by the M10 consumer.

## Prospective parameter refinement after the first full integration

GPU attempt02 completed all five grids, using59.288297284 GPU wall seconds
and66.172461721 CPU seconds. Nine CPU/GPU parity comparisons pass. State and
convolution normalization changes are below8e-15. The analytical posterior
omission bound is5.657e-9, and the largest omission allowance is7.529e-7.
However, parameter40/64 normalization changes1.427e-6, above1e-6, and the
64-node marginal interpolant has minimum density-0.0003652. The24/40 changes
also leave10 of13 function allowances above their limits. This is a parameter
quadrature failure, not a latent-reference or surrogate-HMC failure.

The next numerical candidate uses the same parameter square and rules with
40/64/96 nodes. Reuse the immutable40/64,20/128 grids only after verifying
their data, numerical likelihood source, order, shape, axis and file hashes.
Evaluate96^2 points at20/128,32/128 and32/64. Freeze summaries/cuts from the
new96,32/128 grid and re-evaluate every ladder function at these common cuts.
Require the same contraction, normalization, interpolation and allowance
criteria. No old failure is reclassified; this is a fresh reference candidate.

Allow this third total execution as one explicitly planned numerical refinement,
within the unchanged cumulative700 CPU/600 wall and600 GPU stage ceilings.
The separate40-CPU-second focused-test ceiling still applies. Forecast only
the new grids, since the two reused grids are hash-checked saved evaluations;
retain the1.5 multiplier and60-second overhead. If this refined candidate fails,
preserve it and reassess the remaining family budget before another action.

## Subnormal validity-guard repair

Attempt03 stopped at parameter(-3.3648781186133125,-3.192114385833288),
state32/convolution128. Its likelihood was6.014e-320 and each unnormalized
predictive integral6.0143e-320: independent reductions differed by one
FP64 subnormal increment. The inverse residual was2.867e-16. This is a
rounding false veto, not evidence of a materially invalid likelihood.
Preserve the failure and its55.556120143 CPU/51.994043099 GPU wall seconds.

Keep all computed values unchanged. In the validity check only, use absolute
tolerance1e-12*max(likelihood, smallest positive normal FP64), rather than
pure relative tolerance. Its floor is approximately2.23e-320; no result is
clipped, renormalized, or supplied with a gradient. Test rejection of finite
substantive overshoot, negative and nonfinite values, the exact failure, and
unchanged healthy likelihood/score parity. Recheck the failed point explicitly
on the designated GPU before the integration ladder. This is a Class B
numerical validity guard, not a change to the model or quadrature algebra.

Permit one fourth total execution with --refine-from pointing to attempt02
and --subnormal-guard-repair. Recompute all40/64/96 grids under current source;
do not reuse prior grids after the guard-source change. Keep cumulative
700 CPU/600 wall/600 GPU limits and every posterior accuracy criterion.
Allocate at most30 additional CPU seconds for this localized regression
repair, within the original1800-CPU M10 family ceiling. The earlier focused
tests used37.268141519 of40 seconds; preserve that accounting separately.

Skeptical audit: a subnormal normalization ratio cannot certify predictive
accuracy and is never used here; full posterior summaries divide integrated
numerators by the ordinary positive evidence. The guard permits only negligible
absolute error at underflow. Fresh integration and all existing refinement,
tail, interpolation and function-precision checks remain required. If another
material failure occurs, report its component before planning further work.

## Reference-stage resource reallocation

Attempt04 passed the exact GPU subnormal regression and all nine CPU/GPU
parity checks, then stopped before integration: forecast601.244 CPU/569.409
wall seconds exceeded the stage's remaining553.172/467.113 seconds. Actual
cost15.627339705 CPU/12.431436516 GPU wall. The maximum per-point timing,
including the slower32/64 kernel, remains in the forecast; do not relax the
forecast multiplier or accuracy checks to force admission.

Reallocate within the already authorized M10 family7200 GPU/1800 CPU ceiling:
the posterior-reference stage now has cumulative850 CPU/800 wall/800 GPU
seconds and at most five executions. This replaces only the earlier700/600/600
stage allocation. The remaining M10 allowance before attempt05 is
7025.219138718 GPU/1362.252055738 CPU seconds, so this redistribution fits
both the family and global budget. It reduces resources available to subsequent
M10 controls and inference, whose forecasts must use actual remaining costs.
Keep the same source/data/functions, fresh full40/64/96 grids, guard repair,
parity checks, and1.5-times forecast. Attempt05 uses the same explicit repair
arguments as04 and a fresh root. A cost stop may be retried only within this
fifth execution, and a qualified reference may not be automatically repeated.

Skeptical audit: independent R is needed for B/R and C/R, but its completion
is not posterior-HMC success and cannot gate the scientifically separate C/D
question. This resource redistribution has no new compute authorization or
method change; the family and campaign ceilings are unchanged. Preserve the
cost stops as such. Forecast all later stages before reserving their budgets.

## Master integration after qualification

Attempt05 qualified the nominal posterior reference under all six checks.
Add P3-M10-posterior as an independent child of the frozen local-reference
preparation. Reuse it only after checking its numerical source/data/grid hashes,
complete check set, and all thirteen allowances against their original margins.
Reuse grants no HMC or tuning authority. The numerical source is unchanged;
orchestration edits do not invalidate completed numerical grids. New preparations
start with the tested40/64/96 ladder and rerun all accuracy checks.

Account nested master children once at their parent, and reduce both worker
CPU and wall ceilings after preceding jobs. Up to60 CPU seconds from the
remaining M10 family are allocated for focused orchestration regressions and
one public master reuse execution. These checks neither rerun the numerical
posterior integration nor use claim data to select controls or HMC settings.

The first orchestration regression process hit its CPU watchdog after51 test
progress markers, with no reported assertion failure. Its19.677892667 CPU
seconds remain charged; this is an incomplete run, not a passing suite. The
original30-second validation allowance was too small for the combined suite.
The60-second allocation prospectively permits one fresh bounded retry and
the public reuse invocation within the unchanged M10 family ceiling. This
repair changes no numerical reference, test expectation, or scientific gate.
