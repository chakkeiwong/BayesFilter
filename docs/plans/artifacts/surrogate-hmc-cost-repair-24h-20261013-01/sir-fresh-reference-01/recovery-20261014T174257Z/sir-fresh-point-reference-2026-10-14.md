# Independent SIR reference at the OT discrepancy point

## Question and evidence contract

Owner request: compute an independent reference for fresh dataset 26101401,
T=50, log-scale theta=(-0.0175,-0.0175,-0.0175), to determine whether either
existing OT configuration has a material likelihood/score error. This is an
extension of the current master's reference diagnostics, not a new HMC campaign.
The exact theta and observation array must be extracted from the SIR result.json
under ot-epsilon1-steps5-01; never regenerate the data or round it again.

Target: the marginal likelihood and its three log-scale parameter derivatives
under NonlinearSQMCSpec(sir_d18), x0~N(initial_mean,I), followed by transition
then observation for each of the 50 steps. Baselines are the two saved N4032 OT
rows. Neither is treated as truth. Before=[-46.24422868673017,
-1.3257831440132581,-13.302526448053706]; after=[4.421015114612653,
-20.980306316072685,-6.4971909416381735].

Reference: existing independent bootstrap PF with analytical complete-data
Fisher scores, likelihood-weighted replication pooling, delete-one-replication
jackknife MCSE. This computes a finite-particle approximation to the model
score, not the derivative of a finite resampling program. It is independent
of OT but shares model callbacks, so fixed-state finite-difference tests and
the exact linear-Gaussian reference fixture must pass before execution.

Precision ladder: start N=131072, 8 independent seeds; continue N=524288,
16 seeds, then N=1048576, 32 seeds if uncertainty or ladder shifts remain
material. Independent seed groups are predeclared per rung. FP64 is an explicit
reference-only exception; original N4032 precision/defaults remain unchanged.
Particle counts here are independent-reference counts, not LEDH test defaults.
GPU/XLA, TF32 off, verified memory growth; stable TensorFlow signatures, no pfor.

Primary evidence: absolute likelihood and score discrepancies with reference
MCSE and finite-particle ladder shifts. A resolved discrepancy requires the gap
to exceed the 1% owner materiality scale max(1,abs(reference component)), plus
three MCSE and the latest particle-ladder shift. Nominal 95% intervals are
descriptive Monte Carlo intervals, not bounds on finite-particle bias. As an
early precision target, seek three MCSE plus ladder shift below one quarter of
the corresponding before/after difference; failure triggers the next rung.
No score-percentage denominator can establish accuracy by itself.

Hard continuation vetoes: input/target/hash mismatch, broken score fixture,
nonfinite outputs, source drift, exhausted budget, incompatible GPU execution.
Low effective sample size, ancestor collapse, high likelihood dispersion or
unstable particle ladders veto reference certification but trigger larger
replication/reference repair within the budget; they do not reject the model.
If the bootstrap ladder cannot resolve the comparison, inspect and reuse the
existing independent trajectory-importance/quadratic route with matching inputs
as a second reference. Document a plan amendment before that additional method.

Heuristic sanity: this is a reference calculation, not a complex-method ranking.
Constructed comparators are the prior 40-step OT, the requested five-step OT,
and replicated plain bootstrap PF. Evaluate at this exact discrepant point;
do not pool with favorable nominal points or claim general dominance. Existing
fixed-state and Kalman fixtures guard the reference implementation itself.
This single-point check cannot certify either full model or HMC/posterior.

## Defaults, audit, budget, artifacts

Exact point and dataset come from the observed maximum; this is diagnostic
selection, not untouched confirmatory data. No retuning or default promotion.
The bootstrap method is an existing tested reference; genealogy and finite-N
bias are risks, checked using particle ESS, ancestor counts and N/seed ladders.
FP64 avoids adding reference rounding error; scope matching still uses the
verbatim saved observations. Three-MCSE and 1% thresholds are diagnostic screens,
not calibrated posterior-impact guarantees. Reference comparisons across the
ladder use fresh randomizations; any ranking remains local and conditional.

Skeptical audit: wrong-dataset and coordinate risks explicitly eliminated;
saved finite-program baseline is not an oracle; uncertainty/particle bias kept
separate; source/initial-law/observation-timing checks required. Runs cannot
support a conclusion merely by finishing. Main pre-mortem is a precise but
biased genealogical score; particle-ladder shifts and alternative reference
are the repair triggers. Audit passes for this bounded diagnostic.

Total diagnostic cap: 4 worker-wall GPU hours and at most 6 launches, including
repairs, inside the existing 15.2207-hour/16-attempt remaining campaign budget.
Serial GPU work only. Existing campaign budget.json is updated on each launch
and exit; reserve runtime before launch and use subprocess timeouts. Continue
the ladder automatically while it is required and resources remain.

Versioned root:
docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/sir-fresh-reference-01/
Store exact inputs, input/source hashes, command, git revision/dirty state,
environment, device/memory policy, seeds, timings, per-replication diagnostics,
summary, decision and inference-status tables, limitations and next action.
No previous evidence is overwritten.

Executor: scripts/run_sir_fresh_reference.py prepare|execute|report.
Reference workers call docs/benchmarks/run_nonlinear_bootstrap_reference.py.
CPU fixtures explicitly set CUDA_VISIBLE_DEVICES=-1 before TensorFlow import.
GPU launch command uses the approved environment Python and trusted permissions.

## Variance repair: independent trajectory importance reference

The first bootstrap rung has score MCSE (15.75,4.91,0.38), despite likelihood
MCSE 0.050. Genealogical score variance is the observed repair trigger.
Complete the running bootstrap ladder, and independently implement this reference
while it runs. GPU work remains serial within the four-hour/six-launch cap.

Freeze first-order SIR transition expansions about EKF filtering means:
f_tilde(x_t|xprev)=N(F_t xprev+b_t,Q). Preserve the exact initial N(m0,I),
observation model N(Hx,R), Q and normalization constants. The resulting linear
Gaussian model has likelihood Z_tilde and smoothing distribution q(x_0:T|y).
Sample its joint trajectories by backward Gaussian conditionals. Define
w(x)=product_t f(x_t|xprev)/f_tilde(x_t|xprev). Substituting
q=mu(x0)*product(f_tilde*g)/Z_tilde into the original likelihood integral gives
Z=Z_tilde E_q[w]. Differentiating the original integral with states held fixed
gives grad log Z=E_q[w S(x)]/E_q[w], where S is the sum of exact transition and
observation partial scores. Differentiation/integration regularity is assumed.
The Gaussian approximation affects only the proposal; its density is explicitly
corrected. No derivative of the proposal is used or required. Finite-sample
self-normalization bias remains. This is a local mathematical derivation.

One-pass linearization is a proposal hypothesis, not a likelihood approximation
or promotion criterion. It can miss posterior regions or produce heavy weights.
Inspect ESS, maximum weight, independent replications, sample-count stability,
and agreement with the bootstrap reference. No ridges/clipping. All covariance
factors must be finite and positive. Required tests: constant weights and exact
likelihood for a linear Gaussian model, plus nonlinear T=2 comparison against
independent numerical quadrature, including Fisher scores. Tests exercise the
same reference implementation used by the actual SIR endpoint.

Sample ladder: 8192 x8 seeds, 32768 x16, then 131072 x16 if needed. GPU FP64/XLA,
fixed signatures, memory growth, no pfor. At most six launches in total across
both reference methods, and four GPU hours. Useful precision requires at least
two trajectory rungs plus cross-method agreement. Preserve both methods if they
disagree; do not choose a favorable reference. No posterior/default promotion.

Skeptical audit: exact density cancellation, x0 timing, parameter coordinates,
Gaussian normalizers and backward conditional covariance checked algebraically.
Finite-sample bias and unobserved importance-weight tails remain limitations.
Linear and nonlinear independent fixtures must pass before GPU execution.
