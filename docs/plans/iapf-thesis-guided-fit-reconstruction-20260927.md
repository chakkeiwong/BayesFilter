# Thesis-guided iAPF fitting reconstruction

Date: 2026-09-27. Status: diagnostic tranche COMPLETE; both scientific replications INCOMPLETE. No downstream fit qualified.
Master: iapf-kdm-replication-master-program-2026-09-24.md.
Source correction: artifacts/iapf-thesis-reconstruction-20260927-01/source-recovery.md.
Execution: scripts/run_replication_master.py run-next.

## Research intent and evidence contract

Question: can the source-described local BFGS approach produce numerically valid
finite fits on the four saved t98 failures, and can a surviving reconstruction
complete fresh backward sweeps? This is independent R/FP64 reference work.
It does not establish the original author's unrecorded settings or a TF runtime.

Comparator: saved profiled LM and profiled BFGS failures on exactly the same
clouds/targets. Mechanism: finite optimization paths depend on joint versus
profiled amplitude, starting point, objective scaling and stopping controls,
even when the mathematical objective is unchanged. Source: Guarniero thesis
Section 7.3, Eq.7.1 with zero explicit penalty for Part III.

Primary numerical criterion: finite positive covariance, finite objective and
a scale-independent first-order residual <=1e-6; a local Hessian must have no
negative eigenvalue below -1e-5 times max(1, largest absolute eigenvalue).
This supports an approximate finite local solution only. Loss, shape agreement,
optimizer success flags and runtime are explanatory, never filter-promotion
criteria. A vanished density, failed derivative/source parity, nonfinite data,
missing output or exhausted budget vetoes the affected continuation. Candidate
failure alone does not invalidate the method.

An unbounded arm must pass all four saved cases to enter two fresh d=5,T=100,
N=1000 backward sweeps, seeds 9274001 and 9274002. It must pass every fitted time
under both preserved floor hypotheses N^-1 and N^-2. Select at most two passing
arms by declared order (joint before profiled, log-quadratic before weighted
moments before cloud moments, default before tight, scaled before raw), without
claiming a statistical ranking. No adaptive pilot or quality claim in this
tranche: a new downstream contract is required after complete sweeps.

If unbounded arms fail, execute the source-suggested L-BFGS-B repair diagnostic
on the same four cases. Bounds are centered on the log-quadratic start, with
mean radii 2 or 4 initial marginal standard deviations and log-variance radius
log(4). These are explicitly local reconstruction hypotheses, not recovered
author constants. Require finite constrained solutions, projected-gradient
residual <=1e-6 and report every active boundary; they cannot close an
unconstrained local-optimum or paper-replication claim. Test no-fire agreement
on a well-conditioned exact-Gaussian control. A successful box diagnostic
nominates a separately declared downstream evaluation, not automatic promotion.

Heuristic adversary set for this fitting diagnostic: unchanged log-quadratic
start, target-weighted moment Gaussian, cloud-moment Gaussian; each is evaluated
on each of four saved cases. They expose optimization damage at negligible cost.
An exact representable Gaussian target is the numerical oracle. These are
fitting sanity checks only, not substitutes for the downstream filtering
heuristics and Kalman-relative likelihood comparison required by the master.
No learned-method quality or stochastic superiority will be inferred here.

## Phases and defaults audit

1. Recover and reconcile sources. COMPLETE: thesis downloaded, relevant
   sections read; final published wording indexed, full final PDF unavailable.
2. Verify independent finite-difference derivatives and profiled/joint
   objective equality. Continue only after all controls pass.
3. Cross three starts (log-quadratic, target-weighted moments, cloud moments),
   joint/profiled lambda, raw/fixed-scaled sum of squared errors, and
   R-default/tight stopping. Four saved cases yield 96 fits. Both modes use
   log-variance coordinates to enforce positivity; joint lambda is linear.
   Controls: default maxit=100,reltol=sqrt(machine epsilon); tight
   maxit=800,reltol=1e-14. These are explicit R-default/diagnostic hypotheses,
   not claimed author settings. Record source status and all settings.
4. For any qualifying unbounded arms, at most two fresh complete-sweep
   reconstructions as above. Otherwise execute the bounded diagnostic repair
   and classify its result. No repeat of a failed unchanged damping recipe.
5. Validate artifacts, issue numerical/source/scientific decisions separately,
   refresh master and concise checkpoint with exact charges and next action.

Initialization hypotheses are generic inexpensive approximations; weighted
moments can overconcentrate on rare support, cloud moments can miss the target,
and log regression can be nonconcave. Reject invalid starts and retain reasons.
Raw SSE is the literal displayed objective; fixed positive scaling preserves
its minimizers but changes a finite BFGS path. Linear lambda is allowed by the
thesis; positivity is checked at a candidate solution. Profiling is exact
algebraically and still an optimization-path hypothesis.
Log-variance coordinates are a domain-preserving local choice with the original
coordinate choice unknown. We are not searching all parameterizations.
R-default stopping can report success on small loss without stationarity;
independent gradients/curvature prevent its promotion. Tight stopping may
follow the collapse longer. Both outcomes answer this diagnostic question.
The retained floor policies are unresolved hypotheses, never a recovered formula.
Explicit bounds are Class C: record changed optimization domain, active walls
and no-fire control; do not tune bounds using downstream error.

## Budget, environment, artifacts and stop conditions

Supplemental cap: 600 CPU wall seconds total, at most three versioned attempts,
zero GPU seconds. Existing combined CPU charge is199.100153/1800, so this is
inside the1600.899847 remaining seconds; GPU balance remains about36.567hours.
At most240seconds are available for calibration/box work; the driver enforces
the overall600seconds including any fresh sweeps. Infrastructure repair may
retry under the same cumulative cap. Scientific failure is a completed result.
Use /usr/bin/Rscript --vanilla; CUDA_VISIBLE_DEVICES=-1; BLAS/OMP threads=1.
Python controller uses stdlib only and records git, frozen sources/input hashes,
R version, seeds, wall time, exit status and output hashes. Unique output root:
docs/plans/artifacts/iapf-thesis-reconstruction-20260927-01/attemptNN.
Source copies and result evidence must be preserved; no overwritten attempts.

Premortem: optimizer success or tiny loss could disguise density collapse; a
finite fit could be useless downstream; a successful invented setting could be
mistaken for source replication. Independent stationarity, source labeling and
explicit nonpromotion address these. Fresh sweeps are prerequisite evidence,
not filter-quality evidence. Full paper repetitions and conditional heuristic/
uncertainty checks remain open regardless of this outcome.

## Skeptical audit before execution

PASS after revising the proposed baseline using the newly recovered thesis.
The final article's mention of regularization does not justify inventing a
penalty for the linear-Gaussian experiment: Part III explicitly used l=0.
The old profiled/tight BFGS probe does not cover joint/default BFGS. The plan
keeps that distinction testable, uses the actual saved failures, excludes
unfair target swaps, retains source uncertainty, enforces explicit CPU limits
and forbids fitting proxies from establishing filter quality. No GPU or
environment change is needed. Review performed by Codex; no independent
reviewer was launched.


## Solver stopping repair, reviewed before execution

Attempt01 completed in 9.431763 CPU wall seconds: all 11 controls passed, but
none of 96 unbounded fits or eight bounded fits passed the independent
stationarity check. No fresh backward sweep was eligible. An optimizer return
code of zero was insufficient: the eight bounded projected-gradient residuals
were 0.0376 to 6.82, above 1e-6. This is a solver-repair trigger, not evidence
that the compact boxes contain no constrained solution.

Execute one bounded repair on those same eight endpoints. Preserve the data,
profiled Eq.15 objective, initial coordinate units, box walls and 1e-6 criterion.
At each restart, freeze a new positive objective multiplier equal to the
inverse current density squared norm; do not differentiate that multiplier or
replace SSE by shape loss. Use R L-BFGS-B with factr=0, pgtol=1e-9, maxit=300,
at most 16 segments per endpoint. These are solver accuracy choices, not
author settings. They test whether absolute stopping tolerances caused the
observed false successes. The fixed positive multiplier leaves the minimizers
and box unchanged, though it changes the optimizer trajectory.

Before the repair, check fixed-scale objective/gradient parity, finite
differences, and the exact-Gaussian control with no active wall. Log each
segment's independent projected gradient, active walls, shape residual,
original log-SSE and density norm; reject a nonfinite endpoint, escaped wall,
or material increase in the original objective. Stop an endpoint at a passing
projected gradient or its segment cap. Box stationarity is only a first-order
necessary condition and does not certify a constrained minimum or useful
filter. Shape and the three initial heuristics remain explanatory.

This repair stays within the existing cumulative 600-second, three-attempt
tranche and 1,800-second combined CPU budget; it uses no GPU. Keep attempt01
immutable and write attempt02. A completed repair closes this diagnostic
tranche; it cannot automatically launch adaptive filtering or promote a box.

Skeptical audit PASS: the stopping/scaling hypothesis is isolated while the
objective and feasible sets stay fixed. No failed gradient criterion is
relaxed, no holdout is reused for promotion, and numerical completion cannot
be mistaken for source replication. Unknown author initialization and floor
remain explicit. The known-good control checks non-harm at a representable
optimum; it does not assert equality of every optimizer trajectory.

## Completed outcome

Both planned phases executed. Calibration: 96 unbounded fits, zero stationary;
8 original box fits, zero projected stationary. Solver repair: 8/8 projected
stationary, but all collapsed to box boundaries with worse shape than the
log-quadratic starts. All 27 controls passed. No fresh sweep or adaptive
filter was eligible. Result and terminal review:
[completed result](artifacts/iapf-thesis-reconstruction-20260927-01/result.md).
Two launches consumed 9.996893 CPU wall seconds and zero GPU; combined CPU
209.097046/1800 seconds. This tranche is closed. Its unused attempt and time
allowances do not authorize undeclared scientific alternatives.
