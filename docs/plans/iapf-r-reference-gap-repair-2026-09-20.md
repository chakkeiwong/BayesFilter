# Independent R iAPF gap repair

2026-09-20. The owner requests the remaining gaps, a reviewed repair plan, and
execution. This continues the first linear-Gaussian study. The previous
equation-(15) failures and every previous result remain preserved.

## Gaps and intended result

1. The profiled equation-(15) loss has a vanishing-density escape. Preserve it
   as `paper_eq15`; implement an explicitly different `relative_l2` fit. A
   successful alternative is an independent iAPF reference, not reproduction
   of the paper's fitting equation or reported implementation.
2. The positive floor, optimizer and earliest particle doubling are not fully
   specified by the paper. Expose their identities in results, retain the
   original choices, and measure sensitivity without selecting on test data.
3. An optimizer convergence code does not establish a good fit. Preserve
   relative residual, gradient, boundary and convergence diagnostics; reject
   a nonconverged alternative fit. Save the actual inputs of any failed fit.
4. No complete repeated fitted comparison exists. Execute fresh complete
   batches, retain failures, compare with Kalman, bootstrap, fully adapted APF
   and sequential importance sampling, and report conditional prefix errors.
5. Exact author settings/data/code and all five published 1000-repeat results
   remain unavailable. A reconstructed model and alternative fitter cannot
   close these source and replication gaps. No nonlinear/score/HMC claim.

## Research intent and evidence contract

Question: does removing the irrelevant density amplitude from Gaussian fitting
resolve the saved fitting failures and produce a reliable independently labeled
iAPF reference on the paper's first model?

For density vector p, positive target y, a=p'p, b=p'y and c=y'y, define
`R=1-b^2/(a*c)=min_lambda ||p-lambda*y||^2/||p||^2`. Optimize this relative
squared residual. It is invariant to separate positive scaling of p and y.
Its derivative in a parameter theta is
`2*b^2/(c*a^2)*sum(p^2*dlogp)-2*b/(c*a)*sum(p*y*dlogp)`.
The public comparator's criterion is `c*R/(1-R)` and has the same minimizers
where R<1; neither criterion equals equation (15). Proposition 1 still applies
to any fixed positive twists with the exact proposal/weight correction.

Primary repair criterion: all three saved failed fitting inputs terminate with
finite parameters, no variance boundary, convergence, and a relative residual
no worse than their initial fit, to numerical tolerance. This permits a pilot;
it does not establish likelihood accuracy. The empirical criterion is complete
32-repeat batches and a 95% bootstrap interval for mean Zhat/Z entirely inside
[0.9,1.1]. A statistical ranking additionally needs a paired variance-ratio
interval excluding 1. Failed/incomplete batches get no success-only interval.

Heuristic adversaries, constructed from this model: bootstrap PF (unmodified
transition, N=10000), fully adapted APF (exact one-step Gaussian update,
N=5000), SIS (same transition without resampling, N=10000), and exact Kalman
likelihood. Compare prefix log-error conditionally on ordinary vs large
Kalman innovations using the pre-existing chi-square 90% cutoff. Underperformance
against a heuristic vetoes promotion. It remains a diagnostic repair trigger,
not a reason to suppress later explanatory experiments.

Nonfinite arithmetic, failed identities, mismatched source/data, or invalid
importance corrections are continuation vetoes until repaired. Fit failure,
iteration caps and runtime caps reject the current candidate/batch; preserve
partial evidence and use only fresh data for a repaired repeated comparison.
Floor probability, optimizer counts, particle histories and runtimes explain
results, and are not tuning or promotion targets.

## Default and assumption audit

| Choice | Provenance and justification | Failure and earliest diagnostic | Status |
|---|---|---|---|
| Relative L2 objective | Derived above to identify Gaussian shape independently of amplitude | Flat shape or local minima remain; saved-input and exact-Gaussian checks | Explicit method extension |
| Diagonal covariance | Paper Section 5.1 | Correlation misspecification; exact Kalman and full-covariance oracle | Paper restriction |
| Log-quadratic initialization | Existing reconstruction; exact for a diagonal Gaussian target | Nonquadratic target; report fallback and initial residual | Hypothesis |
| L-BFGS-B, maxit 200, factr 10, pgtol 1e-8 | Existing local solver; analytic gradient, relative objective has natural units | False/nonconvergence; independent derivative and residual checks | Numerical reconstruction |
| Log variance bounds +/- -log(eps) | Existing numerical guard, not a scientific constraint | Scale dependence; reject rather than clip/accept a boundary | Fail-closed guard |
| Floor tail power 2 | Existing reconstruction, Gaussian chi-square contour | Too much/too little transition mixing; sensitivity powers 1,2,3 | Frozen hypothesis |
| Doubling at l>=k | First complete window in Algorithm 4 | Extra early doubling vs published counts; compare labeled l>k alternative | Ambiguous reconstruction |
| New observations, paper T/N/model | Original observations unavailable | Data-dependent SD; preserve data and exact Kalman | Reconstructed study |
| 32 repeats and 2000 bootstrap draws | Previous bounded plan | Weak variance/tail inference; intervals, no published-scale claim | Bounded screen |

## Execution

1. Add explicit fit and doubling modes without changing the default equation
   (15) path. Add value/derivative/scale tests and consumer wiring tests.
2. Replay the three saved failures from the previous mechanics02 directory.
   Check exact-diagonal targets, diffuse directions and failed convergence.
3. On calibration data only, run T100 d5/d10 pilots and a sensitivity set:
   baseline (floor2, earliest window), floor1, floor3 and delayed doubling.
   Report all settings; retain floor2/earliest-window for the untouched batches.
4. Freeze sources, run fresh 32-repeat d5 and d10 batches if pilots are viable.
   If resources permit, run one fitted pilot at d20/40/80. Scaling to all 1000
   repeats is conditional on complete reliable bounded results and measured
   feasibility, never on a successful subset or exact-twist controls alone.
5. Review results, update the implementation note, master and active checkpoint.

Explicit CPU reference: Rscript --vanilla, CUDA_VISIBLE_DEVICES=-1,
OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1. Python driver uses the existing
/home/chakwong/anaconda3/envs/tftwogpu/bin/python. No installs, GPU, author
messaging, production defaults or canonical LEDH edits.

New repair output root:
`docs/plans/artifacts/iapf-r-reference-gap-repair-20260920-01/`.
Budget (amended within existing authorization below): at most 1550 summed
experimental worker-seconds, at most 10 launches,
at most two simultaneous workers, at most 400 seconds per worker; focused
mechanics limited to 120 seconds. This uses less than the previous allocation's
remaining worker time; the owner-requested repair replaces its one remaining
launch slot with this bounded multi-stage schedule. Charge every attempt.
Each launch records command, HEAD, source snapshots/hashes, environment, CPU
choice, seeds, data hash, elapsed time, method identity, plan and result path.

## Skeptical review before execution

Reviewed against the source PDF's Section 5.1 equations (15)--(16), Algorithms
3--5, Section 5.2 and the actual R call chain. The original proposed action
"repair fitting and replicate" would be misleading without an objective label;
this plan repairs that flaw. It does not treat relative residual as evidence of
likelihood accuracy or success-only samples as unbiased comparisons. Kalman,
three heuristics, source/data identity and incomplete-status handling answer
the main baseline and validity risks. Floor/controller sensitivity is separate
from frozen heldout comparisons. No unexamined numerical setting is promoted.
Verdict: proceed with the explicitly labeled alternative and retain the strict
paper-replication gap. This is a self-review, not an independent review.

## Calibration repair 1

All three old failures pass the relative fit (residuals .00177/.00238/.00241).
The first fresh sensitivity launch fails at the maxit=200 limit, with finite
variances and residuals 1.23e-5 (d5) and 9.37e-4 (d10) in its baseline arms.
This is a solver resource limit, not the old variance escape. Before any repeat
batch, replay these two saved inputs at maxit=1000, then 5000 only if needed.
Compare convergence, iterations/evaluations, gradient and parameter changes.
This budget calibration does not select for likelihood performance. Retain
failure rejection and the same objective/derivative/floor/controller. Freeze
the smallest tested solver allowance that completes both inputs before a fresh
calibration pilot and untouched comparison data. All work is within the existing
120-second mechanics and 1200-second experiment budgets. Self-review: the inputs
are calibration-only, and raising a finite cap changes an explicit numerical
reconstruction choice; it must be recorded in method settings and source hashes.

Replay result: maxit1000 completes d10 but caps d5; maxit5000 completes both.
d5 requires 1237 evaluations, final residual 1.1000e-5 and gradient 8.63e-9;
d10 requires 230 evaluations, residual 9.3663e-4. Freeze maxit5000 for the
alternative campaign, preserving maxit200 for the default equation-(15) route.
Continue fresh calibration data seed62000000+d and the planned unchanged floor
and controller. Saved evidence: mechanics02/solver-budget.csv.

## Calibration repair 2

The second sensitivity run completes all four d5 arms, but d10 still reaches
maxit5000 at backward time87 (relative residual2.736e-5, gradient1.30e-6,
finite log variances [-.783,1.012]). This reveals an additional inherited
optimizer default: L-BFGS-B stores only five curvature pairs even though the
fit has 2*d parameters. Test memory equal to parameter dimension (at least5)
on the three old and three fresh failed inputs, holding the relative objective,
initialization and maxit fixed. This costs O(N*d+d^2) storage in this reference
and targets convergence on a correlated optimization problem. Compare with
the saved five-pair outcomes; accept as a reconstruction only if all converge,
healthy exact-Gaussian fits remain identical, and residual/gradient checks pass.
It changes the solver approximation, not the specified objective. No likelihood
metric participates in this calibration. Self-review: this is a localized
numerical repair and proceeds within the remaining recorded budget.

All six memory-calibration cases pass. The d10 maxit5000 failure converges in
1585 evaluations with memory20, gradient5.71e-8 and residual2.72256e-5. The
previous d5 cap case falls from1237 to227 evaluations. Adopt memory=max(5,2*d)
only for relative_l2, keep the equation-(15) solver unchanged, and record both
memory and maxit in every fit. Fresh calibration seed64000000+d precedes the
untouched comparisons. This is a convergence repair, not a variance ranking.

## Timeout repair and completion budget

The first d10 half (16 untouched repetitions) completes in342.512 seconds.
The d5 worker hits its400-second timeout with30 complete four-method
comparisons and the iAPF member of repeat431 complete. Resume431:432 with
identical code, data, method settings and seeds; duplicate saved values must
agree exactly. A time limit is an infrastructure/resource interruption, not
outcome-based selection or a request for new fit settings. Run d10 repeats
417:432 unchanged; combine only if all32 completed.

The initial1200-second repair sub-allocation is too small to reserve both
continuations. Use1550 seconds of the previously authorized1551.617 remaining
worker-seconds instead. The previous stage used248.3825, so the combined ceiling
is1798.3825, below the original1800. No additional compute class, paid resource,
data, method or scientific criterion changes. Retain10 repair launches and
400 seconds per worker. This replaces the initial sub-allocation; it is not a
new1550 seconds on top of it. Skeptical review: complete the prespecified samples
without tuning or dropping slow/failed seeds; source/data checks prevent mixing.

## Dimension20 diagnostic after the repair

The optional d20 pilot fails at backward time92 after5000 iterations, despite
relative residual7.30e-8 and finite log variances[-.320,.556]. The squared-target
effective sample size is1.272/1000. Before interpreting this as a general method
failure, inspect the singular values of the normalized-residual Jacobian and
its independent finite-difference gradient at the saved input. As a bounded
solver diagnostic, compare base-R nlminb (same relative objective and bounds,
analytic gradient, iter.max5000,eval.max10000,rel.tol1e-10) on that exact input.
No retuning of ongoing d5/d10 heldout comparisons, no substitution into their
source, and no promotion from a small fitting residual. This is explanatory
mechanics under the120-second allowance. Any accepted new solver would need
the old healthy/saved-failure tests and fresh pilot/repeats. Self-review:
concentration and local conditioning diagnose identifiability; they do not
prove likelihood failure, a global optimum, or insufficient particles by
themselves. Preserve parameters and the full probe result.

## Same-objective solver repair after the d20 diagnostic

The independent Jacobian/finite-difference check agrees to 2.61e-10. Its local
singular-value ratio is 48,446, while the squared target weights have effective
size 1.272. On the saved input, base-R nlminb converges from the same original
initialization in 3,887 iterations, reducing the residual from 7.302e-8 to
6.657e-8 with maximum gradient 8.57e-10. This establishes a local conditioning
and solver issue, not global optimality or the adequacy of particle coverage.

Implement an explicitly selected `relative_l2_nlminb` mode, with the identical
relative objective, analytic gradient, variance bounds, initialization, floor,
and controller. Use iter.max=5000, eval.max=10000, rel.tol=1e-10 as in the
diagnostic; record solver identity. The existing strict and relative-L-BFGS
modes keep their behavior. There is no automatic fallback. First replay all
seven saved problematic fits and check exact diagonal-Gaussian recovery,
nonconvergence rejection, and the actual controller-to-fit call chain. Failed
qualification prevents a pilot. Fit residuals remain explanatory only.

If qualification passes, use fresh d20 observations (seed 67000020), repetition
1, with at most 200 worker-seconds. If it completes, the final launch may probe
d40 on fresh observations (seed 67000040), at most 75 seconds. If d20 fails,
inspect its preserved inputs before any last bounded retry. The existing
1,550-second/10-launch campaign ceilings still apply: 1,269.628 seconds and
eight launches have been consumed, leaving 280.372 seconds and two launches.
The 120-second mechanics allowance includes solver qualification and tests.

Skeptical self-review: this repair changes an explicit numerical solver, not the
fitting target, paper fidelity, or heldout d5/d10 evidence. New pilot sources
must remain separate from the completed comparisons. A successful d20/d40 pilot
would establish completion only; it cannot establish likelihood accuracy,
variance reduction, paper replication, or safe extrapolation to d80. Budget
exhaustion and failed validity checks remain stop conditions. Proceed within
these bounds.

The nlminb qualification passes all seven saved fits and the six additional R
solver/wiring checks, but fresh d20 attempt09 fails at backward time68. Its
relative residual is 3.006e-10, gradient maximum 1.786e-7 and squared-target ESS
1.409/1000. A small loss is not permission to ignore nonconvergence. Preserve
the solver's message, evaluations and iterations in future failure records.
Replay this new saved input with the unchanged nlminb objective at maxit5000
and 10000, then test 10000 on all seven prior cases only if the new input
converges. If this narrowly resolves the cap with finite non-boundary fits,
the last pilot may use maxit10000 on fresh data seed68000020, capped at250
worker-seconds (258.803 remain). This is calibration of a finite numerical
allowance, not selection on likelihood results. If it does not resolve the
saved failure, preserve the failure and close the bounded stage with optimizer
conditioning unresolved; no arbitrary relaxation of convergence is authorized.
No d40 launch follows a failed d20 pilot. Self-review: the explicit solver
allowance and fresh data prevent a successful replay from being reported as
generalization. There is one launch remaining; the established d5/d10 inference
does not change.

## Terminal state

The last saved-input test fails at both 5000 and10000 iterations. The conditional
last pilot is therefore not launched. This closes the bounded repair stage with
complete32-repeat comparisons at d5/d10 and unresolved d20 solver qualification;
it does not reject the iAPF research direction. Nine repair launches consumed
1291.196979/1550 worker-seconds, leaving258.803021 and one launch. Together with
the earlier248.382526 seconds, total experimental use is1539.579505 within the
original1800-second authorization. Mechanics are conservatively charged40/120
seconds. No process is running. See the terminal result and checkpoint under
artifacts/iapf-r-reference-gap-repair-20260920-01 for the decisions, inference
limits, remaining source/replication gaps and exact next question.
