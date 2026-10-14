# Controlled-SMC replication: author-relative acceptance reset

Date: 11 October 2026. This is the active controlled-SMC acceptance policy for
the [master program](controlled-smc-kdm-replication-master-program-20260928.md).
**Planning-error determination:** the owner-approved 6 October rule was wrong
planning for controlled SMC. It imposed uniform 1% mean-likelihood and 5%
`Var(log Z)` pass/fail margins before establishing that the released source's
tails, moments, and feasible sample size could resolve them. The owner withdrew
those margins; this does not mean the frozen Q2 arithmetic was wrong. The
[6 October criteria](controlled-smc-kdm-acceptance-criteria-20261006.md) remain
historical for controlled SMC; their KDM NLL and RMSE criteria are unchanged.
This policy does not alter the author algorithm, source-intrinsic-failure rule,
candidate code, or existing experimental records.

## Question and evidence contract

The implementation question is whether the TensorFlow port reproduces the
released author's controlled-SMC method and adds no failures on author-valid
inputs. The separate stochastic question is whether its downstream behavior
is no worse than the author's at a defensible, testable resolution. The
comparator is the pinned author code in the recorded Octave/LAPACK runtime,
with the same observations,
parameters, particle count, policy stages, stopping rule and random-input law.
The candidate is the existing centered-QR, float64, GPU/XLA port. The 40 Lorenz
forcing-grid cells and generating-parameter cell remain distinct; no pooling
may hide a failing cell. Paper Figure 6 reports sample variability from 100
independent runs; it does not prescribe a 1% or 5% translation tolerance
([published paper, Section 6.2](../../.localresources/papers/heng2020-controlled-smc-published.txt)).
The source fits a quadratic with normal equations and applies adaptive
refinements in `cSMC_Resample.m:185-193,239-264`; these operations and actual
downstream consumers remain the implementation target.

The pass requirements independent of stochastic margins are source-method and
consumer fidelity, valid fixed-input values and invariants with
conditioning-aware bounds, all predeclared source controls, and **no additional
invalid port outcome on an author-valid matched input**. Record every source
failure, port failure and denominator. Report the one-sided 95% upper bound on
an extra-failure rate under its stated independent-trial model; observed zero
is not a population proof of zero. An intrinsic author failure is accepted as
a baseline outcome, not a port defect or a reason to repair the author's code.

For continuous outcomes, retire `abs(log(E Z_port/E Z_author)) <= log(1.01)`
and `Var(log Z_port)/Var(log Z_author) <= 1.05` as pass/fail conditions. Retain
the likelihood-location estimate, log-likelihood variance, paper variance
statistic, influence concentration, tail diagnostics and their intervals as
**explanatory** results. A wide interval is inconclusive, not an equivalence
pass; a favorable point estimate is not evidence of no-worse performance.
There is no substitute fixed percentage or unreported significance test.

The correction is a **two-part verdict**. Source-method and consumer fidelity,
fixed-input numerical checks, source controls, and the added-failure rule can
support a bounded implementation-replication finding on tested inputs. A
separate stochastic no-worse finding needs estimable downstream quantities and
a predeclared, scientifically defensible degradation bound with enough
precision to exclude material harm. Neither a non-significant difference nor a
wide source-self-repeatability envelope supplies that bound automatically.

First use independent author-only runs at matched settings and cost to assess
the tails, moments, and source-self differences for candidate likelihood and
dispersion summaries. This is a **feasibility and noise-floor study**, not a
new acceptance threshold. Before seeing fresh port qualification outcomes,
freeze the selected summaries, any independently justified degradation bounds,
sample sizes, cellwise and simultaneous-inference method, decision code, and
cost/precision stop rule in a versioned plan. An author-repeatability envelope
may become an acceptance bound only with a separate argument that its implied
maximum degradation is substantively acceptable; otherwise it remains a
diagnostic. Use fresh author/port cases for the resulting comparison. Mean
likelihood and variance may be decision summaries only where their source-only
tail and moment checks pass. Predeclared robust distributional summaries can
answer a narrower distributional question; they must not be silently renamed
mean-likelihood or variance replication.

For a stochastic no-worse claim, every predeclared required comparison must
meet its justified bound with valid uncertainty accounting; the fixed-input and
added-failure vetoes must also pass. A supported bound violation triggers
repair. Invalid source controls, corrupted artifacts, unestimable required
summaries, or a budget that cannot resolve the bound stop that *claim*, not the
whole replication investigation. If no defensible bound or feasible precision
exists, report the implementation/failure evidence and leave stochastic
no-worse **unresolved**. Solver coefficients, run time, trace count and
exploratory residuals explain discrepancies but do not replace downstream
evidence. Any pass is restricted to the tested settings and resolution; it
does not establish equality for all inputs, exact likelihood accuracy, or
superiority to APF/iAPF.

## What the completed Q2 run establishes

The [frozen Q2 summary](artifacts/controlled-smc-kdm-replication-20260928/csmc-qualification-20261009-final-013640/summary.json)
has 13,120 paired trials in 41 cells, 14 intrinsic source failures, five port
rejections on source-failed inputs, and zero additional port failures on
source-valid inputs. The source `Var(log Z)` spans approximately `4.18e-5` to
`4057` across cells. Source residual effective counts fall to `1.45`; 18 cells
fail the run's likelihood-interval adequacy check and 34 fail its variance
adequacy check. The old **NOT_QUALIFIED** result and 5/41 pass count are
correctly computed historical decisions *under the wrong planning rule*.
They do not establish a controlled-SMC port failure under the active rule or
a retroactive pass. Q2 port outcomes have already been inspected, so Q2 is
development evidence for designing a new comparison, never untouched
confirmation of a newly selected bound.

## Defaults and assumptions to audit before launch

| Choice and status | Provenance and reason | Failure mode and earliest check |
|---|---|---|
| Pinned Octave/LAPACK author source; baseline | Q2 source snapshot preserves released code and intrinsic failures | This is not the author's original MATLAB runtime; exact source-control replay and runtime provenance must pass |
| Centered-QR float64 GPU/XLA port; candidate | Frozen Q2 candidate, algebraically solving the source fit on regular systems | Ill-conditioned solver behavior can differ; check fixed inputs and downstream outcomes |
| 40 forcing cells plus generating point; tested scope | Paper Figure 6 and Q2 design | Pooling can hide harm; preserve cellwise decisions and multiplicity treatment |
| Common source/port input tapes; diagnostic coupling | Q2 paired design reduces comparison noise | Divergent policies generate different clouds; verify each method follows its own sampling law |
| 95% uncertainty convention; reporting choice | Historical acceptance protocol | This is not a practical effect size; freeze simultaneous inference before qualification |
| Sample size, source noise, and degradation bound; unknown | Q2's 320 pairs per cell are cost/tail diagnostics, not acceptance calibration | Sparse extremes can make a summary unestimable; source-only feasibility, an independent bound rationale, and a precision/budget check come first |

## Next bounded phase and skeptical audit

First perform a source-only feasibility analysis, using Q2 source outcomes
for exploration only. Any inferential source calibration requires fresh
author-only runs, disjoint from both Q2 and future port qualification.
Predeclare a small, cost-bounded diagnostic before any long run; do not repeat
the 41-cell GPU campaign merely to tighten the wrong-planning margins. If a
scientifically justified and statistically resolvable no-worse decision is
feasible, freeze a new qualification plan and code before generating or
inspecting its port outcomes. Otherwise stop that decision and report the
narrower implementation-replication evidence. Each launch needs unique outputs, exact command/environment/seeds, CPU/GPU
status, memory policy, wall time, budget debit and stop conditions. The
historical Q2 supervisor and its inference module implement the withdrawn
margins and must not issue the new decision.

Skeptical audit: the old 1%/5% bounds were practical preferences, not paper
targets, and Q2 shows that many corresponding moment intervals are not
reliable. Replacing them with mere non-rejection would mistake low power for
agreement; treating a very wide author-self envelope as a permissive margin
would repeat the planning error. Using Q2 to choose and then certify a new
bound would reuse claim data. The source-only/fresh-port split, independent
bound rationale, and separate failure/fixed-input requirements address those
errors. Author-only calibration can still be too noisy or too costly; that
outcome leaves a narrower implementation-replication statement, not a
fabricated statistical pass. No new campaign is launched by this policy change.
