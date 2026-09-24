# Proposed tests of unspecified iAPF numerical choices

Status: COMPLETE after the authorized execution, skeptical self-review and a
localized diagnostic-guard repair. See the
[terminal review](artifacts/iapf-r-author-choice-hypotheses-20260922-01/terminal-review.md)
and [revised report](artifacts/iapf-r-author-choice-hypotheses-20260922-01/runs/report-v3/results/result.md).
No candidate met the frozen all-dataset condition for untouched validation.
The executed source snapshots retain this plan's pre-run form.

## Question and source boundary

Can a small number of plausible, disclosed numerical choices explain the gap
between the independent R reconstruction and the first linear-Gaussian study?
The immediate aim is a useful reconstruction with the right mathematical
target, followed by comparison with the published accuracy, particle counts,
and resampling counts. Matching a table cannot identify the authors' code.

Source: the inspected local Guarniero--Johansen--Lee TeX,
`artifacts/iapf-r-source-reconciliation-20260920-01/sources/arxiv-extracted/iapf_arxiv.tex`,
lines 515--547 (Algorithm4), 655--704 (Equations15--16 and implementation),
and 742--822 (first study). The optimizer, initial values, positive-floor
formula and precise finite-sample `sd` convention remain unspecified in this
source. Author identity of the public R comparator remains unverified.

Keep T=100, alpha=.42, A_ij=alpha^(|i-j|+1), initial/process/observation
covariances I, diagonal Gaussian plus constant guides, N0=1000, k=5,
tau=kappa=.5, and the fresh final APF. Use the printed six-estimate window;
delayed doubling remains the disclosed local convention. Preserve iteration20
and particle16000 caps as resource/candidate limits, not author settings.

## What the new campaign must not repeat

The previous amendment tried joint and analytically profiled Equation15 with
QR and target-weighted-moment starts, including maxit200/600 retries. None of
32 full-filter attempts completed. Earlier work already tested relative-L2
optimization, local box widths .5/1/2, chi-square floor powers 1/2/3/4/8,
early/delayed doubling, and a five-estimate stopping extension. Coverage differs
between these earlier tests; they are not all full-study comparisons.

The cause motivating the next hypotheses is structural. With p_i the fitted
Gaussian density and b_i the backward target, profiling Equation15 gives

    L = ||p||^2 - (p'b)^2 / ||b||^2.

Moving the Gaussian away from the particles or letting its variance grow can
send p and L to zero without recovering the target's shape. Changing solvers
or running longer cannot remove that escape. A finite local optimizer could
nevertheless return a useful guide; that is a testable procedural hypothesis.
See the completed amendment's `mathematical-findings.md` for the derivation.

## Hypotheses, in priority order

| ID | Exact proposed comparison | Rationale and classification | Earliest falsification |
|---|---|---|---|
| F1 | Profiled Equation15 with L-BFGS-B, QR start, fixed loss/parameter scaling; compare factr=10 with factr=1e7. Hold pgtol=1e-8 and maxit=200 fixed. | Test whether finite numerical termination preserves a useful local fit before amplitude escape. Earlier premature stopping is known; its downstream effect is the question. Same printed residual, unverified finite optimization procedure. | Exact diagonal-Gaussian no-fire check, objective/gradient checks, accepted termination reason, relative shape residual and heldout error. A loose stop is not evidence of a minimizer. |
| F2 | Repeat F1 using the previous outer iteration's Gaussian at the same time as the start; use QR when the previous guide is constant. | Test reuse of a learned guide, instead of restarting every backward fit. Plausible unspecified initialization; no new target. | Trace actual controller-to-backward-fit wiring; record start identity, parameter movement, solver status and any escape. |
| F3 | Weighted log-quadratic fits with w_i proportional to b_i and b_i^2; ordinary unweighted QR is the comparator. | Explicit alternative objective, not Equation15. Near a good fit, its density residual is approximately lambda^2 b_i^2 times squared log residual. Exponent1 is a milder weighting hypothesis; exponent2 is the local least-squares approximation. | Effective weight count, weighted-design rank/conditioning, positive fitted variances, heldout errors. Reject rank failure/nonconcavity; do not clip to a Gaussian silently. |
| P1 | Existing chi-square-tail floor8 versus c = p(m)/N^a for a=2,4,8, where p(m) is the fitted Gaussian peak. | Equation16 specifies positive c(N,m,Sigma) but gives no recovered formula. These have the right density units and a simple N dependence. The exponents are sensitivity hypotheses, not calibrated defaults. | Actual transition-mixture probability c/(integral(f p)+c), by time and quantiles; check finite values and whether the floor makes the guide nearly constant. |
| C1 | Six-estimate stopping with sample SD (denominator5) versus population SD (denominator6). Keep all other controller operations fixed. | A finite-sample convention for the unspecified `sd`, not a shorter history. Population CV is sqrt(5/6), about .913, times sample CV on the same six values. | Replay saved histories to locate changed decisions; actual alternate-controller runs are required to measure later particle counts and likelihoods. |
| D1 | Independent, preassigned observation datasets; hold each dataset fixed across filter replicates and methods. | The paper's realized observations are missing. Between-dataset variation could explain part of the table discrepancy. This is robustness assessment, not seed hunting. | Preserve every dataset and outcome. Never select a dataset because its numbers resemble the paper. |

F1's strict arm is a control, not a claimed new solution. Do not silently accept
an iteration-limit/error status. Record a solver's successful numerical stop
separately from actual stationarity and from filter validity. If deliberately
truncated optimization is later desired, specify it as a new finite-step arm
before execution rather than relabeling a failed optimizer.

For F3, fit an intercept, linear terms and diagonal quadratic terms to log b.
The intercept absorbs the unknown scale. The density-loss Taylor expansion is
local; it does not establish equivalence of the two global minimizations.
Tiny effective weight counts are an explanation and warning, not by themselves
a theorem that a returned Gaussian or filter is invalid.

For P1, the floor must affect every backward target, transition integral,
proposal and weight consistently. A frozen-guide floor-only replay is a cheap
mechanism diagnostic; it cannot substitute for rerunning the full learning
controller with the new floor.

## Sequential design and evidence contract

1. Implement the named arms as explicit independent-reference options and
   verify their consumer call chains. Use exact Gaussian controls, analytic
   derivatives where applicable, weight-scale invariance, and exact floor
   mixture identities. Preserve all known failure fixtures.
2. Screen F1--F3 on fresh d5/d20/d80 clouds, including terminal time100,
   time99 and time50. Use separate predictive/smoothing evaluation draws.
   Absolute training loss is explanatory only; no choice can qualify solely
   by reducing it. Keep floor8 and the six-estimate controller fixed.
3. Run short full-filter probes for surviving fit hypotheses at d5/d20 on
   two new datasets and four filter replicates per dataset. Carry at most two
   fits in deterministic order F1-loose, F2-loose, F2-strict, F3-exponent1,
   F3-exponent2, after validity/completion screens. This order prefers the
   printed objective and simpler choices; it is not a performance ranking.
   QR remains a comparator even if every new fit fails.
4. Test P1 on those fits, or QR alone if no fit survives. Then test C1 on
   frozen fit/floor combinations. Use calibration datasets only and disclose
   interactions; do not run a large fit-by-floor-by-controller grid. Carry at
   most two combinations that pass the calibration validity and practical
   screens. Prefer floor8 then peak exponents8/4/2, and sample SD before
   population SD, as a fixed simplicity order rather than table matching.
5. Freeze every selected choice before untouched validation: d5/d20/d80,
   two datasets per dimension, sixteen replicates per dataset and method.
   Run QR, BPF(N10000), fully adapted APF(N5000), and SIS(N10000) on the same
   observations with the exact Kalman likelihood as authority. Evaluate every
   dataset separately. If budget cannot finish this stage, retain the partial
   evidence without promoting it. A later all-five-dimension/1000-repeat
   expansion depends on this screen; it is not included automatically.

Use disjoint dataset seed families 91100000+100*j+d for cloud diagnostics,
92100000+100*j+d for calibration, and 93100000+100*j+d for validation (j=1,2).
Check for prior use before launch; if a collision exists, document and freeze a
replacement family before drawing data. Record method-specific filter seed
streams and all observations. Do not regenerate data between filter replicates.

Primary endpoint is terminal Zhat/Z against Kalman. The bounded practical
screen retains the existing first-study reference requirements: bootstrap95%
mean interval contained in [.8,1.2], SD upper bound at most twice the published
SD for the dimension, and mean final N at most1.5 times the published mean.
These are project tolerances for a useful reference, not paper claims or proof
of replication. Wide intervals mean insufficient evidence, not proven bias.

Constructed heuristic comparators have distinct roles: BPF tests whether
learning adds value over prior propagation; fully adapted APF tests whether
future-guide learning adds value over exact one-step adaptation; SIS tests the
value of resampling; Kalman supplies the exact likelihood. Their particle
budgets follow the comparison design, so no matched-cost claim is available.
Observed conditional losses veto promotion under the existing conservative
project screen, but do not establish statistical inferiority.

Report terminal mean/SD/MSE, final N, learning iterations, resampling count and
algorithm time by dimension and dataset. Report prefix errors by the existing
ordinary/large-innovation strata as an additional filtering-use assessment.
The existing prefix veto remains binding for general filtering promotion;
terminal paper reconstruction and prefix filtering adequacy get separate
verdicts. Exact-guide prefix variance is not a reason to redefine the paper's
terminal endpoint or to discard inconvenient prefix errors.

Within-dataset paired uncertainty and explicitly limited between-dataset
summaries accompany comparisons; two datasets cannot establish population-wide
robustness. Tables1--2 discrepancies are descriptive, never a tuning loss.
No authorship identity, exact paper replication, superiority, later-study,
LEDH/KDM, GPU, HMC or default-readiness conclusion follows from this screen.

## Budget, artifacts and repairs

Proposed ceiling:7200 aggregate worker seconds, at most100 numerical launches
and two CPU R workers. Reserve1200 for fit diagnostics,3600 for calibration,
1800 for untouched validation and600 for checks/reporting. Unspent stage time
may move forward, without exceeding the total. Charge unsuccessful attempts.
This fits within the last recorded65410.204 remaining campaign worker seconds;
recheck the live ledger before execution. The original deadline remains
2026-09-21T20:04:26Z. This proposal does not renew it or promise all validation
cells will fit before it. No automatic expensive paper-scale expansion.

Use Rscript --vanilla with CUDA_VISIBLE_DEVICES=-1,
OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. This is the explicitly permitted
independent CPU R reference. Extend the existing runners only after the named
options and tests exist. Save immutable execution sources, exact commands,
environment, Git/diff provenance, seeds, checksums, status and measured time in
a fresh `artifacts/iapf-r-author-choice-hypotheses-20260922-01` root (allocate a
new suffix if it exists). Do not write into completed experiment directories.

Localized harness failures trigger repair, focused regression and a fresh
attempt within budget. A candidate failure advances to the next hypothesis.
Invalid invariants, corrupt evidence, a scientific-contract change, deadline or
budget exhaustion are continuation vetoes. Underpowered outcomes are recorded
as unresolved. No routine phase-by-phase human approval is introduced.

## Skeptical review

PASS as a proposed bounded design. Revisions incorporated during review:
relative-L2, arbitrary box sweeps and more iterations are already explored and
are not presented as new repairs; warm starts require actual controller wiring;
weighted log fitting is explicitly a changed objective; floor replays must be
followed by full learning runs; SD convention replays cannot predict changed
trajectories; published table entries are not optimization targets; all
controls stay fixed while each hypothesis is isolated. Paper and project
prefix questions remain distinct without changing prior results.

Material risks remain rare likelihood tails, a numerical stop mistaken for a
valid minimizer, unstable weighted designs, floor/fit interactions, and limited
dataset coverage. The early diagnostics above address these risks; none is
silently promoted to a correctness certificate. A successful process exit or
lower fitting loss cannot nominate a failed filter.

## Pre-execution audit and operational clarification

Reviewed 2026-09-21 16:31 UTC against branch surrogate-hmc at
6fbcf3147660c40d5d5644bbcbcc9fadbcb06aef. The original deadline is still in the future;
7200 worker seconds remain within the existing campaign balance. No approval
or source-recovery dependency blocks these explicitly labeled hypotheses.

PASS after the following corrections, before implementation or numerical runs:

- F1/F2 share QR-derived coordinate and fixed loss scales, even when their
  start differs. Floating-domain log-variance bounds are +/-(-log(double eps));
  touching a bound rejects the fit. These bounds protect representability and
  are not a scientific covariance regularizer. Exact Gaussian starts may stop
  at relative residual<=1e-26, the existing roundoff check.
- All six fit options receive full-filter probes if their algebra/wiring tests
  pass. A fixed-cloud fit failure is explanatory and does not prove failure on
  the controller's different training clouds. Four replicas are attempted per
  dataset; failures remain visible. F2 may fail before its first warm start,
  in which case initialization beyond that point has not been evaluated.
- Carry one new fit (the first fully complete option in the stated order), or
  QR if none completes, into the floor sweep. This is within the proposed
  maximum of two and keeps total launches below100. Floor8 is the comparator;
  peak2/4/8 are all tested at d5/d20/d80, two calibration datasets and four
  repeats each. Reuse only identical completed calibration controls.
- Test population SD on the first two practically eligible fit/floor options;
  if none qualifies, perform this diagnostic on the baseline floor8 option.
  This fallback is not validation admission. Select at most two combinations
  for untouched validation using complete practical screens on every tested
  calibration dataset and the fixed order; do not rank on published resemblance.
- Gaussian-limit tail margins are a conservative promotion veto, not a proof
  of infinite variance of the actual positive-floor algorithm. Prefix heuristic
  losses veto general-filtering promotion while allowing the paper-terminal
  hypothesis tests to continue. Preserve both separate verdicts.
- Use2000 deterministic bootstrap resamples per dataset for mean/SD and paired
  differences. Four-replica calibration intervals are only screens; terminal
  validation and rare-tail uncertainty remain necessary. No superiority claim.
- Record floor-mixture quantiles from the proposal actually used at each time.
  These are passive observations and must not alter sampling or RNG streams.
- Each worker is limited to180 seconds, or300 for a fixed-cloud/validation job,
  further reduced by its stage/budget/deadline balance. Candidate numerical
  failure advances the plan; a timeout preserves partial evidence and permits
  a bounded infrastructure/resource repair without changing numerical settings.

Expected maximum before repairs:87 numerical/test launches, including6 cloud
jobs,24 fit-probe jobs,6 calibration comparator jobs,20 floor jobs,12 SD jobs,
18 validation jobs and1 focused-test job. Branches may require fewer. Report
jobs and repairs share the remaining13 launches and the same7200-second cap.

Next concrete action: implement and test F1--F3, floors, and SD options with
actual controller wiring, then execute the sequential campaign.

## Terminal review and repair record

Completed 2026-09-21 17:17 UTC (2026-09-22 Hong Kong). The final self-review
found one misplaced Equation15 diagnostic veto in the weighted-log route.
The [bounded repair note](artifacts/iapf-r-author-choice-hypotheses-20260922-01/post-run-diagnostic-plan.md)
records the defect, saved-fixture regression, identical-seed replay and fresh
report. That replay completes; all-dimension selection remains empty.
The report distinguishes 288 final records from eight resource-censored scheduled
replications and preserves the original failure. Used 2188.624 worker seconds
in 69 launches. The program ended on its scientific decision, not on budget
or a procedural permission stop.

The F1/F2 log-variance bounds are a disclosed numerical-domain hypothesis,
|log(v)| <= -log(double epsilon), not the full representable exponential range
and not a recovered author setting. A boundary rejection rejects that configured
candidate; it does not establish failure of every Equation15 implementation.
