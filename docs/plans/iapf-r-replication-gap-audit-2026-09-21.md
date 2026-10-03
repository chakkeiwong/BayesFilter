# iAPF replication: source audit and discriminating experiments

The owner's current request authorizes tracing the code and papers, reviewing
this plan, and executing it through localized repairs. This is the next bounded
stage of the independent R reference, not a change to BayesFilter's GPU backend.

## Research intent and evidence contract

Question: which remaining discrepancies arise from an implementation error,
an unspecified numerical choice, the printed controller, or insufficient
replication? The target is GJL's first dimension experiment (Section 5.1,
T=100, alpha=.42, d=5/10/20/40/80), with exact Kalman likelihoods. The frozen
working reconstruction is diagonal log-quadratic fitting, positive floor power
8, and doubling only after k. It is not the paper's Equation 15 optimizer.

1. Trace the actual worker -> reconstruction -> controller -> APF/backward fit
   call chain against Equations 5/6/15/16 and Algorithms 3/4/5. Check the source
   theory and appendix, not just the numerical tables. Add executable checks
   for proposal/weight telescoping, the fully adapted endpoint, and the exact
   future-guide oracle.
2. Revisit the previously rejected local Equation 15 fit. Reaching an explicit
   constraint is a sensitivity diagnostic, not intrinsically an invalid
   constrained optimization. Keep the old rejecting route unchanged by default;
   add an explicit diagnostic opt-in that reports active bounds. Compare box
   multipliers .5, 1, 2 on fresh d10/d20 backward targets at t=99, against QR,
   using heldout predictive and smoothing clouds. If this route is finite and
   converged, execute one d5 end-to-end smoke under a 180-second cap. Numerical
   nonconvergence remains a veto for that fit; an active constraint alone does
   not. Neither cloud residuals nor this smoke can nominate a scientific winner.
3. Replay four fixed guides (the first four saved d80 validation replicates,
   IDs1401--1404, no selection by outcome) at N=1000 and 2000, 16 independent
   runs per N and guide. Keep the guides, including their floors, unchanged.
   Compare each likelihood ratio to Kalman. This isolates final-filter accuracy
   from learning/controller cost; it cannot establish an end-to-end controller
   repair. Include the exact future-guide oracle at N=32 and an observation-only
   guide at N=1000. The latter is explanatory; the full baseline ladder remains
   in the validation cells.
4. Complete fresh validation with 32 repeats each: d20-A data87000020, IDs1601+
   and d20-B data88000020, IDs1701+; d40 data87000040, IDs1801+. Use the frozen
   reconstruction and BPF10000, FA-APF5000, SIS10000, plus exact Kalman. Preserve
   each dataset separately. If the earlier d20 descriptive loss recurs, report
   it and its uncertainty; do not tune on these outcomes.
5. Close the audit with a complete gap register, decision/inference tables,
   exact commands and hashes. Refresh the master and concise checkpoint.

The practical-reference primary screen is inherited, not retuned: bootstrap
95% mean-ratio interval wholly within [.8,1.2], SD upper interval <= twice the
paper's SD, and mean N <=1.5 times its reported mean N. Report the stricter
[.9,1.1] mean screen separately. Literal variability/resampling agreement is a
different descriptive comparison, not a prerequisite for unbiased estimation.
No observed conditional MSE loss to the heuristic set is the conservative
promotion veto. A confidence interval spanning zero does not establish a true
performance difference. Use 4000 replicate-level bootstrap draws; never treat
correlated observation times as independent replicates. Intervals are pointwise,
conditional on simulated datasets, and do not provide simultaneous familywise
or cross-dataset guarantees.

Constructed heuristic set: BPF uses the unmodified dynamics; FA-APF uses the
tractable current observation; SIS removes resampling to expose degeneracy.
Evaluate original-prefix likelihood-ratio MSE separately for ordinary and
top-decile Kalman innovation times. Kalman is the exact likelihood authority.
These are falsification checks, not tuning targets. Unequal particle counts and
R implementations do not support an equal-cost efficiency ranking.

Hard continuation vetoes: wrong target/model/seed, broken weight identity,
nonfinite or corrupted evidence, missing required fields, violated isolation,
or exhausted bounded compute. Optimizer failures, active bounds, failed mean
screens, and a candidate losing to FA-APF do not invalidate unrelated phases.
Infrastructure failures trigger focused repair and a fresh attempt directory.

## Defaults and assumptions

| Choice | Provenance and role | Failure risk and early check |
|---|---|---|
| Model, T, N0, k=5, tau=.5, kappa=.5, diagonal fit | Paper, baseline | Wrong indexing/weights: deterministic oracle and telescoping checks |
| QR/floor8/after-k | Previously frozen reconstruction, hypothesis | Different fit/controller from authors; explicit identity and fresh validation |
| Local boxes around QR, multipliers .5/1/2 | New diagnostic hypotheses | Bound-driven result; report active coordinates and heldout errors at every radius |
| QR initialization and optimizer maxit200 | Existing reconstruction choices | Rare-target coverage/nonconvergence; target effective sample size and solver status |
| Frozen-guide N1000/2000 | Paper initial count / observed doubling | Good guide selected retrospectively; use first four, preserve all outcomes |
| No change to six-estimate CV | Printed k+1 window | Mixes learning transients with final noise; fixed-guide repetition distinguishes them |
| Exact oracle has no floor | Analytic Gaussian future likelihood, reference only | Must produce constant weights and Kalman agreement before use |
| Bootstrap, 32 validation repeats | Bounded diagnostic evidence | Rare tails and low power; no 1000-replicate/table-reproduction claim |
| CPU R with one BLAS thread | Owner-authorized independent reference | No GPU, production, gradient or HMC claim |

A shorter five-estimate stopping window is an extension of the printed rule.
It is not substituted for the paper in this campaign. First determine whether
the retained guide itself requires twice as many particles.

## Skeptical pre-execution review

PASS after revision. The initial idea of simply finding settings that match
Table 1 would select on noisy published summaries without the original data.
Instead, source gaps are separated from correctness and practical accuracy.
The old active-bound veto cannot be used as evidence of filter failure; the
new opt-in changes the diagnostic question explicitly and retains prior results.
The baseline uses the actual paper model and fresh final APF. Fixed-guide replay
is conditional evidence, never a replacement for full-controller validation.
Fresh d20/d40 cells do not select parameters. All phases have bounded attempts,
preserved failures, and real stop conditions. Remaining authors' unknown data,
optimizer and floor are disclosed rather than silently guessed.

Pre-mortem: a small variance can conceal collapsed likelihood estimates, so
check mean ratios and conditional MSE first. Good on-cloud fits can miss relevant
states, so include independent predictive and smoothing clouds. Active bounds
can improve loss while changing shape; preserve both. A guide replay can look
excellent while its learning process is expensive; retain that distinction.

## Execution allocation

New allocation for this explicit follow-up request: 2400 summed single-thread
R-worker seconds, 300 seconds for focused tests/reporting, at most ten launches
(six planned plus four localized retries). At most two workers concurrently;
reserve each full timeout before launch. Per-worker cap 500 seconds; usual
controller limits remain 20 iterations/N<=16000. This is about 45 CPU-minutes
maximum, within the authorized local reference direction. The previous stage's
72.641587 unused seconds are not transferred or spent again.

Use Rscript --vanilla and /home/chakwong/anaconda3/envs/tftwogpu/bin/python,
CUDA_VISIBLE_DEVICES=-1, OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1. Capture sources
before launching, including the plan and all direct/transitive local code.
Output: docs/plans/artifacts/iapf-r-replication-gap-audit-20260921-01/.
Every attempt gets a unique subdirectory and full log. The manifest records
commit, dirty-source hashes, CPU/R environment, seeds, commands, time, plan and
result paths. No installation, external messaging, or public release is involved.

## Source scope

Primary authority: Guarniero, Johansen and Lee, *The iterated auxiliary particle
filter*, DOI10.1080/01621459.2016.1222291, cached PDF and arXiv1511.06286 v1/v2
source under the source-reconciliation campaign. Relevant source anchors and
the public unverified R comparator will be recorded in the gap register.
No original-author implementation has been verified. The public R comparator
does not compute the printed Equation15 objective. The alpha sweep, 1000-run
published scale, PMMH and stochastic-volatility studies remain separately
identified coverage gaps; they are not silently added to this bounded stage.

## Causal-audit amendment before fixed-guide interpretation

A final guide may owe its quality to learning after particle doubling. Therefore
replaying final guides alone cannot locate the cause of doubling. Under the same
allocation, reconstruct the first four original d80 histories with a read-only
filter-call hook that retains the guide entering iteration6, before the first
eligible doubling. Require identical history/final output to the saved original
run, then replay each retained guide16 times at N1000 with fresh seeds87800000+
100*(guide_id-1400)+replication. Compare this pre-doubling evidence separately
with the final-guide evidence. It remains conditional on these four learning
histories and cannot validate a modified stopping controller.

The source sensitivity rerun also freezes optimizer parameter scaling across
box widths (see repair-note.md). These are two localized repairs, within the
ten-launch and2400-worker-second limits. Reserve120seconds for the sensitivity
rerun and400seconds for the pre-doubling diagnostic after completed workers
release their unused reservations. If400seconds cannot be reserved, do not
claim that final-guide replay resolves the controller mechanism. Skeptical
review passes: no threshold or candidate selection changes; the amendment
removes a specific causal confound identified before interpreting replay data.

## Bounded completion and mean-deficit check

The original d40 attempt reached its 500-second timeout after completing all
four methods for IDs1801--1827; ID1828 had started. This is a resource failure,
not a numerical failure. Rerun IDs1828--1832 with the same settings and seeds
in attempt09 (200-second reservation). Preserve the original attempt and merge
only complete, nonduplicated replicate/method records into a derived 32-repeat
dataset with source provenance. Require equality of observations and Kalman
references before merging; no partially written fit may enter the comparison.

d20-A's initial32 mean-ratio interval [.9136,.9684] excludes1, although it passes
the inherited practical screen. This deserves an independent check. Attempt10
uses exactly64 additional iAPF-only repeats on the same dataset87000020,
IDs1901--1964 (400-second reservation), unchanged delayed/QR/floor8 settings.
No new heuristic comparison is needed to answer this mean question. Report
fresh-batch mean/SD with4000 replicate-level bootstrap draws, whether its95%
mean interval contains1, and a separately labeled exploratory pooled96 result.
Do not run until an interval passes; this is one fixed confirmation batch.
A repeated deficit is a repair trigger for estimator/tail diagnostics, not a
proof of mathematical bias. A fresh interval containing1 does not prove
unbiasedness. These are conditional, pointwise intervals; initial screening and
the triggered choice of dataset are disclosed. No fitting or policy selection
uses these observations.

Skeptical review passes: the d40 repair restores the original planned sample;
the mean check answers an observed discrepancy that a loose practical screen
would otherwise hide. Completed initial workers consumed1276.942164 seconds;
the four remaining full reservations120+400+200+400 total1120, within the
1123.057836 remaining worker seconds. Ten launches total, at most two workers.
Neither the scientific target nor the total budget changes.
