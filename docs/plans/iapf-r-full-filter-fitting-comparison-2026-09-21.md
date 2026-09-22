# Full-filter comparison of constrained Equation 15 and QR fitting

The owner's request to continue the master program authorizes this next bounded
CPU R reference stage. It follows the completed source audit, whose next step is
already recorded in the master. The previous stage's 549.198570 unused worker
seconds remain separate; they are not spent again here.

## Question and evidence contract

Does minimizing the paper's density least-squares objective over an explicit
compact box produce a reliable complete iAPF, and how does its full learning
cost and likelihood accuracy compare with the frozen log-quadratic QR reference?
The paper's unconstrained Equation 15 admits diffuse-density escape. A bounded
optimizer is an explicitly non-author reconstruction, not a recovered original
implementation. This comparison changes fitting only: diagonal Gaussian plus
positive floor, floor power 8, delayed doubling, k=5, tau=.5, kappa=.5, N0=1000,
T=100, alpha=.42, maximum 20 iterations and N<=16000 remain fixed.

The exact comparator is the Kalman marginal likelihood. The direct algorithmic
comparator is delayed/QR/floor8. Constructed heuristic adversaries are BPF with
10000 particles (no learned guide), FA-APF with 5000 (observation-only optimal
one-step proposal), and SIS with 10000 (no resampling). Report their likelihood
prefix MSE separately for ordinary and large innovations, defined before the
run by Kalman's squared standardized innovation exceeding chi-square(.9,d).
These fixed particle counts reproduce the preceding diagnostic ladder; they
are not matched-cost comparisons. Record the entire iAPF learning plus fresh
final-filter time and compare it with each baseline's total time.

The primary validation screen is the inherited practical-reference criterion:
the replicate-bootstrap 95% mean-ratio interval lies within [.8,1.2], the SD
interval's upper bound is at most twice the paper's dimension-specific SD,
and mean final N is at most 1.5 times its reported mean. Also report whether the
mean interval contains one and the stricter [.9,1.1] interval screen. This is a
bounded reference screen, not a test of exact unbiasedness or literal replication.
Use 4000 replicate-level bootstrap draws; paired comparisons share data and
replication IDs. No method ranking is allowed without a paired interval excluding
zero, and no multiple-comparison or paper-wide claim follows from these small cells.

Vetoes: non-finite results, nonconverged fits, controller/particle caps, invalid
source or data identity, missing required records, and the Gaussian-limit tail
screen failing. The tail test is a guide-quality veto, not a claim of infinite
variance for a positive-floor guide. An observed conditional mean MSE loss to
any constructed heuristic is the inherited conservative promotion veto; its
paired interval determines whether inferiority is statistically established.
Active bounds alone are explanatory. Training loss, active coordinates,
projected gradients, ESS/resampling, and runtime explain outcomes and do not
select a winner. Literal paper SD/resampling agreement is reported separately.

Candidate rejection triggers the next predeclared repair or independent arm;
it does not reject iAPF or halt the entire research direction. Continuation
vetoes are corrupted evidence, a wrong likelihood/weight invariant, an invalid
common implementation, exhausted allocation, or a scientific-scope change.
Even passing all screens does not establish original-author identity, paper-scale
replication, production TF/GPU correctness, HMC readiness, LEDH or KDM validity.

## Sequence, fresh data, and repair

1. Run four calibration replicas for each QR and box .5/1/2 at d=5 and d=20.
   Data seeds 89100005/89100020; replication IDs 2101--2104/2201--2204.
   All boxes use fixed optimizer scaling and explicit acceptance/reporting of
   active constraints. Reuse the common controller and filter, not forked copies.
2. Nominate a box only if every planned calibration run in both dimensions
   completes with converged finite fits and passing tail screens. Prefer box 1
   if admissible, then .5, then 2. This predetermined nominal-width rule does
   not optimize likelihood performance or proximity to the paper's table.
   QR remains the frozen comparator regardless of descriptive calibration ranks.
3. If no box survives because of a solver iteration limit, first replay the
   saved failed fitting inputs with maxit=800. Preserve all old attempts. A
   successful local replay licenses a full calibration retry of the same arm;
   it cannot stand in for complete-filter validation. Freeze the optimizer limit
   before opening validation. Other failures receive a bounded mathematical or
   implementation diagnosis; do not silently change objective, floor or controller.
4. Validate the frozen nominated arm and QR plus all three heuristics on fresh
   d20 data seeds 89200020 and 89300020, IDs 2301--2316 and 2401--2416,
   respectively. Launch each cell only with a full timeout reservation. Preserve
   partial results on a timeout and, if allocation permits, complete precisely
   the missing IDs in a fresh directory, requiring equal data/Kalman hashes.
   If no constrained arm survives calibration, do not spend the validation budget
   pretending that a rejected candidate is viable; diagnose the failure instead.
5. Review all results and update the master/checkpoint. A fitting failure leaves
   QR available as a distinct reference, with its known higher-dimensional
   limitations. Controller changes are a separate subsequent experiment. Do not
   launch the five-dimension 1000-replica study before reference settings and
   its materially larger compute allocation are fixed.

Filter RNG: 53000000 + 100000*d + 10*replication + method index, where fitted
arms/QR use index 1, BPF 2, FA 3, SIS 4. Equal seeds pair algorithm runs but do
not imply identical random draws after their trajectories diverge. Data seeds
and calibration/validation separation are fixed before results are inspected.

## Defaults and skeptical review

| Choice / provenance | Reason and possible failure | Early check / status |
|---|---|---|
| Paper model, dimensions and T | Exact Kalman authority; only two calibration dimensions | Oracle identity tests; source baseline |
| QR initialization, boxes .5/1/2 from prior sensitivity | Bounds prevent escape but may dictate shape; no original-author provenance | Report active bounds and full-filter outcomes; hypotheses |
| L-BFGS-B maxit200, factr10, pgtol1e-8 | Inherited convenience, possible premature termination or weak absolute scaling | Preserve optimizer message/loss/gradient, bounded maxit800 replay; not an endorsed default |
| Floor8, delayed controller from preceding reconstruction | Fixed causal comparison; controller can still inflate N | Record history/floor probabilities/N and veto limits; baseline, not safety certification |
| Four calibration and sixteen validation replicates | Bounded feasibility then fresh accuracy check; weak tail/SD precision | Bootstrap intervals and explicit inconclusive outcomes; convenience budget |
| Fixed heuristic particle counts | Comparable to prior reference ladder; not equal compute | Full learning cost in every row; diagnostic comparators |
| CPU R / single BLAS thread | Explicit independent-reference exception | Environment and source capture; no production claim |

Pre-mortem: successful constrained training fits might merely shrink densities
or concentrate on a few particles, and small reported SD might hide a likelihood
deficit. Full-filter likelihood ratios and conditional heuristic MSE address
those failures. An optimization failure may reflect the iteration cap rather
than the mathematical method; saved-input replay separates them. A small fresh
sample cannot certify tails or recover unpublished author choices.

Skeptical audit: PASS for a diagnostic comparison. The baseline and exact target
are distinct; no loss proxy selects the method; fresh validation data are frozen;
active constraints are not falsely treated as invalidity; matched-cost claims
are excluded; failure repair and true stopping conditions are explicit. The
heuristic screen is conservative and must not be narrated as an established
ranking when its interval spans zero. This review precedes implementation and
numerical launches. Independent external review is not required for this bounded
follow-up; terminal self-review must record the strongest alternative explanation.

## Execution and preservation

New allocation under this continuation request: at most 2400 summed single-thread
R-worker seconds, 300 seconds of mechanics/tests/reporting, 16 launches and two
workers at once. Reserve full caps before launch; release unused reservations
on completion. Calibration caps: 180 seconds per d5 arm, 360 per d20 arm.
Validation caps: 600 per constrained cell, 200 per QR-plus-heuristics cell.
Repair caps must fit the remaining allocation; no new paid/hardware resources.

Use Rscript --vanilla, CUDA_VISIBLE_DEVICES=-1, OPENBLAS_NUM_THREADS=1,
OMP_NUM_THREADS=1 and the tftwogpu environment's Python for supervision.
Artifact root: docs/plans/artifacts/iapf-r-full-filter-fitting-20260921-01/.
Capture all source dependencies and this plan, record git HEAD/dirty hashes,
R version, command, seeds, device mode, time, attempt outcome and data hashes.
Every launch uses a fresh directory; checkpoints give the next exact action.
Preserve full logs and failed-fit RDS objects. Result notes include decision
and inference-status tables and clearly separate implementation, numerical
validity, and scientific interpretation.

## Explanatory follow-through within the same allocation

After validation, if 120 worker seconds and one launch remain available, inspect
the saved final guides without generating or fitting additional data. Compare
their Gaussian components with the exact future-likelihood Gaussian available
in this linear model. Report KL(exact component || fitted component), separated
mean/covariance terms, and log determinant inflation at each time and replica.
These comparisons ignore the explicit positive floor and therefore are
explanatory shape diagnostics, not a KL for the actual mixture proposal or a
replacement likelihood criterion. No setting is selected or modified using
validation guides. Check KL identity/nonnegativity, exact terminal observation
guide, and complete saved-guide coverage. This cheap diagnosis distinguishes
density-shape distortion from an implementation failure or a controller-only
explanation. Skeptical audit passes: exact authority, scope and limits are
explicit; no new inferential or compute allocation is created.

## Recorded execution state, 2026-09-21

The fitting comparison and first controller dataset are complete. The controller
timeout was repaired automatically within the same allocation by completing only
missing method/replica pairs. Total2354.902653/2400 worker seconds,15/16 launches;
45.097347 seconds remain. The second fixed d80 dataset is unrun because its full
reservation cannot fit. Results, review, uncertainty and exact next action are in
[the result note](artifacts/iapf-r-full-filter-fitting-20260921-01/result.md).
No controller or production default was promoted.
