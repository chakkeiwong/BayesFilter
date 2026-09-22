# Separate iAPF controller-window diagnostic

This is the next controller experiment already named by the master program.
The owner authorized continuation without stopping at phase boundaries. Execute
only after the full-filter fitting comparison finishes, using its unspent
2400-worker-second allocation, remaining mechanics allowance and launch slots.
No new compute allocation is created and no previous completed campaign balance
is spent. The usual complete-cell cap is 400 seconds. A second cell may use a
smaller cap within the remaining allocation only when that cap still exceeds
1.25 times the measured first-cell duration; this resource rule is fixed before
either controller result is observed. Otherwise record the budget boundary and
leave that cell unlaunched. Its eight replicas and scientific criteria never change.

## Research question and evidence contract

In d80, can discarding the earliest estimate in the stopping window reduce
controller-driven particle doubling without invalid likelihood estimates?
Earlier replay found that each of four fresh fixed-guide CVs was .19--.23 while
the corresponding six-estimate learning-history CV was .52--.69. That finding
nominates a test; it does not validate the shortened rule.

Baseline: the current independent-reference QR/floor8 controller, stopping only
after iteration k=5 when CV of the last k+1 estimates is below .5. Candidate:
use the last k estimates for that stopping check. Keep the earliest eligible
iteration, doubling test/window, fitting objective, floor, particle caps,
resampling and fresh independent final filter unchanged. This is an explicit
extension to the paper's controller, not a claim of source faithfulness.
Use the same common controller with a tested optional window argument, never a
copied controller implementation. Default behavior remains k+1.

Two fixed fresh d80 datasets: seeds 89400080 and 89500080, T100, alpha .42.
Eight paired replicas per dataset, IDs 2501--2508 and 2601--2608. QR and shortened
QR share seed 53000000+100000*d+10*ID+1. BPF10000, FA5000 and SIS10000 use the
same formula with indices 2/3/4. Every run includes all learning and final-filter
cost. Run the first dataset, then the second if its full cap fits. Do not choose
a window from these validation results or repeat until a desired interval appears.

Primary screen: each complete cell's likelihood mean-ratio bootstrap95% interval
must lie in [.8,1.2], SD upper interval <=.70 (twice the paper's d80 SD), and mean
N<=1713 (1.5 times paper mean1142). Particle use and full runtime are explanatory
comparisons until paired uncertainty is reported. Use 4000 paired replicate
bootstrap draws for particle-count, terminal squared-error and conditional
prefix-MSE differences; eight repetitions remain a small diagnostic sample.
An interval containing1 does not prove unbiasedness. Report literal paper
SD/resampling-pattern agreement separately, never tune toward those numbers.

Constructed heuristic set: BPF (no guide learning), FA (optimal one-step proposal)
and SIS (no resampling), all at the counts above. Conditional situations are
Kalman standardized innovation <= or > chi-square(.9,d). Observed mean prefix-MSE
loss to any heuristic is a conservative promotion veto; its uncertainty may
leave ranking unsupported. Exact Kalman likelihood is the authority.

Numerical/engineering vetoes are non-finite values, incorrect common-controller
wiring, nonconverged QR fits, particle/iteration caps, missing data/seed identity,
or Gaussian-limit tail-quality failure. A candidate failure blocks its promotion
but does not stop the second independent cell. Stop the campaign only for a
common-implementation/identity failure, corrupted evidence, or exhaustion.
Allowed repairs restore the same declared algorithm and missing replica IDs in
fresh directories within the original total time and launch allowance.

Resource-repair clarification before interpreting controller results: early
observed d80 runtimes are about70 seconds per five-method replica, so the first
400-second worker may time out. Preserve its completed (replication,method)
pairs and finish only missing pairs in one fresh attempt if its cap can be fully
reserved. Each method resets its seed, making this continuation independent of
skipped executions. Require matching data/Kalman hashes, exact40-pair identity,
fit/tail/prefix coverage, and source provenance before merging. Never discard
an unfavorable completed pair, or count an unfinished pair as evidence. Reserve
up to the remaining allocation minus1 second, capped at400, for this localized
resource repair; it has priority over starting the second dataset. The second
dataset remains explicitly unrun if its complete reservation cannot fit.

Nonclaims: original-paper replication, mathematical superiority, a production
default, TF/GPU/HMC validation, or correctness of LEDH/KDM. Positive-floor
conditional unbiasedness follows from a fresh final APF under the previously
checked assumptions, independently of this stopping rule; a small empirical
mean check does not prove its practical tail behavior.

The mathematical distinction is precise. Let H contain the entire learning
history up to the selected stopping time, including its particle count N(H) and
positive guide psi(H). The fresh final APF uses independent randomness U. The
fixed-guide likelihood identity checked against Equations5/6/16 and Algorithm5
in the preceding source audit gives E[Zhat(psi(H),N(H);U) | H]=Z for any valid
fixed H. Taking expectation over H gives E[Zhat]=Z. This reasoning covers either
window because it does not reuse a learning estimate as the final likelihood;
it does not bound practical variance or guarantee a finite-budget implementation
always completes. The controller wiring test must preserve that fresh call.

## Defaults, pre-mortem and skeptical audit

| Choice | Provenance / reason | Failure and early diagnostic | Status |
|---|---|---|---|
| Window k instead of k+1 | Prior exact controller replay implicates early learning estimates | Omits useful variance signal; fresh likelihood/conditional checks | Explicit extension hypothesis |
| Same k, tau, doubling, QR, floor | Isolates one controller change | QR or floor failures remain possible; numerical/tail records | Frozen comparator |
| d80 only | Existing inflated particle-count discrepancy | No cross-dimension generality; two fresh datasets | Target-specific scope |
| Eight replicas/cell | Fits remaining authorized allocation | Broad or unreliable tail intervals; no ranking by means alone | Diagnostic convenience |
| Fixed baseline counts | Same published/preceding comparison ladder | Not matched cost; retain whole learning runtime | Diagnostic baselines |

Pre-mortem: the candidate may stop sooner with an inadequate guide and merely
look cheap. Exact likelihood and conditional heuristic errors can veto it.
Shared seeds do not guarantee identical draws after trajectories diverge.
The first dataset might be unusually easy, so the second is fixed in advance;
an unrun second dataset remains missing evidence, never silently waived.

Skeptical audit: PASS as a separate diagnostic. It tests the recorded mechanism,
keeps fitting/controller effects separate, preserves the default and fresh final
draw, does not use a proxy to certify likelihood quality, and has explicit caps,
uncertainty, unfair-cost disclosure and failure continuations. Before launch,
add a deterministic executable test showing the default window retains its old
decision, the new window changes only the intended decision, and both routes
take a fresh final filter call. Run existing R core/identity regressions.

## Execution records

CPU R only, CUDA_VISIBLE_DEVICES=-1, single-thread BLAS; the Python supervisor
uses /home/chakwong/anaconda3/envs/tftwogpu/bin/python. Two serial launches capped
at400 seconds each, with the predeclared smaller second-cell cap rule above.
Capture the changed source closure separately after fitting
workers finish; preserve their earlier source snapshots. Store under the fitting
campaign root in new attempt directories, with a controller budget-transfer note
and source hashes. Account every second in the original2400 total and16 launches.
Terminal result must include decision/inference-status tables, source identity,
actual commands/seeds/environment, comparison intervals and strongest alternative
explanation. Update the master and concise checkpoint after completion.

## Recorded execution state, 2026-09-21

The fitting comparison and first controller dataset are complete. The controller
timeout was repaired automatically within the same allocation by completing only
missing method/replica pairs. Total2354.902653/2400 worker seconds,15/16 launches;
45.097347 seconds remain. The second fixed d80 dataset is unrun because its full
reservation cannot fit. Results, review, uncertainty and exact next action are in
[the result note](artifacts/iapf-r-full-filter-fitting-20260921-01/result.md).
No controller or production default was promoted.
