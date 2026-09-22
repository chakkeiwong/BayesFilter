# Independent validation of the repaired R iAPF reference

Status: COMPLETE. The owner-authorized campaign finished all 80 paired cells,
160 full learners and 880 additional evaluations with no learner failures.
All five primary paired intervals include zero; no accuracy ranking follows.
The repaired reference has 14 observed conditional heuristic vetoes. See
`artifacts/iapf-r-independent-validation-20260922-01/result.md` for the result
and terminal review. This is an independent CPU R reference campaign. The
weighted-log/ridge extension is different from the paper's Equation 15; its
validation cannot establish original-author identity or full paper replication.

## Research question and evidence contract

Does the frozen repaired learner complete independent runs across dimensions
5, 10, 20, 40 and 80, and what accuracy and cost does it achieve relative to
the existing unweighted QR learner? The preceding repair campaign resolved
eight selected failures but did not establish an accuracy advantage. Its source
audit and numerical checks remain at
`artifacts/iapf-r-targeted-fitting-repair-20260922-01/result.md`.

The model is the already audited first-study linear Gaussian model: T=100,
alpha=.42, A[i,j]=alpha^(abs(i-j)+1), identity initial/process/observation
covariances and zero initial mean. Kalman gives the exact terminal log
likelihood. Freeze both fitters, floor rule, controller and solver safeguards
at the final source hashes in the preceding root manifest. No fitting or
threshold tuning uses this validation data.

Primary numerical criterion: complete all planned runs with finite values,
positive guide covariance and the existing fit checks. Report failure counts
and binomial uncertainty; zero observed failures is not proof of reliability.
Primary accuracy comparison: per dimension, the paired mean difference in
absolute terminal log-likelihood error, repaired minus QR, from each complete
learner's original fresh final estimate. Use stratified paired bootstrap
intervals, resampling the eight replication indices within each of two fixed
datasets, with 20,000 resamples and 99% two-sided intervals (five-dimension
Bonferroni family). Only an interval wholly below zero supports a conditional
accuracy advantage, and failures or heuristic vetoes still prevent promotion.
An incomplete planned cell precludes that dimension's confirmatory ranking;
complete-case summaries then remain descriptive.

Candidate promotion vetoes: fit/domain/KKT rejection, non-finite candidate
values, particle/iteration caps, observed conditional underperformance against
an applicable declared heuristic, or poor likelihood calibration. Candidate
failures remain results and do not stop other independent cells. A failed
exact-guide/Kalman check, changed frozen source, corrupt/missing required
evidence, or exhausted aggregate budget/deadline is a continuation veto.
Infrastructure and serialization repairs can continue under the same frozen
scientific method, preserving every attempt. A new numerical algorithm would
be a separate candidate and cannot silently replace this frozen validation.

ESS, prefix errors, active constraints, ridge strengths, KKT values, condition
numbers, resampling counts and timing explain results; none substitutes for
terminal accuracy. Prefix behavior cannot refute a terminal-likelihood claim.

## Independent design, controls and cost

There are two fresh datasets per dimension, eight repetitions per dataset:
80 paired cells and 160 full learners. Dataset seeds are 92800000+10000*j+d,
j=1,2. Learning seed is 92840000+10000*j+101*r+d, r=1,...,8, shared between
the two learners. Counterbalance method order by j+r. The shared controller
uses N0=1000, k=5, tau=.5, kappa=.5, tail8, sample SD, after-k doubling,
k+1 history, maximum 12 iterations and 4000 particles. Every final estimate
remains fresh as implemented by the checked shared controller.

After learning, keep each guide fixed and evaluate independent fresh filters
at N=250,1000,4000. Grid filter seeds are
92900000+100000*h+10000*j+1000*r+d, h=1,2,3. Both learned guides and all
controls at N=1000 share the relevant seed. These evaluations do not update
or select the guide. They separate learning cost from final integration cost.

Constructed heuristic set, evaluated separately by dimension and dataset:

- Constant guide: ordinary bootstrap filtering, the simplest usable baseline.
- Observation-only guide: locally adapted use of the current likelihood.
- QR learner: the established simple regression comparator.
- Moment-diagonal exact-future guide: attainable diagonal approximation when
  future Gaussian moments are known, a privileged analytic control.
- Precision-diagonal exact-future guide: the other KL projection, also privileged.
- Full Gaussian guide and Kalman: exact terminal oracle and validity check.

All five nonlearned guides run at N=1000 in every cell. Comparisons with the
learned guides use their N=1000 fresh evaluation. Preserve which controls use
model-specific analytic information; they are not general-purpose competitors.
The full Gaussian/Kalman oracle supplies exact-error and harness checks, not a
stochastic competitor in the heuristic win/loss table.
Record observed mean absolute-error underperformance per dataset as the
heuristic veto. Do not optimize against this table.

For cost, record the actual full-learning time, each internal APF time, and
each independent final-filter time. A timer wrapper must preserve values and
RNG state on a focused parity check. Training cost subtracts the original
fresh-final call from full-learning time. Cost of one new estimate is training
plus its fresh final-filter time; common model/data/Kalman setup is excluded.
Guide-construction cost is retained for analytic controls. Serial single-thread
execution and counterbalanced order reduce timing confounding; timing remains
machine-specific and is not the primary ranking criterion.

Define relative likelihood error R=exp(logZhat-logZ)-1. Report its empirical
RMSE and bootstrap uncertainty along with log error and mean exp(log error).
The fixed precision target is relative RMSE <=.5. This is an explicit
engineering target motivated by the existing .5 controller scale, not a
published accuracy criterion. For the two learned methods and three N rungs,
compute one-sided bootstrap upper RMSE limits at 1-.05/30, accounting for five
dimensions, two methods and three rungs. Report cost among rungs whose upper
limit meets the target; if neither/both lack qualifying rungs, say so. These
are approximate bootstrap precision screens on only 16 learned guides per
dimension, not guarantees about unseen tails or exact matched-MSE efficiency.
No cheapest-rung choice changes defaults or triggers reuse of validation data.

## Assumptions, skeptical review and limits

The inherited fitting/controller settings are frozen comparator hypotheses,
not recovered author choices. Their known failure modes are floor dependence,
stopping variability, poor learned shape and caps; the experiment retains all
corresponding diagnostics. The ridge, bounds and active-set fallback retain
the preceding derivation and tests. Their purpose is numerical validity;
likelihood performance was not used to calibrate them.

The two datasets and eight repetitions are a bounded independent check, not
the paper's 1000-repetition design. Intervals are conditional on these fixed
datasets; no general data-population claim follows. Bootstrap intervals can
miss rare likelihood tails. In particular a collapsed bootstrap filter may
show spuriously small variance: always report mean likelihood ratio and the
near-zero fraction, never efficiency from variance alone. N=250/1000/4000 is
a declared geometric cost probe, not a tuned optimum. The .5 precision target
and all seeds/rungs are frozen before execution.

Skeptical audit: passed for this bounded validation. It corrects the previous
selected-failure design with disjoint data and paired repetitions, uses the
unchanged shared learning consumer, distinguishes log and likelihood errors,
keeps oracle controls explicit, includes all learning work in cost, and treats
method failure separately from experiment invalidity. A successful command
could still mislead through unobserved tails, weak bootstrap coverage or two
unrepresentative datasets; the report must preserve those limits. A timer
parity check and exact-guide check are the earliest harness falsifications.

## Execution, budget and evidence

Root: `artifacts/iapf-r-independent-validation-20260922-01`. Use unique per-cell
attempt directories, executed source snapshots, exact command/environment,
git HEAD and source hashes, seeds, wall time, R session info, result CSV/RDS,
and a terminal decision/inference report. CPU reference only:
CUDA_VISIBLE_DEVICES=-1, OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1.

Maximum 4100 aggregate numerical worker seconds and 90 launches, within the
4513.601569557 seconds remaining from the prior allocation. Cumulative use at
entry is 110076.194833523/172800 worker seconds. Preserve the original deadline
2026-09-21T20:04:26Z; reserve its final 120 seconds for closure. Pair-job caps
are 180 seconds at d80 and 120 seconds otherwise. Run cells serially, cycling
all dimensions and both datasets within each replication, starting with d80
to expose expensive failure early. End-of-budget omissions are resource-censored,
not failures or successes. No launch extends the deadline or starts another
campaign. The final report must record completion, vetoes, uncertainty, cost,
strongest alternative explanation and the next justified action.

## Bounded scheduling repair, 2026-09-21T20:02Z

The initial supervisor completed 78 cells; its 120-second closure reserve
censored d10/dataset2/rep8 and left d5/dataset2/rep8 unstarted. Both fitters
and scientific criteria remain frozen. Skeptical review: these two short
cells can finish under the original hard deadline and phase/launch caps;
retaining a 30-second closure reserve is sufficient given the measured
1.274-second report. Reduce the reserve only for this bounded completion.
Preserve cells.csv and the first report unchanged; cells-completion.csv
selects fresh replacement attempts. Do not interpret resource censoring as
a candidate failure. No budget or hard-deadline renewal is authorized.

Completion: the fresh d10 and d5 attempts both passed, followed by the final
statistical report. Total numerical execution was 2209.885535 seconds across
84 launches, including the initial timed attempt and both reports. The final
numerical process finished at 2026-09-21T20:03:51Z, before the unchanged
20:04:26Z deadline. The final selection is cells-completion.csv; the initial
cells.csv and first report remain unchanged. All recorded source/input/output
hashes and completed-cell pairing, seed, oracle and cost-decomposition checks
passed during terminal assembly. No numerical worker remains active.
