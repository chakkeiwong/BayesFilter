# Independent R iAPF continuation: 24-hour campaign

The owner authorized another 24 hours on 2026-09-21 Hong Kong time. This
supersedes the previous remaining-time boundary, not the previous evidence.
The campaign starts at 2026-09-20 20:04:26 UTC and ends no later than
2026-09-21 20:04:26 UTC. At most two single-thread CPU R workers may execute
concurrently: maximum 172800 summed worker-seconds, additionally bounded by
that elapsed deadline. Old campaign balances are neither transferred nor spent.
Use at most 10000 worker launches, fresh directories, and the existing R 4.1.2
environment. CPU-only independent reference is explicit; CUDA_VISIBLE_DEVICES=-1,
OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. No package, GPU, public API, or
production-default changes are authorized by this extension.

Output root: docs/plans/artifacts/iapf-r-24hour-campaign-20260921-01/.
Manifest, active checkpoint, source snapshots, method/replica records, phase
reports and a terminal result preserve execution. The master remains
docs/plans/younis-kdm-score-master-program-2026-09-14.md.

## Question, evidence and continuation rules

The question is whether the frozen independent R reconstruction produces
reliable likelihood estimates under the paper's first linear-Gaussian study,
and which observed replication discrepancies arise from fitting, controller
behavior, or insufficient Monte Carlo evidence. The paper's Equation15 fit
and QR are different objectives. The preceding constrained-fit failures are
preserved; no failed arm is relabeled faithful or silently promoted.

Exact Kalman likelihood and prefix likelihoods are the numerical authority.
Compare QR with stopping windows six and five while holding k=5, tau=.5,
N0=1000, T=100, floor-tail-power8, delayed doubling, resampling and the fresh
final APF fixed. BPF10000, fully adapted APF5000 and SIS10000 form the constructed
heuristic set: no learned guide, optimal one-step proposal, and no resampling.
Counts follow the existing study comparison; costs are not matched. Evaluate
prefix squared relative errors separately for ordinary and large standardized
Kalman innovations (chi-square90% cutoff), including whole learning cost.

Promotion requires a complete valid cell, the predeclared practical primary
screen and no conditional heuristic veto. Observed conditional loss is a
conservative promotion veto even when its paired interval spans zero. Numerical
validity, source/data/seed identity, complete prefix/fit/tail records and fresh
final-filter wiring can veto promotion. A common-core or oracle-identity failure,
corrupt evidence, exhausted time/attempt allocation, or scope boundary stops
dependent execution. A failed candidate does not stop independent comparisons
or a replication intended to characterize that failure. Unknown author settings
and QR's different objective remain explicit limitations, not hidden defaults.

Use 4000 replicate bootstrap draws with fixed analysis seed9212026 for means,
SDs and paired errors/costs. Intervals are pointwise; no optional stopping based
on an attractive interval. Prespecified sample sizes are completed regardless
of interim metrics unless a validity or resource stop applies. Non-finite log
values are invalid; exponentiation underflow is retained explicitly with its
finite log ratio and is never called zero theoretical variance. Numerical,
engineering and interpretation decisions remain distinct.

## Execution sequence

1. Finish the untouched second controller dataset89500080, IDs2601--2608:
   eight replicas of both fixed windows and all three heuristics, using the
   same seed recipe53000000+100000*d+10*ID+method-index (QR variants1;
   BPF2, FA3, SIS4). Preserve the previously complete dataset89400080.
   Execute each replica in its own resumable directory, initially reserving
   240 seconds per five-method replica. Primary d80 screen: likelihood-mean
   95% interval inside[.8,1.2], SD upper interval<=.70, mean N<=1713.
   Run every planned replica even when a candidate fails this screen.
2. Before launching the larger controller comparison, record a phase amendment
   with fresh datasets/IDs and fixed replication counts. The intended scope is
   64 paired replicas on each of two fresh d80 datasets, with both windows and
   the same three heuristics. This distinguishes eight-replica uncertainty
   from a reproducible accuracy/cost tradeoff. No tuning of window lengths,
   thresholds, fitting, floors or bounds on previous validation data.
3. Freeze the exact reconstruction and study design before the planned
   five-dimension first-study replication. Intended sizes are1000 independent
   replicas at each of d=5,10,20,40,80 on fresh fixed datasets, with the paper's
   T/model/particle counts and exact Kalman comparison. Recheck primary source
   tables/model definitions and practical/literal criteria before launch.
   Source fidelity, practical utility and numerical correctness are separate.
   A frozen diagnostic reconstruction can be characterized even if it fails a
   practical promotion criterion; that characterization is not promotion.
   Keep the second controller variant separate if source/design or budget
   review makes a two-variant full study uninformative. Record the choice before
   opening new study data; do not select it by fresh study metrics.
4. Analyze all completed cells with uncertainty, conditional heuristic tables,
   paper-table comparisons, learning/particle/resampling diagnostics and exact
   next decisions. Refresh the master and remaining gap list. Source or
   multidimensional R/TF comparison work already in the master may proceed
   independently after its own focused evidence contract; do not substitute
   GPU/TF results for the independent R study or imply original-author identity.

## Defaults, pre-mortem and skeptical review

| Choice | Provenance and justification | Failure mode / early diagnostic | Status |
|---|---|---|---|
| Two CPU workers, single-thread BLAS | Preserves prior hardware class; new elapsed allowance | Contention distorts timing; record worker concurrency and all costs | Resource choice, not speed benchmark |
| QR and positive floor8 | Existing tested reconstruction | Different fitting objective / practical tail risk; exact Kalman and guide-tail checks | Frozen diagnostic comparator |
| Window5 versus6 | Previously derived controller transient mechanism | Premature stopping harms accuracy; independent fresh-final estimates | Explicit extension hypothesis |
| Eight pending replicas | Previous untouched design | Wide intervals; subsequent fresh64 comparison | Preserve existing contract |
| Larger64 then1000 samples | User's approved first-study replication ladder | Lucky data or unobserved tails; fresh data, conditional checks, fixed counts | Planned uncertainty reduction |
| Per-replica resumable launches | Previous timeout lost work | Seed or duplicate errors; exact pair identity and data hashes | Engineering repair |

Pre-mortem: a cheap shortened controller can fail likelihood quality, a correct
unbiased estimator can look biased in a short heavy-tailed sample, and a paper
table can match accidentally despite different fitting semantics. Report all
three separately. A passing sample does not identify unpublished author choices.
Do not optimize toward the published SD/resampling values.

Skeptical audit: PASS for stage1 and this campaign envelope. The baseline and
data/seed identities are frozen, criteria precede execution, inherited choices
are labeled, conditional simple baselines and exact authority are available,
the previous failure remains a promotion veto rather than a continuation veto,
and resumable work addresses the measured timeout without algorithm changes.
Later phases require their source/design amendment before execution, but not a
new routine user approval within this 24-hour allocation.

Localized resource, serialization and reporting failures may be repaired within
the same budget and scientific contract. Every failed launch is charged. Keep
completed pairs, rerun only missing pairs with their original seeds, validate
complete coverage before combining, and preserve attempts. Stop at the elapsed
deadline even if another launch would fit the summed-worker ceiling.
