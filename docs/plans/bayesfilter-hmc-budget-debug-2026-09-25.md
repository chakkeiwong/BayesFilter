# HMC confirmation budget and execution repair

This is a bounded repair of active G–L/C1/C2 work, authorized by the owner's
September 25 request to debug the stopped runs. It does not create a new
scientific study or change the 50-hour grant. Outputs go under
`artifacts/hmc-budget-debug-2026-09-25/`; prior results remain immutable.

## Question and checked starting evidence

Why did C2 miss two fits while its enclosing budget still had time, and why
did C1's direct cost more than double its earlier forecast? C2's 254 completed
fits pass the three original pointwise rate screens, but two outcomes are
missing. The quarter/half last fits received only 214.792/36.620 seconds from
their cells, despite their declared 750-second cumulative fit allowances.
The old reservation has 1,676.085 seconds left. These are cell-budget stops,
not evidence that either fit used its complete numerical allowance.

C1's B development wrapper explicitly used `reuse_leapfrog_graphs=True`.
The later isolated worker called the same pipeline with its default `False`.
Its source-bound execution records confirm this difference. C1's latest
tuning stages cost 447–569 seconds; the earlier development tuning stages
cost 174–227 seconds. Different seeds, source, workload and profiling prevent
treating this as a causal speedup estimate. The current profile shows extensive
TensorFlow execution/compilation, with 124 live reusable-runner objects in the
first Gaussian fit. A paired full-fit check must distinguish graph strategy
from profiling or machine load.

## Repair and evidence contract

1. Propagate an explicit graph-reuse execution flag through the public suite,
   cell and isolated-fit entry points. Keep the default false and the existing
   reference-mean engine's true behavior. Record the policy and refuse a resume
   that changes it. Graph reuse must not change target, geometry, seeds, broad
   L coverage, verification, posterior counts or candidate retention.
2. Add optional sequential sharing of unused cell time within the original
   sum of cell budgets. Record actual allowances and cumulative costs; prohibit
   parallel sharing and double credit on resume. Classify cell-limited stops
   separately from exhausted per-fit allowances. Test real dispatch and resume
   accounting, including failed/interrupted/missing artifacts.
3. Recover both incomplete C2 slots from copies of their original checkpoints
   on **original source-g1**, original design/seeds, and only their unspent
   original 750-second fit allocations. The 254 completed outcomes are reused
   without rerunning or rewriting them; the original incomplete report remains.
   This explicitly amends the earlier blanket timeout disposition: a shorter
   cell allocation did not exhaust either fit. The genuinely exhausted slots
   of the historical 9/256 study remain closed. Recovery is eligible only if
   original-source/design identity and checkpoint integrity checks pass. Both
   missing slots are resumed regardless of their eventual statistical outcome.
4. On a fresh frozen ordinary-runtime source, compare full Gaussian and
   beta-binomial development fits with graph reuse off/on at identical design
   and seed, with profiling held constant off. Inspect native candidates,
   epsilon/L evidence, warmup/retained counts, health and saved numerical arrays.
   Exact array equality is the primary engineering criterion; if arithmetic
   differs, retain the diagnostic and investigate before claim-bearing use.
   Timing and machine-load observations are explanatory; two pairs do not
   establish a performance ranking or runtime tail bound. Extend direct pricing
   to all four existing development identities only if reuse passes parity.
5. Reprice the unchanged 512-fit C1 study using repaired execution, all measured
   costs and remaining allowance. Launch only if its complete measured forecast
   fits a declared enclosing ceiling with a recorded margin. No dropped fits,
   reduced chains, skipped preparation, shared fitted state, new acceptance
   band, or weakened .90 coverage/delivery screen. Confirmation remains on a
   fresh versioned source/output and requires terminal scientific review.

The research intent remains complete-procedure false-alarm/sensitivity (C2)
and actual-stopping delivery, interval coverage and joint success (C1). Exact
analytic references, full denominators, pointwise exact 95% bounds and original
thresholds remain primary. R-hat/ESS/MCSE remain posterior diagnostics, never
epsilon/L tuning gates. Numerical/source/artifact mismatch or exhausted total
compute is a continuation veto for the affected experiment. An unfavorable
candidate/statistical screen triggers diagnosis, not selective replacement.
Costs, utilization and profiles cannot establish statistical calibration.

## Budget and numeric/default audit

The new grant has 177,603.156 GPU seconds left. Reserve at most **3,600 GPU
seconds** for paired development/repricing and **1,800 CPU seconds** for tests
and diagnostic analysis, the latter drawn from the prior CPU balance. These
are convenience engineering ceilings, not numerical defaults. C2 recovery
uses at most **1,500 GPU seconds** of its existing 1,676.085-second reservation;
each fit receives at most 750 minus all its prior attempted wall time, with
no new contention grace. This leaves shutdown/accounting headroom. No remaining
budget or seed is reset by copying checkpoints.

All GPU commands use the previously selected UUID, trusted execution, memory
growth before import, TF/TFP and XLA. Check it is free; never kill other users'
work. Every launch has a unique output directory, exact command, source hash,
hardware/memory policy and enclosing receipt. Charge enclosing time once, not
its nested stages. The repaired optional execution settings are inherited
mechanisms, not newly promoted inference defaults. The 750-second allowance,
256+256 C1 units, C2 128/64/64 units and statistical thresholds retain their
documented provenance in G–L. Pricing seeds are the four existing development
roots; confirmation seeds stay untouched. Current canonical NeuTra policy is
unchanged; no learned-map work or legacy training is launched here.

## Skeptical pre-execution audit

The baseline audit found a real graph-strategy mismatch, so blaming the GPU
or posterior sample counts is unsupported. Host profiling is another possible
cost source and must not vary inside the paired graph comparison. C2's
cell ceilings stranded about 1,446 baseline seconds and its declaration did
not price runtime variation; a per-fit timeout must not be mistaken for an
exhausted original fit allowance. Recovery cannot mix new numerical source
with the 254 old fits or replace their outcomes. The proposed original-source
continuation avoids that mismatch and preserves the initial incomplete result.

The plan passes for bounded engineering repairs and full-fit diagnostics.
Successful commands or lower runtime do not close C1/C2. Continuation to C1
requires measured full-workload affordability and repaired-path numerical
checks; failed validity or budget conditions produce a concrete terminal
diagnosis instead. The exact MacroFinance input and canonical learned-map
requirements are independent and remain open.

### First recovery finding

The quarter-defect checkpoint resumed in 51.395 seconds. The half-defect
attempt failed after 5.286 seconds because it had only a running preparation
progress file, not a resumable tuning checkpoint. The public tuner correctly
refused that occupied directory. The localized repair preserves the failed
attempt, copies the latest recovery state to a fresh output, and archives only
the unfinished progress-only preparation directory before replaying the same
preparation seed. The 750-second cumulative cap includes the original 36.898
seconds and this failed 5.286-second recovery. Completed quarter data are
loaded, not rerun. No posterior outcome is replaced or source changed.

## Repricing decision and confirmation launch audit

All six development fits completed with profiling off. Gaussian graph-off/on
costs were 671.072/272.021 seconds; beta-binomial costs were 584.443/248.072.
The independent saved-evidence audit found exact numerical parity for both
pairs: 82 and 92 tensors, 150 and 165 numerical observations, all candidate
lifecycle decisions, geometry, seeds and posterior decisions. The two additional
reusable fits cost 268.592 and 254.480 seconds. Timing differences remain
descriptive; these are two paired mechanisms checks, not runtime-tail estimates
or a stochastic sampler ranking.

The declared forecast is therefore **134,904.326 seconds (37.473 hours)** for
the full 512-fit inventory. The inherited 165,600-second confirmation ceiling
leaves 30,695.674 seconds above that observed-cost forecast. The new grant has
175,300.798 seconds remaining after both pricing attempts. The full ceiling
plus 11 seconds for the independent cgroup shutdown limit fits while preserving
the 1,200-second canonical-map pricing reservation. These margins are
operational containment choices; they do not guarantee every fit will finish.

Confirmation uses a fresh copy of tested source-r4, `source-confirmation-r1`,
with unchanged Gaussian/beta-binomial scientific designs and root seeds
2026092481/2026092482. Only cell budgets and descriptive purpose are changed
by the previously tested proportional-allocation helper. The public command
is `python -m bayesfilter.testing.inference_validation run confirmation-suite.json
--output confirmation-r1/suite --reuse-leapfrog-graphs --share-unused-budget`.
Profiling remains off. `run_check.py` enforces 165,600 seconds and systemd
enforces 165,610 seconds plus one second of shutdown for the entire cgroup.
Every child must verify selected-GPU placement, XLA and memory growth.

The final skeptical audit checks full-fit receipts, hardware provenance,
untouched confirmation seeds, unchanged statistical screens, source hashes and
the two public execution flags before launch. Pricing source-r3 differs from
source-r4 solely in recognizing a preparation-budget exception during a paid
retry; none of the six completed pricing fits entered that retry branch.
Its regression passed separately. No learned-map source is used. The old
queue's reconciler must not run because it predates these debug receipts;
the new terminal finalizer appends one enclosing charge to the grant ledger
and preserves all previous charges. The audit passes for launching the full
prospective inventory, with no scientific promotion before terminal review.

### Terminal implementation audit: interrupted later-cell credit

The final review found a resume-only accounting defect: a later cell marked
`running` could have consumed predecessor credit, but `max_jobs` could end the
dispatch scan before its reservation was settled, and the sharing calculation
excluded `interrupted` predecessors. The repair settles every interrupted
reservation before dispatch filtering and counts interrupted cells' spending.
A two-cell regression repeats an earlier-cell retry after a later reservation
used all remaining time, with `max_jobs=1`; neither retry may reclaim that time.
This is a bounded engineering fix under the existing CPU allowance. The fresh
C1 run uses frozen source-r4, has no resumed coordinator, and makes no automatic
retry, so this newly checked branch cannot affect its initial execution.
Do not modify its running source or silently resume it with the later repair.
