# q20 recovery, pricing and tuning repair plan

Proposed September 21, 2026, 22:00 Asia/Shanghai. September 22: owner authorized
integration as master phases R1–R6, thorough review and execution. Implementation
and focused engineering checks passed. Current-source GPU qualification and
trained-map reassessment passed; the finite plain tuning cohort is running in
campaign-04. See the September 22 execution result for current status. This plan
supersedes another timeout-only retry.

The objective is one stable estimate of the fixed q20/T30 four-dimensional
float64 UKF-approximate posterior, using plain NeuTra HMC or the permitted
tempered NeuTra ensemble. Either can satisfy the objective. The independent
integration reference validates the same approximate posterior. Both methods
retain frozen learned maps, identity mass in latent coordinates, four batched
chains, TensorFlow/TFP GPU/XLA execution and verified memory growth.

## Findings and concrete repairs

| Finding | Code or evidence inspected | Required repair |
| --- | --- | --- |
| A final capacity exception obscures completed computation | `q20_master_stages.py:18`; attempt 00008 wrote the complete pricing result before `check_gpu_contention` raised | Persist numerical completion and resource observations separately; distinguish prelaunch unavailability, interruption and contention after completion |
| Pricing cannot resume its completed measurements | `q20_campaign_runtime.py:230`, `q20_master_stages.py:253`; attempt 00009 repeated all eight HMC timing rows | Add a resumable pricing ledger at measurement boundaries, with checked identities and explicit timing quality |
| A retry launches even when its remaining cap cannot cover its next useful work | `scripts/q20_campaign_supervisor.py:311`; 1,872 seconds remained after a 1,873-second pricing attempt | Quote the next missing block including process/compile overhead; launch only if it fits all remaining limits |
| An L=25 template is treated as the initial ensemble cost requirement | `q20_master_stages.py:390`, `q20_master_program.py:15`; first assessment reservation is 96.57 hours | Measure an actual short-trajectory ensemble for conditional initial planning, then reprice the exact verified ensemble before sampling |
| The template is not the selected ensemble | `chart_targets = width_targets` at `q20_master_stages.py:355` retains the final width; pricing uses unqualified maps and common L | Record the template explicitly; exact sampling quotes must bind all selected map/member IDs, widths, epsilons, L values, temperatures and chart probabilities |
| Timing units are inconsistent | Individual HMC first-call times are whole-call seconds; `exchange_times.append(.../count)` at line 433 divides the first exchange call too | Store whole first-call seconds, result count and steady seconds per transition separately; count compilation once |
| Local limits prevent funded continuation, and all timeouts consume the infrastructure repair hold | `q20_stage_budget.py:38`, supervisor attempt checks | Classify allocation exhaustion separately from infrastructure failure, preserve every charge, and plan cumulative stage/work allocations explicitly |
| Plain tuning left an unexplored interval | Saved search has no refinement; epsilon 0.050625 moves every chain, while 0.0759375 fails movement at every tested L | Investigate a finite set of interior hypotheses through the public fixed-transport tuner with fresh measurement and verification |

The pricing failure did not establish a target, mathematical or sampler defect.
The plain trial did establish a movement veto for its final six settings. That
does not prove that NeuTra cannot work or that no intermediate epsilon works.
The earlier cap-only repair missed terminal contention and forecast semantics.

## 1. Preserve evidence and isolate the implementation

Use the actual execution snapshot
`/tmp/BayesFilter-q20-staged-budget-20260920-r2` as the baseline. The shared main
worktree contains unrelated HMC changes; do not incorporate them implicitly.
Create a fresh isolated checkout/source copy containing only the reviewed repair.
Preserve campaign-01, campaign-02, all attempts and the failed plain search.

Separate host orchestration changes from changes to numerical programs. Prefer
keeping existing numerical workers unchanged where a bounded host adapter is
sufficient. Where repository worker/pricing code must change, audit the exact
diff and use the checked source-import mechanism in `q20_training_resume.py`.
That importer preserves maps, optimizer/RNG state and eligible cached losses,
but clears old exports and requires current assessment. A directory on its
coordinator allowlist is not proof that arbitrary edits in it are harmless:
inspect numerical callables as well. Do not change hashes in old files to make
them pass a new source check.

New tuner scopes must be issued through `tune_fixed_transport_hmc_kernel` and
its supported numerical binding, as specified by `hmc-tuning-interface.md` and
`HMC_TUNING_INTERFACE_CAPABILITIES`. Check the actual preserved implementation
before adopting a feature documented by newer main-worktree code.

Deliverable: source-diff/import report, fresh execution root and reconciled
accounting, with reusable and non-reusable evidence listed explicitly.

## 2. Repair completion and resource handling

Before a terminal capacity probe, atomically preserve the completed numerical
result and initialization/wall-time information. Record numerical validity,
timing quality and resource availability as separate fields. A prelaunch wait
has no computed result; a completed calculation with contention has computed
work whose timing needs qualification. These states must not share the same
automatic restart behavior.

Continue enforcing existing finite-value, target-status, memory-growth and
scientific checks. A capacity observation alone must not relabel an intact map,
chain checkpoint or completed calculation as numerically invalid. For ordinary
training/sampling work, preserve and validate the completed state before
deciding whether another worker is needed. For cost measurements, mark affected
timings `contention_observed` or `resource_history_unknown`; do not advertise
them as uncontended benchmarks or as reliable upper bounds.

Before and after each timing block record process/device observations. An
ordinary per-GPU advisory lock can prevent cooperating BayesFilter workers
colliding, but external jobs can ignore it; the lock is not evidence of exclusive
occupancy. Recheck available devices when scheduling a missing block. Use the
existing three-GPU inventory and never assume GPU 1 must become free.

Regression acceptance: a deterministic fixture computes once, then receives a
terminal contention exception. The saved numerical result survives, its timing
is marked, the next phase is correctly identified, and another full computation
is not silently dispatched. Resource failure must not choose another method.

## 3. Make pricing incremental and correct its units

Break pricing into resumable work records: training-price scope, individual
HMC setting, actual ensemble-mixture setting, reference batch and reporting.
Each record binds the target/data, numerical dependency hashes, dtype/XLA/device
settings, map state, beta, L/epsilon, batch/chain dimensions and deterministic
seed. Keep source identities and measured timings together. A restart restores
completed eligible records and computes only missing or specifically invalidated
ones. Use existing ordinary checksum/atomic-write machinery.

For attempt 00008 import the saved result with its original failure and resource
qualification. It is a source of completed work and provisional cost hypotheses,
not an automatically passed clean timing stage. Its global final probe cannot
tell which earlier blocks were uncontended. Attempt 00009's row files likewise
lack complete per-block resource observations. Missing metadata stays unknown.
Do not fabricate the absent `worker_initialization_seconds`: use a labelled
bound from recorded process and numerical wall times, or measure startup afresh.

Represent first-call elapsed time in seconds for the entire call, steady runtime
in seconds per transition, and the number of transitions explicitly. Use the
first call as a conservative compilation/setup allowance when separate compiler
timing is unavailable. Do not divide it and later treat it as whole-call time.

Add cooperative budget checks to pricing, leaving the existing external timeout
as a backstop. Before each new block check whether its forecast plus shutdown
grace fits. Commit each completed block immediately, so a timeout loses at most
the unfinished block. Recovery without numerical progress is bounded to the
existing one extra attempt per block; repeated failure produces a repair task.

Acceptance: forced interruption after a known block, followed by resume, executes
only missing blocks; changing identity rejects reuse; corrupt data is rejected;
spent time, including failed work and recompilation, is charged exactly once.

## 4. Correct forecasts and cumulative phase allocation

Maintain distinct estimates for an initial viable cost scenario, the actual
selected procedure, and maximum-length stress scenarios. For the ensemble,
price the actual two-chart, beta=(0, 0.5, 1) program at L=3 for initial conditional
planning. L=3 is the existing minimum search length, not a newly selected kernel
or a convergence claim. A template width must be recorded explicitly; use an
existing declared width with its cost uncertainty, then measure the actual
selected widths/maps. Reuse the historical L=25 result only with its recorded
contention qualification. Do not approximate mixture cost by multiplying a
single-chart cost or scaling L=25 by 3/25.

After complete tuning and fresh verification, measure the exact frozen mixture
that sampling will execute, including each chart's own epsilon/L. Refresh both
the first posterior-assessment reserve and the chunk quote with this result.
Keep timing draws separate from posterior draws. If the exact verified procedure
is unaffordable, preserve it and stop or use another already authorized,
predeclared viable candidate; do not silently shorten L or lower evidence gates.

For serial execution, allocate each next work block under

`cap = min(stage remaining, campaign remaining minus downstream protection,
diagnostic remaining if applicable, calendar remaining minus stop grace)`.

The next block must fit the cap, including restart overhead. Protect the first
posterior check, reference and final assessment. Also account for all remaining
mandatory tuning scopes, rather than spending the whole tuning allowance on
the first of the ensemble's four scopes. Initial cost estimates are hypotheses;
if measured costs rise, refresh the forecast and preserve unfinished work.
Maximum caps are limits, not expected durations or mandatory up-front spending.

Record separate reasons for cooperative allocation exhaustion, hard external
timeout, infrastructure exception, capacity wait and completed candidate failure.
Only real infrastructure failures debit an infrastructure-repair reserve; all
work always debits the total ledger. Preserve historical charged time and do not
refund it. Do not retroactively restore the exhausted repair hold. Any future
repair suballocation must be an explicit transfer within the existing total,
with downstream funds still protected.

Replace the launch-count-driven diagnostic stop with a finite declared set of
diagnostic work records and bounded retries, still capped by the existing
diagnostic seconds. Currently five of six counted attempts are used; relabelling
another full rerun as a new phase must not reset that history or create funds.
Use actual completed work to determine progress and cost, not just process count.

Acceptance: budget-conservation tests, deadline clipping, partial-scope replay,
non-renewing retries, and a fixture where the measured short ensemble is fundable
while its L=25 stress scenario is not. A separate fixture must refuse sampling
when the actual verified mixture exceeds available funds.

## 5. Investigate the plain tuning gap before paying for ensemble training

The existing map is `direct-w16-lr0.0005-r0`. At epsilon 0.050625 all four chains
moved for every L in (3, 5, 9, 13, 18, 25); acceptance remained too high for the
declared tuning screen. At 0.0759375 at least one chain had zero movement in
every setting. The controller correctly refused those settings, but the saved
search has `refinement_rounds=0`, so it supplies no evidence inside this gap.

First inspect the existing per-chain target/energy traces and frozen-map scale.
Determine whether the discrepancy indicates rejected proposals, defective state
handling, or residual geometric variation. Do not infer poor whitening solely
from acceptance or treat an immobile chain's pooled acceptance as healthy.

If the checked target and traces remain valid, propose one finite interior
cohort through the existing public tuner: the geometric midpoint
`sqrt(0.050625 * 0.0759375) = 0.062002709114199195`, at each of the six existing
L values. This is a derived exploration hypothesis with six candidates, not a
root-finding guarantee: the failed endpoint is not valid opposing acceptance
evidence and acceptance need not be monotone. One interior cohort is a
convenience bound on new exploration; any further refinement needs its own
recorded evidence-based plan within remaining funds.

Use a new search identity and fresh measurement/verification streams. Preserve
the old search as completed evidence. Keep the current measurement lengths,
evidence rungs (1, 2, 4), movement/energy/status criteria and fresh verification.
Reserve the finite cohort's required work using observed same-L costs before
launch; if not affordable, record that result instead of running a shortened
unqualified substitute. A valid lower-cost setting may then reach sequential
sampling; it is not a statistically ranked winner.

This is a q20 search proposal, not a change to the global HMC acceptance policy.
The existing failed-interval exploration option handles specified nonfinite
proposal failures; the observed movement/acceptance conflict must not be
mislabelled to trigger that option. Use explicit supported epsilon proposals.
If an implementation defect is discovered, repair and regress it before any
such numerical trial. If the finite trial has no verified member, continue to
the permitted ensemble route only when its measured staged plan is affordable.

## 6. Validate and connect the master

Run focused CPU engineering tests with GPUs hidden, followed by the smallest
required trusted GPU/XLA checks. Reuse the actual worker-entry/checkpoint test
and add forced final-contention, pricing-interruption and budget-boundary cases.
Test an end-to-end known-target fixture through tuning, fresh verification,
sampling, reference and final decision, including a restart between phases.
The CPU fixture establishes engineering behavior, not q20 posterior validity.

Keep four-chain batching. Multiple-GPU candidate dispatch is a separate optional
performance improvement, not required to fix these stops. If added later, run
only independent peers in the same declared cohort concurrently, preserve
deterministic seeds and cohort closure, and charge the sum of GPU-worker wall
times. Three GPUs can reduce elapsed time; they cannot triple the authorized
GPU-hour budget. Sequential repairs still depend on their parent observations.

The repaired master must persist current phase, completed blocks, numerical and
resource outcomes, next action, cumulative spend, remaining funds, and last
progress timestamp. `status` must clearly show running, waiting, repair required,
under-budgeted or complete. `ensure` stays idempotent and follows the reviewed
new execution directory. It must not restart a terminal failure without an
inspected repair, and repairable phase pauses should resume funded saved work
without asking the owner again. No external notification channel is added.

## Budget, numerical provenance and evidence contract

| Choice | Value or rule | Provenance / failure check |
| --- | --- | --- |
| Remaining campaign funds | 175,624.1062574485 s = 48.7845 h | Settled campaign-02 ledger; reconcile again before launch |
| Remaining included GPU diagnostics | 2,537.861876872681 s = 42.30 min | Same ledger; no extra GPU diagnostic hours authorized here |
| CPU engineering test cap | 600 process-seconds | Inherited bounded-test convenience; debit the separate authorized CPU reserve after reconciliation |
| GPU diagnostic allocation | At most the remaining included diagnostic seconds, and only priced missing checks | Derive individual caps from block costs plus overhead; no whole-price rerun by default |
| Safety reserve multiplier | 2 | Inherited engineering hypothesis, not a probabilistic runtime bound |
| Stop grace | 5 s | Existing supervisor policy; included in caps and deadline checks |
| Timing probe shape | Four chains, four transitions/call, two calls initially | Existing pricing protocol; separate compilation from steady timing and record weak timing evidence |
| Conditional ensemble length | L=3 | Existing minimum search length; not automatically selected for sampling |
| Exact sampling price | Actual verified maps and per-chart kernels | Measured after tuning, before retained sampling |
| Plain repair cohort | Six L values, one epsilon 0.062002709114199195 | Derived midpoint proposal; finite uncalibrated exploration, fresh evidence required |
| Posterior checks | Existing warm-up minimum 2,000; retained minimum 1,000; chunks 1,000; maxima 10,000 each; R-hat 1.05/1.01; bulk/tail ESS 400 and existing precision/health/reference checks | Inherited scientific protocol; unchanged by resource repairs |
| Final deadline | September 25, 18:00 Asia/Shanghai | Owner directive; no compute allowance implied by calendar extension |

Engineering pass: completed valid work survives resource transitions; resumes
do not repeat eligible blocks; units and source identities are correct; all
attempts stay within cumulative limits; next actions are recorded. Scientific
pass: one independently verified procedure passes the existing posterior and
reference contract. Engineering success, short timings and acceptance alone
cannot establish posterior convergence, exact nonlinear-likelihood correctness,
transport whitening, method superiority or a finish ETA.

The old contended forecast gives about 19.04 hours for training, 96.57 hours for
the L=25 first posterior assessment, and 1.34 hours for initial reference checks,
before tuning. Those are scenario reservations, not an executable 116.94-hour
request and not the budget proposed here. Until the short/exact mixture is
measured and tuning succeeds, a full-campaign completion forecast is unsupported.

## Skeptical review and execution order

Audit findings were material: increasing a timeout cannot repair discarded
completion; changing resource treatment cannot supply missing clean timings;
measuring L=3 cannot establish that a verified L=3 kernel exists; source edits
can invalidate otherwise reusable state; switching to the ensemble does not
answer the untested plain step interval; and more GPUs do not create compute
funds. The plan addresses each with explicit identities, separate result roles,
measured staged costs, a bounded fresh tuner cohort and cumulative accounting.

Recommended order: isolate sources and reconcile evidence; implement completion,
pricing-resume and budget repairs; run the CPU end-to-end regressions; investigate
the saved plain failure; execute the finite plain repair only if affordable;
then either continue verified plain sampling or obtain the missing short-ensemble
price and proceed to ensemble training/tuning. Refresh costs after every phase
and stop at the first validated estimate. The exact verified procedure is always
repriced before an expensive sampling allocation.

Review verdict: suitable as a repair proposal. Numerical execution remains
conditional on passing the implementation checks, a valid scope import, and
an affordable finite next phase. A real target/source/health invalidity,
exhausted total/diagnostic funds, or the calendar deadline is a continuation
veto. A rejected numerical candidate is a repair/fallback trigger, not automatic
rejection of either research direction.

September 22 execution review and current status are recorded in
[the recovery result](bayesfilter-q20-recovery-execution-result-2026-09-22.md).

Implementation artifacts use a fresh versioned directory beneath
`docs/plans/artifacts/q20-recovery-and-affordability-2026-09-22/`, preserving a
source/import audit, engineering test results, phase plan, cumulative budget,
run manifest and result note. Record actual commands, environment, seeds,
GPU/memory/XLA settings, wall times and next decisions there. This proposal
has created no numerical run and has not changed the active campaign pointer.

## R7. September 22 ensemble pricing import repair

At 05:49 Shanghai the finite plain midpoint cohort completed without a verified
member. This is candidate rejection, not invalidity of the target or a rejection
of either permitted research direction. The automatic ensemble fallback passed
beta-0.5 GPU qualification, then stopped at 05:52 with `KeyError: timing_quality`.
In `price_complete`, map construction overwrote the historical timing `origin`
captured by the import helper. The second temperature consequently supplied map
metadata where timing provenance was required. Baseline: the preserved
`/tmp/BayesFilter-q20-recovery-20260922-r1` source and campaign-04 receipts.

Repair only host provenance bookkeeping, and explicitly carry the completed
plain rejection across the source refresh so that its expensive tuning is not
repeated. Check its completed result and saved evidence before skipping that
method. Preserve the original trained state through the checked importer, all
spent time, historical files and acceptance/health/posterior criteria. Use a
fresh source snapshot and successor directory; current-source qualification is
required before new GPU work. Existing numerical identities are not restamped.

Engineering criterion: a two-temperature ensemble pricing fixture imports each
timing block with its original resource qualification, retains distinct map
provenance, and completes the missing exchange measurement. Interrupted or
invalid tuning must not be mistaken for completed rejection. Test replay and
budget conservation. Tiny CPU fixtures hide GPUs and support no posterior or
performance claim. The inherited 600 process-second engineering cap applies;
charge actual CPU usage to the existing CPU reserve. GPU checks and the missing
pricing block consume the remaining 2,228.768696 diagnostic seconds inside the
remaining 157,018.617735 campaign seconds; no new allowance is created. Reconcile
the remaining block reservation and already charged failed attempt before launch.

Skeptical audit: accepting a missing quality field with a default would conceal
lost provenance; rerunning plain tuning would answer an already settled question;
copying main's unrelated numerical edits would invalidate the baseline. Separate
variables, a focused multi-temperature test, checked completed-method carryover,
and an isolated source refresh address those risks. Continue only if current
qualification and the existing staged affordability calculation pass. Candidate
rejection is a fallback trigger; corrupted evidence, numerical invalidity, an
unfunded next phase, exhausted allowance, or the calendar deadline stops execution.
Record the repair, checks and actual continuation in the existing recovery result.
