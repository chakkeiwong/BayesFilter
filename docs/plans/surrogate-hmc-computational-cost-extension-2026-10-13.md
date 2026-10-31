# Surrogate-HMC computation repair: additional 24 GPU hours

Execution refresh, 2026-10-15: the executable master now supports the accepted
**epsilon=1/five-iteration OT continuation**, with fresh per-model mixture
calibration and automatic phase-local/campaign-level next-phase refresh.
The completed SIR reference comparison is settled and is not a new stopping gate.
See [current continuation plan](surrogate-hmc-accepted-ot-continuation-2026-10-15.md).

Remaining authorized compute before this phase: 51,669.679990 seconds (14.352689 h).
The 86,400-second campaign ceiling is unchanged; attempt cap is 140 to accommodate
fifteen planned workers and bounded repairs. N4032/K2016, FP32 default, TF32 off,
GPU/XLA and the existing SIR mixed-precision exception are retained. Latest saved
controls seed fresh tuning at offset 100; no old selection is current-source
admission. This continuation leads directly to actual HMC mechanics/cost checks.

The goal remains testing surrogate HMC. Full posterior inference is unfinished:
the normal prior in q remains a mechanics fixture pending the target decision.
That decision does not block this computational phase. Live execution state:
[phase plan](artifacts/surrogate-hmc-cost-repair-24h-20261013-01/next-phase-plan.json).

## Preserved earlier execution record

The following dated records describe their original sources and controls.
Their completion statements do not apply to the new OT continuation.

Owner authorization: “you have 24 more GPU hours. refresh the master program and continue.”
This continues the existing five-model master. The goal is testing surrogate HMC;
this allocation first closes its unfinished computation repairs. Mechanics cannot
establish posterior validity.


## Current completion, 2026-10-14
The four computation phases are complete, with practical long-chain cost still an
open finding rather than an assumed success. Corrected-source scope selection,
mechanics, paired actual-consumer comparisons and selected-setting FP64 comparisons
pass all five models. The final master command exited 0 with no issues:

~~~bash
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_performance_repair.py finalize --campaign-ledger docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/budget.json --advance-dir docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/fixed-design-scope-advance-01 --paired-dir docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/fixed-design-consumer-paired-01 --selected-precision-dir docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/fixed-design-selected-precision-01 --output docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/fixed-design-computation-result-01
~~~

This host-only command does not launch GPU work or overwrite a prior result.
Budget: 8.3656 h used; 15.6344 h remain; 99/120 attempts. No active worker.
Result: [computation report](artifacts/surrogate-hmc-cost-repair-24h-20261013-01/fixed-design-computation-result-01/result.md).
158 CPU tests and 3 subtests pass in computation-terminal-regression.log.

The existing master remains incomplete. Before full posterior sampling, resolve
whether the recorded benchmark-normal prior becomes the intended benchmark target
or is replaced by model-specific scientific priors; an owner question is pending.
The prior affects the target and cannot be silently chosen from a mechanics fixture.
Budget authorization is already granted. Public-authority target-specific HMC
tuning, practical trajectory cost and cumulative posterior/reference diagnostics
remain. Failed cost candidates did not invalidate the research direction. No
posterior launch occurs until the target decision is resolved.

Earlier execution entries below are a preserved chronology; use this section,
the master opening and the active checkpoint for current status.

## Scope, budget and execution
FP32, TF32 off, N4032/K2016; LGSSM/KSC/predator/SIR T50, range T20.
Use the existing optional SIR FP64 recursion with FP32 pair kernels; range
bearing uses uniform FP32 after the shared ancestor-selection repair.
The current covariance-proposal force is not canonical Contract-E LEDH admission.
No target, particle count, reset-iteration, mixture or correction changes are cost
repairs. Analytical all-parameter tangents remain in one observation recursion.

New root: artifacts/surrogate-hmc-cost-repair-24h-20261013-01/.
New ledger: budget.json, 86400 worker-wall seconds including compilation/failures,
maximum 120 attempts, one GPU worker at a time. Prior ledger remains unchanged.
Use the escalated tftwogpu absolute interpreter, memory growth, stable XLA
signatures, and unique outputs. No package changes or new hardware.

Executable master: scripts/run_surrogate_hmc_performance_repair.py
run/advance/regression, explicit --campaign-ledger, --output and bounded
--per-model-timeout. Extend the component diagnostic for full sequences and
the paired diagnostic for preserved batch-kernel comparison. Exact commands,
source snapshots, seeds, devices, settings, wall times and results are recorded
by workers. Refresh next-phase-plan.json after every phase.

## Evidence contract and research intent
Question: reduce actual all-parameter force cost without changing the finite
filter, then verify savings in the surrogate-HMC consumer.
Baseline: preserved source from extension start, including compact covariance
and retained-force repairs, with identical data/designs/controls/precision.
The rejected full N-by-N tangent cache is not repeated unchanged.

Promotion: finite full-horizon values and every score within owner 1% scaled
arithmetic tolerance abs(new-old)/max(1,abs(old)), no newly invalid parameter
row, bounded memory, measured saving in fused consumer. Record finer errors.
Compilation, warm latency, allocator and host RSS are separate. Timings from
few repeats are descriptive; no statistical ranking is asserted.
FP64/finite differences test arithmetic/derivative consistency. Exact Kalman
and refined KSC test model approximation separately. Particle/tangent errors
and component shares explain behavior; they do not prove inference validity.

Promotion vetoes: nonfinite results, omitted derivatives, newly invalid rows,
changed target, corrupted artifacts or missing provenance.
Repair trigger: failed candidate or local infrastructure failure.
Continuation vetoes: budget exhaustion, irrecoverable invalid experiment or
material scope change. Candidate failure is not direction rejection.
Nonclaims: no general speed superiority, posterior correctness, convergence or
canonical LEDH admission from component diagnostics or tiny chains.

Constructed cheap comparators: unchanged fused force (repair baseline);
value-only filter (score overhead); exact Kalman for LGSSM; refined KSC grid;
existing bootstrap and simple Gaussian filters where available. Report models
and healthy/off-centre regions separately. Current work promotes implementation
equivalence/speed only, not the complex method over these scientific comparators.

## Phases and soft budgets within the hard 24-hour cap
1. Full-sequence shared proposal/density/reset profiling and fused parity, 2 h.
   Preserve per-time costs, accumulated value/score, final clouds and health.
   Staging alters fusion; report staged/fused ratio and label attribution diagnostic.
2. Bounded transport/mixture repairs: reuse K-sized calculations, combine repeated
   reductions, remove large derivative broadcasts while retaining every term, 10 h.
   Start tiny FP64/finite-difference checks, then paired N4032 GPU tests.
   Preserve/revert failed candidates; do not repeat the rejected full tangent cache.
3. Actual consumer call-chain audit, all five full horizons/off-centre rows,
   repeated paired timing, compilation/memory and FP64 checks, 6 h.
4. Freeze final source; refresh affected scope tuning, finish predator selection,
   and short four-chain consumer mechanics/cost, 6 h. Genuine HMC tuning must use
   public tuning authorities. Normal benchmark prior and L1 mechanics are
   non-inferential. Full posterior needs the existing adaptive warmup/cumulative
   sampling protocol and a valid target/prior; do not substitute tiny chains.

Unused phase time carries forward. Later phases are existing master work, not
a direction change. The new allocation is an upper bound, not a duty to burn it.

## Defaults, pre-mortem and skeptical audit
N4032/FP32/1%: owner choices. K2016: repository chunk policy. Fixed inherited
controls: paired baselines, not fresh tuning authority. Mixed precision:
existing optional hypothesis, tested in its own scope. Two rows: diagnostic
region, not posterior coverage. No pfor/autodiff claim score/scalar-row fallback.
A staged speed saving can disappear in fused XLA: fused timing decides.
Reordered arithmetic can amplify through time: full-horizon FP64 and per-time
checks decide. GPU contention must be recorded and avoided.

Audit passed with corrections: first-observation attribution is insufficient,
so phase 1 spans the sequence; old source-bound tuning cannot authorize new
consumer claims, so phase 4 refreshes it; the prior failed cache is preserved;
LG/KSC model-score gaps and HMC convergence remain separate from timing.

## Automatic remaining-phase continuation

The active fresh-scope controller is followed by
scripts/run_surrogate_hmc_consumer_followups.py. This serial controller waits for
the current five-model advance to finish and the shared ledger to have no active
worker. It then runs paired actual-consumer comparisons with each available
fresh selection, using final-scope-execution.json. Per-model caps are2400s
within the existing86400s ledger. Its output is consumer-paired-followups-01;
its next-phase-plan.json records all completed and deferred models. See
automatic-followup-audit.md (148 CPU tests passed). This is existing phase3
work after the fresh selections from phase4, with no scientific source change.

## 2026-10-13 refresh after fresh-scope completion

The fresh tuning, short four-chain mechanics and actual consumer-cost phase
finished for all five selected scopes in final-scope-advance-01. The automatic
paired comparisons finished in consumer-paired-followups-01: all five pass the
1% scaled force-parity screen against the preserved source. The observed
candidate/baseline warm ratios are 0.797 LGSSM, 1.111 KSC, 0.866 predator-prey,
0.660 SIR and 0.835 range bearing. These few timings are descriptive; KSC is a
measured cost concern, and no all-model speed-promotion claim follows.

Phase 3 now also includes the selected-consumer arithmetic check under
selected-consumer-precision-contract.md. The serial executable
scripts/run_surrogate_hmc_selected_precision.py checks each frozen selection
on the saved HMC data at q=-0.15,-0.05,0.05,0.15 in every coordinate. It
compares likelihood, every physical score and coordinate force to the *same*
finite program in FP64 at identical physical inputs, and verifies actual
consumer wiring separately. Its per-model timeout is 3600 seconds, with no
budget expansion. A failure triggers localization; no retuning on this holdout.
Successful checks do not establish posterior sampling, convergence or the
canonical LEDH route. Follow the result with a terminal cost-and-arithmetic
assessment and refresh the posterior-phase plan only within an admitted target
and public HMC tuning authority.

## 2026-10-14 current computation phase

All five selected actual-consumer arithmetic checks pass the 1% same-program
FP64 screen. KSC repeated cost ratios 1.0054 and 1.0141 do not reproduce the
initial 11% slowdown and show no measured gain. Three transport primitive
candidates failed their predeclared cost nomination thresholds; none changed
the runtime. Continue phase 2 with sir-local-arithmetic-plan.md to localize
SIR's all-FP32 discrepancy (48.69% at the saved nominal row). The selected
mixed route passes, but restoring FP32 could reduce its remaining cost.
This is computation repair; posterior HMC remains unexecuted.

## 2026-10-14 fixed-design repair

The SIR localization exposed a real comparison/implementation defect: FP32
rounding before floor selected different proposal ancestors than FP64 from the
same representable uniforms. The events at observations 1 and 7 coincide with
the large local proposal differences. Follow
`artifacts/surrogate-hmc-cost-repair-24h-20261013-01/ancestor-selection-repair-plan.md`.
Both actual draw paths now share binary64 discrete-index selection; continuous
kernels remain in their requested precision. This changes the old FP32 finite
design at boundaries and therefore requires refreshed arithmetic comparisons,
fresh tuning identities and consumer checks. It is phase 2 of the existing
master, not a change of scientific direction. 150 CPU regressions passed.
The corrected SIR T10 GPU localization is running; full-horizon all-model
precision checks follow, then any remaining targeted repair. No posterior
convergence or canonical admission is implied.


## Corrected-source continuation, 2026-10-14

The shared inverse-CDF selector now uses FP64 for discrete ancestor selection.
Four full-horizon uniform-FP32 scopes pass the 1% arithmetic screen. SIR retains
a 4.50994% maximum scaled score discrepancy, despite a 2.13425e-7 likelihood gap.
FP64 here is the same-program arithmetic comparator, not an independent oracle.
Previous tuning and selected-consumer checks are source-stale.

Saved-input localization identifies transport tangent arithmetic as the dominant
local residual. Gauge centering and direct pairwise tangent candidates fail the
predeclared fivefold reduction criterion. Neither changes the runtime; see
`artifacts/surrogate-hmc-cost-repair-24h-20261013-01/transport-arithmetic-review.md`.

Continue the existing master with `fixed-design-scope-execution.json`: fresh
calibration/validation seed offset 60, uniform FP32 except SIR mixed recursion,
N4032/K2016/TF32off/XLA. Launch `advance` into `fixed-design-scope-advance-01`,
using corrected full-horizon precision, then fresh selected-precision and paired
consumer checks. Preserve next-phase plans, one active GPU worker, and budget
accounting across every attempt. A failed candidate triggers the planned local
repair; it does not reject the research direction.

The preserved batch-cost baseline imports the shared `fixed_design_indices`, so
both arms use corrected discrete selection. Pair identical data, designs,
controls, and precision. This comparison measures the computational repair only,
not the scientific accuracy of either method.

Skeptical audit: fresh artifacts bind current sources and disjoint calibration /
validation partitions; scope-specific precision is explicit; the 1% criterion is
unchanged. No promotion veto becomes a continuation veto. These checks establish
neither posterior validity, statistically supported speed ranking, nor canonical
LEDH admission. Full posterior sampling still requires the actual target/prior
and a compatible public artifact-authority tuner. The remaining worker-wall
budget, rather than elapsed calendar time, limits this continuation.
