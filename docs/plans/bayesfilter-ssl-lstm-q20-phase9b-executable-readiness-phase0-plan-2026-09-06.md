# SSL-LSTM q=20 Phase 9B M4-P0 Executable Readiness Plan

Date: 2026-09-06  
Updated: 2026-09-09 (Asia/Shanghai)  
Status: `M4_P0_GPU_RUNNING_P1_AUTO_AFTER_READINESS`  
Parent master: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`  
Phase 9B plan: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-sequential-validation-plan-2026-09-05.md`  
P1 repair plan: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-p1-sequential-canary-plan-2026-09-05.md`  
P1 repair result: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-p1-sequential-canary-result-2026-09-06.md`  
Prior source-authority preflight: `docs/plans/artifacts/ssl-lstm-q20-phase9b-sequential-validation-2026-09-05/p0-source-preflight/run_manifest.json`

## Current Phase 0 disposition and approved amendment

The prior four-hour diagnostic is closed historical evidence. The owner now
authorizes a fresh eight-hour per-arm cap (`28,800` seconds) and a fresh
`86,400` aggregate GPU-worker-second budget under
`bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-8h-amendment-plan-2026-09-09.md`.
Its prepared campaign root is
`docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-24gpuh-20260909T135803Z/`.
GPU work started at 2026-09-09 16:00:10 UTC (September 10, 00:00:10
Asia/Shanghai), service `bayesfilter-q20-phase9b-24gpuh-20260909.service`.
The previous ledger is not edited or combined with the new one. Read the
amendment result for live placement, stage and budget status.

The old strict reserve-inclusive forecast of `16,202.12` seconds fits the new
cap, but this is an entry arithmetic check, not a Phase 0 pass. The remaining
work is fresh current-source GPU recovery/health and timing. Prepared-ledger
initialization and actual P1 checkpoint/parallel integration now pass 128 CPU
tests. Keep full sampled-state status recomputation; status reuse is optional,
not an execution blocker. The actual P1 entrypoint's `--campaign-root` route
performs Phase 0 and enters bounded P1 automatically after all checks pass.
The amendment plan governs this route, not the old r14/legacy closeout schema.

## September 9 four-hour diagnostic disposition (historical)

September 9 terminal result:
`bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-result-2026-09-09.md`.
The owner-authorized four-hour arm ceiling and separate ten-GPU-hour allocation
were implemented. Both hard-interruption canaries, both full two-call timing/
health measurements, and both endpoint-bank component profiles completed. Do
not repeat these completed stages unchanged. The terminal CPU suite passes 152
tests. No worker from that campaign remains.

| Phase 0 requirement | Terminal evidence | Disposition |
|---|---|---|
| P0-I interruption and persistence | Actual two-arm mid-call SIGKILL/resume gives exact sample/trace equality; 34 committed bundles and 372 tensors verify; first archive precedes next call; parent accounting reconciles | Passed for checkpoint-aware diagnostic/shared-controller route, not every legacy consumer or hardware failure |
| P0-J complete sampled-state health and startup | Two actual 500-transition calls per arm, all four required health checks pass; growth, one-trace XLA and allocator receipts preserved | Passed; proposed-state scores/native divergences remain unexposed, not claimed checked |
| P0-K component localization | Two endpoint banks, four warmed repeats, physical/transformed target, transport and status measured on both GPUs | Complete; accepted-state status reuse is nominated, not implemented or validated |
| P0-L affordable complete schedule | The old strict forecast failed the four-hour cap; the new authorized cap is 28,800 s per arm and 86,400 aggregate seconds | Pending fresh repair/remeasurement; do not treat the cap change alone as a runtime pass |
| P0-M process-parallel tuning | Independent factor/strict workers pass real GPU tuning; September 9 runtime also overlaps separate processes on GPUs 1 and 0 | Passed for measured routes; candidate pairs inside each worker remain sequential |
| Actual P1 integration and current-source readiness | Shared checkpoint API is implemented, but old P1 consumer/audit/closeout is not current launchable evidence | Integrate and validate the actual consumer before issuing a Phase 0 pass |

The closed historical ledger consumed 11,352.55558313601 of 36,000 aggregate
worker-seconds. The new amendment ledger has 86,400 seconds, zero consumed and
zero reserved. Prior ledgers remain unchanged and their balances are not
transferable. The executed four-hour plans remain frozen at their launch hashes;
the amendment plan governs only new work.

### Detailed cost-repair and integration work

The next hypothesis is that carrying already-computed accepted-state target
status through HMC can avoid another expensive target evaluation. Profiling
localizes the cost but does not establish equivalence or savings in the fused
controller. The source anchors and actual component means are in the result.

1. Audit the existing status-carrying mechanics and the HMC tuning interface.
   Specify exact accepted/rejected-state status selection, value/score and seed
   equivalence against the current source-bound controller. No target, bridge,
   batch shape, precision, schedule, health threshold or tuning authority changes.
2. Implement the smallest optional reuse path and connect the actual P1 consumer
   to the tested durable checkpoint route. Preserve arm-level process execution,
   chunk commits, full health, external timeouts and parent-owned accounting.
   Rejected-proposal status must never replace the accepted state's status.
3. Run focused CPU rejection/status/trajectory, stale-handoff, corruption,
   interruption and consumer-integration checks before any GPU work. Freeze the
   new source revision; use fresh manifests/output roots, preserving old evidence.
   A changed numerical/tuning scope requires fresh verified tuning, not edited
   handoff hashes. Failure of equivalence rejects the repair, not the target.
4. Declare bounded GPU canary and two-arm full-call remeasurement allocations
   under the remaining ledger before launch. Require numerical/status parity,
   real interruption continuation, memory growth, one stable XLA trace and full
   health. Use eligible non-display GPUs first under the unchanged 40%/5 GiB
   headroom policy; unrelated contexts remain untouched.
5. Recompute six-chunk setup-inclusive costs, one-chunk interruption reserve and
   cleanup grace from the repaired full calls. Both arms and their aggregate
   must fit the then-remaining ledger. If not, preserve the result and state the
   remaining cost/budget limitation; no automatic cap increase or reserve waiver.
6. Issue a passing M4-P0 closeout only for the actual tested P1 consumer with
   matching source/audit evidence and a feasible complete schedule. Then refresh
   P1's exact command/seed/output/budget plan. P2 and posterior/default promotion
   remain subject to their own later scientific gates.

This is an engineering cost repair, not a new sampler or statistical ranking.
Recovery, health and component diagnostics cannot stand in for downstream
posterior/reference agreement. Each new serious measurement still needs its
concrete bounded command and evidence contract before execution.

## September 7--8 diagnostic history

The entries and original blocker-discovery table in this section preserve what
was known at the time. Their old caps, incomplete-runtime statements and calls
for new funding do not supersede the September 9 disposition above.

September 8 update: P0-M process-parallel tuning now passes a real GPU startup
smoke and complete fresh factor/strict cold-scope tuning on non-display GPUs
1 and 0. Result:
`docs/plans/bayesfilter-ssl-lstm-q20-phase9b-parallel-tuning-gpu-result-2026-09-08.md`.
The new user-authorized 14,400-second campaign consumed 844.217169 aggregate
worker-seconds, with 13,555.782831 left and no active reservations. The original
5,200-second ledger is unchanged. No full-controller timing run or P1 closeout
was issued; the P0-I/J/K/L requirements below remain open for their actual
controller workload. Do not repeat the already-passing tuning diagnostic just
to consume the new reserve, or treat its handoffs as full readiness clearance.

This section supersedes the pre-run pending/authorization wording below. The
September 7 diagnostic executed on eligible non-display GPU 1, completed the
factor arm's two calls and archives, and stopped during strict-arm work.
Factor first/steady times are 1,360.86/1,390.70 seconds; six chunks extrapolate
to 8,314.36 seconds before overhead, above the unchanged 2,600-second arm cap.
The settled 5,200-second campaign has only 183.80 seconds left. M4-P0 did not
pass; there is no complete strict result or P1 launchable closeout.

Current result:
`docs/plans/bayesfilter-ssl-lstm-q20-phase9b-multigpu-continuation-result-2026-09-07.md`.

| Repair | Problem exposed | Required implementation/check | Exit evidence |
|---|---|---|---|
| P0-I interruption and stage records | The launcher omitted the external timeout; the arm receipt appears only after both calls; Python TERM handling did not finish during compiled work | Launch under an external wall bound; persist call-start/call-end records and archive a completed call before another expensive call; settle from the parent on forced exit | Synthetic termination/accounting tests, one preserved bounded runtime receipt, no double debit |
| P0-J complete health and startup evidence | The factor receipt checks finite states/target values and movement, but does not persist full status/log-acceptance/declared energy checks or the startup memory-policy return | Use the shared trace-health semantics; persist growth verification before expensive work; classify finite extreme energy diagnostics consistently rather than invent a threshold | Focused invalid-trace tests and complete per-call runtime records; movement alone cannot close this row |
| P0-K cost localization | Similar first/steady costs invalidate a compile-only explanation for this attempt | Under a new declared allocation, profile target/score, transport, status evaluation, and the controller at unchanged target/shape/precision/XLA; test source-equivalent repairs before timing them | Per-transition breakdown and reproducible source-equivalence checks |
| P0-L affordable complete schedule | Factor alone fails the arm forecast; strict timing is incomplete; 183.80 s remain | Obtain an explicit compute decision; fresh two-arm measurements must fit both the remaining aggregate and arm caps; retain all historical debits | Passing complete-schedule forecast and fresh M4-P0 closeout before P1 |

The order is routine instrumentation repair and CPU checks, a concise profiling
plan with a new compute allocation, bounded measurements, and only then a
complete P1 feasibility decision. No target, posterior criterion, warmup length,
or precision may be weakened to fit the old cap. No new serious GPU run is
authorized by the historical 3,307.39-second allocation. This failure concerns
readiness/resources and incomplete evidence, not rejection of the factor
candidate or the research question. P2 remains independently blocked.

## P0-M process-level tuning parallelism

At discovery, the legacy runners were serial at the scope, arm and candidate-
evaluation levels; the reusable TensorFlow runner pool was not multiprocessing.
The separate-process diagnostic and its funded GPU validation now close this
gap for the measured tuning route. The legacy P1 consumer still needs explicit
integration rather than inheriting concurrency from a runner-pool name.

The binding repair plan is
`docs/plans/bayesfilter-ssl-lstm-q20-phase9b-parallel-tuning-execution-plan-2026-09-07.md`.
It establishes the following Phase 0 requirements:

- use separate OS processes, with at most one process per selected GPU;
- select eligible non-display GPUs first under
  `bayesfilter_non_display_first_load40_headroom5g_v1`, and use a display GPU
  only when no non-display GPU is eligible; queue excess work rather than
  using display merely to fill a slot;
- pin each worker to one physical GPU UUID and set/verify memory growth before
  TensorFlow initialization;
- give each worker a fresh output root and disjoint seed namespace;
- keep measured-joint-grid candidate selection inside the worker until every
  declared pair is measured; do not merge partial candidate rows in the parent;
- let the parent own timeout, termination, budget settlement, and complete
  receipt reconciliation; and
- split a larger scope set into successive waves rather than sharing a GPU.

Coordinator implementation and CPU checks alone did not reopen P1 or make the
September 7 partial factor receipt claim-bearing. The subsequent funded GPU
wave passed, as recorded below; affordability and actual P1 integration remain
separate requirements.

September 8 integration: the tuning-only entrypoint is now
`docs/benchmarks/run_ssl_lstm_q20_phase9b_parallel_tuning_2026_09_08.py`.
Factor and strict each run their fresh chart construction and complete scope
tuning in separate subprocesses, with private roots and source/seed bindings.
Admission budgets an inherited 4 GiB worker peak in addition to the owner's
5 GiB free headroom. The coordinator rechecks live free memory, reserves both
worker caps, and sums their individual lifetimes including termination.
CPU tests cover actual simultaneous subprocesses, worker failures, ignored
TERM, artifact loss, incomplete grids, and once-only settlement. The old ledger
is not changed. The subsequent funded GPU validation also passed (latest
runtime result above); P0-I/J/K/L and P1/P2 are not closed by this tuning-only
validation. Candidate-level tuning and HMC time steps
inside each worker remain sequential. The implementation result is
`docs/plans/bayesfilter-ssl-lstm-q20-phase9b-parallel-tuning-implementation-result-2026-09-08.md`.

## Decision and purpose

The earlier Phase 9B P0 source-authority preflight passed, but that result did
not establish that the P1 launcher can execute a complete sequential arm. The
three P1 attempts then exposed harness defects, and the repaired runner's
observed first sequential chunk is too expensive for the existing budget
forecast. It is therefore correct to make resolution of executable-readiness
blockers a new Phase 0 before another P1 launch.

This plan is the binding `M4-P0` gate. It makes the smallest Phase 9B P1
execution technically admissible without using P1 draws as evidence. It does
not perform posterior validation, choose chart-quality thresholds, promote a
sampler, or decide whether the factor route is scientifically better.

No material P1 GPU launch is valid until this plan has a passing closeout
receipt. CPU tests, static audits, and source inspection are allowed while the
gate is active. The owner instruction to continue authorizes one bounded
runtime-readiness diagnostic within the existing campaign remainder; it does
not authorize P1 until the measured complete two-arm schedule fits the ledger.
The diagnostic requires a fresh trusted placement inventory under
`bayesfilter_non_display_first_load40_headroom5g_v1`. A cap increase requires
the user's compute-budget approval; routine repairs within an unchanged
authorized budget do not require another approval token or review chain. A
failed diagnostic is classified as a harness, interface, route-governance,
resource, or provenance failure before scientific interpretation.

The September 7 resumption leaves runtime measurement pending and the complete
P1 budget conditional on that measurement. The prior trusted check at
`2026-09-07T07:45:41Z` found GPU 0 busy, but the current placement amendment
allows a fresh eligible non-display GPU and a display fallback. The 5,200-second
campaign has an estimated historical debit of 1,832.61 seconds, leaving at
most 3,367.39 nominal seconds. One diagnostic may use at most approximately
3,307.39 seconds, retaining a 60-second settlement margin. P1 and P2 remain
blocked until the diagnostic and closeout gates pass.

## Research-intent ledger

| Item | Binding definition |
|---|---|
| Main question | Can the current q=20 P1 program execute one fresh factor arm and one fresh strict-comparator arm through the repository sequential controller with durable, identity-consistent, budget-valid evidence? |
| Mechanism under test | The repaired P1 runner, the shared `bayesfilter_neutra_sequential_hmc_v1` controller, the fixed-transport tuner, and the factor/strict same-scope handoff boundary. |
| Exact baseline | The P1 chart-0, beta-1.0 cold scope with four chains, the declared warmup/retained schedule, fresh scope-bound tuning for each backend, TensorFlow/TFP, a GPU selected by the active placement policy, XLA, TF32, and verified memory growth. |
| Primary Phase 0 criterion | Every executable-readiness blocker below has a passing focused check or a recorded reviewed exception, and the resulting P1 command has a durable budget forecast that fits its declared cap and campaign ledger. |
| Promotion veto | Missing or stale source receipt; unresolved callback/axis/archive defect; absent stable TensorFlow signature; unclassified new route; stale audit receipt; seed collision; non-durable budget accounting; failed GPU memory/XLA/TF32 preflight; output collision; or a forecast that cannot fit the complete declared P1 schedule. |
| Repair trigger | A localized test, route-audit, signature, serialization, timing, or runtime-preflight failure that leaves the target, bridge, method, hardware class, and total authorized budget unchanged. |
| Continuation veto | A required source or artifact cannot be verified; the P1 contract would need a changed target, data, bridge, hardware class, privacy boundary, or scientific question; the required budget is unavailable; or the remaining choice is a material direction or compute decision not covered by the user's authorization. |
| Explanatory diagnostics | Compile time, steady-state chunk time, retracing count, allocator telemetry, memory-growth state, source hashes, route-ledger findings, and per-attempt wall time. These diagnose readiness and do not establish posterior validity. |
| Must not conclude | No posterior correctness, convergence, whitening, mode discovery, replica-exchange validity, sampler ranking, superiority, scaling law, production readiness, default readiness, or scientific validity. |

## Entry state

The following facts are fixed at entry and must not be silently rewritten:

- The source-authority Phase 9B P0 receipt passed at
  `docs/plans/artifacts/ssl-lstm-q20-phase9b-sequential-validation-2026-09-05/p0-source-preflight/run_manifest.json`.
- The source-synchronized factor receipt and audit remain the narrow numerical-
  backend authority. Their recorded hashes are
  `e0225382192ceb9da1e075cb9c7a91ed424e2c5c67adcfec7a0f34c05003a4e1` and
  `9848779a1fe1611f3b3acfb666bec8b08bef6a42296fe30e84d3762d892c5933`.
- The repaired P1 runner has six focused CPU regression tests and the current
  source-only P1 plan audit passes. Those checks do not replace an integrated
  post-repair sequential arm.
- Three P1 attempt roots and the fail-closed budget-preflight receipt are
  preserved under
  `docs/plans/artifacts/ssl-lstm-q20-phase9b-p1-sequential-canary-2026-09-05/`.
- The first observed sequential chunk took approximately `1,423` seconds after
  tuning. Six required chunks per arm imply approximately `8,538` seconds per
  arm before the strict comparator's completion is accounted for. This is a
  forecast from a single observed chunk, not a steady-state timing result.
- The nominal `3,367.39` seconds remaining in the old P1 cap is timestamp-based
  bookkeeping, not a durable campaign ledger and not authorization for another
  material launch.
- The full NeuTra route-policy test now passes after the new Phase 9B diagnostic
  and P1 runner were classified and the unrelated legacy paths were explicitly
  classified without making them active routes.

## Blocker-resolution table

| ID | Tracked readiness issue | Current evidence | Phase 0 resolution | Exit evidence |
|---|---|---|---|---|
| B0 | No complete post-repair P1 arm exists | The first two attempts failed at chart selection; the third reached the controller callback boundary; no sequential archive or strict result exists | Run focused callback, chart, axis, MCSE, archive, failure-provenance, and output-collision checks; then perform only the bounded integration-readiness diagnostic specified below | Focused suite passes and a typed readiness receipt shows the repaired path can reach the controller/archive boundary without the old defects |
| B1 | Stable graph contract repaired; target GPU behavior unmeasured | Explicit fixed state/seed signature and repeated-shape CPU regression are present | Preserve the signature and verify bounded tracing and XLA in the funded runtime diagnostic | Controller receipt records the signature, concrete-function count, repeated-shape behavior, and XLA status |
| B2 | Route classification repaired | The full route audit passes with new Phase 9B routes and historical/non-HMC legacy dispositions recorded | Preserve those dispositions and rerun the full audit after qualifying source changes | Full route-policy audit passes with no unclassified new path |
| B3 | Complete P1 feasibility remains unresolved | At most `3,367.39` nominal seconds remain from the shared `5,200`-second cap; the first-chunk extrapolation does not fit the `2,600`-second arm cap | Run one owner-authorized diagnostic with at most `3,307.39` seconds, then measure each cost and reconcile the complete schedule before P1 | Explicit diagnostic allocation within the existing ledger and a measured schedule fitting the authorized remaining aggregate and per-arm caps |
| B4 | Durable accounting implementation repaired; live campaign ledger not yet initialized | Both launchers use the shared ledger implementation; historical spend remains timestamp-estimated, and no new GPU attempt has run | Initialize with the historical debit on the first authorized launch, reserve before work, and settle every attempt once, including failures and overhead | Ledger attempt table, consumed/reserved/remaining seconds, and fail-closed pre-launch/chunk checks |
| B5 | Fresh GPU placement and valid runtime provenance remain open | Prior trusted probe at `2026-09-07T07:45:41Z` found GPU 0 occupied by PID `6826`; the active policy now permits eligible non-display selection and display fallback | Obtain fresh trusted inventory, select by UUID under `bayesfilter_non_display_first_load40_headroom5g_v1`, and record memory/XLA/TF32/allocator evidence without stopping existing processes | Passing trusted preflight and complete, explicitly non-scientific runtime manifest |
| B6 | Strict comparator path is unmeasured | Factor tuning reached the controller boundary; strict tuning and sequential execution were not reached | Exercise strict and factor profile construction independently in the readiness diagnostic, verify disjoint handoffs/seeds/output roots, and time both only under the declared budget | Both arm profiles pass identity and handoff checks; actual sequential comparison remains P1 evidence, not Phase 0 evidence |
| B7 | Source closure and audit receipts can become stale | The P1 plan and runner are coupled to the immutable P1 audit receipt; changing the plan invalidates its recorded hash | Rotate the P1 plan-audit receipt after this plan is linked, update the runner's receipt pointer, and record all changed source/plan hashes | Fresh P1 audit passes against the exact current plan and runner |
| B8 | Chart-quality thresholds remain unresolved | Current centered-density and pullback-score residuals are very large; no independent threshold receipt exists | Preserve this as a downstream P2 promotion veto. Do not invent a threshold in Phase 0 or use it to block technical readiness for the P1 mechanics canary | Closeout states explicitly that P1 can be executable while P2 remains scientifically blocked |

B0--B7 track executable readiness; some static repairs are complete while
runtime integration, budget, and device evidence remain open. B8 is a later scientific entry
blocker, not a reason to confuse a technical launch gate with posterior
validation. A passing Phase 0 does not waive B8.

## Default and assumption audit

| Choice | Provenance | Failure mode | Earliest diagnostic | Status |
|---|---|---|---|---|
| Phase 0 precedes P1 | Direct consequence of the three P1 harness failures and failed budget forecast | P1 could be relaunched with a repaired-looking runner but no valid integrated contract | Phase 0 closeout and readiness receipt | Binding governance repair |
| Explicit controller signature | TensorFlow graph policy and the current controller source | Retracing, unstable graph/resource cost, or an interface that works only for one incidental shape | Source inspection plus repeated-shape concrete-function test | Required repair unless a reviewed exception passes |
| Full route audit is reported separately from scoped P1 audit | Current route-policy output contains unrelated legacy migration debt | A green-looking scoped result could conceal unclassified active routes, or unrelated debt could be silently rewritten | Exact route-policy command and preserved error list | Binding transparency rule |
| Compile/steady-state decomposition | The single `1,423`-second chunk conflates compilation, tuning context, device setup, and steady execution | A cap could be widened for the wrong reason or a viable route could be rejected on a contaminated first-call estimate | Per-component monotonic timers and repeated same-shape call | Required measurement |
| Persistent campaign ledger | Global scientific policy requires durable attempt and budget provenance | Timestamp arithmetic can overstate remaining budget and allow a second process to collide with the same reserve | Ledger write/read/hash test and pre-chunk reservation check | Required infrastructure |
| Active-policy GPU, XLA, TF32, memory growth | Repository owner directives and prior factor receipt | Whole-device preallocation, device mismatch, stale placement, or non-XLA fallback can invalidate cost and launch claims | Trusted placement before TensorFlow import and allocator receipt | Required execution condition |
| P1 four-chain schedule | Repository sequential-controller policy and reviewed P1 plan | The schedule may be infeasible for this target or too short for later scientific claims | Phase 0 budget forecast; P1 diagnostics | Policy input, not target evidence |
| 5,200-second shared cap and 1,832.61-second historical debit | Cap inherited from the authorized P1 plan; debit estimated from preserved attempt timestamps | Treating the remainder as fresh funding, or the estimate as measured runtime, could overstate resources | Reconcile the attempt ledger and fund diagnostic plus complete P1 before launch | Authorized cap; estimated spend; no cap increase implied |

## Phase 0 work packages

### September 7, 2026 resumption audit

Source inspection found two additional execution defects. P0-H required P0-F
to pass before the GPU diagnostic, although P0-F needs that diagnostic's
measurements. The P1 runner also checked only the earlier source-authority P0
receipt; it did not enforce this executable-readiness closeout. Consequently,
the previous statement that GPU occupancy was the only remaining blocker was
too strong.

Resolve the dependency in this order: static authority/interface/graph/route
checks, budget-accounting infrastructure, trusted placement preflight, bounded
two-arm runtime measurements, complete-schedule budget reconciliation, and
executable-readiness closeout. Only then may P1 run. Add a CPU regression that
rejects a missing or stale closeout before TensorFlow import or campaign
reservation. Keep the source-only verifier callable by the Phase 0 diagnostic;
requiring an executable-readiness pass there would recreate the dependency
cycle.

The diagnostic also omitted first-call and steady-state times from the timing
mapping consumed by its forecast, which would cause a post-run `KeyError`.
Its per-output ledger minted a fresh 5,200-second allowance on every retry and
did not charge import or cleanup time. Repair these before any GPU workload:
test the forecast against complete synthetic timing records, use the existing
P1 aggregate ledger, reserve the diagnostic allocation before TensorFlow import,
and settle the whole attempt exactly once on success or failure.

These localized source repairs are implemented. The diagnostic now records
first-call and steady-state timings, charges the whole attempt once to the
shared P1 ledger, writes structured environment/trust provenance, and rejects
a stationary chain on either controller call. The CPU checks are engineering
evidence only. The audit preserves the cold-scope strict comparator, independent
tuning/seeds, unchanged target and XLA/TF32 policy, and the boundary between
timing diagnostics and posterior promotion; no scientific gate is relaxed.

The inherited 5,200-second cap remains the total P1/readiness repair campaign
cap, not an allowance per retry. Approximately 1,832.61 seconds were consumed
by the historical attempts, leaving at most 3,367.39 nominal seconds for both
readiness measurements and a later P1 attempt. The historical spend remains
explicitly an estimate, not a measurement retroactively made precise by a
ledger. Require an explicit diagnostic `--max-seconds` allocation within the
live remainder; the owner has authorized the bounded `3,307.39`-second
allocation for this continuation. No additional campaign or compute cap is
created. A passing static audit cannot close the two-arm timing and complete-P1
budget questions. If the measured schedule cannot fit, stop for a compute-
budget decision rather than minting another allowance. The closed factor
campaign's cap is not transferable. Obtain fresh device evidence immediately
before the diagnostic and do not stop existing display or other-user processes.

### P0-A: authority, date, and source closure

1. Verify the prior Phase 9B P0 manifest, source-synchronized factor manifest,
   source-synchronized factor audit, and their recorded SHA-256 values without
   regenerating or overwriting any receipt.
2. Record the actual execution date and time, the current Git revision, the
   dirty-worktree hash, the exact changed-source boundary, and the plan hashes.
   September 6, 2026 is the activation date, not a fixed date for later runs.
3. Confirm that the target signature, bridge identity, factor backend, strict
   comparator, tuner, controller policy, dtype, and XLA/TF32/memory policy agree
   across the current plan and source receipts.
4. Treat all pre-September-6 P1 failures as preserved infrastructure evidence;
   do not merge their partial scopes or handoffs into a new attempt.

### P0-B: repaired-runner contract closure

Run the smallest CPU-only checks that answer the known defects:

- chart-object selection for beta `1.0` and rejection of a missing beta;
- callback type and status-mapping semantics at the shared-controller boundary;
- conversion from controller samples `[draw, chain, parameter]` to diagnostic
  samples `[chain, draw, parameter]`;
- finite MCSE and posterior-mean diagnostic serialization;
- disjoint role, arm, and attempt seed namespaces;
- `run_start.json`, incomplete-failure provenance, and explicit missing
  allocator telemetry;
- refusal to overwrite an existing output root; and
- fail-closed pre-chunk budget forecasting.

The six existing focused tests are necessary but not sufficient if the
controller interface changes. Add or update the narrowest test that proves the
actual callback object accepted by `run_sequential_neutra_hmc`, rather than
testing only a mock with a compatible name.

### P0-C: stable TensorFlow graph contract

Inspect and repair `_build_batched_hmc_program` in
`bayesfilter/inference/neutra_hmc.py`.

The preferred implementation passes the fixed state shape into the builder and
decorates the compiled function with an explicit signature equivalent to a
`float64` state tensor of shape `[chain_count, dimension]` and an `int32` seed
tensor of shape `[2]`. The signature must be stable for every P1 chunk; changing
the number of results must create only the declared bounded set of programs,
not unbounded retracing.

The check must record:

- the exact input signature and state/seed shapes;
- concrete-function count before and after repeated same-shape calls;
- whether the compiled path is XLA-enabled;
- finite and movement checks on the tiny reference result; and
- peak host/device allocator readings where the runtime exposes them.

If an explicit signature cannot be implemented without changing a frozen
scientific interface, stop at this work package and write a reviewed exception.
The exception must explain why a native TensorFlow loop or stable signature is
not suitable and must include bounded retracing, numerical-equivalence,
memory, and runtime checks. It is not enough to retain `reduce_retracing=True`.

### P0-D: route-policy classification

Run the repository route-policy audit and preserve its exact output. The
repaired entry-state failures included:

- the new P1 runner;
- the Phase 9B source-authority auditor; and
- unrelated older hardbound, surrogate-force, performance-diagnostic, and
  stepwise migration-debt paths.

The P1 runner must receive one explicit route classification consistent with
its canary-only claim boundary. The source-authority auditor must be excluded
or classified as a read-only audit with a reason that it does not execute HMC.
Each unrelated path must either receive a historical/diagnostic classification
or a discovery exclusion with a source-specific reason. Do not mark an old path
`active_claim_bearing` merely to make the test green, and do not delete old
ledger findings.

The Phase 0 closeout must contain both the scoped P1 result and the full audit
result. The full route-policy audit must pass before P1 is launchable. If an
unrelated legacy path cannot be dispositioned within this repair, Phase 0
remains blocked for a claim-bearing P1 launch; the limitation is recorded as a
real governance blocker, not converted into a hidden green status. No new
Phase 9B path may remain unclassified.

### P0-E: provenance and artifact boundary

Create a fresh Phase 0 artifact root:

`docs/plans/artifacts/ssl-lstm-q20-phase9b-executable-readiness-2026-09-06/`

The root must contain separate receipts for authority, runner contract,
controller signature, route policy, budget, runtime preflight, and closeout.
Every receipt records the actual command, current Git revision, dirty-worktree
hash, environment, target signature, policy identifiers, source/plan hashes,
seed namespace, wall time, output paths, and claim boundary. A failed attempt
gets a new child directory; no prior artifact is overwritten.

The Phase 0 claim boundary is:

`phase9b_m4_p0_executable_readiness_only`

The artifact must say that no posterior draws, sampler ranking, convergence,
whitening, or default decision follows from the readiness result.
Run-start, success, and failure receipts must include the Python executable and
version, conda environment/prefix, platform, host, working directory, relevant
bounded GPU environment settings, and the actual managed-session trust basis.

### P0-F: compile-versus-steady-state budget diagnostic

The current `1,423`-second observation is not sufficient to set a new cap. The
source-owned bounded diagnostic
`docs/benchmarks/diagnose_ssl_lstm_q20_phase9b_executable_readiness_2026_09_06.py`
must measure, separately for factor and strict:

1. process start and Python/TensorFlow import;
2. GPU/device and memory-policy initialization;
3. chart construction and fresh same-scope tuning;
4. first compiled controller call;
5. repeated same-shape steady-state chunk call;
6. archive and diagnostic serialization; and
7. cleanup/closeout.

The diagnostic must use the same chart-0, beta-1.0 scope, target signature,
dtype, XLA/TF32 settings, and batch shape as P1, but it must use a fresh
readiness output root and may not produce a claim-bearing posterior stream.
Factor and strict must not share tuning artifacts, seeds, or output paths.
Both the first compiled call and the steady-state call must pass the shared
controller's per-chain movement check. A stationary chain invalidates this
readiness attempt; movement alone does not establish convergence.

The budget calculation must include the complete P1 schedule: all required
warmup and retained chunks for both arms, tuning, first-call compilation,
serialization, reserve, and termination grace. A positive single-chunk
forecast is not enough. If the schedule does not fit the current cap, refresh
the P1 plan with a new cap and explicit budget ledger before any P1 launch; do
not silently widen the cap in the runner.

### P0-G: durable campaign accounting

Replace timestamp-only remainder arithmetic with a persistent campaign ledger
under the Phase 0 artifact root. It must record:

- campaign identifier and declared total budget;
- each prior attempt and its fresh output root;
- monotonic and wall-clock start/end times;
- failure class and repair applied;
- measured, reserved, consumed, and released seconds;
- remaining budget before every new chunk; and
- the source/plan hash and seed namespace bound to each attempt.

The pre-chunk guard must fail closed if the ledger cannot be read, the source
hash has changed, the requested reserve exceeds the remaining budget, or the
output root is not fresh. A failed candidate or localized infrastructure retry
consumes the declared campaign budget; it does not mint a new budget.

### P0-H: trusted runtime preflight

The read-only trusted device probe may run while Phase 0 is active. The
bounded GPU diagnostic requires the static P0-A through P0-E checks and the
P0-G accounting infrastructure, the owner-authorized explicit diagnostic
allocation within the existing ledger, and a fresh passing trusted placement
check under `bayesfilter_non_display_first_load40_headroom5g_v1`.
It supplies P0-F's measurements; P0-F and the final P0-G budget reconciliation
therefore close after the diagnostic, not before it.
It must set `TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow import and let
the placement helper set `CUDA_VISIBLE_DEVICES` to the selected physical UUID.
Before any logical GPU or tensor initialization, it
must configure and verify memory growth on every visible physical GPU. It must
record:

- physical and logical device names;
- verified memory-growth values or an explicit reviewed logical-device-limit
  exception;
- allocator current and peak bytes when available;
- TensorFlow/TFP versions;
- XLA and TF32 state;
- source and plan hashes;
- exact command, environment, seed roles, and wall time; and
- trust basis `owner_designated_managed_session_visible_gpu_trusted` when that
  managed-session condition actually holds.

The readiness diagnostic must stop on failed memory growth, device mismatch,
missing allocator provenance, XLA fallback, output collision, or a source
closure change. It is not a P1 posterior run and its finite values cannot
promote the candidate.

## Required command set

The following commands are the CPU/static sequence. They are allowed
while Phase 0 is active and must write fresh, versioned output roots where the
command accepts an output directory. Regenerate the source audit before the P1
tests because the runner verifies that receipt. Preserve prior revisions and
write the post-run static source audit as `r14`.

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/audit_ssl_lstm_q20_phase9b_p1_canary_plan_2026_09_05.py \
  --output-dir \
  docs/plans/artifacts/ssl-lstm-q20-phase9b-p1-sequential-canary-2026-09-05/p1-plan-audit-20260907-r14
```

```bash
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_campaign_budget_ledger.py \
  tests/test_ssl_lstm_q20_phase9b_p1_canary.py \
  tests/test_ssl_lstm_q20_phase9b_readiness.py \
  tests/test_neutra_hmc.py \
  tests/test_q20_bridge_identity.py \
  tests/test_ssl_lstm_q20_phase9a_repair_runner.py
```

```bash
CUDA_VISIBLE_DEVICES=-1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_neutra_hmc_route_policy.py
```

The route-policy command now passes. Its full discovered/classified route set is
preserved by the route ledger, including the historical and non-HMC legacy
dispositions.

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m py_compile \
  bayesfilter/inference/neutra_hmc.py \
  bayesfilter/runtime/campaign_budget_ledger.py \
  docs/benchmarks/run_ssl_lstm_q20_phase9b_p1_sequential_canary_2026_09_05.py \
  docs/benchmarks/diagnose_ssl_lstm_q20_phase9b_executable_readiness_2026_09_06.py \
  docs/benchmarks/audit_ssl_lstm_q20_phase9b_plan_2026_09_05.py \
  docs/benchmarks/audit_ssl_lstm_q20_phase9b_p1_canary_plan_2026_09_05.py
git diff --check
```

The controller-signature and compile/steady-state commands are implemented in
the shared controller and the named source-owned diagnostic. The diagnostic is
the next runtime step under the active multi-GPU placement amendment. The last
trusted probe found GPU 0 occupied by `/usr/NX/bin/nxnode.bin` (`PID 6826`),
but that observation no longer forbids selecting another eligible GPU. No
process is to be stopped or displaced.
The exact probe is preserved in
`docs/plans/artifacts/ssl-lstm-q20-phase9b-executable-readiness-2026-09-06/resumption-20260907/gpu-probe.json`.

## Exit criteria

`M4-P0` exits only when every applicable item below is true and recorded in one
immutable closeout manifest:

1. P0-A verifies the current authority chain and all required source/artifact
   hashes without modifying historical receipts.
2. P0-B focused tests pass, including the actual shared-controller callback,
   chart selection, axis conversion, MCSE, seed, archive, failure, and budget
   contracts.
3. P0-C records an explicit stable TensorFlow signature and bounded concrete
   function/retracing behavior, or a reviewed exception satisfying the stated
   checks.
4. P0-D classifies every qualifying route exactly once or gives it an explicit
   discovery exclusion with a source-specific reason, passes the full
   route-policy audit, and preserves the pre-existing legacy dispositions.
5. P0-E writes complete, parseable, non-colliding manifests with current Git,
   source, plan, environment, seed, and claim-boundary provenance.
6. P0-F separates compile from steady-state cost for both arms and proves that
   the complete P1 schedule fits the declared budget, including reserve; if it
   does not, the P1 plan is refreshed before this gate can pass.
7. P0-G supplies a durable campaign ledger whose consumed and remaining budget
   agrees with the attempt table and whose pre-chunk guard fails closed.
8. P0-H supplies a valid active-policy GPU memory/XLA/TF32/device/allocator receipt, or a
   documented non-GPU exception that is explicitly limited to readiness and
   does not authorize P1. The current GPU-ownership receipt is a blocker, not
   an exit receipt.
9. The P1 plan points to the fresh audit receipt and states that this Phase 0
   closeout is a mandatory entry condition.
10. The closeout decision table says `P1_LAUNCHABLE_PENDING_FRESH_ATTEMPT` only
    for technical readiness. It must continue to say P2 is blocked by chart
    thresholds, downstream posterior/reference checks, and uncertainty gates.

## Closeout decision table

| Decision | Primary criterion | Veto status | Next action | Not concluded |
|---|---|---|---|---|
| Executable readiness | All B0--B7 checks pass and a valid budget exists | Any unresolved interface, route, provenance, device, or budget veto blocks P1 | Refresh P1 entry receipt or preserve the exact blocker | No sampler or posterior claim |
| P1 factor arm | Not assessed by Phase 0 | Requires a fresh post-Phase-0 attempt | Run only the refreshed P1 factor arm | No factor superiority or default |
| P1 strict comparator | Not assessed by Phase 0 | Requires fresh strict tuning and handoff | Run only under the same refreshed budget contract | No factor-versus-strict ranking |
| P2 scientific validation | Closed | Chart threshold and downstream evidence gates remain unresolved | Write a separate P2 plan after P1 closeout | No posterior correctness, convergence, or whitening |

## Post-run red-team requirements

The closeout must answer all three questions:

- Could the readiness diagnostic pass while a complete P1 run still fails?
  Yes: a short or single-shape diagnostic can miss cumulative resource growth,
  archive pressure, or later-arm cost. The budget reserve and fresh P1 attempt
  remain mandatory.
- Could a failure be only infrastructure or tuning rather than evidence against
  the research idea? Yes: classify the failure and preserve the attempt before
  changing any scientific choice.
- What would overturn the Phase 0 decision? A source-hash mismatch, unbounded
  retracing, route-policy gap for a new path, missing allocator provenance,
  infeasible complete schedule, or a changed target/data/hardware/privacy/
  scientific contract.

The closeout must separately state engineering correctness, numerical validity,
and scientific interpretation. A Phase 0 pass addresses only the first and the
resource/provenance part of the second.

## Handoff

When this plan passes, write:

- a Phase 0 result note beside this plan;
- one immutable closeout manifest under
  `docs/plans/artifacts/ssl-lstm-q20-phase9b-executable-readiness-2026-09-06/`;
- a refreshed P1 plan and source-only P1 audit receipt; and
- a master-program/reset-memo entry naming the exact fresh P1 command.

Until those files exist, the only valid next work is Phase 0 repair or focused
diagnostics. Do not launch the P1 runner merely because its six CPU tests pass.

## Execution ledger

### 2026-09-06 activation and initial checks

- The six focused P1 runner tests passed. Python compilation for the touched
  controller/runner/auditor paths and `git diff --check` also passed.
- The rotated source-only P1 audit at
  `docs/plans/artifacts/ssl-lstm-q20-phase9b-p1-sequential-canary-2026-09-05/p1-plan-audit-20260906-r8/run_manifest.json`
  passed after the P1 plan was made subordinate to M4-P0.
- The controller now has an explicit state/seed signature, the full NeuTra
  route-policy test passes, the durable campaign ledger is implemented and
  tested, and the source-owned compile/steady-state diagnostic is present.
- The trusted TensorFlow GPU probe failed closed because GPU 0 is occupied by
  `/usr/NX/bin/nxnode.bin` (`PID 6826`, user `ubuntu`, observed elapsed time
  about 2h20m). `nvidia-smi` showed GPU 0 using 337 MiB, including 312 MiB for
  that compute process. No BayesFilter GPU workload was launched.
- The initial record described GPU ownership as the only remaining blocker;
  the September 7 audit corrects that incomplete interpretation. The blocked
  September 6 probe receipt is
  `docs/plans/artifacts/ssl-lstm-q20-phase9b-executable-readiness-2026-09-06/gpu-preflight-blocked-20260906/run_manifest.json`.

### 2026-09-07 resumption and dual-blocker refresh

- Repaired closeout enforcement, the diagnostic dependency order, complete
  timing records, shared-budget accounting, environment/trust provenance, and
  per-chain movement checks for both controller calls.
- The standalone readiness regression passed with `12 passed`. The refreshed
  `r12` source audit passed with no findings. The combined CPU suite passed
  with `72 passed, 191 warnings in 8.60s`; the full route-policy suite passed
  with `6 passed in 2.10s`. Python compilation and `git diff --check` passed.
  Commands and logs are recorded in the Phase 0 result note; these checks use
  intentionally hidden GPUs and do not establish q=20 GPU readiness.
- The trusted probe at `2026-09-07T07:45:41Z` again returned
  `requested_gpu_compute_busy:0`. GPU availability is an external resource
  blocker, separate from the missing campaign-budget decision.
- No runtime diagnostic or P1 launch occurred. P0-F, final P0-G reconciliation,
  P0-H runtime evidence, and the passing M4-P0 closeout remain open. P2 stays blocked.
