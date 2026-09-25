# Master-program reset memo: q=20 factor admission and reboot recovery

## Governing update: master repaired, reviewed and executed, 2026-09-16

Engineering status: `REPAIRED_REVIEWED_EXECUTED`. The final real q20 master
exited normally with `UNDER_BUDGETED`, after trusted GPU readiness, both
positive-temperature HMC qualifications and the repaired early affordability
check. No full training cohort, posterior comparison or confirmation was run.

The final source is isolated commit `71e0fba399489a8f25fcbdea1185f6bbb600e487`,
integrated on main as `04d59bcc`. Run `real-q20-r3` used the RTX 4080 SUPER,
TensorFlow 2.20.0, GPU/XLA, float64 and verified memory growth. Its total
supervised worker time was 766.7371110460954 seconds; pricing exited normally
after 129.03990818205057 seconds. Source checks, saved artifact checksums and
attempt accounting match; none of the owned supervisor processes remains alive.

The first measured production-batch scope already implied a reservation of
268997.4853 seconds (74.72 h), exceeding the initial 124650.7274-second
allowance. Earlier r2 records cover all four production-batch scopes and imply
1617749.0855 seconds (449.37 h) for the full declared training reservation,
before downstream costs. Most is heldout evaluation. These are conservative
reservations for all bank expansions and rungs, with a factor of two; they
are not statistical lower bounds on actual adaptive runtime.

Remaining allowance is 123883.99030228473 campaign seconds (34.41 h), including
42907.537327354854 diagnostic seconds (11.92 h). All previous charges and
holds remain deducted. The saved `settled-allowance.json` and final run are
under `docs/plans/artifacts/ssl-lstm-q20-executable-master-2026-09-16/`.
The four initial diagnostics and three amended cost-repair diagnostics are
consumed. Further execution needs a concrete cost/performance plan within the
remaining allowance; rerunning this unchanged reservation rule cannot admit
the full campaign. No additional total budget was created.

The [plan](bayesfilter-ssl-lstm-q20-executable-master-repair-plan-2026-09-16.md)
and [whole-program review and final execution record](bayesfilter-ssl-lstm-q20-executable-master-review-result-2026-09-16.md)
preserve commands, reviews, focused tests, limitations and the next decision.
The complete Gaussian fixture exercised 25 actual worker stages through
confirmation and cached resume. The final accounting repair passed 14 focused
checks, including actual early-stop and complete-pricing workers.

The current CLI is `docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py`.
Numerical evidence belongs to the isolated committed checkout
`/tmp/BayesFilter-q20-master-repair-20260916`; main retains unrelated active work.
No current security rejection occurred. Execution and qualification do not
establish q20 learning, whitening, posterior accuracy, method superiority or
production readiness. This outcome rejects the funded reservation plan, not
the NeuTra research direction.

## Historical recovery audit — superseded by the governing update above

All current/next-step statements below belong to their recorded older snapshot.
They preserve the evidence that motivated this repair and must not select an
older runner or override the governing update.

Updated: 2026-09-16 (Asia/Shanghai)  
Status: `PRODUCTION_REPAIR_INCOMPLETE_INTEGRATION_AND_LAUNCH_GAPS`

## Current recovery instructions

Start with the September 16 section of the
[governing master](bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md).
All older P0/P1 instructions below, including the final instruction to resume
the old service, are historical. Do not automatically resume the old recovery
runner as the production pipeline.

The production training repair is on local `main` at
`965ba2949244ee2a1d911515b6865c268a4af8d2` (one commit ahead of origin at this
audit). It was developed on `main` in
`/tmp/BayesFilter-q20-production-repair-20260915`; earlier audit work used
`/tmp/BayesFilter-phase9b-tuning-telemetry`. Main contains unrelated concurrent
changes, including pre-existing edits in this memo and the master. Preserve
them. Use an isolated, inventoried source snapshot for future numerical runs.

The current [CLI](../benchmarks/run_ssl_lstm_q20_production_2026_09_15.py) has
only `validate`, `price`, `train`. There is no end-to-end executable master.
Training/checkpoint/export tests pass on a tiny Gaussian; current public tuning,
verified-member sampling and ensemble adapters have not passed a connected
test. `sample_member` supplies a positive value to a required negative
log-acceptance threshold, and the ensemble adds an undeclared finite-energy
veto. The comparison module reports a missing reference rather than computing
one. See the [corrected result](bayesfilter-ssl-lstm-q20-production-repair-result-2026-09-16.md).

Resume engineering in this order:

1. Repair the current HMC consumer contracts, connect and test the actual
   training/export/public-tuning/verified-member/posterior/ensemble stages, and
   implement explicit CLI dispatch with honest incomplete statuses.
2. Enforce external launch deadlines and persistent aggregate cost accounting.
   `price` ignores `--max-seconds`; `train` is only cooperatively bounded.
   Initialization and compilation need an enclosing supervisor. Reconcile prior
   spending and freeze the execution source before numerical work.
3. Run the bounded trusted q20 GPU/XLA diagnostic under the existing repair
   allocation, then price all remaining stages and implement/validate the
   independent reference. Only a fully funded protocol may start the development
   training grid; numerical and scientific checks retain their distinct roles.
4. Execute qualified training, per-scope tuning, matched development comparisons
   and fresh confirmation as specified by the master and numerical ledger.

The saved budget is a September 15 snapshot: 135,275.83289109988 campaign
seconds remaining, including 54,299.37991617 diagnostic seconds; later costs
are not fully settled. The initial repair allocation is 7,200 aggregate worker
seconds, with at most four real q20 GPU diagnostic attempts capped externally
at 600 seconds each. These are existing limits, not new grants. The master
links the recovered amendment and requires subsequent costs to be imported.

The latest known GPU probe returned `no_idle_policy_permitted_gpu` at
2026-09-16 00:48:21 Shanghai. Availability now is not checked by that receipt.
No repaired q20 GPU training/HMC result exists. A fresh trusted probe is needed
immediately before a diagnostic; no gateway or environment change follows
from the historical availability result.

Safe immediate metadata check, already passed on main:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py validate
```

The owner's repair/retry authorization persists within the unchanged target
and budget. These gaps require engineering and evidence, not another approval
chain. No training loss, smoke, tuning receipt or GPU-ready result establishes
q20 whitening, posterior coverage, a completed campaign or a method ranking.

## Historical recovery record — superseded

All “current”, “active”, budget and resume directions below describe earlier
snapshots. They do not override the recovery instructions above.

Date: 2026-09-06  
Updated: 2026-09-13 (Asia/Shanghai)  
Status: `M4_P1_TERMINAL_FAILURE_SOURCE_CLOSURE_AND_STRICT_MOVEMENT_VETO`
Governing master: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`  
Latest runtime result: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-8h-amendment-result-2026-09-09.md`  
Active cap amendment: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-8h-amendment-plan-2026-09-09.md`  
Factor plan: `docs/plans/bayesfilter-ssl-lstm-q20-factor-route-fresh-tuning-admission-plan-2026-09-04.md`  
Active executable-readiness plan: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-executable-readiness-phase0-plan-2026-09-06.md`
Completed GPU continuation: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-plan-2026-09-09.md`
Validated parallel-tuning plan: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-parallel-tuning-execution-plan-2026-09-07.md`
Latest engineering result: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-result-2026-09-09.md`

## Current continuation state

The September 12 P1 retry is terminally stopped, not done. Factor completed
four warmup chunks but failed source-closure validation after the repository
changed during execution. Strict completed four chunks but fired the preserved
`chain_without_movement` veto in chunk 3 for chain 3 (zero movement); all
sampled states, target values, and log-acceptance values were finite. Preserve
both arms as failed evidence with separate classifications and keep P2 blocked.
The next step is a source-freeze/provenance repair and focused regression,
followed by a fresh versioned P1 attempt only after the closure guard passes.

At **September 12, 18:47 Shanghai**, service
`bayesfilter-q20-phase9b-p1-r2.service` is actively running bounded P1. Factor
and strict passed the refreshed setup identity and are executing in parallel on
non-display GPUs 1 and 0; display GPU 2 remains untouched. Both first P1
chunks are in progress. P1 has no terminal result yet, and no posterior,
convergence, ranking or default-readiness claim is authorized.

The earlier r2 P1 launch failed at 14:06 Shanghai after 10.84 measured
GPU-worker seconds total across both arms with
`checkpoint identity changed`. The cause was that accounting repair changed
the coordinator/ledger source closure while committed setup checkpoints still
used the pre-repair setup identity. The source migration now binds that exact
reusable setup identity, preserves current source identity for streams, and
archives the stale readiness/execution records. All 80 focused recovery/ledger
tests pass in 38.52 seconds, including six new setup-identity and
successful-sibling reuse regressions; GPU devices were intentionally hidden.

Phase 0 now passes for bounded P1. Both fresh training/tuning arms and actual
SIGKILL/resume canaries complete. Factor and strict each complete two healthy
500-transition-per-chain runtime chunks in 4,370.6522 and 6,639.4178 worker
seconds. Their reserve-inclusive forecasts are 16,519.6636 and 25,283.0138
seconds, within assigned allocations of 18,000 and 28,800 seconds. Evidence:
r2 `runtime-complete.json` and `phase0-readiness.json`, status
`PASS_CURRENT_SOURCE_PHASE0_FOR_BOUNDED_P1`. These forecasts and health
screens establish bounded mechanics readiness, not convergence.

Current ledger: 24,329.48797829113 consumed, 46,800 reserved, and
15,270.51202170887 unreserved seconds. The active launch is
`numerical-repairs/eigh-refinement-r2/launches/p1-d1884c1974/` under the
original campaign ledger. Continue until P1 reaches a terminal screen or a
declared veto; do not treat the reservation as spent compute.
The total unspent balance before live settlement is 62,070.51202170887
seconds. No P2 launch or new campaign allocation follows from this state.

### Previous runtime snapshot

Latest check: **September 12, 02:45 Shanghai**. Both arms completed fresh
training/tuning and passed the actual interruption/resume canary with exact
sample/trace equality. Both canaries reuse the first committed chunk and
recompute the interrupted second. In r2 `launches/runtime-c355a7aa08/`, factor
completes both 500-transition-per-chain runtime chunks and all health checks;
strict passes its first chunk and is running its second. All three completed
chunks have finite required values and valid target status. The active service
continues unchanged. P1 has not started; wait for both full timing/health
results and affordability before proceeding. No convergence claim follows.

Ledger at this check: 13,308.582441157134 settled, 57,600 reserved, and
15,491.41755884286 unreserved seconds. The completed factor runtime worker's
4,370.652188197 seconds are not yet settled; runtime accounting settles after
the wave. Preserve the live workers, source binding, original budget and
eight-hour per-arm cap. There is no new completion-time guarantee.

### Earlier launch snapshot

The owner completed the reboot and requested continuation. Matching driver/
NVML 580.178.04 is verified. The eight-sweep GPU/XLA diagnostic passes both
32-row fixed banks, both original four-row endpoint checks and odd-dimensional
sign controls. It costs 53.115703790001135 GPU-worker seconds; no invalid row
remains on those inputs. Evidence:
`launches/eight-sweep-gpu-validation-5c70410169/`. All 137 CPU regressions bind
the same current sources. This is numerical equivalence, not sampler stability.

Active migration: `source-migration.json` now binds the eight-sweep core to
`numerical-repairs/eigh-refinement-r2/`. Archive/activation receipt:
`runtime-health-diagnostic-20260911-r1/eight-sweep-migration-7a685780c8/`.
It verifies 124 bundles/3,109 tensors and preserves original start/ledger
bytes. Do not run either historical activation helper again or reuse r1
setup/tuning/stream/completion receipts.

Service `bayesfilter-q20-phase9b-eigh8-20260912-r1` is active from September 12,
00:44:39 Shanghai. In r2 `launches/reference-67d95ab1f2/`, factor uses GPU 1 and
strict GPU 0; both are non-display and start in parallel. Fresh setup/tuning
is in progress. At 00:48, both charts have committed completed training and
all six 32-row preflights at beta 0/0.5/1 pass. Both public tuners are active;
completed calls inspected so far have finite required quantities and valid
target status. Full runtime health remains pending. The service proceeds to actual SIGKILL/resume,
two-arm runtime/health and affordable bounded P1, stopping on any declared
veto. `Restart=no`; a failure requires inspection before a local repair/retry.

Launch accounting: 9,523.699322834134 consumed, 7,200 reserved,
69,676.30067716587 unreserved seconds. Unspent balance before settlement is
76,876.30067716587 seconds (21.3545 GPU-hours). Keep the original ledger and
28,800-second per-arm cap. Source and migration are frozen while workers run;
only progress notes may change. Recompute forecasts from the eight-sweep
runtime; old timings cannot establish completion time or affordability.

### Historical pre-reboot state

September 11, 15:36 Shanghai: no GPU worker is running. The loaded NVIDIA
module (580.173.02) and installed NVML (580.178.04) still mismatch after a
06:02 unattended driver upgrade. The host requests reboot; coordinate that
with the operator rather than changing packages, reloading drivers or
rebooting without approval. This happened after the 04:52 preflight failure
and cannot explain it.

CPU tracing reproduces the factor row 15 failure at time index 8 with exact
instrumented/uninstrumented value and score equality. Four Jacobi sweeps leave
too large an eigenpair residual on a positive-definite covariance. Eight
sweeps pass without relaxing any threshold. Both complete 32-row banks now
pass CPU/XLA against same-input native CPU reference (64/64 valid), with
maximum likelihood residual 2.38742e-12 and scaled score residual 1.00143e-13.
Evidence: `runtime-health-diagnostic-20260911-r1/preflight-eight-sweep-bank-r1/`.
All **137 focused CPU regressions pass** in 121.77 seconds after the minimal
eight-sweep edit; the new fixed-row tests reproduce both old failures.
`git diff --check` passes. Earlier CPU timeouts are preserved, not interpreted as
numerical outcomes.

The old four-sweep source migration is stale. Do not run the old activation
helper or resume `eigh-refinement-r1` against edited sources. After driver
repair, validate the eight-sweep banks on GPU/XLA, create a fresh numerical
migration and versioned namespace under the original ledger, then redo both
arms' tuning, recovery and full runtime forecast. CPU evidence cannot replace
those GPU checks. The ledger is unchanged: 9,470.583619044133 consumed, zero
reserved, and 76,929.41638095587 aggregate GPU-worker seconds remaining.

The read-only resume check confirms stale-source rejection before launch and
byte preservation of the original start, ledger and migration. Current source
snapshots and validation are in
`runtime-health-diagnostic-20260911-r1/eight-sweep-validation-a8df05dcf2/`.

The false-indefiniteness bug is in the GPU/XLA eigensolver, **not an actually
indefinite saved covariance**. At the original failed endpoint, 60-digit
arithmetic gives minimum eigenvalue 4.67714e-14 while the old solver gives
-1.42567e-11. Its binary32 internal Jacobi cutoff is inadequate here in float64.
The bridge correctly emits NaN after the erroneous invalid-row classification;
the rejected proposal then gives the stored negative-infinite acceptance.
No guard, tolerance, target, analytic derivative or final epsilon is relaxed.

The preceding shared four-sweep binary64 refinement was implemented for strict
and factor-cached principal-root paths. All 122 CPU regressions and the integrated
GPU/XLA fixed-point diagnostic pass. Both paths return four valid rows and
agree with the independent CPU reference within 7.11e-15 in likelihood and
6.10e-15 in scaled analytic score. Evidence is
`launches/candidate-diagnostic-a7e029690b/strict/` within the active campaign.
Four sweeps and the residual check remain tested local hypotheses, not a
universal accuracy claim.

The preceding attempt resumes **both arms freshly** in
`numerical-repairs/eigh-refinement-r1/` beneath
`campaign-24gpuh-20260909T135803Z/`. That migration passes 135 combined
CPU tests and preserves the original start/ledger; 101 bundles/2,697 tensors
verify. Keep the original
start/ledger/lock. Never label this serializer-only, reuse old setup/tuning/
completion/canary receipts, or create another budget for the nested directory.
Both fresh GPU workers start in parallel on non-display GPUs 0/1 at September
11, 04:52:31 Shanghai, then fail beta-zero chart preflight before an optimizer
update or tuning. The service stops at 04:52:55; no GPU worker is now active.
Failed attempt: `launches/reference-b6ab845731/` in the numerical namespace.
The subsequent fixed-bank localization and eight-sweep repair are recorded
here; do not retry using different seeds or relaxed gates.
Recovery/runtime/P1 remain pending. Original failed strict and successful factor evidence
remain intact, but neither supplies current numerical-scope clearance.

Settled budget: 9,470.583619044133 consumed, zero reserved,
76,929.41638095587 aggregate seconds remaining; eight-hour per-arm ceiling.
No P2, posterior admission, superiority or default-readiness claim. Read
`bayesfilter-ssl-lstm-q20-phase9b-runtime-health-repair-2026-09-11.md` and the
eight-hour amendment result before continuing. Earlier snapshots below are
historical, not the current status or next command.

## Historical snapshot: first invalid covariance localized

The September 10 service stopped at **17:34:11 Asia/Shanghai**, after the
serializer repair and both actual SIGKILL/resume canaries succeeded. Recovery
has exact tensor/trace equality, first-chunk reuse and second-chunk replay.
The subsequent `runtime-c48f1f0535` wave completed factor's two healthy
500-transition calls but rejected strict's first committed chunk.

Verified stored evidence: exactly one negative-infinite acceptance ratio at
transition 159/chain 4 (one-based), no NaNs or positive infinities in acceptance,
proposal rejected, sampled state unchanged. All sampled states/targets/status
pass. The original full-health function independently reproduces
`nonfinite_log_accept_ratio` and `nonfinite_delta_h`. These are actual runtime
vetoes, not the old diagnostic-ESS serializer exception. Do not suppress them.

Current investigation:
`bayesfilter-ssl-lstm-q20-phase9b-runtime-health-repair-2026-09-11.md`;
evidence: campaign `runtime-health-diagnostic-20260911-r1/`. Exact prefix and
cached-gradient diagnostics now localize the first invalid evaluation to the
second leapfrog of transition 159, chain 4. The finite parameter vector
[-3.6130528940902975, -0.2409610860997648, 0.5636597915347656,
0.7942885593780721] produces a placement-covariance minimum eigenvalue
-1.4256699903299468e-11. The bridge's invalid-row NaN guard then contaminates
momentum and the third position. Investigate the covariance recursion before
choosing an equivalent numerical repair or fresh public scope-specific tuning.
Never widen the tolerance, reuse failed runtime/P1 draws for tuning or overwrite the failed
stream. The current migration permits only the previous wrapper-only repair;
do not disguise a numerical change as serializer-only.

The completed prefix in `launches/prefix-diagnostic-83873ea777/` matches every
saved prefix sample, acceptance ratio/decision and sampled target bitwise.
`launches/cached-gradient-diagnostic-2dededa013/` preserves the original initial
gradient and finds the invalid second endpoint. Its preceding-transition
control is exact; one unrelated finite L=3 acceptance ratio differs by about
4e-16. The earlier one-step replays are not original-failure reproductions.
No seed bug is established. The unlaunched retuning draft was withdrawn;
the original serializer-repaired wrapper and migration are restored. Any new
strict tuning needs fresh seeds, stream/completion paths and recovery canaries.

Settled ledger: 9,269.455504760117 consumed, zero reserved,
77,130.54449523988 remaining seconds. No GPU worker is active at this update.
The active source closure is verified; 33 bundles/376 tensors verify and 56
CPU regressions pass (`postrun-validation.json`). No new allocation is minted.
P1 is not running. Preserve the
successful factor sibling rather than recomputing it. Phase 0 and P1 remain
incomplete; candidate rejection does not reject the research direction.

Current result:
`bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-8h-amendment-result-2026-09-09.md`.

## Historical September 10 resumed state

At **September 10, 2026, 08:23:01 UTC / 16:23:01 Asia/Shanghai**, execution
resumes under user service `bayesfilter-q20-phase9b-serializer-20260910-r1.service`,
coordinator PID 804934. Active attempt: `reference-4f43b85252`. Strict reruns
only its uncommitted tuning on non-display GPU 1; the completed factor reference
and both chart checkpoints are preserved and reused. Memory growth is verified
before GPU initialization, and XLA compilation is observed. The display GPU
is untouched by this launch. Independent later arms run in separate processes
when their original comparator devices are eligible.

The NaN is in candidate 1's tiny-window upper-tail ESS diagnostic, not evidence
of nonfinite input draws or HMC target values in the saved artifact. Tagged
nonfinite serialization and the public tuner's existing hash repair the failure
without dropping the veto or reselecting the candidate. The patch passes 161
CPU tests and reloads both the real finite factor checkpoint and a separate
diagnostic round-trip of the actual strict artifact. An unlaunched draft that
broke legacy tuple hashes is preserved separately; it is not the active repair.

`source-migration.json` records the final wrapper-only repair. It preserves
`campaign-start.json`, the frozen plan, original setup/stream identities, old
attempts and spent accounting. Do not edit bound execution code while workers
run. Validation and provenance are under
`serializer-repair-validation-20260910-r1/` in the existing campaign; the new
supervisor log is `supervisor-serializer-20260910-r1.log`.

Pre-resume unspent balance: 84,212.35038186601 seconds (23.3923 GPU-hours).
At 08:25 UTC, 3,600 seconds is reserved, 80,612.35038186601 remains unreserved,
and the new live worker time is not yet settled. Do not add the reserved time
to the remaining allocation or transfer older campaign balances. Continue
strict reference, the true interruption/resume canaries, both full runtime and
health measurements, Phase 0 closeout and bounded P1 if affordable. P2 and
posterior promotion do not follow automatically. No reliable finish ETA yet.

Repair plan: `bayesfilter-ssl-lstm-q20-phase9b-nonfinite-checkpoint-repair-2026-09-10.md`.
Live result: `bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-8h-amendment-result-2026-09-09.md`.

## Historical terminal state from ETA inspection

The detached service stopped at 2026-09-09 16:36:39 UTC (September 10,
00:36:39 Asia/Shanghai). Factor reference completed; strict completed public
tuning but failed typed-checkpoint hashing on NaN diagnostic data. The public
strict result selected candidate 0 and recorded a nonfinite-efficiency veto on
candidate 1; preserve that veto when repairing serialization. This is not a
budget/cap stop or a conclusion against the research direction.

The ledger settles 2187.64961813399 seconds, reserves zero and retains
84212.35038186601 seconds (23.3923 GPU-hours). Canary comparison, full runtime
measurement and P1 remain unfinished. The run has no reliable finish ETA until
repair and current timing. Next: lossless nonfinite typed-serialization repair,
focused regressions, recorded source migration under the same allocation and
resume unfinished work without rerunning the completed factor reference.
Read the amendment result's terminal failure section; older running snapshots
below do not describe the current service state.

## September 9 approved eight-hour amendment

The owner approves **28,800 seconds per arm** and a fresh **86,400 aggregate
GPU-worker-second** budget. The prepared campaign root is
`docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-24gpuh-20260909T135803Z/`.
The previous four-hour campaign and ledger remain immutable historical evidence;
their unused time is not transferred. GPU execution started at
**2026-09-09 16:00:10 UTC** (September 10, 00:00:10 Asia/Shanghai) in service
`bayesfilter-q20-phase9b-24gpuh-20260909.service`, coordinator PID 704521.
At 16:04:50 UTC the factor worker PID 704530 runs fresh tuning on non-display
GPU 0, with its chart committed and nine tuning calls complete. GPU 1 was at
64% launch utilization; strict queues and the display GPU is untouched. Device
availability controls actual concurrency. Memory growth is verified before
logical-device initialization. The first wave reserves 7,200 seconds; unsettled
live time is not a zero-spend claim.

The prior measured strict reserve-inclusive forecast, 16,202.12 seconds, fits
the eight-hour cap. This removes the old resource arithmetic blocker, but the
actual P1 consumer now delegates to the checkpoint-aware parallel coordinator.
Prepared-ledger adoption, interrupted initialization, third/fourth/first-retained
chunk replay, cumulative archive continuation, budget and route-policy checks
pass (128 CPU tests). The HMC kernel and full sampled-state telemetry are
unchanged; status reuse is optional future work, not a prerequisite. The prepared
root is now initialized and the actual P1 entrypoint runs with `--campaign-root`
and `--resume`. It performs fresh canary/runtime/health, issues current-source
Phase 0 readiness, and automatically enters bounded P1 if affordable. No P2 or
posterior promotion follows automatically.

Live amendment result:
`docs/plans/bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-8h-amendment-result-2026-09-09.md`.
Do not edit frozen amendment code/plan while workers run. Update this memo and
the result instead. Candidate warmup/R-hat failure is terminal for that attempt,
not permission to retry the same statistical test until it passes.

Active plan: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-8h-amendment-plan-2026-09-09.md`.

## September 9 terminal state and next repair (historical four-hour run)

The requested work is complete: cap raised to **14,400 seconds per arm**, both
real mid-call SIGKILL canaries pass exact resumed sample/full-trace equality,
and both arms complete two actual 500-transition calls with all required
sampled-state health checks passing. The terminal CPU suite passes 152 tests;
independent verification checks 34 checkpoint bundles and 372 tensors. Full
runtime, source, memory and cleanup evidence is preserved under

`docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-10gpuh-20260908T192200Z/`.

Read `terminal-summary.json`, `postrun-verification-r1.json` and the result note
before any further execution. **No campaign GPU worker remains. Do not rerun
the completed canary/runtime/profile commands.** The executed plans and the
runtime-stage `validated-result.json` remain frozen historical snapshots;
their earlier pending wording or pre-profile budget is not current status.

| Current quantity | Factor | Strict |
|---|---:|---:|
| First / repeated 500-transition call, seconds | 1,398.57 / 1,413.21 | 2,140.30 / 2,129.13 |
| Six chunks including fresh setup and observed overhead, seconds | 9,305.42 | 14,040.61 |
| With interruption reserve and cleanup grace, seconds | 10,751.04 | 16,202.12 |
| Fits four hours with reserve | Yes | **No** |

The forecasts are descriptive, not confidence bounds. Strict exceeds the
reserve-inclusive arm ceiling by 1,802.12 seconds. The conservative fresh
two-arm reserve-inclusive total is 26,953.15 seconds, also above the remaining
balance. The separate 36,000-second campaign has consumed
**11,352.55558313601 seconds (3.153487662 GPU-hours)** including failed attempts
and profiling, leaving **24,647.44441686399 seconds (6.846512338 GPU-hours)**
with zero reserved. Both older ledgers remain unchanged and their balances
must not be combined with this one.

P0-I/J checks pass for the checkpoint-aware route and P0-K localization is
complete. Status evaluation costs about another target value/score evaluation
on both saved endpoint banks. **The next step is a source-equivalent accepted-
state status-reuse repair**, with rejection/status/trajectory tests, a bounded
GPU parity/recovery canary and fresh full-controller timings. No optimization
or cap clearance is established by isolated component timings. Every attempt
must fit the remaining ledger; do not quietly omit the interruption reserve or
raise the cap. P0-L also needs the actual P1 consumer wired to the tested
checkpoint path and refreshed source/readiness evidence before it can close.
**P1/P2 remain blocked and no posterior sampling has been launched.**

Recovery resumes committed stages/chunks and replays the unfinished unit, not
a live CUDA instruction pointer. The GPU canary uses eight-transition chunks;
the full runtime uses 500, so up to one full chunk can require recomputation.
Unfinished tuning/training stages are replayed as stages. Do not claim universal
instruction-point, hardware/disk-loss or legacy-entrypoint recovery.

Both required full health screens pass, but strict has finite extreme energy
errors; these remain explanatory under the predeclared policy. Native TFP
divergences and proposed-state scores are unexposed, not zero or verified.
The scalar innovation eigen-gap's positive-infinity sentinel is documented in
the result. This does not establish convergence, chart quality or superiority.

The grid, timeout, queue/sibling and telemetry-inspector repairs are recorded
in the result and grid-repair note. Preserve source migrations r1/r2, all
failed attempts and original stream identities. The September 6--8 sections
below are historical wherever they describe pending measurements, old arm caps,
missing funding or an earlier current/next execution state.

## September 8 parallel-tuning continuation (historical)

**Completed:** the two-worker GPU/XLA smoke and fresh factor/strict tuning wave
both passed on non-display GPUs 1 and 0. Display GPU 2 was not used. The
validated campaign root is
`docs/plans/artifacts/ssl-lstm-q20-phase9b-parallel-tuning-2026-09-08/campaign-4gpuh-20260908T090200Z/`.
Both arms measured eight pairs, retained viable indices 0 and 2, and selected
epsilon 0.055 / L=3 with fresh heldout verification. No statistical ranking,
posterior result, or full P1 readiness conclusion follows.

The 14,400-second ledger consumed **844.217169 aggregate worker-seconds** and
has **13,555.782831 seconds (3.765495 GPU-hours)** unused. All four worker
settlements are unique; reserved time is zero; all worker PIDs have exited.
No retry was required. `validated_result_summary.json` and `gpu-after.json`
preserve verification and cleanup evidence. The old readiness ledger is
unchanged. The next justified work is full-controller instrumentation and
bounded cost localization for P0-I/J/K/L; P1 and P2 remain blocked. Keep the
parallel plan and executed sources at their recorded identities until any
new bounded stage or source-equivalent repair is documented.

The serial execution gap now has an integrated tuning-only diagnostic at
`docs/benchmarks/run_ssl_lstm_q20_phase9b_parallel_tuning_2026_09_08.py`.
Its parent launches complete factor/strict tuning jobs in separate Python
processes, with one UUID per worker and no shared TensorFlow state. It prefers
eligible non-display GPUs, queues excess jobs, and uses display only when none
of the non-display GPUs is eligible. Memory admission includes a 4 GiB peak
estimate plus 5 GiB headroom; growth is verified before expensive work.

CPU-only tests verify real subprocess overlap, cleanup, result reconciliation,
and single-parent budget settlement. They do not demonstrate q=20 multi-GPU
performance or numerical equivalence. Candidate pairs inside each worker and
the old readiness/P1 sampler remain serial. The old campaign balance remains
183.799655 seconds. The user has now authorized a **new 14,400-GPU-second
(four aggregate GPU-hour) campaign** for tuning validation. Reserve 600
GPU-seconds for the two-worker startup smoke, 7,200 for the first fresh
factor/strict tuning wave, and initially 6,600 for local repair/retry. Actual
worker lifetimes are charged and unused reservations released. No passing
Phase 0 closeout exists. Parallelizing tuning alone cannot resolve the factor controller's
8,314-second six-chunk forecast. Preserve P0-I/J/K/L, P1, and P2 blockers and do
not reuse old charts/tuning/draws as fresh evidence.

The sections below preserve September 6--7 history. Their old pending-probe,
3,307.39-second launch, and ledger-initialization wording is superseded by this
September 8 section and must not be executed as current instructions.

## Purpose

This memo is the recovery anchor after a process or machine reboot. It records
the last valid scientific state, the evidence that supports that state, the
artifacts that must be preserved, and the only work that may be proposed next.
It does not authorize a new long experiment. A future agent must read this
memo, the governing master, and the factor result before inspecting or running
the q=20 program.

## September 6--7 terminal state (historical)

Terminal execution update, September 7: PID `203299` was the bounded runtime
diagnostic on non-display physical GPU 1 and is now stopped. Output:
`docs/plans/artifacts/ssl-lstm-q20-phase9b-executable-readiness-2026-09-06/runtime-diagnostic-20260907T143204Z/`.
The actual start is `2026-09-07T14:34:25.180687Z`; the directory suffix is
only an attempt label. Exit was confirmed at `2026-09-07T15:27:28.771055Z`.
The shared ledger charged 3,183.590345 seconds plus the historical 1,832.61-
second estimate: 5,016.200345 seconds spent, 183.799655 remaining, zero reserved.
Do not repeat the former allocation or treat earlier uninitialized-ledger
descriptions as current state.
Audit r13, 52 focused checks, 73 broader checks, and 6 route-policy checks
passed; these are overlapping CPU suites, not GPU validity evidence.
The deadline supervisor and parent settlement completed and their receipts are
preserved. First/steady factor calls took 1,360.86/1,390.70 seconds; the minimum
six-chunk factor forecast before overhead is 8,314.36 seconds versus the
2,600-second arm cap. The strict arm is incomplete; full trace/startup evidence
also needs repair. No P1 launchable closeout exists. Additional serious GPU
work requires an explicit compute allocation, not another approval token.

Current next work is the detailed P0-I through P0-L repair in the Phase 0 plan:
interruption-safe records, startup memory verification, full trace checks, a
bounded per-transition profile, and an affordable complete schedule. No current
result rejects the target, factor backend, or research direction. Earlier
sections below preserve the pre-launch history; this terminal update and the
latest result govern resumption.

The active M4-P0 continuation remains open for its runtime diagnostic and
complete-schedule reconciliation. The owner has authorized bounded continuation
within the existing campaign cap and has replaced GPU-0-only placement with
`bayesfilter_non_display_first_load40_headroom5g_v1`: eligible non-display GPUs
are preferred at utilization `<=40%` and free memory `>=5,120 MiB`, with an
eligible display GPU used only as fallback. The prior GPU-0 occupancy check is
historical resource evidence. P1 and P2 remain blocked until their respective
gates pass. The completed factor campaign below is unchanged.

The source-synchronized factor campaign is complete and independently audited:

- Profile: `phase9a_factor_tuning_full_source_sync_v1`.
- Attempt: `source-sync-20260905T203000Z`.
- Target signature: `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`.
- Backend: `tensorflow_eigh_strict_factor_cached`.
- Coverage: all six `(chart, beta)` scopes, all eight declared `(epsilon, L)`
  pairs per scope, six factor-bound handoffs, held-out checks, and the Phase
  9A transition mechanics controller.
- Runtime: GPU0, TensorFlow 2.20.0, XLA and TF32 enabled, memory growth
  verified before device initialization.
- Wall time: `3551.299326295033` seconds, below the source-sync cap of 4000 s.
- Allocator peak: `1402668544` bytes.
- Git revision recorded by the manifest:
  `2dae412e450a5b44f46e375b810a7ad81aa78aeb`; the working tree was dirty.

The narrow decision is:

`PROMOTED_Q20_PHASE9_NUMERICAL_BACKEND`

The factor eigensystem backend is admitted for a separately planned q=20
Phase 9B candidate lane. The strict backend remains the explicit comparator and
fallback. The generic public API default has not changed.

## Authoritative artifacts

The durable receipt directory is:

`docs/plans/artifacts/ssl-lstm-q20-factor-route-fresh-tuning-2026-09-04/source-sync-20260905T203000Z/`

Required files and hashes:

| File | Required status | SHA-256 |
|---|---|---|
| `run_manifest.json` | `PASS_PHASE9A_SCOPE_PREFLIGHT` | `e0225382192ceb9da1e075cb9c7a91ed424e2c5c67adcfec7a0f34c05003a4e1` |
| `source-sync-audit.json` | `PASS_FACTOR_SOURCE-SYNC_AUDIT` | `9848779a1fe1611f3b3acfb666bec8b08bef6a42296fe30e84d3762d892c5933` |
| `run_start.json` | source-synchronized launch record | recorded in the receipt directory |

The result note contains the complete attempt ledger, scope table, decision
table, inference-status table, red-team analysis, and the terminal audit. Do
not reconstruct scope results from logs or merge partial attempts into this
receipt.

## What the evidence does and does not show

The selected and held-out calls were finite and moving under the declared
mechanics rules, and the transition controller had one compiled trace with no
hard controller veto. This supports only the numerical-backend admission.

The following remain open repair triggers:

- selected acceptance is extreme in several scopes (`0.99995`, `0.99987`,
  `0.23315`, and `0.27253` are representative values);
- the short mechanics receipt has maximum folded R-hat `3.3187` under a
  deliberately permissive threshold of `100` and only four retained draws per
  chain;
- centered log-density and pullback-score residuals are very large (chart 0
  RMS `535.98`, chart 1 RMS `661.94`, with maximum pullback RMS/coordinate
  `1438.64` and `2551.14`);
- TensorFlow emitted retracing warnings while independent trainer instances
  were constructed, even though the reusable per-scope transition trace passed;
  and
- the tuning and transition streams are short mechanics streams, not posterior
  samples.

Therefore Phase 9B is still blocked. No current artifact establishes IID
Gaussian whitening, a well-trained NeuTra map, posterior correctness,
convergence, exhaustive mode discovery, sampler superiority, high-dimensional
scaling, production readiness, or a repository-wide default change.

## Preserved failed attempts

These attempts are historical evidence and must not be combined with the
successful receipt or reused as tuning/confirmation data:

1. `full-attempt-20260904T185852Z`: stopped at chart 0, beta 0.5 after
   `1617.6309049129486` s with nonfinite proposed values, no movement, and raw
   NaN/Inf receipt serialization failures.
2. `r2-attempt-20260905T095452Z`: completed the six scopes but rejected the
   final manifest hash because raw NaN values were not normalized.
3. `r2-attempt-20260905T190500Z`: stopped before tuning because the source
   closure had changed during the concurrent HMC interface repair.

The source-synchronized run was deliberately launched with fresh seeds and the
current source closure. No old handoff, checkpoint, scope record, or partial
call was silently promoted.

## September 6--7 budget and execution boundary (historical)

The factor plan declares an aggregate serious-GPU budget of 11,800 material
seconds. Its ledger reported approximately 4,126.89 seconds before the final
source-synchronized run; that run consumed `3551.299326295033` seconds. The
accounting remainder is therefore approximately `575.59` seconds. This is not
authorization for a new material run and is not enough for Phase 9B. Treat the
factor campaign as closed unless a new reviewed plan declares a new budget and
fresh output root.

The active P1/readiness campaign is separate: its authorized total is 5,200
seconds, with an estimated 1,832.61 seconds consumed by historical attempts
and at most 3,367.39 nominal seconds remaining. Both the diagnostic and P1
consume the same ledger. No reviewed allocation currently funds both jobs
under the available forecast; a cap increase needs user approval. Do not
transfer the closed factor campaign's 11,800-second budget or interpret GPU
availability as a budget decision. Historical spending remains an estimate
even after it is recorded in the durable ledger.

No process was active at memo creation. Do not resume an interrupted shell,
reuse a stale PID, or launch the old M3/M3P/M3Q replay scripts. The old GPU
10000 performance plan remains subordinate historical evidence; its pending
strict canary is not an implicit next command.

## Worktree and source-closure rules

The working tree contains concurrent HMC tuning-guide/interface repairs as well
as the factor runner, auditor, tests, plans, and receipt files. Preserve all
user and agent changes. Do not run `git reset`, `git checkout`, destructive
cleanup, broad formatting, or a source-closure bypass. The source-synchronized
manifest is bound to the executable closure at its launch; changing source
files requires a new attempt and a new manifest rather than editing the old
hash.

## Reboot recovery checklist

For the current continuation, first inspect the September 9 campaign's
`terminal-summary.json`, settled `campaign_budget_ledger.json`, source migration
r2, runtime manifests and cleanup receipt. Compare source hashes before reusing
any checkpoint; corruption or mismatched scientific identity is a real stop.
There is no pending runtime worker or automatic P1 resume command. The checklist
below additionally preserves the earlier factor-admission prerequisite; it is
not sufficient by itself to reopen P1.

After reboot, perform only these read-only checks before proposing new work:

```bash
cd /home/ubuntu/python/BayesFilter
git status --short
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m json.tool \
  docs/plans/artifacts/ssl-lstm-q20-factor-route-fresh-tuning-2026-09-04/source-sync-20260905T203000Z/run_manifest.json \
  >/dev/null
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m json.tool \
  docs/plans/artifacts/ssl-lstm-q20-factor-route-fresh-tuning-2026-09-04/source-sync-20260905T203000Z/source-sync-audit.json \
  >/dev/null
sha256sum \
  docs/plans/artifacts/ssl-lstm-q20-factor-route-fresh-tuning-2026-09-04/source-sync-20260905T203000Z/run_manifest.json \
  docs/plans/artifacts/ssl-lstm-q20-factor-route-fresh-tuning-2026-09-04/source-sync-20260905T203000Z/source-sync-audit.json
```

Compare the printed hashes with the table above. If a file is missing,
modified, or unparsable, stop and classify the artifact problem before any
scientific interpretation. Do not regenerate the receipt in place.

## Phase 9B P1 execution update

The reviewed P1 subplan was attempted three times on 2026-09-06 under fresh
output roots. The first two attempts reproduced a chart-object indexing defect.
The third repaired that defect, completed factor tuning, and then exposed an
incorrect shared-controller callback contract. No complete sequential arm,
retained posterior stream, strict comparator result, or P1 pass exists. The
terminal repair result is
`docs/plans/bayesfilter-ssl-lstm-q20-phase9b-p1-sequential-canary-result-2026-09-06.md`.

The next runtime step requires a fresh trusted placement inventory and the
owner-authorized bounded diagnostic, not a blind GPU retry. The repaired runner must use the current P1 audit
receipt, retain its fresh output-root rule, and preserve the following
requirements:

- long per-chain warmup and cumulative retained sampling under
  `bayesfilter_neutra_sequential_hmc_v1`, with warmup excluded from posterior
  estimates;
- modern split/folded R-hat, bulk and tail ESS, MCSE, finite-state/status,
  movement, energy, and divergence checks;
- chart-quality and pullback-score thresholds, including a decision for the
  large residuals already observed;
- independent calibration, held-out, and confirmation seeds and a fresh output
  root;
- physical HMC, strict single-chart, replica-exchange, and factor multi-chart
  comparator arms as applicable;
- retracing and steady-state compilation diagnostics; and
- downstream posterior/reference and mode-travel checks with uncertainty
  analysis.

The focused regression passed on CPU. The measured first sequential chunk took
approximately 1,423 seconds after tuning; three attempts consumed an estimated
1,832.61 seconds, leaving approximately 3,367.39 nominal seconds in the P1
budget. The minimum schedule needs six chunks per arm and therefore cannot fit
the existing 2,600-second arm cap under that forecast. Separate compile and
steady-state timing is now the next bounded diagnostic, with an initial maximum
of approximately 3,307.39 seconds after a 60-second settlement margin. A
passing mechanics receipt alone cannot open Phase 9B.

## M4-P0 executable-readiness activation

The remaining work is now explicitly organized as Phase 0 rather than treated
as an informal P1 retry. The active plan is
`docs/plans/bayesfilter-ssl-lstm-q20-phase9b-executable-readiness-phase0-plan-2026-09-06.md`.
It is a technical entry gate and carries no posterior claim.

Its required blocker closure is:

1. verify the repaired runner's actual shared-controller callback, chart/axis,
   MCSE, archive, seed, failure, and output-root contracts;
2. add an explicit stable TensorFlow input signature to the shared batched HMC
   program, or record a reviewed exception with bounded retracing and resource
   checks;
3. classify the new Phase 9B paths in the NeuTra route policy and preserve the
   unrelated legacy audit findings rather than silently making them active;
4. separate compile, first-call, steady-state, tuning, and serialization costs
   for both factor and strict arms;
5. replace timestamp-only retry arithmetic with a persistent campaign budget
   ledger; and
6. produce a valid GPU memory-growth, XLA, TF32, device, allocator, source, and
   environment receipt.

No material P1 GPU command is valid until the Phase 0 closeout, a rotated P1
source-only audit, and a complete-schedule budget forecast all pass. The old
P1 attempts remain preserved and cannot be merged into the next run.

## Phase 0 initial execution record

The initial non-GPU checks on September 6, 2026 are:

- the focused P1 runner suite passed with `6 passed`;
- Python compilation for the controller, P1 runner, and auditors passed;
- `git diff --check` passed;
- the earlier rotated P1 source-only audit `r4` passed, followed by `r8` after
  the first executable-readiness repairs; and
- the full NeuTra route-policy test initially failed on unclassified new and
  legacy paths, which were subsequently classified and rechecked.

The September 6 static repairs covered the controller signature,
route-policy classifications, persistent campaign ledger, source-owned
readiness diagnostic, and source-only P1 audit `r8`. The trusted GPU probe
then failed closed because GPU 0 is occupied by `/usr/NX/bin/nxnode.bin`, PID
`6826`. No BayesFilter GPU process was launched, and GPU 1 was not substituted.

The September 6 blocked receipt is
`docs/plans/artifacts/ssl-lstm-q20-phase9b-executable-readiness-2026-09-06/gpu-preflight-blocked-20260906/run_manifest.json`.
The exact Phase 0 result is
`docs/plans/bayesfilter-ssl-lstm-q20-phase9b-executable-readiness-phase0-result-2026-09-06.md`.

## September 7 pre-launch resumption history

The source audit found and repaired missing P1 executable-closeout enforcement,
a circular diagnostic prerequisite, incomplete compile/steady timing records,
and a diagnostic ledger that would renew the allowance per output directory.
Both launchers now share the historical-debit ledger, and the diagnostic
settles the full attempt once on success or failure. Its receipts also retain
Python/conda/host/platform/GPU-environment provenance and the managed-session
trust basis. Per-chain movement is required on both controller calls.

The standalone readiness regression passed with `12 passed`. After the current
placement and identity-binding repair, the runner and plan must use the fresh
passing source audit
`docs/plans/artifacts/ssl-lstm-q20-phase9b-p1-sequential-canary-2026-09-05/p1-plan-audit-20260907-r13/run_manifest.json`.
The combined focused suite passed with `72 passed, 191 warnings in 8.60s`;
the full route-policy suite passed with `6 passed in 2.10s`. Compilation and
`git diff --check` passed. The Phase 0 result preserves commands and logs; GPUs
were deliberately hidden for these engineering tests. The live campaign ledger
has not been initialized because no runtime diagnostic or P1 retry has launched.

The last trusted probe is
`docs/plans/artifacts/ssl-lstm-q20-phase9b-executable-readiness-2026-09-06/resumption-20260907/gpu-probe.json`.
It returned `requested_gpu_compute_busy:0` at `2026-09-07T07:45:41Z`; the
accompanying process snapshot identifies `/usr/NX/bin/nxnode.bin`, PID `6826`,
using 312 MiB. Do not kill that process or any display/other-user process. A
fresh inventory must be collected immediately before the diagnostic.

The static validation is being refreshed after the multi-GPU integration. Run
the two-arm diagnostic with the explicit `3,307.39`-second allocation, reconcile
the measured complete schedule against the remaining budget, and issue M4-P0
closeout if all required evidence passes. The closeout must precede P1, not the
diagnostic that supplies its measurements. A fresh GPU selection alone is
insufficient.
P1 and P2 remain blocked; neither blocker rejects the target, factor backend,
strict comparator, or research direction.

## Recovery stop conditions

Stop and request a new direction only if the target, data, bridge, hardware
class, privacy boundary, scientific contract, or campaign budget would change;
the durable receipt cannot be verified; a required source or dependency is
missing; or a new plan cannot state a valid evidence contract. A failed
candidate or localized harness issue within an authorized future campaign is a
repair trigger, not evidence against the transport research direction.

## Authority order after reboot

1. Repository `AGENTS.md` and owner directives.
2. The governing master program and this reset memo.
3. The active M4-P0 executable-readiness plan named by the master.
4. The reviewed P1 subplan, after the M4-P0 entry gate passes.
5. The factor result and source-synchronized manifest/audit.
6. Source code and focused tests consistent with the active plan.
7. Older plans, logs, and partial attempts as historical evidence only.

The rebooted session must inspect the service, ledger and latest stage receipts
before reporting live state. At the latest documented snapshot it was:

`M4_P0_STOPPED_STRICT_TUNING_CHECKPOINT_SERIALIZATION`

Resume the initialized campaign with the same P1 entrypoint and `--resume` only
if no coordinator remains. Never create another budget/root to bypass an active
reservation. The transient service survives this session, not necessarily a host
reboot; after reboot recreate a trusted service with a fresh console log and the
same campaign root. Committed work replays and unfinished units recompute.
