# SSL-LSTM q=20 72-core staged process-parallel campaign

Date: 2026-09-03  
Status: `P3_FULL_CAP_BLOCKED`  
Supersedes for execution: no earlier plan; this is a new continuation design  
Parent master: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`

## Purpose and boundary

The preceding Phase 9A replay stopped because one GPU process executed the
whole measured grid serially and could not finish under its fixed wall cap.
This plan tests a different execution design: independent spawned CPU
processes, with a maximum of 72 worker cores and explicit barriers.  It does
not change the target, bridge, transport, HMC transition, candidate grid,
seed roles, or the repository GPU default.  It is an engineering and
mechanics/performance continuation; it cannot reopen Phase 9B or turn a CPU
diagnostic into a production or posterior result.

The six scopes are the two chart identities crossed with the three bridge
temperatures `(beta=0, 0.5, 1)`.  The measured joint grid remains
`epsilon=(0.25, 0.55, 0.85, 1.20)` crossed with `L=(3, 8)`, eight pairs.
Every declared pair is attempted independently.  No result is inferred from
another pair, and no map or kernel is averaged across workers.

## Research-intent ledger

| Item | Binding definition |
|---|---|
| Question | Can the current q=20 fixed-transport tuning work be split into isolated CPU processes under the requested `8x4 + 2x8 + 6x4 = 72` worker-core budget while preserving numerical and seed semantics? |
| Mechanism | Spawned workers with explicit CPU affinity and TensorFlow thread caps; three sequential barriers for screens, replicated selection, and scope held-out finalization. |
| Exact comparator | The serial `measured_joint_grid_v1` call semantics in `fixed_transport_hmc_tuning_tf.py`, using the same target, chart checkpoint, chain bank, HMC counts, and stateless seed derivation. |
| Primary canary criterion | All barrier workers start with the declared affinity/environment, the current q=20 worker route is finite and XLA-compiled, and a deterministic serial/process fixture agrees within the predeclared tolerances. |
| Full-run criterion | All 48 scope-candidate screens, all required replicated-selection calls, and six held-out scope finalizations produce durable typed records with complete identities and no hard veto. This is a mechanics/performance pass, not posterior admission. |
| Hard vetoes | Missing CPU IDs; nested/oversubscribed workers; CUDA visible in a CPU child; absent memory-growth/CPU launch declaration; target/chart/bridge signature mismatch; non-finite state/value/score; seed collision; output collision; worker crash/timeout; XLA disabled; incomplete task set; or serial/process parity failure. |
| Repair triggers | Startup contention, affinity drift, excessive resident memory, retracing, a localized serialization error, a candidate-local numerical failure, or a descriptive throughput regression. Repairs preserve the scientific contract and use a fresh attempt directory. |
| Continuation veto | A parity or identity failure that cannot be repaired without changing the target, bridge, chart, kernel, data, hardware class, worker-core budget, or campaign cap; durable artifacts cannot be written; three localized infrastructure repairs make no progress; or the declared campaign budget is exhausted. |
| Nonclaims | No whitening, mode-discovery, ergodicity, convergence, posterior correctness, sampler ranking, high-dimensional scaling law, CPU default, GPU speedup, production readiness, or Phase 9B readiness. |

## Evidence contract

The canary and full run must each write a fresh versioned artifact root and a
manifest containing the Git commit/status, exact command, Python environment,
target/bridge/chart identities, grid and seed namespace, worker topology,
logical CPU IDs, thread settings, CUDA visibility, XLA setting, memory status,
per-task timing/status, and output hashes.  Prior failed attempts are never
overwritten or resumed as fresh tuning evidence.

The numerical comparison is made on the computed object, not on acceptance
alone.  For a fixed seed and fixed chart, the serial and worker calls compare
the returned samples, target log probabilities, scores/gradient telemetry,
and finite/status flags.  The default tolerances are `rtol=1e-9` and
`atol=1e-9` for float64 samples/values and `rtol=1e-8`, `atol=1e-8` for
float64 scores.  If CPU XLA produces a documented backend rounding
difference, the canary fails closed until a target-specific tolerance is
measured against the serial reference; it is not silently relaxed.

Acceptance rate, elapsed time, RSS, and worker throughput are explanatory
diagnostics.  They may trigger a resource repair, but they do not establish
convergence or rank candidates.  Selection remains the declared replicated
minimum-bulk-ESS-per-gradient policy, with all selection draws discarded.

## Resource topology

The expression supplied by the user is interpreted as three **sequential
barriers**, not three nested pools:

| Barrier | Work units | Workers | Cores per worker | Worker-core total |
|---|---:|---:|---:|---:|
| `screen` | 48 `(scope, epsilon, L)` calls (six scopes x eight pairs) | 8 | 4 | 32 |
| `selection` | two independent replication streams over the measured screen results | 2 | 8 | 16 |
| `scope_finalize` | six scope-specific nominated-candidate held-out calls | 6 | 4 | 24 |
| **peak/phase sum** | barriers are mutually exclusive |  |  | **72** |

The controller never starts a later barrier before all tasks in the current
barrier have returned or have been durably classified as failed.  A worker is
one OS process, not a process pool.  Its TensorFlow intra-op/OpenMP/BLAS
thread count equals its assigned core count and inter-op is one.  CPU IDs are
allocated from the controller's current affinity set at launch, in disjoint
sets; hard-coded `0..71` assumptions are forbidden.  The manifest reports
logical CPU IDs and does not call them physical cores.  The 72-core value is a
worker budget; the controller and operating system are outside the worker
allocation and are recorded separately.

Every child sets `CUDA_VISIBLE_DEVICES=-1` before importing TensorFlow and
checks that TensorFlow reports no GPU.  CPU-only execution is therefore an
explicit diagnostic exception under the repository's GPU-default policy.
No pfor, `tf.vectorized_map`, implicit Jacobian pfor, or scalar row loop is
introduced.  Each target call remains the existing batch-native rank-2,
`tf.function(jit_compile=True)` route.

## Execution phases

### P0: authority refresh and audit

1. Add this plan to the parent master and write a reset receipt naming it as
   the only active continuation.  Keep M3/Phase 9B blocked.
2. Record the default/assumption audit below and the skeptical audit below.
3. Verify that the current target signature, strict `tensorflow_eigh` backend,
   C5 chart protocol, grid, and seed roles are unchanged.

### P1: implementation and focused checks

1. Add a repository-owned topology allocator that validates disjoint affinity
   sets, the three barrier budgets, no nested pools, and available CPU IDs.
2. Add a spawned worker/controller harness.  Workers receive only typed JSON
   task payloads and source-owned checkpoint paths; they reconstruct the q=20
   bridge, restore and hash-check the frozen chart, build the existing
   reusable HMC runner, and return typed JSON envelopes.
3. Persist one record per task, including `started`, `complete`, or `failure`
   status.  Sort results by declared task identity after asynchronous return.
4. Add unit tests for topology arithmetic, affinity validation, seed
   disjointness, task coverage/order, crash/timeout handling, and a synthetic
   serial/process parity fixture.  Tests must not import NumPy into runtime
   code and must not use pfor.

### P2: composite canary (must pass before P3)

The canary uses a new output root and a mechanics-only frozen chart fixture.
It exercises all three barriers with reduced chain lengths, plus one exact
serial-vs-process q=20 call.  It must verify:

- `8` screen workers each own exactly four distinct allowed logical CPUs;
- `2` selection workers each own exactly eight distinct CPUs;
- `6` finalization workers each own exactly four distinct CPUs;
- the realized worker-core maximum is `32`, `16`, and `24` by barrier and
  never exceeds `72` overall;
- no child sees a GPU, all children report XLA enabled and the same target,
  bridge, chart, and runner source identities, and every worker exits zero;
- the serial and process fixed-seed outputs meet the tolerances above;
- all canary records, logs, and the manifest are durable and collision-free;
- peak per-worker RSS and barrier wall times are recorded (descriptive only).

The canary is a hard engineering gate.  A high or low acceptance rate does
not decide it.  A canary failure invokes P2-R repair and a fresh retry; it
does not authorize the full run.

### P2-R: canary repair and refresh

Classify the failure as harness, environment, numerical, resource, or
artifact.  Apply only a localized repair that leaves the target, grid, seed
roles, topology, hardware class, and cap unchanged.  Run the smallest
focused regression, preserve the failed root, write a repair receipt, and
refresh this plan's command/assumptions.  Three unsuccessful localized
repairs fire the continuation veto.

### P3: full staged diagnostic execution (only after a passing P2)

1. **Fresh chart preparation:** in an isolated GPU child, build the two
   compact-high charts at beta `(0, .5, 1)` with a new seed namespace and
   write six checkpoint files.  Verify each checkpoint hash and parent chain.
   Preparation is outside the 72 CPU worker budget and is not itself a
   posterior claim.
2. **Screen barrier (`8x4`):** dispatch all 48 scope/candidate tasks.  Each
   task runs the exact screen counts and writes one candidate record.  The
   controller rejects missing or duplicated `(scope,candidate)` identities.
3. **Selection barrier (`2x8`):** dispatch replication `0` and replication
   `1` streams.  Each stream evaluates every screen-passing candidate in every
   scope with its own stateless seed domain.  No replication result is reused
   as a posterior draw.  Candidate-local failures are recorded and do not
   invalidate unrelated candidates unless a declared hard veto applies.
4. **Scope-finalization barrier (`6x4`):** for each scope, select the nominee
   using the existing measured-grid ordering and run its fresh held-out
   verification.  Write a scope handoff only when all required identity and
   held-out checks pass.  These handoffs remain mechanics/tuning diagnostics.
5. Join all records, compute descriptive wall/CPU/RSS summaries, and write a
   decision table and inference-status table.  Do not launch Phase 9B.

The full campaign wall cap is `14,400 s` after canary, with a total campaign
cap of `15,600 s` including the `1,200 s` canary.  These are conservative
hypotheses derived from the previous serial lower bound and are refreshed
after the canary using its measured per-task window; the refreshed cap may
only decrease or remain within the declared total.  A cap stop is a resource
failure, not a numerical failure, and leaves all completed records historical
to this diagnostic attempt.

### P3-R: full-run repair and master refresh

Preserve every attempt.  Run focused schema/hash/tests, classify failures,
repair only localized infrastructure under the remaining cap, and write a
closeout receipt with actual commands, wall time, CPU allocation, hashes, and
remaining budget.  Refresh the parent master to either (a) a new reviewed
Phase 9B subplan if all required mechanics gates pass, or (b) an explicit
continued block if they do not.  A successful parallel schedule alone never
opens posterior validation.

## Default and assumption audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|---|
| `8x4 + 2x8 + 6x4` | Explicit user request | Exactly 72 worker cores and matches the three logical barriers | Nested interpretation or CPU oversubscription | Allocator unit test and canary affinity snapshot | Reviewed execution hypothesis |
| CPU children with CUDA hidden | Repository GPU-default policy; CPU is allowed for diagnostics | Independent CPU processes avoid GPU0 contention and TensorFlow context sharing | CPU may be slower or numerically different | q=20 XLA parity canary and RSS/timing telemetry | Diagnostic exception |
| Four/eight TensorFlow threads per worker | Derived from cores-per-worker | Prevents each process from oversubscribing its assigned set | Native libraries may create extra threads | `/proc/<pid>/task` affinity and env snapshot | Resource hypothesis |
| Eight-pair grid and two replications | Existing target-specific Phase 9A profile | Preserves the measured-joint-grid evidence question | Work may remain too expensive | Complete task-count check and cap accounting | Frozen baseline |
| `14,400 s` full cap | Conservative provisional budget from prior measured work | Bounds a serious local campaign | Cap may be too small or unnecessarily large | Refresh from canary timing before P3 | Unproven budget hypothesis |
| New fresh chart checkpoints | Derived from no-reuse rule | Prevents partial prior calls becoming new tuning evidence | Preparation may fail independently | Checkpoint hash/parent-chain validation | Required |

## Skeptical audit before execution

The plan was reviewed against the required failure questions before any code or
run:

- **Wrong baseline:** the comparator is the exact serial measured-grid
  semantics, not the invalid fast grouped-HMC path (which previously failed
  parity) and not a historical CPU result.
- **Proxy promotion:** acceptance, timing, ESS, and RSS are explicitly
  explanatory; only identity, finite/status, coverage, parity, and durable
  artifact checks can pass the canary.
- **Stop conditions:** worker crashes, cap exhaustion, parity failure, missing
  tasks, and three failed repairs are explicit; a candidate failure does not
  silently become a research-direction veto.
- **Fairness:** every scope/candidate and every declared replication has an
  explicit task identity and seed; asynchronous completion order cannot alter
  selection order.
- **Hidden assumptions:** CPU IDs are discovered from affinity, logical versus
  physical cores is stated, controller overhead is recorded, and fresh chart
  preparation is separated from CPU scheduling.
- **Environment mismatch:** children hide CUDA before TensorFlow import,
  declare thread settings, require XLA, and report target/chart/source hashes.
- **Artifact question:** per-task envelopes plus a manifest answer whether all
  work was attempted, what failed, and whether outputs can be reproduced;
  partial work is never promoted.

This audit passes for the stated engineering/mechanics question.  It does not
certify the implementation; P1 tests and P2 canary remain mandatory.

## Execution update

P1 focused tests passed (`5 passed`, Python compilation, and `git diff
--check`).  The first canary attempt exposed and repaired only a launcher
import-path/child-join harness defect.  The fresh second attempt passed all
canary gates in `581.3556926490273` seconds.  Its result and repair receipt
are:

- `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-canary-result-2026-09-03.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-canary-reset-memo-2026-09-03.md`

P3 full execution is authorized with the unchanged target, grid, seed roles,
and staged `8x4 + 2x8 + 6x4` topology.  The fresh-chart and full-run caps are
unchanged provisional resource hypotheses and remain subject to the terminal
P3 repair/refresh closeout.

### P3 execution repair update (2026-09-03)

Full attempt-01 stopped before TensorFlow or GPU initialization.  The
controller attempted to open `fresh_chart_preparation/prepare.stderr.log`
before the preparation child created its reserved output directory.  This is
classified as a localized harness/artifact failure, not a numerical or
resource result.  The failed root is preserved under the full attempt-01
directory and is not usable as tuning evidence.

The repair moves the preparation stderr log beside the reserved directory,
records a structured failure receipt when a child exits without a manifest, and
uses one global full-campaign deadline across preparation and all three
barriers.  Focused compilation, topology tests, and diff checks must pass
before a fresh attempt-02 launch.  The scientific target, bridge, chart
protocol, grid, seeds, worker topology, hardware class, and caps are unchanged.

The first post-repair full attempt then exposed a second localized defect:
TensorFlow's CPU and GPU reduction order gave mathematically identical bridge
facts different decimal sums, and the bridge hash rejected every worker chart.
The source-owned fact builder now uses a deterministic host-side compensated
sum for fixed sigma-point metadata.  The prior full result is therefore
classified as an identity/harness veto, not a numerical result.  A fresh canary
is required after this identity repair; only that canary can authorize the next
full attempt.

### P2 canary attempt-03 concurrency interruption (2026-09-03)

Attempt-03 reached the parity, screen, and selection barriers, but its
finalization workers could not start because an unrelated concurrent Git merge
removed the untracked launcher from the shared worktree while child processes
were being created.  The preserved stderr records contain only the
missing-source error; there is no canary summary and no numerical or topology
evidence from this attempt.  It is classified as an infrastructure/worktree-
concurrency failure, preserved under `canary/attempt-03/`, and excluded from
promotion.

The merge has finished and the launcher has been restored and repaired.  The
focused compile, bridge-identity, and topology checks pass.  A fresh canary is
mandatory before any full launch; it must use a new attempt directory and the
unchanged `8x4 + 2x8 + 6x4` contract.  If it passes, the next full attempt
receives a fresh directory; if the worktree changes again during execution,
the attempt stops and is recorded rather than interpreted as candidate
evidence.

## Pre-mortem and decision rules

The likely misleading success is a fast schedule that changed the target or
used a different chart.  Signature and fixed-seed parity checks detect it.
The likely implementation failure is TensorFlow/XLA startup memory or thread
contention rather than the process design; readiness telemetry and a reduced
fixture distinguish those causes.  The likely scientific misinterpretation is
calling a finite, fast, high-acceptance run converged; the result templates
forbid that inference and keep Phase 9B closed.

At closeout the result must include this decision table:

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Canary admission | topology + q=20 serial/process parity | pass/fail | CPU backend rounding and contention | full run only on pass | no posterior or scaling claim |
| 72-core schedule feasibility | complete staged task set under cap | pass/fail/resource stop | fresh-chart preparation cost | refresh Phase 9B plan or repair | no sampler ranking |

and this inference-status table:

| Inference class | Required statement |
|---|---|
| Hard veto screen | Which identity, finite, resource, and artifact vetoes fired |
| Statistically supported ranking | `none` unless a separately powered analysis exists |
| Descriptive-only differences | Wall time, RSS, acceptance, and throughput only |
| Default readiness | `not assessed`; GPU default unchanged |
| Next evidence needed | Scope-specific numerical/tuning review, then sequential Phase 9B validation |

## P2 canary attempt-04 and P3 authorization (2026-09-03)

A direct absolute-path launch first exposed the same import-path defect before
the controller could create an artifact.  The launcher now establishes the
repository root on `sys.path` before repository imports; compilation passed.
The subsequent fresh canary completed in `585.6644140318967` seconds under the
`1,200` second cap.  All three barriers were durable and failure-free, the
CPU-only/XLA declarations and disjoint affinity assignments matched the
topology, and fixed-seed serial/process parity passed at the declared
tolerances.  The authoritative attempt-04 receipt is
`docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-canary-attempt-04-result-2026-09-03.md`.

This pass authorizes P3 full execution with a fresh chart-preparation root and
the unchanged target, bridge, grid, seed roles, hardware class, worker budget,
and caps.  It does not authorize Phase 9B or any posterior, convergence,
whitening, mode-discovery, or sampler-ranking claim.

## P3 full attempt-03 serialization repair (2026-09-03)

The first full launch after canary attempt-04 reached the screen workers, but
candidate diagnostics containing non-finite values could not be serialized:
`json.dumps(..., allow_nan=False)` raised `ValueError` while writing the
failure row.  This is a localized artifact-boundary defect.  It is recorded in
`docs/plans/artifacts/ssl-lstm-q20-72core-process-parallel-2026-09-03/full/attempt-03/controller_failure.json`;
the attempt supplies no candidate or posterior evidence and is quarantined.

The repair preserves non-finite diagnostics as explicit tagged JSON values and
keeps their typed failure/status fields, while finite payloads and numerical
kernels are unchanged.  The focused compile, bridge, topology, and
serialization tests pass (`7 passed`).  Because the artifact boundary changed,
a fresh canary is required before another full launch.  The next canary uses a
new attempt directory and the unchanged `8x4 + 2x8 + 6x4` schedule.

## P2 canary attempt-05 and renewed P3 authorization (2026-09-03)

The fresh post-serialization-repair canary completed in `587.6819816830102`
seconds under the `1,200` second cap.  Its 8-worker screen, 2-worker
selection, and 6-worker finalization barriers all reached readiness and wrote
complete durable records; no worker failed.  The fixed-seed serial/process
parity checks for samples, target values, log acceptance, and scores passed at
the declared tolerances.  Non-finite candidate diagnostics, where present,
were retained as tagged JSON values rather than aborting artifact writing.
The authoritative receipt is
`docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-canary-attempt-05-result-2026-09-03.md`.

This pass renews authorization for a fresh P3 full attempt under the unchanged
target, chart protocol, candidate grid, seed roles, hardware class, staged
`8x4 + 2x8 + 6x4` topology, and `14,400` second full cap.  It remains a
mechanics/tuning diagnostic and cannot open Phase 9B or support posterior,
convergence, whitening, mode-discovery, or sampler-ranking claims.

## P3 full attempt-04 interruption (2026-09-03)

The fresh full attempt reached 46 of 48 screen records before its controlling
session was interrupted while the final workers were still running.  It has no
complete barrier summary and is quarantined as an operator/session interruption.
A fresh full attempt under the same contract is the next execution step;
partial records from attempt-04 are not reused.

## P3 full attempt-05 cap-closeout defect and repair (2026-09-03)

The uninterrupted fresh attempt completed the full screen barrier and 26 of
32 selection task records (13 per replication stream).  At the declared
14,400-second campaign deadline both streams were in the next long candidate;
the controller terminated them as expected.  The implementation then raised
`selection worker failure: [{'returncode': None}, {'returncode': None}]` and
returned without a typed barrier or full summary.  This is wrong relative to
the plan's stated rule that a cap stop is a resource outcome with durable
partial coverage.  The completed task records are retained for timing and
diagnostic inspection only; the attempt cannot support a complete selection,
finalization, or candidate-ranking claim.

The localized repair adds `ParallelCampaignDeadline`, writes a
`barrier_timeout.json` receipt with the deadline stage, worker return codes,
started/durable task IDs, and missing coverage, and emits a typed
`full_summary.json` with status `CAP_STOP_INCOMPLETE`.  Full-run closeout now
returns this summary without pretending that partial work is a worker crash or
that it passed the staged criterion.  A focused regression exercises both the
partial-coverage encoder and the controller readiness-cap path.  The target,
bridge, chart protocol, grid, seeds, topology, hardware class, and caps are
unchanged.

Because this changes the artifact/status boundary, a fresh canary is mandatory
before another full launch.  The canary must pass the existing topology,
identity, finite, XLA/CUDA visibility, durability, and serial/process parity
checks, plus the new timeout-receipt regression.  If it passes, the next full
attempt may use a fresh directory under the same contract.  The 72-core
selection schedule itself remains a resource-cap blocker until a complete
barrier is feasible; no cap increase or task repartition is authorized by
this repair.

### Skeptical repair audit

The repair was audited before retry.  It does not convert incomplete output
into a pass, does not relax numerical tolerances, and does not treat process
termination as candidate failure.  It records both durable and missing task
identities, preserves the global deadline, and leaves canary/full promotion
criteria and nonclaims unchanged.  The new regression uses a fake process and
does not import TensorFlow or alter the runtime target.  This audit passes for
the narrow artifact-boundary defect; it does not establish that the 72-core
full schedule can finish within the cap.

## P3 terminal closeout (2026-09-03)

Canary attempt-06 passed after the cap-closeout repair in
`585.254545083968` seconds.  Its reader-facing receipt is
`docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-canary-attempt-06-result-2026-09-03.md`.
The full attempt-05 evidence remains a
cap-stopped schedule: 48/48 screen tasks completed, 26/32 selection tasks
completed, and 0/6 scope-finalization tasks started before the 14,400-second
global cap.  Both selection workers reached the cap while evaluating the same
long candidate class.  The repaired timeout regression and focused suite pass
(`9 passed`), but the repaired timeout path was not exercised by a second long
run because repeating the unchanged schedule would not answer a new question.

The terminal result and reset memo are:

- `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-result-2026-09-03.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-reset-memo-2026-09-03.md`

The staged topology is retained as a valid mechanics diagnostic.  The full
schedule is not promoted because the cap is a resource continuation veto; no
cap increase, task repartition, candidate ranking, or Phase 9B launch is
authorized by this plan.  A future continuation must use a new reviewed
subplan with an explicit resource/evidence change.  This plan is closed at
`P3_FULL_CAP_BLOCKED`.
