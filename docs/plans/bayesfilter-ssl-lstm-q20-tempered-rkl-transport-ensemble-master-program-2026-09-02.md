# SSL-LSTM q=20 Tempered Reverse-KL Transport Ensemble Master Program

Date: 2026-09-03  
Status: `M3_TERMINAL_M3C_CLOSED_M3P_FULL_CAP_BLOCKED_PHASE9B_BLOCKED`
Latest continuation result: `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-result-2026-09-03.md`
Governing reset memo: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-reset-memo-2026-09-03.md`
Closed continuation plan: `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-plan-2026-09-03.md`

## Scope and purpose

This is the active master program for the q=20 SSL-LSTM tempered reverse-KL
transport and fixed-transport HMC work. It governs which plan is active, what
may be executed next, how a failed phase is repaired, and which scientific
claims remain closed. It is a control document, not evidence that the
transport is a good Gaussianizing map or that the posterior sampler has
converged.

The earlier particle-authority master program dated 2026-08-25 remains the
historical record for its phases 0--26 and its direct full-state LEDH blocker.
It does not govern this newer tempered-transport continuation. The current
implementation plan and execution record are subordinate documents:

- `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-implementation-plan-2026-08-28.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-execution-2026-08-28.md`

The post-terminal performance/whitening investigation was a bounded M3-C
continuation. It was subordinate to this master and diagnosed execution
without reopening the failed M3 replay. Its closed plan and result are
`docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-next-plan-2026-09-02.md`
and
`docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-next-result-2026-09-02.md`.

The 72-core staged process-parallel continuation is a new, explicitly
diagnostic performance design. It is subordinate to this master, uses fresh
attempt roots, and leaves the M3 replay result and Phase 9B block unchanged.
Its plan is
`docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-plan-2026-09-03.md`.

## Current state

| Area | Binding state | Evidence |
|---|---|---|
| Target and bridge | q=20 SSL-LSTM posterior on `theta in R^4`; proper likelihood-tempered bridge and strict square-root backend are admitted for mechanics | Phase 0 and C5 receipts; target signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278` |
| Transport protocol | C5 `phase8-k2-compact-high-l3-pure`, two `(16,16)` tanh charts, betas `(0,.5,1)`, pure continuation, fixed `gamma=(.5,.5)` | `docs/plans/bayesfilter-ssl-lstm-q20-phase8-c5-freeze-result-2026-08-31.md` |
| Tuning policy | `measured_joint_grid_v1`; directional acceptance repair is diagnostic-only | `docs/plans/bayesfilter-hmc-tuning-guide-policy-repair-result-2026-09-01.md` |
| Phase 9A localized repair | Complete for chart 1, beta 0 as mechanics evidence only; four of six scopes were intentionally not attempted | `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-chart1-beta0-program-repair-result-2026-09-01.md` |
| Full Phase 9A replay | Corrected canary retry reproduced the bounded resource failure; M3 is terminal and full replay is blocked | `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-performance-result-2026-09-02.md` |
| M3-C performance/whitening continuation | Closed bounded diagnostic; no nomination and no fast grouped-HMC admission | `docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-next-result-2026-09-02.md` and repaired manifest |
| M3P 72-core staged process continuation | Canary passed after timeout repair; full CPU-only diagnostic stopped at the declared cap, no Phase 9B admission | `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-plan-2026-09-03.md` and `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-result-2026-09-03.md` |
| Phase 9B posterior validation | Blocked until all six scope handoffs and their gates pass; no retained confirmation stream is open | parent implementation plan and current result notes |
| Whitening and mode claims | Closed; large pullback residuals and short, seed-sensitive chains remain diagnostic evidence | M3-C result and manifests |

## Authority order

When documents disagree, apply this order:

1. Active repository policy in `AGENTS.md` and applicable project directives.
2. This master program and its latest dated reset memo.
3. The active phase subplan named by this master, including its evidence
   contract and frozen identities.
4. The mathematical note and parent implementation plan.
5. Source code, tests, and run manifests that satisfy the active plan.
6. Historical plans and results, which explain prior decisions but cannot be
   reused as current tuning or confirmation evidence.

Engineering correctness, numerical validity, and scientific interpretation are
separate ledgers. A passing import, finite value, or mechanics screen never
promotes a posterior or whitening claim.

The prior M3/M3-C state and the M3P continuation are closed.  M3P's repaired
canary passed in `585.254545083968` seconds, but its fresh full attempt stopped
at the declared wall cap before selection and finalization were complete.
Phase 9B remains blocked.  No new long command is valid until a reviewed plan
changes the M3P resource/evidence contract.

## Research-intent ledger

| Item | Binding definition |
|---|---|
| Main question | Can fresh Gaussian reverse-KL charts, a proper temperature bridge, and fixed state-independent chart kernels provide reliable q=20 exploration after target-specific tuning? |
| Candidate mechanism | Independent tempered reverse-KL transport ensemble, optional joint mixture refinement, fixed multi-chart HMC, and exact adjacent replica exchange. |
| Baseline ladder | Physical-coordinate HMC; single cold NeuTra; physical replica exchange under the same bridge; single-chart tempering; cold multi-chart HMC; plain ensemble; optional joint-RKL ensemble. |
| Expected failure | Chart collapse, poor bridge overlap, invalid inverse/log determinant, chart-specific tuning failure, static XLA/retracing cost, high acceptance with little effective movement, or mode locking. |
| Promotion criterion | For a declared phase, all exact identity/health gates pass, every required scope is measured, disjoint held-out checks pass, and any later stochastic comparison has uncertainty evidence. |
| Promotion veto | Wrong target/bridge identity, non-finite value/score/map, stale or reused scope, unmeasured candidate, missing movement/status/energy telemetry, invalid transition or swap ratio, memory-growth/XLA violation, output collision, or claim beyond the computed quantity. |
| Continuation veto | Exact fixture contradicts the contract; target or common measure is unavailable; required artifacts cannot be made durable; three scoped infrastructure repairs make no progress; the declared campaign budget is exhausted; or a future repair would change target, data, hardware class, privacy boundary, or scientific contract. |
| Repair trigger | High acceptance, poor ESS/movement, seed sensitivity, retracing, no viable candidate, inadequate grid resolution, or a localized implementation/resource failure. |
| Must not conclude | No IID-Gaussian whitening, exhaustive mode discovery, posterior correctness, convergence, sampler superiority, high-dimensional scaling, production readiness, or default readiness from the current evidence. |

## Evidence contract

Every serious phase must state these fields before execution:

- exact target, bridge, chart, beta, dtype, backend, XLA, and data identities;
- comparator and target-call/wall-time accounting;
- primary pass criterion and hard veto diagnostics;
- explanatory diagnostics whose values cannot promote a candidate alone;
- disjoint calibration, selection, held-out, and confirmation seed roles;
- fresh versioned output root and a manifest containing command, Git state,
  environment, device/memory policy, seeds, timings, and hashes; and
- explicit nonclaims and the smallest next repair.

For the current q=20 route, acceptance is descriptive or a repair trigger.
Selection uses measured joint `(epsilon,L)` pairs and replicated efficiency
diagnostics. Modern R-hat, ESS/MCSE, declared-region travel, initialization
forgetting, and replica round trips become promotion gates only in a separately
approved sequential Phase 9B plan.

## Frozen identities and defaults

| Choice | Provenance and status |
|---|---|
| `theta` dimension 4 and target signature above | Frozen target fact; a 60-dimensional internal filtering state is not the sampling measure. |
| C5 compact-high K=2/L3 protocol | Frozen calibration representative for Phase 9A; not a universal architecture default. |
| `measured_joint_grid_v1` | Repaired guide and shared tuner policy; every declared pair is measured before selection. |
| q=20 localized grid | `(epsilon=(0.55,1.20), L=(3,8))` in the completed v4 mechanics probe; the next replay draft uses `(0.25,0.55,0.85,1.20) x (3,8)` as a target-specific hypothesis, not a promoted default. |
| Four varied starts and four chains | Mechanics baseline for the localized probe; too short for posterior inference. |
| TensorFlow/TFP, GPU0, XLA, TF32, memory growth before initialization | Repository execution policy; CPU-hidden fixtures are explicit exceptions. |
| No pfor or row-mapped scalar target | Repository policy and route-scan requirement. |
| Prior 18-hour particle-authority budget | Closed historical campaign; it is not silently transferred to this continuation. A new serious campaign must declare its own cap. |

## Phase map and ownership

| Master stage | Purpose and exit | Active document | State |
|---|---|---|---|
| M0 | Bridge, density, trainer, lineage, fixed-kernel, replica-exchange, and analytic mechanics foundations | Parent implementation plan phases 0--7 and execution record | Complete for stated mechanics gates |
| M1 | q=20 calibration, diversity/overlap checks, optional joint-arm feasibility, and C5 freeze | `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-phase8-calibration-subplan-2026-08-29.md` and C5 freeze subplan/result | Complete; K=2 protocol frozen for tuning only |
| M2 | Fresh chart construction and scope tuning preflight | `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-chart1-beta0-program-repair-subplan-2026-09-01.md` | Complete localized mechanics pass; Phase 9A partial |
| M3 | Performance repair, chart-1/beta-0 canary, then fresh six-scope replay | `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-performance-subplan-2026-09-01.md` | Terminal: second bounded canary resource failure; full replay blocked |
| M3-C | Post-terminal performance and whitening diagnostics | `docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-next-result-2026-09-02.md` | Complete; no N2 nomination, fast grouped-HMC path rejected |
| M3P | Staged `8x4 + 2x8 + 6x4` process-parallel mechanics/performance diagnostic | `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-plan-2026-09-03.md` | Closed at full-cap blocker; Phase 9B remains blocked |
| M4 | Sequential warmup, retained beta-one sampling, comparator runs, region/travel and posterior checks | New Phase 9B subplan required after M3 pass | Blocked |
| M5 | Controlled dimension/scaling ladder with uncertainty and matched baselines | New plan required after M4 | Blocked |

No stage may skip its entry gate because an earlier mechanics artifact is
finite. M3 replay is terminal, M3-C is closed without a nomination, and M4 and
M5 remain closed.

For this terminal state, “M3 continuation” refers only to the explicitly named
M3-C diagnostic plan. It does not authorize the old six-scope replay, a widened
cap, reuse of partial calls, or any Phase 9B command.

### Mandatory stage protocol

Every master stage is a pair of substages: `E_k` is the declared execution and
`R_k` is the mandatory repair-and-refresh closeout.  The closeout is required
after a pass as well as after a failure.  The next execution cannot start until
`R_k` has written its receipt and refreshed the next subplan.

| Stage | Execution | Mandatory repair/refresh closeout | Entry to next stage |
|---|---|---|---|
| M0 | `E0` mechanics foundations | `R0`: verify identities, fixtures, and source closure; repair localized defects; refresh M1 | M0 receipt and M1 entry checks |
| M1 | `E1` calibration and C5 freeze | `R1`: classify calibration evidence, preserve failed arms, freeze only justified controls; refresh M2 | C5 receipt and M2 entry checks |
| M2 | `E2` fresh-chart/localized repair | `R2`: audit fresh seeds, hashes, numerical health, and claim boundaries; refresh M3 | localized result and reset memo |
| M3 | `E3` performance, canary, and six-scope replay | `R3`: recompute cost, repair localized defects, and refresh or stop the Phase 9B plan | M3 terminal result and a valid M4 plan, or a stated continuation blocker |
| M4 | `E4` sequential posterior validation | `R4`: audit convergence, posterior, comparator, and uncertainty evidence; refresh M5 | M4 terminal result and a valid M5 plan |
| M5 | `E5` dimension/scaling ladder | `R5`: audit uncertainty and default decision; refresh the next research program | terminal decision or explicitly scoped continuation |

Subplans may contain finer-grained repair steps, but they may not omit the
stage closeout.  Each closeout records preserved attempts, failure
classification, focused regression, changed-file and artifact hashes, remaining
budget, refreshed assumptions/defaults, the next exact command, and the
condition that would constitute a real continuation blocker.

## Immediate next steps

These are the ordered actions captured in the M3 subplans and their
post-terminal M3-C/M3P continuations:

1. **M3-E0/R0 evidence audit:** validate the completed attempt-04/05 records,
   separate compile from steady-state cost, and reserve fresh output/seed
   namespaces. Close with the mandatory audit receipt.
2. **M3-E1/R1 performance/profile repair:** add stable timing and, if needed,
   the smallest numerically equivalent TensorFlow-native change. Define and
   test separate canary/full source-owned profiles and the launcher. Close with
   focused regression and a refreshed canary command.
3. **M3-E2/R2 and R2a repair:** preserve the invalid first canary, repair the
   launcher envelope, run the exact regression, and retry once with the
   unchanged eight-pair profile and 1,800-second cap. This retry reproduced the
   bounded resource failure, firing the M3 continuation veto.
4. **M3 terminal closeout:** preserve attempt-02, write the terminal result and
   reset memo, and keep the six-scope replay and Phase 9B blocked. A future
   replay requires a new reviewed plan with a changed performance design or
   budget.
5. **M3-C-E0..E4/R0..R4 continuation:** closed under its result note. It
   measured exact grouped-transition equivalence and a target-specific training
   ladder; no sampler was promoted and M4 was not reopened.
6. **M4-E4/R4 posterior validation:** only after a passing M3 closeout and a
   separately refreshed Phase 9B plan; close the phase before any scaling work.
7. **M5-E5/R5 scaling and terminal review:** write decision and inference-status
   tables, then complete the closeout before any new direction.
8. **M3P-E/R closeout:** the repaired `8x4 + 2x8 + 6x4` canary passed, but the
   full schedule hit its 14,400-second cap at 26/32 selection tasks.  Preserve
   the cap-stopped result, close M3P, and require a new reviewed resource/
   evidence plan before any further long run.

The detailed commands, artifact roots, caps, and repair triggers for the closed
M3-C run are in its subordinate plan and result. No Phase 9B command is
currently valid, and no new M3-C command is authorized without a new dated
subplan.  The same M3P task order and cap must not be relaunched: its measured
long-candidate cost makes that attempt non-discriminating.

## Between-phase repair protocol (binding)

After each executed phase:

1. Preserve the attempt directory and never overwrite a prior receipt.
2. Run the smallest exact regression and inspect the manifest/schema/hash.
3. Classify the outcome as harness, implementation, numerical, tuning,
   resource, candidate, or scientific-evidence failure.
4. Decide whether the failure invalidates the harness or only rejects the
   current candidate. Continue to the declared repair when no continuation
   veto fired.
5. Make the smallest repair without changing frozen target, data, measure,
   correction, hardware class, privacy boundary, or total phase cap.
6. Use a new output directory and seed namespace for the repair; preserve the
   failed attempt as evidence.
7. Run focused tests and record the actual command, wall time, hashes, and
   remaining budget in a closeout receipt.
8. Refresh the next subplan's defaults, assumptions, evidence contract,
   commands, budget, and stop conditions from that receipt.
9. Advance only after the receipt and refreshed subplan exist and the next
   phase entry gate passes. A localized pass is not a whole-program pass.

High acceptance, poor whitening, low ESS, a missed mode, or a rejected arm is a
candidate repair trigger, not automatically a whole-program blocker. A true
blocker is limited to the continuation-veto conditions in the research ledger.

## Budget and execution boundary

The completed localized attempt-05 consumed `494.5689085649792` seconds and
used a 1,402,670,592-byte peak TensorFlow allocator reading on GPU0. Those
figures are evidence for the localized route only. The M3 campaign executed
one 1,800-second canary attempt and one permitted corrected retry; the latter
hit the declared continuation veto. The unexecuted 7,800-second full replay is
not authorized under this terminal state.

Routine document, CPU fixture, compile, and focused-test work remains allowed
within the current scope. The user's explicit instruction authorized the
unchanged M3 canary/retry campaign; that authorization was consumed by the
terminal closeout. A long GPU launch still crosses the platform's trusted GPU permission boundary, with
`TF_FORCE_GPU_ALLOW_GROWTH=true` set before TensorFlow import. Retries under
this unchanged contract remain within the campaign cap; a new cap, profile,
target, hardware class, or privacy boundary requires a refreshed authorization.
Broad shell, interpreter, package-manager, network, or arbitrary-GPU
permissions are not needed. The M3-C continuation had its own 900-second GPU
and 300-second CPU diagnostic budget and fresh artifact root; it did not inherit
the M3 replay cap or partial-call budget. It consumed `834.5439817190636` GPU
seconds across its two GPU attempts. The first attempt is quarantined for
cross-arm comparison because of an arm-dependent validation bank; the second,
repaired attempt is the sole valid N2 comparison. No remaining budget
authorizes a changed contract.

The subsequent M3P process-parallel campaign used a separate declared cap of
1,200 seconds for each canary and 14,400 seconds for the full staged run.  The
authoritative canary attempt-06 passed in `585.254545083968` seconds.  Full
attempt-05 consumed its cap after 48/48 screen and 26/32 selection records;
the repaired closeout classified this as `M3P_FULL_CAP_BLOCKED`.  No further
M3P run is authorized under the same task order and cap.  A new cap,
partition, or hardware contract requires a new reviewed subplan and explicit
campaign budget.

## Skeptical master-program audit

The audit found four governance gaps and repaired them in this document:

| Finding | Repair |
|---|---|
| The 2026-08-25 particle-authority master was terminal for a different campaign | Mark it historical for this lineage and establish this dated master as the active authority. |
| The 2026-08-28 implementation plan contained phases but did not identify a single current master or explicit state transition after the localized repair | Add the phase map, ownership, current state, and entry/exit boundaries above. |
| The next replay draft initially lacked a concrete profile, launcher, and artifact root | The M3 subplan now binds `phase9a_full_replay_v1`, launcher interface, output root, per-call durability, and separate caps. |
| A localized mechanics pass could be misread as permission to run Phase 9B | The master keeps M4 blocked until six scope handoffs, sequential diagnostics, and a new Phase 9B plan pass. |
| Repair was described as guidance rather than a state transition | Add the binding `E_k -> R_k` protocol and require a closeout receipt before every next-stage entry. |

The audit also checked wrong baselines, proxy metrics, hidden cap widening,
stale seed reuse, output collisions, missing stop conditions, environment
mismatch, unexamined defaults, and whether a phase could advance without a
repair receipt. The result is
`PASS_MASTER_REPAIR_PROTOCOL_AND_M3_PROFILE_AUTHORIZATION`, followed by the
terminal `BLOCK_M3_CANARY_RESOURCE_VETO` and the M3P cap blocker recorded
below; it governs the recorded closeouts and keeps M4 closed.

## Execution ledger

### 2026-09-02 master refresh

- Created this active master program and linked the current parent, M2 result,
  reset memo, and M3 draft.
- Preserved the historical particle-authority master and added a successor
  pointer rather than rewriting its phase evidence.
- Confirmed the current M2 attempt-05 manifest hash and fresh-seed audit.
- Verified the M3 draft's concrete execution envelope and corrected its stale
  `7,200`-second reference to `7,800` seconds.
- Focused tests: 10 Phase 9A repair tests and 18 HMC-policy tests passed;
  documentation rendering, Python compilation, shell syntax, whitespace, and
  independent attempt-05 manifest/hash checks passed.
- Amended the program so every master stage has a mandatory repair/refresh
  closeout, including passed stages. Added separate canary/full replay profile
  identities, fresh seed namespaces, profile-bound material caps, and a
  source-owned launcher. The user's latest instruction supplies the plain-
  language authorization for the unchanged M3 campaign; platform GPU trust is
  still requested at launch.

### 2026-09-02 R0/R1 repair closeout

- R0 immutable evidence audit completed and recorded in
  `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-r0-audit-result-2026-09-02.md`.
- R1 added compile/steady-run telemetry, disjoint canary/full profiles, and a
  profile-bound-cap launcher. Focused tests and analytic mechanics checks
  passed; the adjacent ordinary-tuner oracle failure was classified as
  pre-existing budget/R-hat migration debt, not an M3 route failure.
- Mandatory closeout receipt exists in
  `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-r0-reset-memo-2026-09-02.md`.
- R2 attempt-01 exposed an invalid literal-brace launcher root and a bounded
  resource failure; it was preserved, not promoted, and classified in
  `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-r2a-repair-result-2026-09-02.md`.
- R2a repaired the root contract and added a `13 passed` regression receipt;
  the corrected retry then reproduced the bounded resource failure. The
  terminal M3 result and reset memo are now recorded, and no third retry is
  authorized. M4 remains blocked.

### 2026-09-02 M3 terminal repair/refresh closeout

- Attempt-02 is preserved under the corrected source-owned root. It matched the
  frozen identities and completed 21 calls in `1645.967775` seconds before the
  fixed `1800` second cap terminated the next call; outer wall time was
  `1852.926691` seconds including termination grace.
- The second bounded resource failure after R1 fired
  `BLOCK_M3_CANARY_RESOURCE_VETO`. The terminal result and reset memo include
  the evidence contract, cost decomposition, decision/inference tables,
  uncertainty limits, and red-team analysis.
- The full six-scope replay and Phase 9B remain blocked. The original M3 replay
  program has no valid GPU command. Any continuation must start with a new reviewed
  plan that explicitly changes the performance design or budget; it may not
  widen the cap or reuse partial calls silently.

### 2026-09-02 M3-C authority refresh

- The completed bounded diagnostic in
  `docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-repair-result-2026-09-02.md`
  is incorporated as subordinate evidence. It measured target batching,
  repaired a diagnostic graph-construction defect, validated affine and
  finite-difference score checks, and left grouped-HMC integration unadmitted.
- The continuation authority was
  `docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-next-plan-2026-09-02.md`.
  Its N0--N4 phases have now closed under
  `docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-next-result-2026-09-02.md`.
- M3 remains terminal for the original replay. M4/Phase 9B remains blocked.
  The M3-C diagnostics named by the continuation plan are now closed; no new
  continuation command is valid without a new dated subordinate plan.

The M3-C refresh was audited before execution: it preserves the frozen target
signature, separates exact-equivalence gates from descriptive timing and loss,
declares fresh seeds/data partitions, rejects validation leakage, bounds GPU/CPU
cost, and states that TFP batch RNG semantics may make the fast grouped path
unadmittable. Audit verdict: `PASS_M3C_BOUNDED_CONTINUATION`.

### 2026-09-02 M3-C execution and repair closeout

- N1's fresh CPU and GPU receipts agree: the scalar and explicit row-loop
  controls are exactly equivalent at the declared tolerance, while the fast
  grouped TFP transition differs in state, target, gradient, and
  log-acceptance. The grouped integration veto is therefore active.
- The first GPU ladder attempt is preserved but quarantined because its
  arm-dependent validation bank made cross-arm comparison invalid. The harness
  was repaired to share validation seeds `98000`, `98100`, and `98200` by seed
  index, and the second GPU attempt is the sole valid N2 comparison.
- The repaired ladder completed 9/9 finite candidates with 12/12 valid updates
  each. Arms A, B, and C each had 0/3 seeds meeting the provisional 10%
  score-RMS nomination threshold. No candidate was nominated; no default or
  active route changed.
- The route-specific regression passed `38 passed` across the tempered
  lineage/ensemble and fixed-transport step-cap tests. A broader five-file
  focused suite returned `113 passed, 2 failed`; the two failures are known
  ordinary-tuner migration debt and an absent private LGSSM fixture outside
  M3-C. They are recorded, not hidden, and do not satisfy a claim of a green
  repository-wide suite.
- The authoritative result and reset memo are
  `docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-next-result-2026-09-02.md`
  and
  `docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-next-reset-memo-2026-09-02.md`.
  The execution-time/current-document hash reconciliation is in
  `docs/plans/artifacts/ssl-lstm-q20-performance-whitening-next-2026-09-02/n4-r4-closeout.json`.
  M3 remains terminal, M3-C is closed, N3 is blocked, and M4/Phase 9B remain
  closed. Any further work requires a new reviewed subordinate plan.

## Required closeout artifacts

Every future serious phase must leave a result note and reset memo beside its
subplan, plus a versioned manifest under
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-<date>/`.
The terminal note must include decision and inference-status tables, a
post-run red team, uncertainty limits, and a precise next action or real
blocker. The M3 terminal result and reset memo now exist; M4 remains closed
until a new reviewed plan changes the performance design or budget.

### 2026-09-03 M3P 72-core execution and terminal closeout

- The staged process topology was implemented as three sequential barriers:
  screen `8x4` (32 worker cores), selection `2x8` (16), and scope finalization
  `6x4` (24), for a 72-core worker budget. CPU IDs were discovered from the
  controller affinity set; workers hid CUDA before TensorFlow import and
  enabled XLA.
- Canary attempt-05 passed before the full run. Full attempt-05 completed all
  48 screen records and 26/32 selection records before the declared 14,400
  second cap; both selection streams were still in long candidates and scope
  finalization did not start. The completed selection records are descriptive
  only and do not nominate a candidate.
- The first cap closeout exposed a harness defect: unset child return codes
  were reported as a generic worker failure and no typed partial summary was
  written. The localized repair added deadline receipts and non-throwing
  partial coverage. Focused tests passed (`9 passed`), and canary attempt-06
  passed in `585.254545083968` seconds after the repair.
- The authoritative M3P result and reset memo are
  `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-result-2026-09-03.md`
  and
  `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-reset-memo-2026-09-03.md`.
  The canary receipt is
  `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-canary-attempt-06-result-2026-09-03.md`.
  The subordinate plan is closed as `P3_FULL_CAP_BLOCKED`; no same-contract
  relaunch, cap increase, task repartition, or Phase 9B command is authorized.
