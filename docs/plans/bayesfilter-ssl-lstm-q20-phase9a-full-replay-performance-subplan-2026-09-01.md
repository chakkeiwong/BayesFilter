# SSL-LSTM q=20 Phase 9A full-replay performance subplan

Date: 2026-09-01 (amended 2026-09-02)  
Status: `M3_TERMINAL_R2_RESOURCE_CONTINUATION_VETO`  
Predecessor: `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-chart1-beta0-program-repair-result-2026-09-01.md`
Master program: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`
R0 result: `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-r0-audit-result-2026-09-02.md`
R0 reset memo: `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-r0-reset-memo-2026-09-02.md`
R2a result: `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-r2a-repair-result-2026-09-02.md`
R2a reset memo: `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-r2a-reset-memo-2026-09-02.md`
M3 result: `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-performance-result-2026-09-02.md`
M3 reset memo: `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-performance-reset-memo-2026-09-02.md`

## Purpose

This is the next campaign after the localized chart-1/beta-0 runner repair. It
asks whether the same frozen q=20 target and C5 transport protocol can produce
all six scope-specific fixed-kernel tuning handoffs within a measured GPU/XLA
budget, followed by one mechanics-only shared transition. It does not authorize
Phase 9B retained sampling and it does not treat the four-draw attempt-05
handoff as adequate tuning evidence.

## Mandatory repair/refresh closeout

This subplan follows the master program's binding `execute -> repair/refresh ->
next execution` protocol.  Each phase below has a closeout, including a phase
that passes.  The closeout must preserve the attempt, classify the outcome,
run the smallest exact regression, repair what is locally repairable without
changing the frozen scientific contract, and refresh the next command,
assumptions, budget, and stop conditions.  No later phase may start while its
predecessor's closeout receipt is missing.

| Execution phase | Required closeout before advancing | Real blocker that stops continuation |
|---|---|---|
| R0 evidence/timing audit | Record immutable hash/seed audit, cost decomposition, and fresh namespace receipt; refresh R1 profile inputs | Required records are corrupt or cannot answer identity/cost questions |
| R1 profile/performance repair | Run focused tests, record equivalence and trace telemetry, bind profile/launcher hashes, refresh R2 command | Numerical equivalence cannot be established or the repair changes the frozen target/route |
| R2 canary | Preserve canary, classify resource/candidate evidence, recompute projected cost, and refresh R3/full seed ledger | A second bounded resource failure after R1, corrupted artifacts, or exhausted M3 budget |
| R2a repair/refresh after an invalid launch envelope | Repair the localized launcher defect, run the exact regression, preserve attempt-01, and refresh one unchanged R2 retry | The launcher defect cannot be repaired, or the retry would require changing target, route, seed, hardware, or cap |
| R3 six-scope replay | Write terminal decision/inference tables and refresh or explicitly block Phase 9B | Target/bridge identity failure, missing durable scopes, or declared campaign budget exhaustion |
| R4 terminal review | Preserve the result and write the next Phase 9B entry or blocker | A stated continuation veto only |

High acceptance, seed sensitivity, poor whitening, and a rejected tuning arm are
repair triggers; they do not stop continuation by themselves.

## Research intent ledger

| Item | Definition |
|---|---|
| Main question | Can a performance-repaired measured-grid route complete fresh, disjoint tuning for every `(chart,beta)` scope without target, map, movement, memory, or provenance vetoes? |
| Mechanism under test | Stable compiled fixed-kernel execution with durable per-scope resume points; no target, bridge, chart objective, or HMC acceptance-policy change |
| Expected failure mode | Long static TFP/XLA chain graphs compile too slowly, repeated trainer construction adds overhead, or some scope has no mobile candidate in the declared grid |
| Promotion criterion | Six fresh scope handoffs plus one finite mechanics-only shared transition, with every declared pair measured and disjoint held-out verification |
| Promotion veto | Target/backend/signature mismatch; nonfinite value/score/map; reused tuning state; missing pair; selected-candidate no movement; held-out failure; XLA or memory-growth failure; output collision |
| Continuation veto | Performance equivalence cannot be established; repeated canary exceeds its bound after the planned repair; artifacts cannot be resumed without changing frozen maps; campaign budget exhausted |
| Repair trigger | High acceptance, poor movement/ESS, trainer retracing, long trace/compile time, one scope with no viable pair, or inadequate grid resolution |
| Explanatory diagnostics | Acceptance, binary acceptance, ESJD, ESS per gradient, R-hat from short tuning rows, compile/steady timing, allocator use, and pullback residuals |
| Must not be concluded | No IID-Gaussian whitening, mode discovery, convergence, posterior correctness, sampler superiority, high-dimensional scaling, production, or default readiness |

## Fixed identity

- Target signature:
  `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`.
- Principal-square-root backend: `tensorflow_eigh_strict`.
- C5 protocol: `phase8-k2-compact-high-l3-pure`, two `(16,16)` tanh charts,
  betas `(0,.5,1)`, pure continuation, fixed chart weights `(0.5,0.5)`.
- Tuning policy: `measured_joint_grid_v1`; identity z-mass remains a tested
  baseline rather than a universal default.
- Execution: TensorFlow/TFP, GPU0, XLA on, TF32 recorded, memory growth before
  initialization, no pfor or row-mapped scalar target route.

Changing any of these identities requires a different plan. In particular,
this campaign cannot solve poor tuning by widening the epsilon cap or reverting
to directional acceptance-only adaptation.

## Concrete execution envelope

R1 creates two source-owned profiles, one for the canary and one for the full
replay, and a launcher named
`scripts/run_ssl_lstm_q20_phase9a_full_replay_gpu.sh`. The historical profile
is not suitable: it has an old seed namespace and is retained only for
reproduction of archival diagnostics. The new profiles bind fresh roots for
both charts, all six `(chart,beta)` scopes, and the separate canary, selection,
held-out, and transition roles. The launcher interface is fixed before R2:

```text
BAYESFILTER_PHASE9A_ATTEMPT_ID=<fresh-attempt-id> \
bash scripts/run_ssl_lstm_q20_phase9a_full_replay_gpu.sh \
  --profile phase9a_full_replay_canary_v1 \
  --scope-start 3 --scope-limit 1
```

R2 uses that command for the chart-1/beta-0 canary and writes below
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-09-02/phase9a-full-replay/`.
After R3 freezes the canary-independent seed ledger, R4 uses the same
launcher with `--profile phase9a_full_replay_v1 --scope-start 0
--scope-limit 6` and a new attempt ID. The
launcher must set `CUDA_VISIBLE_DEVICES=0` and
`TF_FORCE_GPU_ALLOW_GROWTH=true` before import, reject an existing output
directory, and preserve `run_start.json`, per-scope/per-call progress records,
`run_manifest.json`, and a result note on both success and failure. R2 has one
`1,800`-second cap; R4 has one `7,800`-second cap. A retry that changes a
profile, seed ledger, target identity, hardware class, or cap requires a
refreshed plan.

The canary profile is pinned to `(scope-start, scope-limit)=(3,1)` and uses
roots in the `20260902/780xx--785xx` namespace.  The full profile owns the
six-scope default `(0,6)` and uses the disjoint `20260902/790xx--795xx`
namespace.  The runner records the profile-bound material cap and rejects a
profile/scope mismatch; the launcher maps only the two combinations above.

## Defaults and assumptions

| Choice | Provenance | Status | Failure mode | Earliest diagnostic |
|---|---|---|---|---|
| Eight-pair grid `epsilon=(.25,.55,.85,1.20)`, `L=(3,8)` | The uncompleted v2 repair profile; attempt-04 rejected `1.20` for no movement while fresh attempt-05 moved at both `1.20` pairs, so `.85` supplies an interior scale without assuming monotonicity | Target-specific baseline hypothesis, not a default | Seed sensitivity may require more replications or a wider grid | Measure all pairs; do not infer neighbors; classify an empty viable set as a repair trigger |
| Screen `4+2`, two selection replications `16+4`, held-out `16+4` | Preserved v2 hypothesis that exceeded the old runtime boundary | Unproven evidence budget | XLA compile or runtime may dominate; chains remain too short for convergence | Performance canary before full replay; descriptive R-hat only |
| Four varied initial states | Localized repair profile and guide requirement against central-start aliasing | Reviewed mechanics baseline | Starts do not represent separated posterior regions | Record chain movement and maximum displacement by chain |
| 7,800-second full-replay cap | Derived from the attempt-05 sum of `336.644644` seconds for 13 full-chain calls: scaled to 150 calls and doubled for six adapter compilations/training overhead gives `7768.722552` seconds, rounded upward | Historical campaign ceiling; unexecuted after the M3 veto | Underestimates compile cost or hides runaway tracing | Per-call, per-scope, and trace/steady timers; any reuse requires a new plan |
| One GPU0 process | Repository GPU default and attempt-05 capacity evidence | Reviewed execution baseline | Resource contention or allocator growth | Trusted pre/post GPU snapshots and TensorFlow allocator telemetry |

The v2 counts are still too short for a posterior or convergence claim. Their
purpose is to make fixed-kernel selection less fragile than the v3 feasibility
probe. Any claim-bearing sequential run must use the repository sequential HMC
policy after this preflight, in a later Phase 9B plan.

## Phases

### R0: immutable evidence and timing audit (execution, then closeout)

1. Validate attempt-04 hashes, fixed identities, per-call records, and two
   reusable runner trace counts.
2. Separate chart training time, first-call compile time, and steady call time.
3. Confirm no existing output path or seed namespace will be reused.

**Exit:** an exact cost model, a fresh seed/output ledger, and a concrete
launcher/profile receipt. If the records cannot distinguish compile from
steady execution, instrument before any new GPU run.  The R0 closeout records
the audit receipt and refreshes R1.

### R1: stable-budget performance/profile repair

1. Add explicit timing around runner construction, first trace, and steady
   calls for each static `(results,burnin,trace schema)` contract.
2. Define and test the source-owned `phase9a_full_replay_canary_v1` and
   `phase9a_full_replay_v1` profiles and the exact launcher interface in the
   execution envelope above; never reuse historical seeds or output paths.
3. Test whether the v2 `4+2` and `16+4` schedules compile under the same XLA
   route. If static `sample_chain` graph growth is the cause, implement the
   smallest TensorFlow-native `tf.while_loop` or chunked one-step route that
   preserves the exact HMC kernel, stateless seeds, retained/burn-in semantics,
   target-status trace, and diagnostics.
4. Prove numerical equivalence on an analytic Gaussian and the frozen q=20
   mechanics fixture. Require the same accept/reject states and samples when
   exact seed equivalence is available; otherwise predeclare a tolerance and a
   distributional comparison before looking at results.
5. Keep an explicit stable `input_signature`, XLA on, and pfor absent.

**Exit:** focused tests pass, trace counts are bounded, and the repaired route
is equal to the current route for the declared target. A faster result without
equivalence is a continuation veto.  The R1 closeout records the equivalence
receipt, profile/launcher hashes, and the refreshed R2 command.

### R2: chart-1/beta-0 v2 canary

Run `phase9a_full_replay_canary_v1`, the fresh eight-pair v2 profile with its
original fresh counts, in one new
output directory, 1,800-second wall cap, GPU0 memory growth, and durable call
records. All eight pairs must be measured. Selection and held-out draws are
discarded.

**Exit:** at least one finite/mobile held-out handoff and complete performance
telemetry. A high-acceptance or no-viable result refreshes the grid; it does not
authorize cap widening. The first attempt reached the resource bound while also
violating the source-owned output-root contract; it is therefore an invalid
launch envelope rather than a completed R2 closeout. The R2a repair below is
the only retry permitted under this subplan.

### R2a: repair and refresh after attempt-01

Attempt-01 is preserved under its literal-brace directory. Its durable records
show 21 completed calls and `1645.963995` seconds, with exactly two first-trace
events and subsequent runner reuse. This timing is an explanatory performance
diagnostic; it is not a candidate result because the launcher wrote outside the
bound source-owned root.

R2a removes the extra brace, adds a source-contract regression, reruns the
focused Phase 9A suite, and refreshes the exact `attempt-02` command with the
same eight-pair grid, seeds, GPU, XLA, memory-growth policy, and 1,800-second
cap. It does not reduce the grid, omit selection or held-out calls, widen the
cap, or reuse partial calls.

**R2a exit:** the repaired launcher and focused checks pass and the retry
command is recorded in a reset memo. The corrected retry reached the same
bounded resource failure, so the R2 continuation veto fired and the six-scope
replay is blocked. A future attempt requires a new reviewed plan; this plan
authorizes no further retry.

### R3: between-phase repair and plan refresh before full replay

Recompute the six-scope cost from R2, freeze the exact grid and seed namespace,
and update this plan before launch. Check specifically for unfair reuse of the
chart-1/beta-0 canary, unsupported acceptance thresholds, missing held-out
separation, proxy metrics promoted to criteria, and a full-run command that
cannot resume from durable fresh checkpoints.

If R2 data were used to choose the grid, R2 is calibration only. The full
replay must rebuild with a new seed namespace and untouched held-out seeds.
The R3 closeout would have been the entry receipt for the six-scope execution,
but it is closed by the R2 continuation veto in this campaign. No full replay
command is valid under the present plan.

### R4: full six-scope Phase 9A replay

Rebuild both charts, tune all six chart-major scopes, and preserve each scope
immediately. Run the shared transition mechanics only after six valid handoffs
exist in the same frozen campaign. Stop on any hard veto; use candidate or
acceptance failures to trigger the declared repair path while budget remains.

**Pass:** all six handoffs, one finite shared transition, complete manifest,
and no hard veto. **Fail:** classify target, chart, tuning, performance,
memory, or provenance failure and preserve prior scopes. A partial pass cannot
be described as Phase 9A completion.  The R4 closeout writes the terminal
decision and refreshes or blocks Phase 9B; it is mandatory even on pass.

### R5: terminal review and next-plan refresh

Write a result with decision and inference-status tables, exact run manifest,
uncertainty limits, and a post-run red team. A passing preflight permits a new
Phase 9B plan; it does not itself authorize retained sampling.

## Skeptical pre-execution audit

Verdict at pre-execution review: `PASS_FOR_R0_R1_REPAIR_AND_R2_CANARY; R3_FULL_REPLAY_CONDITIONAL`.
That verdict is historical for the launch authorization; the terminal M3
result below supersedes it after the corrected retry.

The audit found that launching the v2 profile unchanged would repeat a known
runtime failure and that the v3 profile would answer only a mechanics question.
The performance discriminator in R1 was required before any new GPU candidate
run. The audit also rejects treating attempt-04 acceptance, ESS, or
R-hat as a tuning promotion result; they are descriptive and repair-trigger
evidence. The target, bridge, C5 architecture, identity mass baseline, and
acceptance bands remain frozen. The 7,800-second ceiling remains a profile-
bound ceiling; it cannot be widened by a caller.

No wrong baseline, proxy promotion criterion, hidden cap widening, stale seed
reuse, missing stop condition, or Phase 9B launch is permitted by this draft.
R1's instrumentation and profile/launcher checks now answer the first
engineering question; R2 must still establish bounded canary cost before R3
can open the full replay.

### Post-audit repair, 2026-09-01

The first draft did not name an executable full-replay profile, launcher, or
artifact root, which would have left those controls to caller improvisation.
That planning defect is repaired above. The profile/launcher contract, fresh
seed and output requirements, per-call durability, separate R2/R4 caps, and
mandatory closeouts are now explicit. The user's 2026-09-02 instruction to
continue is the campaign authorization for the unchanged M3 contract; platform
GPU trust is still requested at the launcher boundary.

## Execution ledger

### 2026-09-02 R0/R1 closeout

- R0 audited attempts 03--05 without modifying them. Attempt-05's embedded
  canonical manifest hash recomputed successfully; historical seed namespaces
  are disjoint from the new profiles.
- R1 added additive first-trace/steady-run telemetry and configuration records
  around the existing `FixedTransportReusableRunnerPool`; no HMC or target
  math was changed.
- Added and tested `phase9a_full_replay_canary_v1` (scope 3/1, cap 1800 s,
  seeds 780xx--785xx) and `phase9a_full_replay_v1` (scope 0/6, cap 7800 s,
  seeds 790xx--795xx), plus the exact launcher.
- Focused Phase 9A tests: `12 passed` at R1 (the R2a root regression raised the
  current focused count to `13`); Python compilation, shell syntax,
  whitespace, analytic transition tests (`12 passed`), and reusable-runner
  mechanics subset (`2 passed`) passed. The adjacent ordinary-tuner oracle
  test remains a pre-existing budget/R-hat migration issue and is outside this
  M3 contract.
- Historical pre-canary closeout status: `R0_R1_PASS_R2_AUTHORIZED`; its planned
  canary was then executed as attempt-01 and superseded by the R2a and terminal
  M3 closeouts below.

### 2026-09-02 R2a repair closeout after attempt-01

- Attempt-01 was preserved under the literal-brace root and classified as a
  combined launcher-envelope and resource/execution failure. It is not a
  tuning candidate and no selection or held-out claim is drawn from it.
- Timing evidence: 21 complete calls, `1645.963995` seconds; eight screen
  calls completed, thirteen of sixteen replicated-selection calls completed;
  first traces occurred only for the two static contracts. The remaining work
  could not be assumed to fit the fixed cap.
- Removed the launcher brace and added an exact-root regression. The focused
  Phase 9A suite now passes `13` tests; shell syntax and whitespace checks pass.
- R2a result and reset memo record the new source hashes, fresh `attempt-02`
  output directory, and exact retry command. The same cap and seed namespace
  are retained; no cap widening or profile substitution is authorized.
- The corrected retry was then executed as attempt-02 and reached the same
  bounded resource failure. The terminal M3 result and reset memo below close
  the campaign; no third retry is authorized.

### 2026-09-02 M3 terminal closeout after attempt-02

- Attempt-02 used the exact source-owned root and fresh output directory. The
  target/profile/seed identities matched; 21 calls completed in
  `1645.967775` seconds and call 21 was terminated by the fixed `1800` second
  cap. The outer wall was `1852.926691` seconds including termination grace.
- This is the second bounded resource failure after R1. The result is
  `BLOCK_M3_CANARY_RESOURCE_VETO`; it rejects this schedule under this cap,
  not the transport research direction.
- The required terminal result and reset memo contain decision and
  inference-status tables, cost and trace evidence, default/assumption limits,
  and a post-run red team. Attempts 01 and 02 remain immutable evidence.
- The six-scope replay and Phase 9B are blocked. Any continuation requires a
  new reviewed plan with an explicitly changed performance design or budget;
  partial calls, hidden call reductions, and cap widening are forbidden.
