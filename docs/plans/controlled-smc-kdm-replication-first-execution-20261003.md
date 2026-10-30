# Replication-first execution contract: controlled SMC and KDM/IWSG

Completed continuation, 6 October 2026: the
[circular KDM consumer checks](controlled-smc-kdm-circular-consumer-result-20261006.md)
pass all 157 float64 checks; four of 157 float32 checks still fail after a
source-order repair. CPU localization also finds float32 disagreement. This is
bounded consumer evidence, not full training or default readiness.
The [DGELSD audit](controlled-smc-dgelsd-arithmetic-result-20261006.md) identifies
a Lorenz arithmetic mismatch without establishing a TensorFlow repair.
The [loader audit](controlled-smc-kdm-checkpoint-wiring-result-20261006.md)
shows that all 22 active bearings evaluations load absent NLL checkpoints,
including MSE-labelled tasks. The
[closure sequence](controlled-smc-kdm-gap-closure-20261006.md) governs next work.
Both replications remain incomplete; no numerical job is running. The owner then added 48 GPU hours and 48 CPU hours. The active balance is now
173464.9919998964 GPU seconds and 324216.641335726 CPU seconds.
[Proposed acceptance criteria](controlled-smc-kdm-acceptance-criteria-20261006.md)
define prospective numerical no-worse tests; previous diagnostic failures remain.
Full training still needs a costed plan. E0 remains deferred.

Updated 5 October 2026 after the fitting/forward and conditional source-solver diagnostics. Authority:
[master](controlled-smc-kdm-replication-master-program-20260928.md).
This contract supersedes the repair-first execution order. Historical plans
and attempts remain readable; historical repair commands are inactive.

The first question is whether our TensorFlow implementation reproduces the
author's method with no worse numerical behavior under matched conditions.
The owner's 4 October clarification accepts faithfully reproduced intrinsic
source failures as baseline outcomes. Successful paper-figure reproduction
and repair of the author's method are not prerequisites for port acceptance. Task 3/E0 numerical enhancements follow accepted source and port
replication. A source warning/failure is evidence, not an instruction to change
the baseline. A correction to our own translation is replication work when it
preserves the source objective, feasible set, update and stopping semantics.

## Dependency graph and scope

Per experiment: G0→C0→C1 baseline accounting→P0→P1; C1 feeds C2 paper analysis separately. C1 baseline accounting includes intrinsic author failures, not only successful runs.
Independently: G0→K0→K1→K2a→K2b. J collects the separate verdicts.
An unavailable or partial result remains incomplete; it does not automatically
activate E0. Source settings and the exact selected figures/models are in the
master. Additional heuristic performance requirements do not redefine replication.

## Bounded executable phases

| Phase | Plan | State after review execution |
|---|---|---|
| C0 source lock | [source lock](controlled-smc-kdm-c0-source-lock-20261003.md) | attempt139 PASS:52 verified source/data files |
| C1 source cost pilot | [pilot](controlled-smc-kdm-c1-source-cost-pilot-20261003.md) | attempts 140/141:12 selected cases; comparator timeout repaired within210-second cap |
| K0 source/config preparation | [K0](controlled-smc-kdm-k0-source-config-20261003.md) | attempt 142 PASS: five original-generator config sets for 11 trials |
| C1 full source matrices | [full source](controlled-smc-kdm-c1-full-source-20261003.md) | Attempt 143 interrupted; verification attempt144 PASS; attempt145 terminal: both neuroscience matrices2000/2000; known Lorenz covariance failure at3537/4000 retained; APF blocked |
| Source process isolation | [memory repair](controlled-smc-kdm-source-memory-repair-20261004.md) | Attempt 144 PASS: 24 author-source evaluations; exact outputs/matrices/RNG; 364.451811 CPU seconds |
| C2 source analysis | [continuation repair](controlled-smc-kdm-source-continuation-repair-20261004.md) | Implemented, 24 software checks PASS; BLOCKED by incomplete Lorenz matrices; ineligible queue retired |
| Frozen source failure replay | [saved-case replay](controlled-smc-lorenz-source-failure-replay-20261004.md) | COMPLETE attempt146: exact source failure reproduced unchanged;4.23 CPU seconds |
| P0 native-draw capture and forward consumer | [forward replay](controlled-smc-p0-forward-replay-20261004.md) | COMPLETE attempts147/148:22 valid stages pass with exact ancestry; failed-policy forward rejection remains unconfirmed;21.90 CPU and17.24 GPU seconds |
| P0 fitting/forward | [fitting plan](controlled-smc-p0-fitting-forward-20261005.md) | COMPLETE attempt149:32 valid TF fits/forwards, both solvers fail strict screens; all16 source-policy controls pass;168.896125 GPU seconds |
| P0 conditional solver reference | [solver plan](controlled-smc-p0-solver-semantics-20261005.md) | COMPLETE attempts150/151:128/128 branches in both;64/128 then128/128 coefficient screens. Octave-linked reference is bitwise exact; all60 SVD fallbacks rank45;0.630285 CPU seconds |
| P0 DGELSD source arithmetic | [arithmetic audit](controlled-smc-dgelsd-arithmetic-20261006.md) | COMPLETE bounded reference diagnostic; D&C mechanism localized on3 selected systems;1.046148 CPU seconds; no port admission |
| K0 checkpoint wiring | [loader audit](controlled-smc-kdm-checkpoint-wiring-20261006.md) | COMPLETE metadata check:22 active NLL checkpoint paths;0 locally present;0.566610 CPU seconds |
| K1 circular density/consumer | [consumer plan](controlled-smc-kdm-circular-consumer-20261006.md) | COMPLETE diagnostic: float64 157/157; float32 153/157; CPU also differs. Full numerical screen fails. 37.764496 GPU and 5.536095 CPU seconds; full neural training still absent |
| Remaining P0/P1, K1/K2 numerical continuation | Master phase table | TF conditional solver and full fitting/forward qualification remain; P1 margins, KDM source/training specifications and costs still required |

These metadata/pilot PASS values do not complete either replication. The full
source command is executable and refuses an insufficient allocation. The
entire roadmap is not yet an unattended executable.

## Evidence and acceptance

Before each numerical phase, freeze its question, source baseline, exact
settings, pass criterion, vetoes, explanatory diagnostics, uncertainty method,
nonclaims, budget, seeds and versioned output root. Preserve defaults with
provenance/risk/early checks. Test the actual consumer call chain.

Source comparison needs explicit case accounting and unchanged numerical
operations. Intrinsic author failures are valid comparison outcomes. Cases
never reached are unevaluated, not inferred failures or passes. C2 compares
paper statistics with appropriate uncertainty when the required results exist;
a blocked full-figure analysis does not block P0/P1 on audited source cases. P0 checks source-valid fixed inputs using output and
invariant tolerances justified by scale and conditioning; coefficient parity
alone is insufficient for ill-conditioned fits. P1 requires no additional invalid
fixed cases plus supported repeated-run downstream non-inferiority.
P0/P1 numerical margins are unresolved and must be frozen before claim runs;
a nonsignificant difference is not equivalence.

K training requires the correct Appendix B or bearings model, initial data and
information, source gradient estimator, objective, all relevant stages,
independent repetitions and downstream density/state evaluation. A saved
checkpoint, exact-reference test or fixed-sample gradient agreement is partial.

## Stop and repair rules

Stop the affected experiment for wrong target/data, corrupt source or output,
invalid reference, missing indispensable diagnostics, or exhausted compute.
A failed candidate can trigger the next planned source-preserving diagnosis
within the same contract and budget. No retry changes the scientific target
or spends the GPU grant on CPU. No source failure authorizes an E0 workaround.

Only external/irreversible work, material scientific changes or additional
compute need another owner decision. Routine repairs within an authorized
budget do not. Use one concise review and focused checks; do not introduce
approval tokens or review chains.

## Provenance and current decision

Supervisor snapshots preserve paths to avoid same-basename collisions; copied
runners take an explicit repository root. Author files are verified against
pinned Git blobs; all attempts have exact commands, seeds, environment,
CPU/GPU choice, wall charges and input hashes. CPU references hide CUDA.
GPU work follows memory-growth and trusted-execution requirements.

Full-source cost is about30.5 CPU hours, with67.9 hours conservative reservation.
The owner approved the 72-hour additional CPU wall-time cap on 4 October; the grant is recorded in the ledger and attempt145 has now finished both independent neuroscience matrices.
The owner added another 48 GPU hours and 48 CPU hours on 6 October.
The live active balance is 48.18 GPU hours and 90.06 CPU hours including the
previous remainder. The older 2812.44-second and 11-minute limits are historical.
The eleven-trial author/TF KDM scope still requires a costed training protocol.
See the live [checkpoint](controlled-smc-kdm-checkpoint.md) and
[execution result](controlled-smc-kdm-restructure-execution-result-20261003.md).


## Approved continuation, 4 October 2026

The owner approved the prepared 72-hour additional CPU allocation. This
supersedes the earlier pending-allocation state; no GPU budget was converted.
The final launch review passed. See
[the original launch record](controlled-smc-kdm-source-execution-20261004.md)
for the initial launch and [the current repair record](controlled-smc-kdm-source-memory-repair-result-20261004.md)
for the verified restart, actual source failure and completed diagnostics. The scientific scope, source settings and replication-first ordering
remain unchanged.

## Persistent monitoring, 5 October 2026

The [monitoring plan](controlled-smc-kdm-monitoring-20261005.md) adds one-minute
health checks, durable terminal/error reports and bounded recovery of dead
waiting queues. It preserves the author baseline, numerical attempt limits
and compute budgets. Desktop notifications depend on Windows notification
settings; the first test found global notifications disabled.

Terminal update,5 October: no numerical job or continuation remains running.
See [the reviewed result](controlled-smc-p0-forward-result-20261005.md). The
watchdog terminated after recording all phase outcomes; its terminal alert
requires review, not an automatic numerical retry. Full replication is incomplete.

## Next bounded work, 5 October 2026

The [readiness review](controlled-smc-kdm-continuation-readiness-20261005.md)
records the next actions. Its routine CPU smoke passed: the exact complex
source policy is rejected at the actual real-valued forward endpoint with
zero traces. This does not change attempt 148 or establish fitting parity.
The initial fitting/forward and conditional-reference phases are now complete.
The next serious numerical phase needs a TF condition/branch/rank specification
against the verified128-system reference, downstream criteria and an executable
bounded allocation. Current attempt bookkeeping is exhausted at151; the
unchanged campaign authorization permits localized repair allocations, not
unspecified full training. KDM source/training specifications and cost remain unfinished.

## Owner-accepted continuation, 6 October 2026

The [acceptance criteria](controlled-smc-kdm-acceptance-criteria-20261006.md) are
now accepted. Execute the [bounded neural consumer phase](controlled-smc-kdm-neural-consumer-20261006.md)
under the existing 48 GPU + 48 CPU hour extension. Its implementation checks
do not substitute for the later untouched no-worse qualification. Intrinsic
author failures remain acceptable; E0 and original iAPF remain deferred.
