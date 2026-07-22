# Contract E--TP All-Model Clean-XLA Visible Execution Ledger

Date: 2026-07-15

Program ID: `contract-e-tp-all-model-clean-xla-validation`

Authorized budget: 96 CPU core-hours, 32 trusted GPU-hours, and three
full-horizon attempts per eligible model.

## 2026-07-15 - Program - Authorization And Runbook Draft

State: `PRECHECK`

Evidence contract:

- Question: can eligible Contract E--TP rows use bounded functional TensorFlow
  loops for the same finite value and total score at declared horizon?
- Baseline: same finite scalar; scientific comparators remain separate.
- Primary criterion: per-row identity, parity, derivative, fail-closed,
  topology, and trusted GPU/XLA gates.
- Vetoes: wrong target, partial derivative, compiled Python unrolling, finite
  invalid state, fallback, invalid evidence, or budget exhaustion.
- Nonclaims: no accuracy, equivalence, superiority, canonical/default, HMC, or
  leaderboard conclusion.

Actions:

- recorded owner authorization of the reviewed 96 CPU/32 GPU-hour campaign;
- found that the master program had a subplan content contract but no binding
  close/draft/review/continue execution state machine;
- drafted the visible runbook and Phase 0 subplan;
- identified the existing six-row target registry as evidence from the earlier
  gradient-comparison program, not a clean-XLA topology registry.
- bounded runbook review returned `REVISE` on deterministic classification
  mapping, numeric budget readiness, terminal negative-result handling,
  evidence-linked readiness decisions, and self-contained review fallback;
- patched the same runbook without changing the target, criteria, or budget.
- focused runbook re-review returned `VERDICT: AGREE`.
- Phase 0 subplan review returned `REVISE`: make the structural fixture's item
  type explicit and bind exact check-log, result, Phase 1 subplan, and readiness
  evidence; patched the same subplan without changing its scientific scope.
- focused Phase 0 re-review returned `VERDICT: AGREE`;
- Phase 0 entry budget check: available `96 CPU, 32 GPU, 3 full-horizon/eligible
  row`; required minimum plus reserve `2 CPU, 0 GPU, 0 full-horizon`; `PASS`.
- Phase 0 source precheck found and repaired a master handoff typo: structural/
  high-dimensional Phase 8 admission comes from the SIR/DSGE Phase 7 re-audit,
  not the predator--prey Phase 6 GPU result.

Gate status: `PHASE0_EXECUTE_MINIMAL`.

Budget used by this entry: no serious CPU/GPU experiment time.

Next action: bounded review of the runbook, repair material findings, then
execute Phase 0.

## 2026-07-15 - Phase 0 - CLOSED_HANDOFF_READY

Gate status: `PASS`; focused suite `5 passed`; registry SHA-256
`704522e3909f446eb1b3d32584bb37cff90159a69b6c1af2f13a8d15fc1aacee`.

Classifications: LGSSM reference pass; actual/KSC/predator eligible; generalized
tested feature family negative; SIR/DSGE target blocked; structural fixture is a
required shared regression item.

Budget: 0.01 CPU core-hours recorded, zero GPU hours, zero full-horizon
attempts. Remaining: 95.99 CPU, 32 GPU, and three attempts per eligible row.

`NEXT_PHASE_READINESS`: all eight clauses pass as recorded in the Phase 0
result. Phase 1 subplan drafted; bounded review pending.

Next action: review Phase 1, repair material findings, then enter Phase 1
`PRECHECK` if ready.

Phase 1 bounded review returned `REVISE`: bind the exact Phase 0 result, exact
CPU-hidden commands, normalized JSON/log/result/ledger artifacts, and a neutral-
name aliased-`range(len(...))` hidden-loop fixture. The same plan was patched;
scientific scope and budget are unchanged.

## 2026-07-15 - Phase 1 - CLOSED_HANDOFF_READY

Guardrail JSON status `PASS_SHARED_SOURCE_AND_GRAPH_GUARDRAILS`; all seven
source controls and all six graph gates pass. Fresh `T=10,50` graphs reproduce
`4014` top nodes, `3712` function nodes, and four functional loops. Focused
suite: `22 passed, 2 warnings`.

Budget: 0.02 CPU core-hours recorded, no GPU, no full-horizon attempt. Remaining
`95.97 CPU,32 GPU,3 attempts/eligible row`.

`NEXT_PHASE_READINESS`: all clauses pass in the Phase 1 result. Phase 2
structural subplan drafted; bounded review pending.

Phase 2 bounded review returned `REVISE`: make the fixed-index support claim
explicit, define the full carried-state tangent and its numerical envelope,
remove the unsupported `128*eps` derivation claim, bind the same compiled
factory for invalid input, complete run-manifest fields, and reconcile cap
versus planned-plus-reserve budget. The same plan was patched without changing
the fixture or campaign scope.

Phase 2 pre-execution chart diagnostic: inherited indices `[1,4,6,11]` pass
step 0 (`min weight 0.1166384`) but fail step 1 (`min weight -0.1599078`).
Classified localized preparation failure. Predeclared repair: offline exhaustive
4-of-12 positive full-rank chart selection maximizing minimum weight with
lexicographic tie-break, then freeze per-step charts before parity/GPU evidence.

Phase 2 source-check preparation exposed a Phase 1 harness defect: nested
function definitions such as `cond/body` were not in the AST reachability
closure. Classified `plan_or_harness_failure`; repaired the guard to index and
audit nested definitions and added a nested dynamic-loop negative fixture.
Phase 1 controlling evidence must be refreshed before Phase 2 source admission.

Phase 2 structured CPU/XLA attempt 1 failed after local numerical tests passed:
XLA CPU rejected variant-backed `TensorArray` history with `TensorListReserve`.
Classified localized loop-history/XLA implementation failure. Repair replaces
variant TensorArrays with fixed-shape tensor carries and indexed updates; the
same scalar, charts, diagnostics, and criteria remain frozen. Attempt 1 is
preserved and the retry will use a fresh directory.

Phase 2 structured CPU/XLA attempt 2 still emitted `TensorListReserve`. The
fixed history repair was correct but insufficient: localization identifies
reverse `tape.jacobian` of loop histories as the remaining TensorList owner.
Repair retains reverse autodiff for the scalar score and computes diagnostic
history Jacobians with four fixed parameter-direction forward JVPs through the
same core. This is a derivative implementation repair, not a target change.

Phase 2 structured CPU/XLA attempt 3 passed focused forward-JVP tests but XLA
rejected reverse differentiation of dynamic `tf.repeat`/`tf.tile` Cartesian
expansion because the gradient reshape was not compile-time constant. Repair
uses static broadcast-plus-reshape with identical parent-major ordering, the
same pattern already validated in the LGSSM clean-XLA repair.

Phase 2 structured CPU/XLA attempt 4 passed parity after static Cartesian
repair but XLA rejected forward autodiff through fixed projection gather as a
higher-order `GatherV2` transpose with nonconstant shape. Repair carries all
four parent-point tangent directions analytically inside the same loop and
gathers fixed selected point tangents. Reverse autodiff remains the independent
scalar-score owner; projected weights remain in that scalar path.

## 2026-07-15 - Phase 2 - CLOSED_HANDOFF_READY

CPU/XLA passed at `T=2,5`; trusted GPU/XLA passed at `T=2` on the first trusted
attempt. CPU/GPU value difference `4.44e-16`, worst score difference `3.33e-16`.
Same concrete factory invalid input returned false and poisoned all claim-bearing
outputs. Final focused suite `21 passed, 2 warnings`.

Budget: 0.10 CPU and 0.01 GPU hours. Remaining `95.87 CPU,31.99 GPU`, three
full-horizon attempts per eligible row. `NEXT_PHASE_READINESS` passes; Phase 3
subplan drafted for bounded review.

Phase 3 skeptical audit repaired a wrong graph baseline before review: `T=2`
has no intermediate filter-loop iteration, so graph scaling now uses `T=3`
versus `T=10`; `T=1,2` remain edge-semantics checks.

Phase 3 bounded review returned `REVISE`: bind provenance/falsification for
order/lookahead/chart choices, label topology ratios as heuristics, and provide
exact CPU-hidden commands and artifact paths. The same subplan was patched
without changing actual/KSC targets or budget.

Phase 3 execution repaired traced model validation, zero-iteration reverse
TensorLists, and static lookahead-dependent topology without changing either
row scalar. Actual SV and KSC-SV both passed the center-scoped CPU/XLA ladder
through `T=100`; the final focused suite passed `98` tests. The controlling
Phase 3 result records all failed attempts and exact hashes.

## 2026-07-15 - Phase 3 - CLOSED_HANDOFF_READY

Actual SV `T=100` value/score are `-226.4735505123` and
`[1.0188616582,2.9593282725]`; KSC-SV are `-226.2889233996` and
`[0.7820828289,3.1248416595]`. Maximum same-scalar FD relative errors are
`1.09e-8` and `7.06e-8`. Top/function-node ratios are `1.0`; GraphDef-byte
ratios are `1.0012233` and `1.0010435`. Both exact factories fail closed at the
frozen invalid theta.

Budget: 0.25 CPU core-hours, no trusted GPU time, and no full-horizon attempt.
Remaining `95.62 CPU,31.99 GPU,3 attempts/eligible row`.

Phase 4 first review returned `REVISE` on CPU/GPU comparison semantics, exact
commands, memory growth, FD definition, artifact fields, budget accounting,
provenance, and invalid-output coverage. A second focused pass required removal
of an unsupported floating-point envelope, create-or-fail attempt roots,
bounded CPU threads/accounting, and poison checks for every increment. The same
plan was repaired. Focused re-review found no material issue and returned
`VERDICT: AGREE`.

`NEXT_PHASE_READINESS`: every Phase 3 clause now passes. Phase 4 status is
`REVIEWED_ACTIVE_EXECUTION`; its center-only boundary and 24 CPU-core-hour / 16
GPU-hour cap are binding.

Phase 4 implementation added the `T=1000` scalar-SV preparation horizon, a
trusted GPU preflight, a CPU/GPU XLA harness, and complete invalid-output
poisoning. Focused implementation checks passed `29` tests before execution.
Actual and KSC preparations both passed 999 center charts.

Actual CPU/XLA center execution passed value, score, topology, replay, and
invalid-control gates but failed the predeclared FD endpoint-validity gate: the
negative gamma endpoint and positive log-beta endpoint were invalid and
poisoned. The first attempt also exposed a reporting-only harness exception on
nonfinite FD values; it was preserved and repaired under a fresh root. No
Actual GPU attempt was consumed and no chart/step was changed post-result.

KSC CPU/XLA passed all gates, then trusted GPU preflight verified the RTX 4080
SUPER, memory growth, and actual GPU placement. KSC `T=1000` trusted GPU/XLA
passed every gate on attempt 1. Value/score are `-2284.098531972001` and
`[2.456831351828945,-4.420876400418747]`; maximum same-scalar FD relative
error is `3.19e-8`. Graph ratios are `1.00448/1.0/0.99972`. Focused close suite
passed `35` tests.

## 2026-07-16 - Phase 4 - CLOSE_RECORD_NEXT_PLAN_REVIEW

Classification: KSC passes the center-scoped full-horizon engineering ladder;
Actual is a row-specific negative candidate at the FD endpoint-validity gate.
No shared continuation veto fired. Conservative phase charge: `0.20 CPU
core-hours`, `0.05 GPU-hours`; KSC consumed one full-horizon attempt.

Remaining: `95.42 CPU,31.94 GPU`; three predator attempts and two KSC attempts.

The Phase 5 draft initially contained an unsupported post-result configuration
selection rule. Skeptical audit removed it: order-5/lookahead-4 is frozen only
as an inherited engineering baseline; adjacent refinements are descriptive and
cannot select a preferred configuration. The revised draft also binds
cross-module RK4 source audit, exact fresh paths, commands, timeouts, and
failure/nonclaim boundaries. Bounded review is pending.

Phase 5 bounded review returned `REVISE`. Material findings: bind exact
theta/data/source/seed hashes; make padded tail counts `4,3,2,1` explicit;
freeze a graph-native invalid theta and preflight artifact; record
HiGHS/SciPy/cutoff semantics; justify topology/parity gates; bind logs/manifests;
and separate numerical candidate rejection from implementation repair. The
same subplan was patched. Two findings were stale or overbroad: the current
draft already created `T=1`, and the reviewed master explicitly permits
NumPy/SciPy for offline preparation. The preparation remains offline but now
has stronger reproducibility and identity requirements. Focused re-review is
pending.
