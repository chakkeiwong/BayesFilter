# Contract E--TP All-Model Clean-XLA Visible Execution Runbook

Date: 2026-07-15

Program ID: `contract-e-tp-all-model-clean-xla-validation`

Status: `ACTIVE_AUTHORIZED_PHASE0_PRECHECK_REVIEWED`

Supervisor and executor: Codex in the current visible conversation. Claude may
perform bounded read-only review. It cannot edit, execute, authorize a phase,
change a target, or change a scientific boundary.

## Purpose And Authority

Execute the reviewed master program at
`docs/plans/bayesfilter-contract-e-tp-all-model-clean-xla-validation-master-program-2026-07-15.md`
under the owner-authorized cap of 96 CPU core-hours, 32 trusted GPU-hours, and
three full-horizon attempts per eligible model.

This runbook supplies the operational loop absent from the master program. It
does not authorize package or environment mutation, destructive operations,
external messages or release, credentials, private-data boundary changes,
additional compute, a new target law, a new client model, or a material change
to a baseline, promotion criterion, veto, hardware class, or scientific scope.

Execution remains visible and recoverable in the current conversation. Do not
launch a detached Codex supervisor, `nohup`, `setsid`, detached `tmux`, or a
copied-workspace campaign.

## Control Artifacts

| Artifact | Role |
| --- | --- |
| this runbook | binding visible state machine and repair policy |
| `docs/plans/bayesfilter-contract-e-tp-all-model-clean-xla-validation-visible-execution-ledger-2026-07-15.md` | chronological states, commands, artifacts, budgets, and gates |
| `docs/plans/bayesfilter-contract-e-tp-all-model-clean-xla-validation-visible-stop-handoff-2026-07-15.md` | current interruption or terminal handoff |
| one phase subplan | exact entry, evidence, checks, budget, repairs, and handoff before phase execution |
| one phase result | close record, checks actually run, classifications, budget use, and next handoff |
| `docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/` | versioned structured experiment evidence and quiet logs |

Every command that can create scientific or engineering evidence uses a fresh
path `phase-<NN>/<row-or-shared>/attempt-<NN>-<timestamp>/`. Failed or superseded
attempts remain referenced; no prior result is overwritten.

## Phase Index

| Phase | Name | Required subplan and result |
| ---: | --- | --- |
| 0 | registry, topology inventory, default audit | `...phase0-registry-topology-default-audit-{subplan,result}-2026-07-15.md` |
| 1 | shared clean-loop primitives and guardrails | `...phase1-shared-clean-loop-guardrails-{subplan,result}-2026-07-15.md` |
| 2 | early structural support regression | `...phase2-structural-support-regression-{subplan,result}-2026-07-15.md` |
| 3 | scalar-SV loop-native core | `...phase3-scalar-sv-loop-core-{subplan,result}-2026-07-15.md` |
| 4 | scalar-SV trusted full-horizon GPU/XLA | `...phase4-scalar-sv-gpu-xla-{subplan,result}-2026-07-15.md` |
| 5 | predator--prey loop-native core | `...phase5-predator-prey-loop-core-{subplan,result}-2026-07-15.md` |
| 6 | predator--prey trusted GPU/XLA | `...phase6-predator-prey-gpu-xla-{subplan,result}-2026-07-15.md` |
| 7 | SIR/DSGE boundary re-audit | `...phase7-sir-dsge-boundary-reaudit-{subplan,result}-2026-07-15.md` |
| 8 | admitted structural/high-dimensional GPU/XLA | `...phase8-structural-highdim-gpu-xla-{subplan,result}-2026-07-15.md` |
| 9 | cross-model regression/compiler red team | `...phase9-cross-model-compiler-red-team-{subplan,result}-2026-07-15.md` |
| 10 | terminal synthesis | `...phase10-terminal-synthesis-{subplan,result}-2026-07-15.md` |

The abbreviated filenames in this table use the fixed prefix
`bayesfilter-contract-e-tp-all-model-clean-xla-validation-`.

## Research Intent Ledger

| Field | Binding interpretation |
| --- | --- |
| Main question | whether every scientifically eligible row can run its already-defined finite Contract E--TP value and total score through bounded functional TensorFlow control flow at its declared horizon |
| Baseline | same finite scalar in a frozen unrolled or independent finite-program route; scientific comparators remain separate |
| Promotion criterion | the per-row target, numerical parity, total derivative, fail-closed, graph topology, and trusted GPU/XLA gates in the master all pass |
| Promotion veto | wrong target/data/time order, partial score, Python-unrolled compiled dynamic loop, finite invalid state, CPU fallback, or failed graph gate |
| Continuation veto | invalid target law, missing executable scalar, infeasible fixed-shape state, rejected feature family, corrupted evidence, or exhausted authorized budget |
| Repair trigger | localized implementation, shape, functional-loop, TensorArray, autodiff, XLA, serialization, harness, or artifact defect that preserves the frozen question |
| Explanatory only | compile/warm time, memory, register spills, condition numbers, feature residuals, and descriptive comparator gaps |
| Forbidden conclusion | clean XLA does not establish filtering accuracy, method equivalence, superiority, canonical/default status, HMC validity, or leaderboard readiness |

## Mandatory Phase State Machine

Every phase executes this exact state machine. A phase is not closed merely
because its local implementation or experiment passed.

1. `PRECHECK`
   Read the current subplan and prior result. Replay artifact hashes, remaining
   budget, environment intent, entry conditions, and row eligibility. Record a
   skeptical audit covering wrong baselines, proxy promotion, missing stops,
   unfair comparison, silent defaults, stale context, environment mismatch,
   and whether the commands can answer the phase question. Patch and re-audit a
   materially flawed subplan before executing it.
2. `EXECUTE_MINIMAL`
   Run the smallest discriminating local check first. Implement and test only
   the current phase scope. CPU diagnostics explicitly hide CUDA before
   TensorFlow import. GPU/CUDA/XLA work uses trusted/escalated execution. Keep
   full noisy output in a declared log and show only a bounded summary.
3. `ASSESS_GATE`
   Run every required local, numerical, source, graph, fail-closed, and device
   check in the subplan. Separate hard veto, descriptive, and statistical
   evidence. Classify each row or shared component directly as `pass`,
   `negative_result`, `target_blocked`, `implementation_blocked`, or
   `invalid_evidence`.
4. `WRITE_CLOSE_RECORD`
   Write the phase result with commands actually run, git/environment/device
   provenance, wall time, attempts, artifacts and SHA-256 hashes, budget used
   and remaining, failures and repairs, decision table, inference-status table
   when stochastic evidence exists, nonclaims, and a post-run red team.
5. `DRAFT_NEXT_SUBPLAN`
   Before closing the phase, draft or refresh the next phase subplan from the
   actual result. It must contain objective, inherited entry conditions,
   eligible rows, assumptions/defaults, required artifacts, checks and reviews,
   evidence contract, forbidden claims/actions, exact handoff, budget, repair
   triggers, and stop conditions. Do not reuse a stale speculative next plan.
6. `REVIEW_NEXT_SUBPLAN`
   Review consistency, mathematical and statistical correctness, feasibility,
   artifact coverage, target boundaries, compute, and whether its commands
   answer the next question. Use one bounded Claude read-only review for a
   material scientific or expensive boundary; Codex remains the decision owner.
   Reviewer unavailability or purely procedural disagreement is recorded and
   does not block adequate local evidence. A material target, mathematics,
   privacy, cost, or destructive-action finding does block until repaired.
7. `ADVANCE_OR_REPAIR`
   If the next subplan is ready, mark the current result `CLOSED_HANDOFF_READY`
   and immediately enter the next phase `PRECHECK` without asking for another
   approval while the frozen campaign scope and budget are unchanged. If it is
   not ready for a fixable reason, enter the repair loop below. Stop only when a
   true stop condition fires.

The phase result and execution ledger must contain a `NEXT_PHASE_READINESS`
table with one pass/fail row and exact evidence path for every clause in the
Ready Gate below. An overall `READY` is legal only when every row passes. A
prose assertion that the next plan is ready is not sufficient.

## Classification-To-Action Map

| Gate classification | Required action classification | Binding next action |
| --- | --- | --- |
| `pass` | none | retain row for the next phase in which it is eligible |
| `negative_result` | `scientific_candidate_failure` | enter a predeclared later repair phase when one exists; otherwise carry the row directly to Phase 10 synthesis as terminal negative evidence and exclude it from intervening implementation/GPU phases |
| `target_blocked` | `true_continuation_veto` for that row | exclude the row until an owner-approved target repair exists; continue other legal rows; if no legal row or shared phase remains, advance to Phase 10 synthesis |
| `implementation_blocked` | `localized_engineering_failure` or `plan_or_harness_failure`, named explicitly in the result | repair/retry within scope and budget; after exhausted attempts or infeasible fixed shape, preserve the blocker and exclude the row from downstream GPU phases |
| `invalid_evidence` | `invalid_evidence` | quarantine the artifact, repair the evidence path, and rerun before the row can advance |

A row-local target blocker is not automatically a program-wide stop. A
program-wide `true_continuation_veto` fires only when shared harness or evidence
is invalid, no legal row or shared phase remains other than synthesis, or
another Human-Required Stop applies. Terminal negative and blocked rows are
still complete classifications for Phase 10; they cannot be silently dropped
or replaced by proxies.

## Repair Loop

`CLASSIFY -> LOCALIZE -> PATCH -> FOCUSED_CHECK -> REFRESH_ARTIFACT -> REVIEW_IF_MATERIAL -> REASSESS`

- `scientific_candidate_failure`: preserve the negative result and continue to
  a later predeclared repair phase when one exists; do not reject the research
  direction.
- `localized_engineering_failure`: repair and retry automatically within the
  same target, method, hardware class, criteria, and campaign budget, using a
  fresh attempt directory.
- `plan_or_harness_failure`: patch the subplan or harness, prove it now answers
  the same question with a focused regression, and retry.
- `invalid_evidence`: quarantine the artifact, preserve it, and rerun only after
  the evidence defect is repaired.
- `true_continuation_veto`: write the result and stop handoff; do not disguise
  it as a candidate failure.

The supervisor may make at most three full-horizon attempts per eligible model.
Prefix, unit, compile, graph-trace, and harness diagnostics do not consume that
full-horizon count but do consume the CPU/GPU hour budget. A repair may not
change a target, feature family, comparator, tolerance, or promotion criterion
after seeing results. Such a change requires a refreshed scientific plan and,
when material, owner direction.

## Budget-Ready Calculation

Every next subplan declares `minimum_entry_budget` separately for CPU core-hours,
trusted GPU-hours, and full-horizon attempts per scheduled row. It also declares
`repair_reserve`, which must be at least one focused retry allocation for every
scheduled material implementation or GPU gate unless the subplan explicitly
has no repairable operation. The ledger computes:

`available = authorized cap - recorded consumption - concurrently reserved work`.

The budget row of `NEXT_PHASE_READINESS` passes exactly when, componentwise,
`available >= minimum_entry_budget + repair_reserve`. Zero is allowed for a
resource the next phase does not use. If the inequality fails, the next phase
is not ready: reduce only optional work before results are observed, advance to
Phase 10 with a budget-blocked classification when terminal synthesis remains
affordable, or stop for expanded-budget direction. The supervisor may not call
an amount "enough" without this calculation.

## Ready Gate For Automatic Continuation

The next phase is ready only when all are true:

- the current result and all required checks exist and are internally
  consistent;
- the next subplan inherits actual classifications and does not schedule
  negative or blocked rows illegally;
- target, scalar, derivative, comparator, device, and artifact identities are
  explicit;
- assumptions and convenience choices are labeled rather than promoted;
- primary criteria, vetoes, explanatory diagnostics, nonclaims, budget, and
  fresh paths are predeclared;
- commands are executable in the repository and their outputs answer the phase
  question;
- no unresolved material review finding or human-required boundary remains;
- the Budget-Ready Calculation passes and is recorded numerically.

Failure of this gate enters repair unless the failure is a true continuation
veto. “A test failed” and “Claude was unavailable” are not by themselves valid
reasons to stop the program.

## Review Protocol

Start a Claude review with one exact path and one question:

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited line: <one path>. Do not
edit, run commands, launch agents, or review the whole repo. Question: Is this
artifact consistent, mathematically and statistically correct, feasible, and
sufficiently bound to its evidence and stop conditions? Findings first. End
with VERDICT: AGREE or VERDICT: REVISE.
```

If Claude produces no output, use the repository Claude probe procedure. A
healthy probe means the review prompt must be narrowed; it does not mean Claude
is dead. Preserve material findings and the final verdict in the ledger or
phase result, without creating review artifacts solely to authorize review.

The required probe ladder is self-contained:

1. run the trusted/escalated noninteractive health probe with a 60-second
   timeout: `claude -p "Return exactly CLAUDE_PROBE_OK."`;
2. if it returns `CLAUDE_PROBE_OK`, resend a narrower one-path, one-question
   review with a 180-second timeout;
3. if it does not return output, repeat the same health probe once in trusted
   context and record the result;
4. if health remains unavailable, record `REVIEWER_UNAVAILABLE` and continue
   only when local review establishes that no material scientific,
   mathematical, privacy, cost, destructive, or external-boundary finding is
   unresolved;
5. if health passes but review still stalls, split the question by exact line
   range or symbol. Do not broaden the prompt.

## Quiet Execution And Run Manifest

Predeclare log and structured result paths. Redirect verbose command output to
the log; preserve exit status and only inspect a bounded tail on failure.
Serious-run manifests record git commit, dirty-worktree status, exact command,
environment, CPU/GPU status and trust basis, XLA/TF32/dtype, data and preparation
identity, seeds, wall time, output paths, controlling plan/result, and hashes.

## Human-Required Stops

Stop and refresh the handoff only for a material scientific choice outside the
program, invalid or missing target law/scalar with no approved repair, campaign
budget exhaustion, package/environment mutation, credentials or private data,
destructive work, external publication/messaging, default-policy change,
unrelated user-work collision, or corrupted evidence that cannot be repaired.

At a stop or terminal completion, the handoff states the last closed phase,
current state, exact artifacts and checks, remaining budget, unresolved
blockers, nonclaims, and the smallest next justified action.

## Skeptical Runbook Audit

Status: `PASS_REVIEWED_FOR_EXECUTION`.

Review record: bounded Claude review round 1 returned `REVISE` on the
classification/action mapping, numeric budget-ready test, terminal negative-row
path, evidence-linked readiness decision, and self-contained reviewer fallback.
The same runbook was patched. Focused round 2 returned `VERDICT: AGREE`.

- The same finite scalar, not Kalman or Zhao--Cui, is the topology baseline.
- Graph/XLA and FD checks cannot become scientific equivalence claims.
- Generalized SV, SIR, and DSGE/NAWM retain their distinct negative/blocker
  states and cannot be filled by proxies.
- The early structural fixture precedes new nonlinear GPU campaigns.
- Full-horizon derivative and fail-closed evidence must execute the exact
  compiled factory/configuration being classified.
- Automatic repair is bounded by unchanged science, three full-horizon attempts
  per eligible row, and the total campaign hours.
- Mandatory next-plan review cannot become ceremonial execution authority;
  material findings block, reviewer unavailability alone does not.
- Gate classifications map deterministically to repair, exclusion, synthesis,
  or stop; negative and blocked rows never disappear from terminal reporting.
- The readiness decision is an evidence-linked checklist whose budget row uses
  a componentwise numeric calculation, not supervisor judgment alone.
- Phase completion means every frozen row is classified with evidence, not that
  every row passed or that Contract E--TP became canonical.
