# BayesFilter Deterministic LGSSM HMC Tuning Visible Gated Execution Runbook

Date: 2026-07-09

Status: `PHASE7_APPROVAL_BOUNDARY`

## Role Contract

Codex in the current conversation is supervisor and executor.

Claude is read-only reviewer only. Claude cannot edit files, run experiments,
launch agents, or authorize human, runtime, GPU, funding, product, release, or
scientific-claim boundaries.

This runbook is visible and recoverable in the current conversation. It must
not launch detached or nested agents.

## Program

Master program:

- `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-master-program-2026-07-09.md`

Execution ledger:

- `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-execution-ledger-2026-07-09.md`

Stop handoff:

- `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-stop-handoff-2026-07-09.md`

Launch review bundle:

- `docs/reviews/bayesfilter-deterministic-lgssm-hmc-tuning-launch-review-bundle-2026-07-09.md`

## Phase Index

| Phase | Name | Subplan | Required result artifact | Runtime approval |
| --- | --- | --- | --- | --- |
| 0 | Governance, Runbook, Review Gate | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase0-governance-subplan-2026-07-09.md` | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase0-governance-result-2026-07-09.md` | Claude review only |
| 1 | Tool Inventory And API Binding | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase1-tool-inventory-subplan-2026-07-09.md` | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase1-tool-inventory-result-2026-07-09.md` | None |
| 2 | Deterministic Config Schema | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase2-config-schema-subplan-2026-07-09.md` | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase2-config-schema-result-2026-07-09.md` | None |
| 3 | LGSSM Fixture Driver | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase3-lgssm-fixture-subplan-2026-07-09.md` | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase3-lgssm-fixture-result-2026-07-09.md` | None for small deterministic fixture |
| 4 | XLA Value/Score Gate | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase4-xla-score-gate-subplan-2026-07-09.md` | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase4-xla-score-gate-result-2026-07-09.md` | Escalated if GPU/CUDA is touched; otherwise CPU-hidden XLA compile |
| 5 | Geometry And Mass Driver | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase5-geometry-mass-subplan-2026-07-09.md` | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase5-geometry-mass-result-2026-07-09.md` | None for small local checks |
| 6 | Kernel Tuning Driver | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-subplan-2026-07-09.md` | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md` | Explicit user approval before serious HMC tuning runtime |
| 6S | Fixed-Mass Candidate Grid XLA Compile Repair | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6s-fixed-mass-xla-compile-repair-subplan-2026-07-09.md` | refreshed Phase 6 result or blocker note | Covered by Phase 6 approval only; no Phase 7 sampling |
| 6T | Log-Accept Telemetry Root-Cause Repair | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6t-log-accept-telemetry-repair-subplan-2026-07-09.md` | refreshed Phase 6 result or blocker note | Covered by Phase 6 approval only; no Phase 7 sampling |
| 6U | Kernel Mechanics Nonfinite Log-Accept Diagnostic | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6u-kernel-mechanics-diagnostic-subplan-2026-07-09.md` | refreshed Phase 6 result or blocker note | Covered by Phase 6 approval only; no Phase 7 sampling |
| 6V | Deterministic Step/Mass-Scale Policy Audit | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6v-step-mass-scale-policy-audit-subplan-2026-07-09.md` | refreshed Phase 6 result or blocker note | Covered by Phase 6 approval only; no Phase 7 sampling |
| 6W | Fixed-Mass Final-Local Budget/Timeout Repair | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6w-fixed-mass-budget-timeout-repair-subplan-2026-07-09.md` | refreshed Phase 6 result or blocker note | Covered by Phase 6 approval only; no Phase 7 sampling |
| 6X | Fixed-Mass XLA Compile-Memory Reuse Repair | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6x-fixed-mass-xla-compile-memory-reuse-repair-subplan-2026-07-09.md` | refreshed Phase 6 result or blocker note | Covered by Phase 6 approval only; no Phase 7 sampling |
| 6Y | Trajectory Handoff XLA Compile-Memory Reuse Repair | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6y-trajectory-xla-compile-memory-reuse-repair-subplan-2026-07-09.md` | refreshed Phase 6 result or blocker note | Covered by Phase 6 approval only; no Phase 7 sampling |
| 6Z | Verification Chunk XLA Compile-Memory Repair | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6z-verification-chunk-xla-compile-memory-repair-subplan-2026-07-09.md` | refreshed Phase 6 result or blocker note | Covered by Phase 6 approval only; no Phase 7 sampling |
| 6AA | SVD Score Wiring Retry | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6aa-svd-score-wiring-retry-subplan-2026-07-10.md` | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6aa-svd-score-wiring-retry-result-2026-07-10.md` | Covered by Phase 6 approval only; no Phase 7 sampling |
| 7 | Burn-In And Sampling Controller | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase7-burnin-sampling-subplan-2026-07-09.md` | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase7-burnin-sampling-result-2026-07-09.md` | Explicit user approval before long sample generation |
| 8 | Serious Recovery Run | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase8-serious-recovery-subplan-2026-07-09.md` | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase8-serious-recovery-result-2026-07-09.md` | Explicit user approval required |
| 9 | Closeout And Handoff | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase9-closeout-subplan-2026-07-09.md` | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase9-closeout-result-2026-07-09.md` | Claude review if pass claim is made |

## Quiet Visible Execution Pattern

For commands that may produce large output, predeclare the log and structured
artifact path, redirect stdout/stderr to the log, and inspect only bounded
summaries in chat. Full output is an artifact.

## Current Boundary

Phase 6AA passed at the kernel-handoff level. The refreshed
`kernel_tuning.json` has `passed=true`, confirmed XLA/JIT execution, no hard
vetoes, and final kernel payload/hash. Phase 7 has not run.

Do not start Phase 7 burn-in or retained sampling without explicit user
approval. Phase 7 is the next runbook phase and must use the deterministic
Python controller, CPU-hidden sample generation, XLA-only target path, and the
predeclared R-hat/ESS/recovery gates.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can BayesFilter produce a deterministic serious LGSSM HMC tuning/recovery program with minimal agent discretion? |
| Baseline/comparator | Existing BayesFilter tuning APIs plus prior-mean `T=120` LGSSM recovery fixture. |
| Primary pass criterion | Phase 8 final artifact passes all predeclared R-hat, ESS, and recovery checks. |
| Veto diagnostics | Manual tuning, non-XLA target-path execution, runtime GradientTape, nonfinite chains, invalid artifacts, failed R-hat/ESS/recovery criteria. |
| Explanatory diagnostics | Geometry fit, mass eigen summaries, acceptance, runtime, compile time, module size, MCSE. |
| Not concluded | No sampler superiority, production readiness, default readiness, DSGE readiness, or public scientific claim. |
| Artifacts | Master program, phase results, JSON artifacts, logs, review bundle, review gate status. |

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| Deterministic Python owns tuning | User directive | Prevents agent-tuned HMC | Hidden manual decision | Config schema review | Required |
| `use_xla=True` target path | User directive and repo policy | Avoids non-JIT fallback | Metadata lies or missing | Phase 4 compile gate | Required |
| CPU-hidden HMC sampling | User directive | Sample generation should be multicore CPU | GPU accidentally used | Environment manifest | Required |
| Quadratic geometry initializer | Existing BayesFilter tool | Uses repo-owned initial covariance path | Geometry rejection | Phase 5 result | Required |
| Prior means as truth, `T=120` | User directive | Identifiable recovery target | Fixture not informative | Phase 3 contract | Required |

## Skeptical Plan Audit

Before executing each phase, Codex must check for wrong baselines, proxy metric
promotion, missing stop conditions, unfair comparisons, hidden assumptions,
stale context, environment mismatch, and artifacts that do not answer the phase
question.

Current launch audit: `PASS_PHASE0_ONLY`. The launch can create and review
planning artifacts. Serious runtime is not approved by this launch.

## Visible State Machine

Each phase follows:

1. `PRECHECK`: read subplan, confirm prerequisites, restate evidence contract.
2. `EXECUTE_MINIMAL`: run only actions required by the subplan.
3. `ASSESS_GATE`: compare artifacts to pass/fail and veto diagnostics.
4. `PASS_REVIEW`: use Claude read-only review for material gates.
5. `REPAIR_LOOP`: patch fixable blockers and retry review, max five rounds.
6. `ADVANCE_OR_STOP`: advance only after gate passes; otherwise write stop handoff.

## Claude Review Policy

Use the smallest exact path that can answer the gate. For Phase 0 launch:

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited line:
docs/reviews/bayesfilter-deterministic-lgssm-hmc-tuning-launch-review-bundle-2026-07-09.md
Do not edit, run commands, launch agents, or review the whole repo.
Question: Does the launch plan enforce deterministic Python-owned tuning and
block manual agent tuning/runtime/scientific boundary crossings?
End with VERDICT: AGREE or VERDICT: REVISE.
```

If Claude does not respond, send a tiny probe. If the probe succeeds, redesign
the review prompt or bundle. If the probe fails, use a fresh Codex read-only
substitute review and record the weaker status.

## Human-Required Approvals Anticipated

- Claude review gate commands require trusted/escalated execution.
- Phase 6 serious HMC tuning runtime requires explicit user approval.
- Phase 7 long burn-in/retained-sampling runtime requires explicit user approval.
- Phase 8 serious recovery run requires explicit user approval.
- Any GPU/CUDA/NeuTra training command requires explicit user approval and
  trusted/escalated execution.
