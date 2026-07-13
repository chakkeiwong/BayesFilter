# BayesFilter LGSSM NeuTra Proper HMC Visible Gated Execution Runbook

Date: 2026-07-08

## Status

`PHASE21_COMPLETE_LGSSM_REFERENCE_HMC_READY`

## Role Contract

Codex in the current conversation is the supervisor and executor.

Claude is a read-only reviewer only.

This runbook must not launch detached or nested execution. Do not use `codex
exec`, `overnight_gated_launch.sh`, detached supervisors, backgrounded phase
runners, or copied-workspace execution.

## Quiet Visible Execution Pattern

Full command output is an artifact, not chat content. For noisy commands,
predeclare logs, preserve them, and report bounded summaries only.

## Program

Master program:

- `docs/plans/bayesfilter-lgssm-neutra-proper-hmc-gap-closure-master-program-2026-07-08.md`

Execution ledger:

- `docs/plans/bayesfilter-lgssm-neutra-proper-hmc-visible-execution-ledger-2026-07-08.md`

## Phase Index

| Phase | Name | Subplan | Required result artifact |
| --- | --- | --- | --- |
| 17 | Frozen GPU/XLA-Trained Affine Payload | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase17-frozen-gpu-xla-affine-payload-subplan-2026-07-08.md` | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase17-frozen-gpu-xla-affine-payload-result-2026-07-08.md` |
| 18 | Fixed-Transport HMC Mechanics Compile Gate | `docs/plans/bayesfilter-lgssm-neutra-hmc-phase18-fixed-transport-mechanics-compile-subplan-2026-07-08.md` | `docs/plans/bayesfilter-lgssm-neutra-hmc-phase18-fixed-transport-mechanics-compile-result-2026-07-08.md` |
| 19 | CPU Multicore HMC Chain Harness | `docs/plans/bayesfilter-lgssm-neutra-hmc-phase19-cpu-multicore-chain-harness-subplan-2026-07-08.md` | `docs/plans/bayesfilter-lgssm-neutra-hmc-phase19-cpu-multicore-chain-harness-result-2026-07-08.md` |
| 20 | LGSSM Reference HMC Validation | `docs/plans/bayesfilter-lgssm-neutra-hmc-phase20-lgssm-reference-validation-subplan-2026-07-08.md` | `docs/plans/bayesfilter-lgssm-neutra-hmc-phase20-lgssm-reference-validation-result-2026-07-08.md` |
| 21 | HMC Readiness Decision Gate | `docs/plans/bayesfilter-lgssm-neutra-hmc-phase21-readiness-decision-subplan-2026-07-08.md` | `docs/plans/bayesfilter-lgssm-neutra-hmc-phase21-readiness-decision-result-2026-07-08.md` |

Phase 17 intentionally reuses the already-created `first-neutra-hmc` path
family because it is the direct continuation from Phase 16. Phases 18-21 use
the shorter `lgssm-neutra-hmc` path family for new proper-HMC gap-closure
artifacts.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can BayesFilter close the packaging, mechanics, CPU-harness, and LGSSM-reference gaps needed before proper HMC testing of the Phase 16 NeuTra path? |
| Baseline/comparator | Phase 16 GPU/XLA training artifact, current manual-score LGSSM target signatures, frozen transport loader, and deterministic quadrature reference posterior over the exact LGSSM likelihood target. |
| Primary pass criterion | Each phase either produces a reviewed artifact for its boundary or records an exact blocker without overclaiming. |
| Veto diagnostics | Stale Phase 10/11 artifact use, `jit_compile=false` fallback in runtime evidence, runtime autodiff in admitted route, hidden training, hidden HMC before mechanics gate, hidden sample generation, missing hashes/signatures, malformed artifacts, unsupported posterior/HMC/product/scientific claims. |
| Explanatory diagnostics | Payload hashes, finite value/score checks, compile timings, chain diagnostics, R-hat/ESS, posterior mean/covariance residuals. |
| Not concluded | HMC convergence, posterior correctness, sampler superiority, production readiness, default readiness, nonlinear SSM validity, DSGE/c603 validity, or scientific validity until a specific reviewed phase proves that narrower claim. |
| Artifacts | Ledger, phase results, review records, logs, tests, payload and validation JSON, HMC diagnostics. |

## Skeptical Plan Audit

Before executing each phase, Codex must record a skeptical audit in the ledger:
wrong baselines, proxy metrics, stop conditions, hidden assumptions, stale
context, environment mismatch, and artifact mismatch.

## Visible State Machine

For each phase:

1. `PRECHECK`: read subplan, confirm prerequisites, restate evidence contract.
2. `EXECUTE_MINIMAL`: run the smallest visible work needed.
3. `ASSESS_GATE`: compare result to primary criterion and veto diagnostics.
4. `PASS_REVIEW`: use Claude only as read-only reviewer for material gates.
5. `REPAIR_LOOP`: patch fixable blockers visibly; rerun focused checks; stop
   after five review rounds for the same material blocker.
6. `ADVANCE_OR_STOP`: advance only after the current phase gate passes.

## Human-Required Stop Conditions

Stop before package installation, network fetch not already authorized,
credentials, environment setup, destructive git/filesystem actions, new
GPU/CUDA jobs not named by the active subplan, new NeuTra training, HMC
sampling/tuning, detached execution, default-policy changes, modifying unrelated
dirty work, live DSGE/c603 runtime work, or unsupported scientific/product
claims.

Phase 17 passed CPU-hidden packaging/loader validation only. Later HMC runtime
work requires its own reviewed subplan and, for GPU/XLA commands, trusted
execution.

## Claude Review Protocol

Use the smallest exact path that can answer the gate. Default prompt shape:

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited line: <one path>. Do not
edit, run commands, launch agents, or review the whole repo. Question: <one
question>. End with VERDICT: AGREE or VERDICT: REVISE.
```

If Claude does not respond, run a tiny trusted probe in this same foreground
conversation. If the probe succeeds, revise the prompt. If the probe fails,
record reviewer unavailability and use a fresh Codex read-only substitute
review only in this same foreground conversation after documenting the
substitution. Do not use fallback review to launch nested agents, detached
execution, copied workspaces, or background supervisors.

Claude cannot authorize human, runtime, model-file, funding, product,
default-policy, or scientific-claim boundaries.

## Phase 20 And 21 Subplan Status

Phase 20 and Phase 21 subplans have been created before Phase 19 execution so
the CPU harness gate has an exact forward boundary:

- Phase 20 owns LGSSM reference posterior validation with CPU-hidden
  multicore chains, `jit_compile=True`, exact reference posterior diagnostics,
  and posterior residual vetoes.
- Phase 21 owns the readiness decision. It is a classification gate over
  Phase 20 evidence, not a new runtime or promotion shortcut.

Phase 19 remained a harness/boundary phase and did not run or claim Phase 20
reference validation.

Phase 20 passed as a narrow LGSSM fixture-local validation with CPU-hidden
multicore HMC, `jit_compile=True`, no GPU sample generation, no post-Phase-16
training, and a deterministic quadrature reference posterior over the exact
LGSSM likelihood target. Phase 21 is now the active readiness-decision gate.

Phase 21 classified the Phase 17-20 evidence as
`LGSSM_REFERENCE_HMC_READY` for the static QR LGSSM fixture and exact Phase
17-20 artifacts only. The runbook is complete at that narrow scope.
