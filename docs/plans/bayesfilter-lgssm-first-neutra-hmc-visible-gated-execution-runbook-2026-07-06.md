# BayesFilter LGSSM-First NeuTra/HMC Visible Gated Execution Runbook

Date: 2026-07-06

## Status

`PHASE16_BOUNDED_GPU_XLA_TRAINING_PASSED_PHASE17_READY`

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

- `docs/plans/bayesfilter-lgssm-first-neutra-hmc-master-program-2026-07-06.md`

Execution ledger:

- `docs/plans/bayesfilter-lgssm-first-neutra-hmc-visible-execution-ledger-2026-07-06.md`

Stop handoff:

- `docs/plans/bayesfilter-lgssm-first-neutra-hmc-visible-stop-handoff-2026-07-06.md`

## Phase Index

| Phase | Name | Subplan | Required result artifact |
| --- | --- | --- | --- |
| 0 | Scope Reset And Launch | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase0-scope-reset-subplan-2026-07-06.md` | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase0-scope-reset-result-2026-07-06.md` |
| 1 | Interface Inventory And Gap Map | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase1-interface-inventory-subplan-2026-07-06.md` | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase1-interface-inventory-result-2026-07-06.md` |
| 2 | LGSSM Exact Target Adapter | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase2-lgssm-target-adapter-subplan-2026-07-06.md` | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase2-lgssm-target-adapter-result-2026-07-06.md` |
| 3 | Plain HMC Mechanics Smoke | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase3-plain-hmc-smoke-subplan-2026-07-06.md` | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase3-plain-hmc-smoke-result-2026-07-06.md` |
| 4 | LGSSM Posterior Reference Validation | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase4-lgssm-reference-validation-subplan-2026-07-06.md` | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase4-lgssm-reference-validation-result-2026-07-06.md` |
| 5 | Frozen Transport Binding | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase5-frozen-transport-binding-subplan-2026-07-06.md` | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase5-frozen-transport-binding-result-2026-07-06.md` |
| 6 | LGSSM NeuTra Training And Freeze | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase6-lgssm-neutra-training-subplan-2026-07-06.md` | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase6-lgssm-neutra-training-result-2026-07-06.md` |
| 7 | First Simple Nonlinear SSM | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase7-simple-nonlinear-ssm-subplan-2026-07-06.md` | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase7-simple-nonlinear-ssm-result-2026-07-06.md` |
| 8 | Same Target Multiple Filters | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase8-multifilter-subplan-2026-07-06.md` | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase8-multifilter-result-2026-07-06.md` |
| 9 | GPU NeuTra Training Preflight | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase9-gpu-neutra-training-preflight-subplan-2026-07-07.md` | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase9-gpu-neutra-training-preflight-result-2026-07-07.md` |
| 10 | Historical Bounded GPU NeuTra Training | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase10-bounded-gpu-training-subplan-2026-07-07.md` | stale/history only: `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase10-bounded-gpu-training-result-2026-07-07.md` |
| 11 | Historical Frozen GPU-Trained Affine Payload | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase11-frozen-gpu-affine-payload-subplan-2026-07-07.md` | stale/history only |
| 12 | CPU Multicore External Sample Boundary | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase12-cpu-multicore-sample-generation-subplan-2026-07-07.md` | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase12-cpu-multicore-sample-generation-result-2026-07-07.md` |
| 13 | XLA/JIT Repair Gate | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase13-xla-jit-repair-subplan-2026-07-07.md` | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase13-xla-jit-repair-result-2026-07-07.md` |
| 14 | XLA TensorList Boundary Repair | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase14-xla-tensorlist-boundary-repair-subplan-2026-07-08.md` | superseded |
| 14A | LGSSM No-GradientTape Target Policy | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase14a-no-gradienttape-policy-subplan-2026-07-08.md` | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase14a-no-gradienttape-policy-result-2026-07-08.md` |
| 15 | Manual-Score LGSSM XLA Compile Gate | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase15-manual-score-xla-compile-gate-subplan-2026-07-08.md` | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase15-manual-score-xla-compile-gate-result-2026-07-08.md` |
| 16 | Bounded GPU/XLA NeuTra Training | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase16-bounded-gpu-xla-training-subplan-2026-07-08.md` | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase16-bounded-gpu-xla-training-result-2026-07-08.md` |
| 17 | Frozen GPU/XLA-Trained Affine Payload | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase17-frozen-gpu-xla-affine-payload-subplan-2026-07-08.md` | pending |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can BayesFilter build a generic Bayesian SSM target adapter and NeuTra/HMC mechanics path from an exact LGSSM base before moving to nonlinear SSMs and later DSGE stress targets? |
| Baseline/comparator | Existing SSM contracts, QR Kalman LGSSM code, fixed transport mechanics, and opt-in QR static LGSSM HMC smoke harness. |
| Primary pass criterion | Each phase either produces a reviewed artifact for its boundary or records an exact blocker without overclaiming. |
| Veto diagnostics | DSGE/c603 as foundation, smoke promoted to readiness, finite probes promoted to posterior correctness, unapproved GPU/training/long-HMC/package/git actions. |
| Explanatory diagnostics | Signatures, manifests, finite probes, reference residuals, review status. |
| Not concluded | Posterior convergence, sampler ranking, production readiness, broad nonlinear SSM validity, DSGE/c603 readiness. |
| Artifacts | Ledger, phase results, review records, logs, tests. |

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

The historical Phase 10/11 non-XLA artifacts are stale diagnostic history after
the Phase 14A no-`GradientTape` repair and the Phase 15 XLA compile pass. They
must not support promotion, packaging, or readiness. Phase 12 is a CPU
multicore boundary-design and smoke phase for external sample generation only;
it must not train NeuTra or run HMC. Phase 13 is superseded blocker history.
Phase 14 was superseded by Phase 14A after the no-`GradientTape` policy
clarification. Phase 15 may run only trusted GPU `jit_compile=True` compile
diagnostics under the current manual-score LGSSM target signatures;
`jit_compile=false` runtime runs are forbidden. Phase 16 passed bounded
trusted-GPU `jit_compile=True` training with no runtime autodiff, no HMC, and no
external sample generation. Phase 17 may package only the Phase 16 XLA-trained
artifact and must not use stale Phase 10/11 non-XLA artifacts.

## Claude Review Protocol

Use `~/python/claudecodex/scripts/claude_review_gate.sh` with bounded bundles
under `docs/reviews/`. If Claude does not respond, run a tiny trusted probe. If
the probe succeeds, revise the prompt. If the probe fails or cannot be run,
record reviewer unavailability and substitute a fresh Codex review.

Claude cannot authorize human, runtime, model-file, funding, product,
default-policy, or scientific-claim boundaries.

## Final Visible Handoff

When execution completes or stops, update:

- `docs/plans/bayesfilter-lgssm-first-neutra-hmc-visible-stop-handoff-2026-07-06.md`
