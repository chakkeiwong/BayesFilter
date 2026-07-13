# BayesFilter Multidimensional Triangular LGSSM NeuTra-HMC Visible Gated Execution Runbook

Date: 2026-07-08

## Status

`LAUNCH_REVIEW_CODEX_SUBSTITUTE_AGREE_PHASE0_READY`

## Role Contract

Codex in the current conversation is the supervisor and executor.

Claude is a read-only reviewer only.

This runbook is visible and foregrounded. It must not launch detached or nested
execution. Do not use `codex exec`, detached supervisors, backgrounded phase
runners, copied-workspace execution, or `overnight_gated_launch.sh` for this
runbook. If true detached overnight execution is later required, write a
separate detached-supervisor plan and stop for approval.

## Quiet Visible Execution Pattern

Noisy commands must write full logs to artifacts and print only bounded
summaries. TensorFlow/GPU/HMC commands must declare log and JSON result paths
before execution.

## Program

Master program:

- `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-master-program-2026-07-08.md`

Execution ledger:

- `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-visible-execution-ledger-2026-07-08.md`

## Phase Index

| Phase | Name | Subplan | Required result artifact |
| --- | --- | --- | --- |
| 0 | Source And Identifiability Inventory | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase0-source-identifiability-subplan-2026-07-08.md` | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase0-source-identifiability-result-2026-07-08.md` |
| 1 | Model Contract | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase1-model-contract-subplan-2026-07-08.md` | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase1-model-contract-result-2026-07-08.md` |
| 2 | Synthetic Data Fixture | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase2-synthetic-data-subplan-2026-07-08.md` | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase2-synthetic-data-result-2026-07-08.md` |
| 3 | Stationary/Lyapunov Implementation | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase3-stationary-implementation-subplan-2026-07-08.md` | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase3-stationary-implementation-result-2026-07-08.md` |
| 4 | Target Score And XLA Compile | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase4-target-score-compile-subplan-2026-07-08.md` | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase4-target-score-compile-result-2026-07-08.md` |
| 5 | Reference Posterior | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase5-reference-posterior-subplan-2026-07-08.md` | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase5-reference-posterior-result-2026-07-08.md` |
| 6 | GPU NeuTra Training | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase6-gpu-neutra-training-subplan-2026-07-08.md` | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase6-gpu-neutra-training-result-2026-07-08.md` |
| 7 | Frozen Transport Packaging | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase7-frozen-transport-subplan-2026-07-08.md` | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase7-frozen-transport-result-2026-07-08.md` |
| 8 | CPU HMC Pilot | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase8-cpu-hmc-pilot-subplan-2026-07-08.md` | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase8-cpu-hmc-pilot-result-2026-07-08.md` |
| 9 | Serious CPU HMC Estimation | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase9-serious-hmc-subplan-2026-07-08.md` | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase9-serious-hmc-result-2026-07-08.md` |
| 10 | Readiness Decision | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase10-readiness-decision-subplan-2026-07-08.md` | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase10-readiness-decision-result-2026-07-08.md` |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can BayesFilter design and validate serious NeuTra-HMC estimation for a stationary, coordinate-anchored multidimensional triangular LGSSM? |
| Baseline/comparator | Source-anchored constrained LGSSM design, local stationary/Lyapunov code, synthetic truth, reference posterior diagnostics, and serious HMC diagnostics. |
| Primary pass criterion | Each phase either writes a reviewed pass artifact or an exact blocker without overclaiming. |
| Veto diagnostics | Unsupported identifiability/stationarity claim, missing stationary initial law, nonstationary draws, runtime autodiff, `jit_compile=false`, GPU sample generation, hidden training/sampling, malformed artifacts, unsupported scientific/product/default claims. |
| Explanatory diagnostics | Eigenvalues, Lyapunov residuals, moment checks, score residuals, compile timing/size, training diagnostics, per-parameter R-hat/ESS, truth/reference residuals. |
| Not concluded | Broad LGSSM readiness, product/default readiness, sampler superiority, nonlinear SSM validity, DSGE/c603 validity, or scientific validity outside the synthetic fixture. |
| Artifacts | Master, runbook, ledger, subplans/results, review records, data/training/HMC JSON and logs. |

## Approval Requests Anticipated

Please approve later, when reached and after subplan review:

- Claude Code review-gate use with trusted/escalated permissions.
- GPU/CUDA probes and GPU-only NeuTra training for Phase 6.
- CPU-hidden multicore HMC pilot and serious HMC for Phases 8-9.
- Any package/environment changes, network fetches, or git commit/push.

## Visible State Machine

1. `PRECHECK`: read subplan, confirm prerequisites, restate evidence contract.
2. `EXECUTE_MINIMAL`: run only visible commands in this conversation.
3. `ASSESS_GATE`: compare outputs with primary criterion and veto diagnostics.
4. `PASS_REVIEW`: use Claude read-only review when available.
5. `REPAIR_LOOP`: patch visible fixable issues, rerun focused checks, max five
   review rounds for the same blocker.
6. `ADVANCE_OR_STOP`: advance only after the current phase gate passes.

## Claude Review Protocol

Use `~/python/claudecodex/docs/claude-review-gate-agent-guide.md`. Start with
bounded review bundles. If Claude does not respond, run a tiny probe. If the
probe succeeds, revise the prompt. If the probe fails, record reviewer
unavailability and use same-foreground Codex substitute review.

Claude cannot authorize human, runtime, model-file, funding, product,
default-policy, or scientific-claim boundaries.

## Human-Required Stop Conditions

Stop before package installation, network fetches, credentials, destructive
git/filesystem actions, new GPU/CUDA jobs, new NeuTra training, HMC
sampling/tuning, detached execution, default-policy changes, unrelated dirty
work edits, live DSGE/c603 work, or unsupported scientific/product claims.
