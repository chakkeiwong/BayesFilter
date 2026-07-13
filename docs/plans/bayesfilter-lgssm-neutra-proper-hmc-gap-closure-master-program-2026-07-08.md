# BayesFilter LGSSM NeuTra Proper HMC Gap-Closure Master Program

Date: 2026-07-08

## Status

`PHASE21_COMPLETE_LGSSM_REFERENCE_HMC_READY`

## Objective

Close the remaining engineering and evidence gaps before proper HMC testing of
the LGSSM-first NeuTra path.

This program starts from the completed Phase 16 GPU/XLA bounded training
artifact. It does not use stale Phase 10/11 non-XLA artifacts for promotion.

## Scope

In scope:

- package the Phase 16 GPU/XLA-trained affine NeuTra state;
- validate frozen transport loader boundaries;
- compile or validate HMC mechanics under reviewed `jit_compile=True` gates;
- build CPU-multicore chain/sample-generation harnesses separately from GPU
  training;
- run LGSSM reference validation only after packaging and mechanics gates pass.

Out of scope unless a later reviewed phase explicitly authorizes it:

- DSGE/c603 runtime work;
- nonlinear SSM expansion;
- public benchmark or product claims;
- default-policy changes;
- HMC sampling before the packaging/mechanics gates pass.

## Current Starting Evidence

- Phase 15 passed manual-score XLA compile gate:
  `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase15-manual-score-xla-compile-gate-result-2026-07-08.md`.
- Phase 16 passed bounded GPU/XLA NeuTra training:
  `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase16-bounded-gpu-xla-training-result-2026-07-08.md`.
- Phase 16 source artifact:
  `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_training_state_seed20260707.json`.
- Phase 16 file SHA-256:
  `727fea040502e4fcb1af2203b9a490d03ab00dca63fd756f501a5bc3c936af7b`.
- Current target signature:
  `275bdd37a82d8c09d2c1b1935b6481f18224644ac659830a921c7958b6ed9038`.
- Current adapter signature:
  `d89bdedcf759566f490ce5222ef753cc8c0c8ea39805d68c064c12d2bec07900`.

## Program Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can BayesFilter close the packaging, mechanics, CPU-harness, and LGSSM-reference gaps needed before proper HMC testing of the Phase 16 NeuTra path? |
| Baseline/comparator | Phase 16 GPU/XLA training artifact, current manual-score LGSSM target signatures, frozen transport loader, and deterministic quadrature reference posterior over the exact LGSSM likelihood target. |
| Primary pass criterion | Each phase writes a pass/blocker artifact with required checks, preserves boundaries, and advances only after review convergence. |
| Veto diagnostics | Stale Phase 10/11 artifact use, `jit_compile=false` fallback in runtime evidence, runtime autodiff in admitted route, hidden training, hidden HMC before mechanics gate, hidden sample generation, missing hashes/signatures, malformed artifacts, unsupported posterior/HMC/product/scientific claims. |
| Explanatory diagnostics | Payload hashes, finite value/score checks, compile timings, chain diagnostics, R-hat/ESS, posterior mean/covariance residuals. |
| Not concluded | HMC convergence, posterior correctness, sampler superiority, production readiness, default readiness, nonlinear SSM validity, DSGE/c603 validity, or scientific validity until a specific reviewed phase proves that narrower claim. |
| Artifacts | Master program, visible runbook, ledger, per-phase subplans/results, payload JSON, validation JSON, logs, HMC diagnostics. |

## Phase Index

| Phase | Name | Objective | Subplan | Result |
| --- | --- | --- | --- | --- |
| 17 | Frozen GPU/XLA-Trained Affine Payload | Package Phase 16 affine state into frozen payload schema and validate loader/reference boundaries without HMC mechanics. | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase17-frozen-gpu-xla-affine-payload-subplan-2026-07-08.md` | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase17-frozen-gpu-xla-affine-payload-result-2026-07-08.md` |
| 18 | Fixed-Transport HMC Mechanics Compile Gate | Bind the Phase 17 frozen payload to HMC mechanics and run a `jit_compile=True` finite transition/mechanics gate without chains. | `docs/plans/bayesfilter-lgssm-neutra-hmc-phase18-fixed-transport-mechanics-compile-subplan-2026-07-08.md` | `docs/plans/bayesfilter-lgssm-neutra-hmc-phase18-fixed-transport-mechanics-compile-result-2026-07-08.md` |
| 19 | CPU Multicore HMC Chain Harness | Build a CPU-hidden multicore chain/sample-generation harness with worker metadata, seeds, return codes, and artifacts. | `docs/plans/bayesfilter-lgssm-neutra-hmc-phase19-cpu-multicore-chain-harness-subplan-2026-07-08.md` | `docs/plans/bayesfilter-lgssm-neutra-hmc-phase19-cpu-multicore-chain-harness-result-2026-07-08.md` |
| 20 | LGSSM Reference HMC Validation | Run bounded CPU multicore HMC chains and compare to a deterministic quadrature reference posterior over the exact LGSSM likelihood target under predeclared diagnostics. | `docs/plans/bayesfilter-lgssm-neutra-hmc-phase20-lgssm-reference-validation-subplan-2026-07-08.md` | `docs/plans/bayesfilter-lgssm-neutra-hmc-phase20-lgssm-reference-validation-result-2026-07-08.md` |
| 21 | HMC Readiness Decision Gate | Classify Phase 20 evidence and decide whether the LGSSM NeuTra-HMC path is validated for this reference target or blocked for repair. | `docs/plans/bayesfilter-lgssm-neutra-hmc-phase21-readiness-decision-subplan-2026-07-08.md` | `docs/plans/bayesfilter-lgssm-neutra-hmc-phase21-readiness-decision-result-2026-07-08.md` |

## Required Subplan Fields

Every phase subplan must state:

- phase objective;
- entry conditions inherited from the previous phase;
- required artifacts;
- required checks/tests/reviews;
- evidence contract;
- forbidden claims/actions;
- exact next-phase handoff conditions;
- stop conditions.

At the end of each phase, Codex must:

1. run the required local checks;
2. write a phase result or blocker close record;
3. draft or refresh the next subplan;
4. review the next subplan for consistency, correctness, feasibility, artifact
   coverage, and boundary safety.

## Human Approval Boundaries

Already approved for this visible execution:

- Claude read-only review usage, subject to bounded prompts and repair loops;
- Phase 17 CPU-hidden packaging/loader checks if the reviewed subplan passes.

Requires explicit approval or a reviewed phase gate before execution:

- HMC runtime sampling/tuning;
- long chains or decision-making sampler validation;
- new GPU/XLA jobs not named by a reviewed subplan;
- package installation, environment mutation, network fetch, credentials;
- git commit/push;
- default-policy changes;
- DSGE/c603 runtime work;
- product, release, public benchmark, or scientific promotion claims.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Only Phase 16 GPU/XLA artifact can feed packaging and later HMC gates. |
| Proxy promotion | Packaging, finite mechanics gates, and short chains cannot become posterior/HMC readiness without Phase 20/21 evidence. |
| Missing stop conditions | Each subplan must include exact stop conditions and pass/block artifacts. |
| Hidden assumptions | GPU training and CPU sample generation remain separate; HMC sampling starts only after mechanics gates. |
| Environment mismatch | GPU/XLA commands require trusted execution; CPU sample generation must hide GPU. |
| Artifact mismatch | Every gate records hashes, signatures, command, environment, seeds, and nonclaims. |

Audit status: draft ready for review.
