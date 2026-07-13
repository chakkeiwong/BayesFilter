# Phase 19 Result: CPU Multicore HMC Chain Harness

Date: 2026-07-08

## Scope

This result records the Phase 19 CPU-hidden multicore fixed-transport LGSSM
NeuTra HMC chain-harness boundary. It is not posterior validation, HMC
convergence evidence, sampler-quality evidence, production readiness, default
readiness, or scientific promotion.

## Decision

`PASS_PHASE19_CPU_MULTICORE_HMC_CHAIN_HARNESS`

## Evidence Contract

| Field | Result |
| --- | --- |
| Question | Can BayesFilter build and run a CPU-hidden multicore harness for fixed-transport LGSSM NeuTra chain workers with deterministic metadata and no forbidden fallback before reference validation? |
| Baseline/comparator | Phase 17 payload, Phase 18 trusted GPU/XLA mechanics compile diagnostic, current LGSSM signatures, and the owner policy that chain/sample generation is CPU multicore. |
| Primary criterion | Met. Two CPU-hidden workers loaded the Phase 17 payload, bound the fixed transport, compiled a tiny batch-native value/score smoke with `jit_compile=True`, and recorded deterministic metadata. |
| Veto diagnostics | Clear. No `jit_compile=false`, no training, no GPU sample generation, no HMC transition, no HMC sampling/tuning, and no posterior validation were run. |
| Explanatory diagnostics | Worker compile-time proxies, finite value/score checks, return codes, process ids, seeds, CPU-hidden environment, signatures, and hashes. |
| Not concluded | No HMC convergence, posterior correctness, sampler quality, transport superiority, production readiness, default readiness, LGSSM reference agreement, nonlinear SSM validity, or scientific validity. |

## Artifacts

- Harness helper:
  `bayesfilter/testing/neutra_cpu_multicore_hmc_chain_harness_tf.py`.
- Focused tests:
  `tests/test_neutra_cpu_multicore_hmc_chain_harness_tf.py`.
- Harness JSON:
  `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase19_cpu_multicore_hmc_chain_harness_seed20260707.json`.
- Harness JSON file SHA-256:
  `38c0400b04ce7438f3bc70236ccfd42916c11c1b9c05d95bd76260b64f8c10b4`.
- Stable artifact hash:
  `sha256:aaa7893d6a6313db74350eb1efac57f4188165170d42056df1b079eef2c313f6`.
- Phase 20 subplan:
  `docs/plans/bayesfilter-lgssm-neutra-hmc-phase20-lgssm-reference-validation-subplan-2026-07-08.md`.
- Phase 21 subplan:
  `docs/plans/bayesfilter-lgssm-neutra-hmc-phase21-readiness-decision-subplan-2026-07-08.md`.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | Not recorded as clean; worktree contains unrelated dirty files. |
| Command | `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m bayesfilter.testing.neutra_cpu_multicore_hmc_chain_harness_tf --payload-path docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_frozen_payload_seed20260707.json --phase18-diagnostic-path docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase18_fixed_transport_hmc_mechanics_xla_compile_seed20260707.json --output-path docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase19_cpu_multicore_hmc_chain_harness_seed20260707.json --seed 20260707 --worker-count 2 --chain-count 2 --num-results 1 --num-burnin-steps 0 --num-leapfrog-steps 2 --step-size 0.1 --jit-compile true` |
| Environment | `CUDA_VISIBLE_DEVICES=-1`, `MPLCONFIGDIR=/tmp`; TensorFlow `2.19.1` inside workers. |
| CPU/GPU status | CPU-hidden worker execution. TensorFlow logged expected CUDA initialization noise under CPU-hidden execution, but XLA service compiled for Host. |
| Seeds | Worker seeds `20261716` and `20262725`. |
| Wall time | Harness artifact elapsed seconds `13.040306944167241`. |
| Plan file | `docs/plans/bayesfilter-lgssm-neutra-hmc-phase19-cpu-multicore-chain-harness-subplan-2026-07-08.md`. |
| Result file | This file. |

## Key Diagnostics

| Diagnostic | Value |
| --- | --- |
| Worker count | `2` |
| Chain count per worker smoke | `2` |
| Worker return codes | `0`, `0` |
| Fixed transport adapter signature | `db6b58a7adc8190f5ed2e48e42482956d32faf02bdf10a7104659a2bd86722c9` |
| Transport hash | `f1780d9eb8ae0f6d5e6865da6dbb3d1d1a22c4c2e5c89beb60c1f887c5f48fc7` |
| Target signature | `275bdd37a82d8c09d2c1b1935b6481f18224644ac659830a921c7958b6ed9038` |
| Adapter signature | `d89bdedcf759566f490ce5222ef753cc8c0c8ea39805d68c064c12d2bec07900` |
| Worker 0 compile-time proxy | `12.7924878988415` seconds |
| Worker 1 compile-time proxy | `12.870044000912458` seconds |
| Worker second-call wall times | `0.0010113921016454697`, `0.001079607056453824` seconds |
| Finite value/score checks | Passed in both workers |
| Full-chain diagnostic authority | Not granted in Phase 19; both workers recorded `accepted_full_chain_xla_diagnostic_authority=false`. |

## Repair History

The first smoke attempt wrote a blocker artifact because the helper used a
static row-unroll path against a batch-native LGSSM adapter. The failure was:

`ValueError: batch-native SSM target requires rank 2 theta [B, D]`

The fix was to use the batch-native fixed-transport value/score compile path in
workers, preserving `jit_compile=True` and not running HMC transitions.

The second smoke attempt reached CPU XLA compilation and worker success, but
the aggregate pass was incorrectly computed by applying `all()` to fields where
`False` is the expected nonaction value. The fix renamed the aggregate boundary
checks to positive predicates such as `training_not_executed` and
`hmc_transition_not_executed`.

No `jit_compile=false` run was used for either repair.

## Local Checks

- `python -m py_compile bayesfilter/testing/neutra_cpu_multicore_hmc_chain_harness_tf.py tests/test_neutra_cpu_multicore_hmc_chain_harness_tf.py`: passed.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/test_neutra_cpu_multicore_hmc_chain_harness_tf.py tests/test_neutra_fixed_transport_hmc_mechanics_xla_tf.py tests/test_neutra_gpu_affine_payload_tf.py -q`: passed, `15 passed, 2 warnings`.
- Phase 19 helper source scan for `GradientTape`, `batch_jacobian`, `tape.`, and `jit_compile=False`: passed with `violations=[]`.
- `python -m json.tool` on the Phase 19 harness JSON: passed.
- Phase 19 JSON field-validation script: passed with `failed=[]`.
- `git diff --check` on Phase 19 helper/tests and Phase 19-21 docs: passed.

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not being concluded |
| --- | --- | --- | --- | --- | --- |
| Pass Phase 19 harness | Met | Clear | This is not a full-chain HMC or posterior validation artifact. | Proceed to Phase 20 LGSSM reference HMC validation under its reviewed subplan. | No HMC convergence, posterior correctness, sampler quality, production/default readiness, nonlinear SSM validity, or scientific validity. |

## Inference Status

| Row | Status |
| --- | --- |
| Hard veto screen | Clear for Phase 19 boundary. |
| Statistically supported ranking | Not applicable; no method ranking was run. |
| Descriptive-only differences | Compile-time proxy and second-call wall times are descriptive only. |
| Default-readiness | Not established. |
| Next evidence needed | Phase 20 retained-chain comparison against exact LGSSM reference posterior. |

## Phase 20 Handoff

Phase 20 may begin because:

- Phase 19 wrote a pass result;
- workers were CPU-hidden and deterministic;
- worker return codes were recorded and zero;
- no `jit_compile=false`, training, HMC transition, HMC sampling/tuning, GPU
  sample generation, or posterior validation occurred;
- Phase 20 and Phase 21 subplans have been drafted and locally reviewed.
