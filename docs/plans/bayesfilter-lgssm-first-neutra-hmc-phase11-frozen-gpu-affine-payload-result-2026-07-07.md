# Phase 11 Result: Frozen GPU-Trained Affine Payload

Date: 2026-07-07

## Scope

This result closes the Phase 11 frozen GPU-trained affine payload packaging
gate for the LGSSM-first BayesFilter NeuTra/HMC program.  It packages the
Phase 10 GPU-trained affine parameters into the reviewed frozen affine-diagonal
transport schema, reloads the payload through `load_frozen_neutra_artifact`,
and runs CPU-hidden loader/mechanics/reference checks.

This is not new NeuTra training, not CPU NeuTra training, not dense IAF
training, not HMC sampling or tuning, not external sample generation, not XLA
repair/readiness, not posterior convergence evidence, not route ranking, not
production/default readiness, and not scientific validity.

## Decision Table

| Field | Status |
| --- | --- |
| Decision | `PASS_PHASE11_FROZEN_GPU_AFFINE_PAYLOAD` |
| Primary criterion status | Passed: the Phase 10 GPU-trained affine parameters were packaged as a frozen affine payload, loaded with the exact Phase 10 target signature, and passed finite forward/logdet plus mechanics/reference checks. |
| Veto diagnostic status | No final Phase 11 veto fired: Phase 10 target/adapter signatures matched, payload tensors were finite, loader accepted the artifact, mechanics/reference diagnostics were finite, and no training, HMC sampling/tuning, XLA/JIT repair, or external sample generation ran. |
| Main uncertainty | This proves frozen-payload usability only. It does not show transport quality, posterior correctness, HMC readiness, XLA readiness, route superiority, production readiness, default-readiness, or scientific validity. |
| Next justified action | Phase 12 should define and smoke-check the CPU multicore external sample-generation boundary, keeping sample generation separate from GPU NeuTra training and from HMC sampling/tuning. |
| What is not concluded | Full NeuTra quality, dense IAF quality, HMC readiness, posterior correctness, sampler convergence, route superiority, XLA readiness, production readiness, default-policy change, or scientific validity. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the Phase 10 GPU-trained affine parameters be packaged into BayesFilter's frozen affine transport schema and loaded/mechanics-checked without new training, HMC, or sample generation? |
| Baseline/comparator | Phase 5 fixed identity/affine transport mechanics and Phase 6 CPU affine payload schema, with Phase 10 GPU training state as the source of learned parameters. |
| Primary criterion | Frozen affine payload loads with the Phase 10 target signature, preserves finite forward/logdet behavior, and passes finite mechanics/reference checks against the LGSSM generic adapter. |
| Veto diagnostics | Missing Phase 10 training state, malformed schema, target-signature or adapter-signature mismatch, nonfinite payload tensors, loader failure, mechanics/reference failure, hidden training, hidden HMC sampling/tuning, hidden sample generation, XLA/JIT requirement, or unsupported readiness/scientific claim. |
| Explanatory diagnostics | Payload hash, loaded artifact signature, transport hash, value/score residuals, transformed target finite checks, and mechanics-only manifest fields. |
| Not concluded | Transport quality, posterior correctness, HMC readiness, XLA readiness, route superiority, production readiness, default execution readiness, or scientific validity. |
| Artifact | Phase 11 payload JSON, validation JSON, result, tests, and Phase 12/13 subplans. |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `e09046088be79f4100a77583063889a37be1de04` on `origin/main` before Phase 11 edits. |
| Source Phase 10 artifact | `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_training_state_seed20260707.json` |
| Source Phase 10 file SHA-256 | `263c492c9789c9b50e245b14efd0bacb114d281b52285267c9ce4c5280496811` |
| Source Phase 10 stable hash | `sha256:5b6bb48c74fc3ddc4d97404d7220a08323d90337e2a72e24f1fcdaa82a7de351` |
| Phase 11 command | `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m bayesfilter.testing.neutra_gpu_affine_payload_tf` |
| Execution target | CPU-hidden artifact packaging, loader, and mechanics checks. |
| GPU training | Not run; Phase 10 GPU artifact is the source. |
| CPU NeuTra training | Not run; forbidden. |
| HMC | Mechanics-only value/score binding; no sampling or tuning. |
| External sample generation | Not run; reserved for Phase 12 CPU multicore boundary. |
| JIT/XLA | Not run; Phase 9 XLA blocker inherited. |
| Plan file | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase11-frozen-gpu-affine-payload-subplan-2026-07-07.md` |
| Result file | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase11-frozen-gpu-affine-payload-result-2026-07-07.md` |

TensorFlow emitted CUDA factory/cuInit messages during the CPU-hidden run. That
is expected framework startup noise under `CUDA_VISIBLE_DEVICES=-1` and is not
GPU evidence, driver evidence, or a GPU failure diagnosis for this phase.

## Phase 11 Artifacts

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_frozen_payload_seed20260707.json` | `1844` bytes | `f0938501e974ad2975283ce2ed2603d784d758f22b3ad9fdcd28ddcea8855157` |
| `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_payload_validation_seed20260707.json` | `7443` bytes | `0fb4e96d8fe3d66318d2102b22a33d87990c29d9845d34258a2427dfceebd767` |

Both artifacts are below the 20 MB repository policy threshold for
claim-supporting artifacts.

## Validation Outcome

| Field | Value |
| --- | --- |
| Target signature | `290a91d2a8f90d5b29243965b258b1ec6fd965aa46ffca69dcb78f7fa1ecabcb` |
| Adapter signature | `0a48b43d2750cad5b452708f7619a1119a259231d8955769809460f256575a97` |
| Artifact signature | `1cccddb4ae606ca084a3d08f7fbdd3c959dafb73b77ad152df6380051c599340` |
| Transport hash | `cdd67ece589a3cb4c474dcd0702686d7d47d1bc406347a3312df38285e75da25` |
| Payload stable hash | `sha256:77cecbfa879e5249d18de580480c14dba79936cd3242c48ca8812ab0366494e8` |
| Validation stable hash | `sha256:fbcc30474cecace4ecbd6994a907bfc3e2851b774814538daf494d8ebe8769e3` |
| Mechanics value | `[-3.2122163938918966]` |
| Mechanics score | `[[0.07575628264232154, -0.022210820660321807]]` |
| Reference value residual | `0.0` |
| Reference score residual | `0.0` |

Finite checks all passed:

- payload shift and raw scale finite;
- transformed `theta` finite;
- log determinant finite;
- base target value and score finite;
- mechanics value and score finite.

Boundary checks all passed:

- target signature matched;
- adapter signature matched;
- source artifact was GPU training;
- no new training ran;
- no HMC sampling or tuning ran;
- no external sample generation ran;
- no JIT/XLA path ran.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Phase 11 used the Phase 10 LGSSM QR GPU training state and existing frozen affine loader/mechanics surfaces; DSGE/c603 and LEDH evidence were not used. |
| Proxy promotion | Successful payload loading and finite mechanics are usability evidence only, not posterior correctness or HMC readiness. |
| Missing stop conditions | Signature mismatch, malformed tensors, loader failure, hidden training, hidden HMC/sample generation, XLA requirement, and unsupported claims were explicit vetoes. |
| Hidden assumption | Phase 11 assumes the Phase 10 artifact is the only learned-parameter source; this is enforced by file hash and signature checks. |
| Environment mismatch | Phase 11 loader/mechanics checks intentionally ran with `CUDA_VISIBLE_DEVICES=-1`; this is CPU-safe artifact validation, not training. |
| Artifact mismatch | The payload and validation JSON preserve nonclaims, source hashes, target/adapter signatures, and no-training/no-HMC/no-sample flags. |

Audit status: passed for Phase 11 execution.

## Local Checks

- `python -m py_compile bayesfilter/testing/neutra_gpu_affine_payload_tf.py tests/test_neutra_gpu_affine_payload_tf.py`: passed.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/test_neutra_gpu_affine_payload_tf.py -q`: passed, `3 passed, 2 warnings`.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m bayesfilter.testing.neutra_gpu_affine_payload_tf`: passed and emitted `passed: true`.
- `python -m json.tool` on the Phase 10 training state, Phase 11 frozen payload, and Phase 11 validation JSON: passed.

The CPU-hidden pytest and CLI commands are test/packaging checks only. They are
not GPU evidence and not training runs.

## Implementation Artifacts

- `bayesfilter/testing/neutra_gpu_affine_payload_tf.py`
- `tests/test_neutra_gpu_affine_payload_tf.py`
- `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_frozen_payload_seed20260707.json`
- `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_payload_validation_seed20260707.json`
- `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase12-cpu-multicore-sample-generation-subplan-2026-07-07.md`
- `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase13-xla-jit-repair-subplan-2026-07-07.md`

## Phase Close Duties

1. Required local checks were run and recorded above.
2. This Phase 11 result records the close decision and nonclaims.
3. The Phase 12 and Phase 13 subplans were drafted.
4. Bounded read-only review agreed with this result and the next-phase
   boundary.

## Read-Only Review

- Claude health probe returned `CLAUDE_PROBE_OK`.
- Claude one-path read-only review of this Phase 11 result returned
  `VERDICT: AGREE`.
- Claude one-path read-only review of the Phase 12 subplan returned
  `VERDICT: AGREE` with one non-blocking clarity suggestion. The Phase 12
  subplan was patched to state that, if no new helper is added, tests must bind
  to an existing boundary surface and identify it in the result.
- The first Phase 13 one-path review prompt stalled. A narrower material-blocker
  prompt against the same single path returned `VERDICT: AGREE`.

## Nonclaims

- No new NeuTra training was run.
- No CPU NeuTra training was run.
- No dense IAF training or packaging was run.
- No HMC sampling or tuning was run.
- No external sample generation was run.
- No XLA/JIT repair or readiness claim is made.
- No route ranking is claimed.
- No posterior correctness, convergence, sampler superiority, production
  readiness, default execution readiness, default-policy change, or scientific
  validity is claimed.
