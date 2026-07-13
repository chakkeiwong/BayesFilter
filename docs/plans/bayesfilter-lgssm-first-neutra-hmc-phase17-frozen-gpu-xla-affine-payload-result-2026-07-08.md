# Phase 17 Result: Frozen GPU/XLA-Trained Affine Payload

Date: 2026-07-08

## Scope

This result closes the Phase 17 LGSSM NeuTra packaging gate. The phase packaged
the Phase 16 GPU/XLA-trained affine NeuTra state as a frozen affine payload,
loaded it through the frozen artifact loader, and ran CPU-hidden loader/
reference finite checks.

This phase did not run new NeuTra training, fixed-transport HMC mechanics, HMC
sampling/tuning, external sample generation, a new JIT compile/timing/size
gate, `jit_compile=false`, DSGE/c603, default-policy changes, or scientific/
product/readiness claims.

## Decision Table

| Field | Status |
| --- | --- |
| Decision | `PASS_PHASE17_FROZEN_GPU_XLA_AFFINE_PAYLOAD` |
| Primary criterion status | Passed: payload and validation JSON were written from the exact Phase 16 source file hash, loader accepted the payload with matching target signature, finite forward/base value/score checks passed, and stale Phase 10/11 artifacts were not used. |
| Veto diagnostic status | Passed: no training, no fixed-transport HMC mechanics, no HMC sampling/tuning, no external sample generation, no `jit_compile=false`, no signature mismatch, no malformed JSON, no oversized artifacts. |
| Main uncertainty | Fixed-transport HMC mechanics with XLA authority and compile/timing/size evidence remain unchecked. That is Phase 18 work. |
| Next justified action | Draft and review Phase 18 fixed-transport HMC mechanics/XLA compile authority subplan. |
| What is not concluded | HMC convergence, posterior correctness, sampler quality, transport superiority, XLA compile readiness, production readiness, default readiness, broad nonlinear SSM validity, or scientific validity. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can BayesFilter package the Phase 16 GPU/XLA-trained affine state as a frozen payload and reload it against the current manual-score LGSSM target signatures without running fixed-transport HMC mechanics? |
| Baseline/comparator | Phase 16 GPU/XLA training-state artifact and current frozen affine artifact loader. |
| Primary criterion | Packaging writes payload and validation JSON, loader accepts the payload with matching target signature, finite forward/base value/score checks pass, exact Phase 16 source hash is recorded, fixed-transport HMC mechanics are not run, and stale Phase 10/11 artifacts are not used. |
| Veto diagnostics | Source state not Phase 16/XLA, target or adapter signature mismatch, malformed payload, nonfinite loader/reference diagnostics, hidden training, hidden fixed-transport HMC mechanics, hidden HMC sampling/tuning, hidden sample generation, `jit_compile=false` fallback, oversized artifacts, unsupported readiness/scientific/product claims. |
| Explanatory diagnostics | Payload hash, loader signature, forward/logdet probes, base value/score checks, reference residuals. |
| Artifact | Phase 17 payload JSON, validation JSON, and this result note. |

## Review

Claude review was attempted using the local review gate, but the sandbox
approval reviewer rejected the command as an external-service disclosure risk.
No workaround was attempted.

A same-foreground Codex substitute review was written at:

`docs/reviews/bayesfilter-lgssm-neutra-hmc-phase17-subplan-codex-substitute-review-2026-07-08.md`

The substitute review returned `VERDICT: AGREE`. It is weaker than an
independent Claude review and does not authorize human, runtime, product,
default-policy, or scientific-claim boundaries.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `e09046088be79f4100a77583063889a37be1de04` |
| Plan | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase17-frozen-gpu-xla-affine-payload-subplan-2026-07-08.md` |
| Command | `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m bayesfilter.testing.neutra_gpu_affine_payload_tf --phase16-training-state-path docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_training_state_seed20260707.json --artifact-dir docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07 --seed 20260707` |
| Runtime target | CPU-hidden packaging/loader/reference check |
| Source training artifact | `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_training_state_seed20260707.json` |
| Source file SHA-256 | `727fea040502e4fcb1af2203b9a490d03ab00dca63fd756f501a5bc3c936af7b` |
| Source stable artifact hash | `sha256:27f2c4364db13d1be14d7ad48b3257bd3f8418c091ad4d075db8504917bdb1c3` |
| Target signature | `275bdd37a82d8c09d2c1b1935b6481f18224644ac659830a921c7958b6ed9038` |
| Adapter signature | `d89bdedcf759566f490ce5222ef753cc8c0c8ea39805d68c064c12d2bec07900` |
| Seed | `20260707` |
| Payload artifact | `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_frozen_payload_seed20260707.json` |
| Payload file size | `2.2K` |
| Payload file SHA-256 | `18992dface97aa8142d714b2fe99b89ee4717c0e5f01a06c7f6e4a868b220aa1` |
| Payload stable hash | `sha256:150ac81e52d8a3f03721c6ade499c6cf27e9632639a4835e9b40c6b2b76b35df` |
| Validation artifact | `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_payload_validation_seed20260707.json` |
| Validation file size | `6.5K` |
| Validation file SHA-256 | `e49bc1ebac7a39671682cb28bc125e8eeba88b51ea3ba6a11111caf60a527fd2` |
| Validation stable hash | `sha256:ee0a9dc890a9d5af560b887be86fa3fd51cf55e6766912b4e5199874072793a8` |
| Frozen artifact signature | `5e36c60aaca37facb3e110138e1b2da2ebe758ace1efb6a8845650553dc3d7e0` |
| Transport hash | `f1780d9eb8ae0f6d5e6865da6dbb3d1d1a22c4c2e5c89beb60c1f887c5f48fc7` |

## Artifact Validation

| Check | Status |
| --- | --- |
| Payload JSON parse | Passed |
| Validation JSON parse | Passed |
| Payload size below 20 MB | Passed |
| Validation size below 20 MB | Passed |
| Source path recorded | Passed |
| Source SHA-256 recorded | Passed |
| Target signature match | Passed |
| Adapter signature match | Passed |
| Finite forward/base checks | Passed |
| Training not run | Passed |
| Fixed-transport HMC mechanics not run | Passed |
| HMC sampling/tuning not run | Passed |
| External sample generation not run | Passed |
| `jit_compile=false` not run | Passed |

Reference residuals:

```text
initial_batch_value_residual = 0.0
initial_batch_score_residual = 4.440892098500626e-16
```

## Local Checks

- `python -m py_compile bayesfilter/testing/neutra_gpu_affine_payload_tf.py tests/test_neutra_gpu_affine_payload_tf.py`: passed.
- Source/config scan for Phase 16/17 required tokens and stale Phase 10/11
  source tokens: passed with `missing=[]`, `violations=[]`.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/test_neutra_gpu_affine_payload_tf.py -q`: passed, `5 passed, 2 warnings`.
- `python -m json.tool` on the payload JSON: passed.
- `python -m json.tool` on the validation JSON: passed.
- Phase 17 field-validation script: passed with `failed=[]`.
- `git diff --check -- bayesfilter/testing/neutra_gpu_affine_payload_tf.py tests/test_neutra_gpu_affine_payload_tf.py docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase17-frozen-gpu-xla-affine-payload-subplan-2026-07-08.md`: passed.

## Nonclaims

- No new NeuTra training was run.
- No fixed-transport HMC mechanics were run.
- No HMC sampling or tuning was run.
- No external sample generation was run.
- No new JIT compile/timing/size gate was run.
- No `jit_compile=false` runtime diagnostic was run.
- No DSGE/c603 target was used.
- No HMC convergence, posterior correctness, sampler quality, transport
  superiority, XLA compile readiness, production readiness, default readiness,
  broad nonlinear SSM validity, or scientific validity is claimed.

## Next Handoff

Phase 18 should repair or validate the fixed-transport HMC mechanics/XLA
authority boundary and measure the reviewed `jit_compile=True` compile gate,
including compile time and available compile-size/proxy evidence. It must not
run HMC sampling/tuning or external sample generation unless its reviewed
subplan explicitly authorizes those actions.
