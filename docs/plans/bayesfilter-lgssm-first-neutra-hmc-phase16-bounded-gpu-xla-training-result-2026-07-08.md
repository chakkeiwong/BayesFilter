# Phase 16 Result: Bounded GPU/XLA NeuTra Training

Date: 2026-07-08

## Scope

This result closes the Phase 16 bounded LGSSM affine NeuTra training gate. The
phase tested the repaired manual-score training route under trusted GPU
execution with `jit_compile=True` only.

The phase did not run `jit_compile=false`, runtime autodiff, HMC
sampling/tuning, external sample generation, DSGE/c603, packaging, route
ranking, default-policy changes, or scientific/product/readiness claims.

## Decision Table

| Field | Status |
| --- | --- |
| Decision | `PASS_PHASE16_BOUNDED_GPU_XLA_NEUTRA_TRAINING` |
| Primary criterion status | Passed: 12 bounded training steps completed on trusted GPU with `jit_compile=True`, current signatures, finite diagnostics, and a parseable training-state artifact. |
| Veto diagnostic status | Passed: no `jit_compile=false` runtime, no CPU runtime evidence, no runtime autodiff, no HMC, no sample generation, no signature mismatch, no malformed artifact. |
| Main uncertainty | This is bounded optimizer training only. It does not establish full NeuTra quality, posterior correctness, HMC convergence, or production readiness. |
| Next justified action | Draft and review Phase 17 frozen-payload packaging against the Phase 16 XLA-trained artifact. |
| What is not concluded | HMC convergence, posterior correctness, sampler quality, transport superiority, production readiness, default readiness, broad nonlinear SSM validity, or scientific validity. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can BayesFilter run a bounded LGSSM affine NeuTra training gate on trusted GPU with `jit_compile=True`, manual scores, and no CPU fallback? |
| Baseline/comparator | Phase 15 compile-gate pass, current manual-score target/adapter signatures, and old Phase 10 non-XLA training as stale history only. |
| Primary criterion | Predeclared bounded training steps complete under trusted GPU `jit_compile=True`, write a parseable training-state artifact, preserve current signatures, and record finite loss/gradient/update diagnostics. |
| Veto diagnostics | Any `jit_compile=false` runtime run, CPU runtime evidence, hidden runtime autodiff route, hidden HMC sampling/tuning, hidden external sample generation, target/adapter signature mismatch, nonfinite diagnostics, malformed artifact, or unsupported readiness/scientific/product claim. |
| Explanatory diagnostics | Loss history, manual gradient norms, device placements, training-state hash, TensorFlow/GPU manifest. |
| Artifact | `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_training_state_seed20260707.json` |

## Review

Claude reviewed the Phase 16 subplan as read-only reviewer.

- First review returned `VERDICT: REVISE` because the plan lacked exact bounded
  command, timeout, output filename, mandatory JSON fields, and explicit
  boundary proofs.
- The subplan was patched to add those execution and artifact constraints.
- A narrowed second review returned `VERDICT: AGREE`.

Claude did not authorize runtime, product, scientific, or readiness claims.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `e09046088be79f4100a77583063889a37be1de04` |
| Plan | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase16-bounded-gpu-xla-training-subplan-2026-07-08.md` |
| Command | `timeout 240 env TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp python -m bayesfilter.testing.neutra_gpu_bounded_training_tf --steps 12 --batch-size 16 --seed 20260707 --learning-rate 0.03 --initial-raw-scale -1.3862943611198906 --device /GPU:0` |
| Python | `/home/chakwong/anaconda3/envs/tf-gpu/bin/python`, Python `3.11.14` |
| TensorFlow | `2.19.1` |
| Trusted GPU probe | `nvidia-smi` passed: NVIDIA GeForce RTX 4080 SUPER, driver `591.86`, CUDA `13.1`. |
| Runtime target | Trusted GPU, `/GPU:0` |
| JIT policy | `jit_compile=True`; `jit_compile=false` runtime was not run. |
| TF32 | Enabled in TensorFlow manifest. |
| Seed | `20260707` |
| Steps | `12` |
| Batch size | `16` |
| Learning rate | `0.03` |
| Initial raw scale | `-1.3862943611198906` |
| Wall time in artifact | `69.49786909413524` seconds |
| Output artifact | `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_training_state_seed20260707.json` |
| Output file size | `19206` bytes |
| Output file SHA-256 | `727fea040502e4fcb1af2203b9a490d03ab00dca63fd756f501a5bc3c936af7b` |
| Stable artifact hash | `sha256:27f2c4364db13d1be14d7ad48b3257bd3f8418c091ad4d075db8504917bdb1c3` |
| XLA log | TensorFlow logged `Compiled cluster using XLA!` |

## Artifact Validation

Required field validation passed with no missing keys or violations.

| Check | Value |
| --- | --- |
| Schema | `bayesfilter.neutra.gpu_xla_bounded_training_state.v1` |
| Phase | `phase16_bounded_gpu_xla_neutra_training` |
| Target signature | `275bdd37a82d8c09d2c1b1935b6481f18224644ac659830a921c7958b6ed9038` |
| Adapter signature | `d89bdedcf759566f490ce5222ef753cc8c0c8ea39805d68c064c12d2bec07900` |
| `jit_compile` | `true` |
| `jit_compile_false_runtime_executed` | `false` |
| `runtime_autodiff_executed` | `false` |
| `keras_optimizer_gradient_route_executed` | `false` |
| `hmc_executed` | `false` |
| `sample_generation_executed` | `false` |
| `external_sample_generation_executed` | `false` |
| All finite checks | `true` |
| All objective outputs on GPU | `true` |

## Training Diagnostics

These diagnostics are descriptive only.

| Diagnostic | Value |
| --- | ---: |
| Initial loss | `4.270573668036143` |
| Final loss | `3.7532094500806354` |
| First-step global gradient norm | `1.6678158366091047` |
| Last-step global gradient norm | `1.0420800003721395` |

Final affine parameters:

```text
final_shift = [0.11725994567583073, -1.288484681090852]
final_raw_scale = [-1.0943541507529184, -1.1246545088489457]
```

Loss history:

```text
[4.270573668036143, 4.185668271792768, 4.263944438523298,
 4.080745947382259, 4.008764836951993, 3.985789812240662,
 4.011308035664472, 3.9095017004703903, 3.7907625843623327,
 3.9222723346607173, 3.768895303037646, 3.7532094500806354]
```

No ranking or superiority claim follows from these values.

## Local Checks

- `python -m py_compile bayesfilter/testing/neutra_gpu_bounded_training_tf.py bayesfilter/testing/neutra_gpu_affine_payload_tf.py tests/test_neutra_gpu_bounded_training_tf.py tests/test_neutra_gpu_affine_payload_tf.py`: passed.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/test_neutra_gpu_bounded_training_tf.py tests/test_neutra_gpu_affine_payload_tf.py tests/test_neutra_xla_repair_tf.py -q`: passed, `15 passed, 2 warnings`. This was source/config guard evidence only, not runtime evidence.
- Source scan for runtime autodiff and Keras optimizer-gradient APIs in `bayesfilter/testing/neutra_gpu_bounded_training_tf.py`: no hits.
- `python -m json.tool docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_training_state_seed20260707.json`: passed.
- Training-state field contract validation: passed with `violations=[]`.

## Nonclaims

- No `jit_compile=false` runtime diagnostic was run.
- No runtime autodiff route was used.
- No HMC sampling or tuning was run.
- No external sample generation was run.
- No DSGE/c603 target was used.
- No full NeuTra quality, HMC readiness, posterior correctness, sampler quality,
  transport superiority, production readiness, default-policy change, broad
  nonlinear SSM validity, or scientific validity is claimed.

## Next Handoff

Phase 17 may plan frozen-payload packaging using the Phase 16 XLA-trained
artifact. It must not use the stale Phase 10/11 non-XLA artifacts for
promotion. Packaging must preserve `jit_compile=True` mechanics checks and must
not run training, HMC sampling/tuning, or external sample generation.
