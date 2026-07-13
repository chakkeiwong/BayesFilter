# Phase 18 Result: Fixed-Transport HMC Mechanics XLA Compile Gate

Date: 2026-07-08

## Scope

This result closes the Phase 18 LGSSM NeuTra fixed-transport mechanics compile
gate. The phase added an explicit evidence-backed XLA-HMC value/score opt-in
for generic SSM adapters, opted the LGSSM fixture in using the Phase 15 trusted
GPU/XLA compile result as evidence, loaded the Phase 17 frozen affine payload,
and compiled a one-point fixed-transport mechanics value/score function with
trusted GPU `jit_compile=True`.

This phase did not run NeuTra training, HMC sampling/tuning, external sample
generation, `jit_compile=false`, CPU runtime compile evidence, DSGE/c603,
default-policy changes, or scientific/product/readiness claims.

## Decision Table

| Field | Status |
| --- | --- |
| Decision | `PASS_PHASE18_FIXED_TRANSPORT_HMC_MECHANICS_XLA_COMPILE` |
| Primary criterion status | Passed: base and fixed-transport value/score capabilities advertise accepted XLA-HMC authority, trusted GPU `jit_compile=True` mechanics compile executed, mechanics value/score were finite, and timing/size proxies were recorded. |
| Veto diagnostic status | Passed: no `jit_compile=false`, no CPU runtime compile evidence, no fallback/GradientTape authority promotion, no training, no HMC sampling/tuning, no external samples, no signature mismatch, no nonfinite mechanics, no malformed/oversized artifact. |
| Main uncertainty | This is mechanics compile evidence only. It does not establish HMC chain correctness, tuning quality, posterior agreement, or production readiness. |
| Next justified action | Draft Phase 19 CPU-hidden multicore HMC chain harness subplan. |
| What is not concluded | HMC convergence, posterior correctness, sampler quality, transport superiority, production readiness, default readiness, broad nonlinear SSM validity, CPU multicore harness readiness, or scientific validity. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the Phase 17 frozen affine payload be bound to the current LGSSM generic SSM adapter as an accepted fixed-transport HMC mechanics target and compiled with trusted GPU `jit_compile=True` without running HMC chains? |
| Baseline/comparator | Phase 17 payload/validation, current LGSSM target signatures, Phase 15 manual-score trusted GPU/XLA compile gate, and current fail-closed fixed-transport mechanics authority policy. |
| Primary criterion | Pass artifact records accepted base and fixed-transport value/score XLA authority, trusted GPU `jit_compile=True` mechanics compile success, finite mechanics value/score, timing/size proxies, and no forbidden runtime actions. |
| Veto diagnostics | Any `jit_compile=false` runtime run, CPU runtime evidence for compile success, hidden training, hidden HMC sampling/tuning, hidden external sample generation, fallback/GradientTape authority promotion, target/adapter/payload signature mismatch, nonfinite mechanics, malformed/oversized artifact, unsupported readiness/scientific/product claim. |
| Explanatory diagnostics | First/second call timing, compile-time proxy, concrete graph bytes, HLO text bytes, value/score capability manifests, mechanics values/scores, payload hashes. |
| Artifact | `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase18_fixed_transport_hmc_mechanics_xla_compile_seed20260707.json` |

## Review

Claude review remained unavailable because the earlier Claude review gate was
rejected by the sandbox approval reviewer as an external-service disclosure
risk. No workaround was attempted.

A same-foreground Codex substitute review of the Phase 18 subplan was written
at:

`docs/reviews/bayesfilter-lgssm-neutra-hmc-phase18-subplan-codex-substitute-review-2026-07-08.md`

The substitute review returned `VERDICT: AGREE`. It is weaker than an
independent Claude review and does not authorize human, runtime, product,
default-policy, or scientific-claim boundaries.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `e09046088be79f4100a77583063889a37be1de04` |
| Plan | `docs/plans/bayesfilter-lgssm-neutra-hmc-phase18-fixed-transport-mechanics-compile-subplan-2026-07-08.md` |
| Trusted GPU probe | `nvidia-smi` passed: NVIDIA GeForce RTX 4080 SUPER, driver `591.86`, CUDA `13.1`. |
| Command | `TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp python -m bayesfilter.testing.neutra_fixed_transport_hmc_mechanics_xla_tf --payload-path docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_frozen_payload_seed20260707.json --output-path docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase18_fixed_transport_hmc_mechanics_xla_compile_seed20260707.json --seed 20260707 --device /GPU:0` |
| Runtime target | Trusted GPU `/GPU:0` |
| JIT policy | `jit_compile=True`; `jit_compile=false` runtime was not run. |
| Payload input | `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_frozen_payload_seed20260707.json` |
| Payload file SHA-256 | `18992dface97aa8142d714b2fe99b89ee4717c0e5f01a06c7f6e4a868b220aa1` |
| Payload stable hash | `sha256:150ac81e52d8a3f03721c6ade499c6cf27e9632639a4835e9b40c6b2b76b35df` |
| Frozen artifact signature | `5e36c60aaca37facb3e110138e1b2da2ebe758ace1efb6a8845650553dc3d7e0` |
| Transport hash | `f1780d9eb8ae0f6d5e6865da6dbb3d1d1a22c4c2e5c89beb60c1f887c5f48fc7` |
| Target signature | `275bdd37a82d8c09d2c1b1935b6481f18224644ac659830a921c7958b6ed9038` |
| Adapter signature | `d89bdedcf759566f490ce5222ef753cc8c0c8ea39805d68c064c12d2bec07900` |
| Fixed-transport adapter signature | `db6b58a7adc8190f5ed2e48e42482956d32faf02bdf10a7104659a2bd86722c9` |
| Output artifact | `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase18_fixed_transport_hmc_mechanics_xla_compile_seed20260707.json` |
| Output file size | `6.8K` |
| Output file SHA-256 | `6d1cb178f1e7b91170450acf8843cb7e7062d9fe6c125d21217489fdf4ade2a5` |
| Output stable artifact hash | `sha256:5059f900fe92df20ef73b326aa5fdaac7f0264a475949c83b0405d994d9b1269` |
| TensorFlow | `2.19.1` |
| XLA log | TensorFlow logged `Compiled cluster using XLA!` |

## Compile Diagnostics

| Diagnostic | Value |
| --- | ---: |
| First call wall time | `2.3196849480737` seconds |
| Second call wall time | `0.003909368999302387` seconds |
| Compile-time proxy | `2.3157755790743977` seconds |
| Concrete graph serialized size | `3087116` bytes |
| Compiler IR/HLO text size | `1262491` bytes |
| Total diagnostic elapsed time | `12.990896784933284` seconds |

Finite checks:

```text
mechanics_value_finite = true
mechanics_score_finite = true
second_mechanics_value_finite = true
second_mechanics_score_finite = true
```

Mechanics probe at `z = [[0.0, 0.0]]`:

```text
value = [-3.386227432795513]
score = [[-0.06925843898065151, -0.10870454061182869]]
```

Both recorded outputs were on `/GPU:0`.

## Authority Checks

Base LGSSM adapter:

```text
value_score_authority = graph_native
xla_hmc_ready = true
accepted_xla_hmc_authority = true
full_chain_xla_diagnostic_ready = false
evidence_path = docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase15-manual-score-xla-compile-gate-result-2026-07-08.md
```

Fixed-transport adapter:

```text
value_score_authority = graph_native
xla_hmc_ready = true
accepted_xla_hmc_authority = true
full_chain_xla_diagnostic_ready = false
evidence_path = bayesfilter/testing/neutra_fixed_transport_hmc_mechanics_xla_tf.py
```

The generic SSM adapter still defaults to `xla_hmc_ready=false`; Phase 18 added
an explicit evidence-backed opt-in, not a global default promotion.

## Local Checks

- `python -m py_compile bayesfilter/ssm/target_builder.py bayesfilter/ssm/__init__.py bayesfilter/testing/lgssm_generic_target_adapter_tf.py bayesfilter/testing/neutra_fixed_transport_hmc_mechanics_xla_tf.py tests/test_general_ssm_target_builder.py tests/test_lgssm_generic_target_adapter_tf.py tests/test_neutra_fixed_transport_hmc_mechanics_xla_tf.py`: passed.
- Source scan for forbidden runtime tokens in Phase 18 helper path: passed with
  `missing=[]`, `violations=[]`.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/test_general_ssm_target_builder.py tests/test_lgssm_generic_target_adapter_tf.py tests/test_neutra_gpu_affine_payload_tf.py tests/test_neutra_artifact_loader.py tests/test_fixed_transport_hmc_binding.py tests/test_neutra_fixed_transport_hmc_mechanics_xla_tf.py -q`: passed, `41 passed, 1 skipped, 2 warnings`.
- Trusted `nvidia-smi`: passed.
- Trusted Phase 18 GPU/XLA compile diagnostic: passed.
- `python -m json.tool` on the diagnostic JSON: passed.
- Phase 18 field-validation script: passed with `failed=[]`.
- `git diff --check` on touched Phase 18 code/tests/docs: passed.

## Nonclaims

- No `jit_compile=false` runtime diagnostic was run.
- No NeuTra training was run.
- No HMC sampling or tuning was run.
- No external sample generation was run.
- No full-chain XLA diagnostic was run.
- No DSGE/c603 target was used.
- No HMC convergence, posterior correctness, sampler quality, transport
  superiority, production readiness, default-readiness, broad nonlinear SSM
  validity, CPU multicore harness readiness, or scientific validity is claimed.

## Next Handoff

Phase 19 may plan a CPU-hidden multicore HMC chain harness. It must keep sample
generation and chain execution separate from GPU NeuTra training, must not use
`jit_compile=false`, and must not claim posterior correctness or HMC readiness
until Phase 20/21 reference validation and decision gates pass.
