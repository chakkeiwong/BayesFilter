# Phase 16 Subplan: Bounded GPU/XLA NeuTra Training

Date: 2026-07-08

## Phase Objective

Run the first bounded LGSSM affine NeuTra training gate under the repaired
manual-score policy: trusted GPU only, `jit_compile=True` only, and no runtime
autodiff in the admitted training route.

This phase is not a fallback rerun of the historical Phase 10 non-XLA artifact.
It supersedes that artifact for any future packaging or promotion work.

## Entry Conditions Inherited From Previous Phase

- Phase 14A replaced the admitted LGSSM target and fixed-affine score route with
  analytical/manual score pullbacks.
- Phase 15 passed the trusted GPU XLA compile gate for the current manual-score
  affine NeuTra objective with `jit_compile=True`.
- Current target signature:
  `275bdd37a82d8c09d2c1b1935b6481f18224644ac659830a921c7958b6ed9038`.
- Current adapter signature:
  `d89bdedcf759566f490ce5222ef753cc8c0c8ea39805d68c064c12d2bec07900`.
- Old Phase 10/11 artifacts with target signature
  `290a91d2a8f90d5b29243965b258b1ec6fd965aa46ffca69dcb78f7fa1ecabcb`
  are stale diagnostic history only.
- No `jit_compile=false` runtime run may be executed.

## Required Artifacts

- Live training helper must default to `jit_compile=True` and reject
  `jit_compile=False`.
- Live training helper must use manual target scores and manual parameter
  updates, not runtime autodiff.
- Trusted GPU/XLA training-state JSON under
  `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/`.
  Exact path:
  `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_training_state_seed20260707.json`.
- Phase 16 result or blocker:
  `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase16-bounded-gpu-xla-training-result-2026-07-08.md`.
- A refreshed next subplan for frozen payload packaging only if Phase 16 passes.

## Exact Bounded Execution Spec

The only authorized runtime command for Phase 16 is:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp python -m bayesfilter.testing.neutra_gpu_bounded_training_tf --steps 12 --batch-size 16 --seed 20260707 --learning-rate 0.03 --initial-raw-scale -1.3862943611198906 --device /GPU:0
```

Run it with trusted/escalated permissions because it initializes and uses the
GPU. The command must not set `CUDA_VISIBLE_DEVICES=-1` and must not pass
`--no-jit-compile`.

Bounded runtime limits:

- steps: `12`;
- batch size: `16`;
- seed: `20260707`;
- learning rate: `0.03`;
- initial raw scale: `-1.3862943611198906`;
- device: `/GPU:0`;
- timeout cap: `240` seconds.

If the command times out, write a blocker result. Do not rerun with
`jit_compile=false`.

## Required Checks/Tests/Reviews

- Source/config guard checks proving:
  - `jit_compile=True` is the default;
  - `jit_compile=False` is rejected;
  - no `tf.GradientTape`, `with tf.GradientTape`, `.gradient(`,
    `tfp.math.value_and_gradient`, `value_and_gradient`,
    `tf.keras.optimizers`, or `apply_gradients` route is present in
    `bayesfilter/testing/neutra_gpu_bounded_training_tf.py`;
  - the output filename contains `gpu_xla_training_state`, so Phase 16 cannot
    overwrite the historical Phase 10 non-XLA state.
- `python -m py_compile` for the touched training helper and tests.
- CPU-hidden pytest for source/config guards only; this is not runtime evidence.
- Trusted `nvidia-smi` before runtime evidence.
- Trusted GPU command for the bounded training helper, with
  `TF_FORCE_GPU_ALLOW_GROWTH=true` and no `CUDA_VISIBLE_DEVICES=-1`.
- `python -m json.tool` on the training-state JSON if produced.
- JSON field check on the training-state JSON requiring the mandatory fields
  listed below.
- `git diff --check`.
- Bounded read-only review of the Phase 16 result and the next payload subplan
  before packaging or HMC work.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can BayesFilter run a bounded LGSSM affine NeuTra training gate on trusted GPU with `jit_compile=True`, manual scores, and no CPU fallback? |
| Baseline/comparator | Phase 15 compile-gate pass, current manual-score target/adapter signatures, and old Phase 10 non-XLA training as stale history only. |
| Primary criterion | Predeclared bounded training steps complete under trusted GPU `jit_compile=True`, write a parseable training-state artifact, preserve current signatures, and record finite loss/gradient/update diagnostics. |
| Veto diagnostics | Any `jit_compile=false` runtime run, CPU runtime evidence, hidden runtime autodiff route, hidden HMC sampling/tuning, hidden external sample generation, target/adapter signature mismatch, nonfinite diagnostics, malformed artifact, or unsupported readiness/scientific/product claim. |
| Explanatory diagnostics | Loss history, manual gradient norms, device placements, first-step compile/warm timing if recorded, training-state hash, TensorFlow/GPU manifest. |
| Not concluded | Full NeuTra quality, posterior correctness, HMC convergence, sampler quality, transport superiority, production readiness, default readiness, broad nonlinear SSM validity, or scientific validity. |
| Artifact | Phase 16 training-state JSON plus result/blocker note. |

## Mandatory Training-State JSON Fields

The training-state JSON must contain these keys:

- `schema = bayesfilter.neutra.gpu_xla_bounded_training_state.v1`;
- `phase = phase16_bounded_gpu_xla_neutra_training`;
- `decision`;
- `passed`;
- `config`;
- `gpu_manifest`;
- `target_signature`;
- `adapter_signature`;
- `parameter_dim`;
- `parameter_names`;
- `initial_shift`;
- `initial_raw_scale`;
- `final_shift`;
- `final_raw_scale`;
- `initial_loss`;
- `final_loss`;
- `loss_history`;
- `gradient_norm_history`;
- `finite_checks`;
- `device_checks`;
- `optimizer`;
- `optimizer_steps_executed`;
- `bounded_optimizer_training_executed`;
- `full_neutra_training_executed`;
- `hmc_executed`;
- `sample_generation_executed`;
- `external_sample_generation_executed`;
- `runtime_autodiff_executed`;
- `keras_optimizer_gradient_route_executed`;
- `jit_compile`;
- `jit_compile_false_runtime_executed`;
- `xla_blocker_status`;
- `evidence_contract`;
- `elapsed_seconds`;
- `artifact_hash`;
- `artifact_hash_semantics`;
- `nonclaims`.

Pass requires:

- `jit_compile = true`;
- `jit_compile_false_runtime_executed = false`;
- `runtime_autodiff_executed = false`;
- `keras_optimizer_gradient_route_executed = false`;
- `hmc_executed = false`;
- `sample_generation_executed = false`;
- `external_sample_generation_executed = false`;
- `optimizer_steps_executed = 12`;
- all finite checks true;
- all objective output devices on GPU;
- artifact file size less than `20 MB`.

## Forbidden Claims/Actions

- Do not run `jit_compile=false` diagnostics or fallback runtime.
- Do not use `GradientTape` or Keras optimizer gradients in the admitted route.
- Do not run HMC sampling or tuning.
- Do not generate external samples.
- Do not package a frozen payload before Phase 16 passes and the next packaging
  subplan is reviewed.
- Do not use DSGE/c603.
- Do not claim posterior, HMC, production, default, or scientific readiness.

## Exact Next-Phase Handoff Conditions

The next phase may begin only if:

- Phase 16 writes a pass/blocker result;
- any runtime evidence used trusted GPU and `jit_compile=True`;
- no `jit_compile=false` runtime command was executed;
- no runtime autodiff, HMC, or sample generation occurred;
- target and adapter signatures match the Phase 15 manual-score signatures;
- result artifact records command, environment, GPU manifest, seeds, wall time,
  output path, finite checks, and nonclaims;
- a next frozen-payload subplan is drafted and reviewed.

## Stop Conditions

Stop if:

- trusted GPU access is unavailable;
- the training helper requires `jit_compile=false`;
- the training helper requires runtime autodiff;
- target or adapter signatures change unexpectedly;
- diagnostics are nonfinite;
- TensorFlow/XLA raises a compile/runtime blocker;
- artifact JSON is malformed or too large;
- review does not converge after five rounds.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Phase 15 manual-score compile pass is the baseline; old Phase 10 is stale history only. |
| Proxy promotion | Bounded loss decrease or finite gradients do not imply posterior/HMC/readiness claims. |
| Hidden fallback | `jit_compile=false`, CPU runtime evidence, and runtime autodiff are vetoes. |
| Environment mismatch | GPU runtime evidence must be trusted/escalated per `AGENTS.md`. |
| Artifact mismatch | The result must preserve a parseable training-state JSON or exact blocker, not just console output. |

Audit status: ready for review before execution.
