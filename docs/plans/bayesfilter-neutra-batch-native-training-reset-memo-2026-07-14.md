# NeuTra Batch-Native Training Reset Memo

Date: 2026-07-14

## Current State

The batch-native training and DSGE knowledge-transfer master program is
engineering-complete through Phase 8. BayesFilter now has a repository-issued
batch target binding, a TensorFlow/XLA dense-IAF trainer, an exact T=120 LGSSM
batch materializer and SVD/eigh value/score/status kernel, trusted GPU evidence,
and a fresh target-specific screen.

The active generic API is available lazily from `bayesfilter.inference` and the
top-level `bayesfilter` package:

- `NeuTraBatchTargetBinding`;
- `bind_batch_native_neutra_target` and
  `require_batch_native_neutra_target`;
- `batch_native_value_status_target_fn`;
- `PlainDenseIAFTrainingConfig`, `PlainDenseIAFTransport`, and
  `train_plain_dense_iaf`.

The exact LGSSM adapter training method is
`neutra_batch_log_prob_and_grad_status(theta: [B,18])`. Its recorded dependency
closure is
`6d5a05a65a15b5fb4378fc08547d5dfd22dc83705d31e7fd662b142df04732b5`.
The scalar `log_prob_and_grad_status` and `target_status_telemetry` methods still
contain row-mapped diagnostics for parity/HMC/status use. They are not in the
bound training dependency closure and remain ineligible for optimizer updates.

## What Was Transferred

The useful DSGE knowledge was topology, not copied defaults:

- leading batch-axis TensorFlow target evaluation;
- reviewed analytical score injection with `tf.custom_gradient`;
- stateless in-graph reverse-KL noise;
- one compiled multi-step optimizer program;
- strict target identity, status telemetry, frozen reload parity, and artifact
  provenance;
- persistent multicore CPU shard evaluation retained only as an alternative
  topology, not the selected LGSSM default.

Batch 480, 96 workers, DSGE model settings, Cholesky/QR/sigma-point target math,
and DSGE timing or scientific claims were not transferred as LGSSM defaults.

## Retired Routes

These entry points are historical migration evidence and fail before optimizer,
GPU, or artifact side effects:

- `bayesfilter.testing.lgssm_neutra_training_tf::train_and_validate_lgssm_affine_neutra`;
- `bayesfilter.testing.neutra_gpu_bounded_training_tf::run_neutra_gpu_bounded_training`.

Do not revive them. Their module headers point callers to the admitted generic
trainer.

## Certified Evidence

- Phase 5 certified value/score/status parity, invalid-row isolation,
  objective-gradient finite difference, deterministic identical-seed optimizer
  state, binding closure, and policy checks.
- Phase 6 found approximately `13.5x` descriptive speedup versus historical row
  mapping. Sequential target `B=128` was `0.1360 s`; the five-step compiled
  training program averaged `0.7431 s/step` including compile.
- Phase 7: all four recipes passed fresh 500-step screens. The deterministic
  proxy nominee is `wide_2x_lr5e3`; all four remain viable and no statistical
  ranking is supported.
- The accepted Phase 7 selection artifact is
  `docs/plans/artifacts/neutra-batch-native-training-2026-07-14/phase7/screen-500/selected_recipe.json`.

The first finalizer execution is preserved and rejected because it indirectly
imported diagnostic NumPy. Accepted attempt 02 uses standard-library-only
selection/admission logic and binds its finalizer source hash.

## Exact Next Step

Use
`docs/plans/bayesfilter-neutra-batch-native-training-fresh-5000-step-handoff-2026-07-14.md`.
It specifies two sequential fresh 5,000-step GPU/XLA jobs, explicit selected-
recipe identity, a 45-minute compiled / 60-minute wall campaign budget, repair
rules, and stop conditions.

Do not run final training without `--selected-recipe`. The harness rejects
implicit recipe lookup before GPU initialization and records the selected
artifact/result hash chain in every successful final training result.

The 2026-07-14 TensorFlow GPU memory policy is also active for the next run.
The strict harness sets the allow-growth environment fallback before TensorFlow
import, applies and verifies memory growth on every physical GPU before logical
device initialization, fails closed on configuration failure, and records the
policy in the GPU manifest. This prevents eager whole-device reservation; it
does not impose a hard memory ceiling. Use a reviewed logical-device memory
limit when a hard sharing boundary is required.

## Remaining Scientific Work

Engineering training readiness is established. The following are not:

- long-training stability across the two fresh seeds;
- frozen transport posterior agreement or calibration;
- tuned NeuTra-HMC convergence using rank-normalized split R-hat;
- comparison with the tuned plain-HMC baseline under uncertainty;
- cross-model generalization or scientific/default readiness.

Training loss and the 500-step heldout proxy cannot close those claims.
