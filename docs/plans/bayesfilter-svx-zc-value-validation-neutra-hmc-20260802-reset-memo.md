# SVX-ZC Value Validation, NeuTra, And HMC Reset Memo

Date: 2026-08-02

## Status

The SVX-ZC T=10 frozen-target campaign is complete. Do not rerun it by default.

Terminal result:
`docs/plans/bayesfilter-svx-zc-value-validation-neutra-hmc-terminal-result-2026-08-02.md`.

Active implementation:

- `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`
- `bayesfilter/testing/zhao_cui_actual_sv_neutra_target_tf.py`
- `bayesfilter/testing/neutra_model_registry_tf.py`
- `bayesfilter/inference/neutra_end_to_end.py`

Target signature:
`deccdda78028706d0987322d30b9798f0f4d8b518c6773451338e83bf14d1cab`.

## Terminal Evidence

- Value: degree 10, rank 2, order 25 passed value-only local validation.
- Initializer: UKF cores are built once on CPU at the center and frozen; this
  prevents device-dependent target identity and runtime retuning.
- Score: diagnostic signs and directions remained aligned; score was not a
  value-capacity veto.
- NeuTra: selected 5,000-step batch-128 GPU/XLA training passed all hard gates.
- Tuning: generic statistical broad grid produced one viable pair,
  `L=25`, `epsilon=0.8434292653387`.
- Sequential HMC: passed after 2,015 warm-up and 2,080 retained draws per chain;
  max R-hat `1.00878`, min bulk ESS `7530.28`, min tail ESS `2343.80`, no hard
  vetoes, truth-tail p-values `0.54284` and `0.49910`.

Terminal artifact:
`docs/plans/artifacts/bayesfilter-svx-zc-value-validation-neutra-hmc-20260802/sequential-hmc-attempt01/SVX-ZC/result.json`.

Provenance limitation: the serious-run manifest records base commit
`fb9a0679adb7c731ff2ac42551f39bdcc15222a1`, but the concurrent dirty worktree
was not snapshotted at launch. The target and input/result artifacts are
hash-bound; a clean-commit reproduction would still be required for stronger
release-grade provenance.

## Repairs Preserved

1. Replaced the batched QR least-squares backward path with a scaled
   regularized Cholesky solve and requested the TensorFlow score inside the tape
   context. Scalar value parity and finite-difference score tests pass.
2. Pinned one-time UKF initializer construction to CPU, making target/core
   hashes independent of prior GPU initialization.
3. Split held-out diagnostics into eight batch-native 128-row calls instead of
   one 1,024-row call, fixing an OOM without changing training.
4. Replaced the legacy exact point-estimate acceptance gate with the reviewed
   statistical L/epsilon broad grid.
5. Used 65-draw GPU/XLA screens/chunks, the smallest size above the broad-grid
   64-draw boundary, to stay within 13.5 GiB while preserving four chains,
   replication, cumulative warm-up, and convergence thresholds.
6. The generic `min_innovation_eigenvalue` status alias carries the minimum TT
   normalizer for SVX-ZC, not an innovation eigenvalue. The adapter now also
   emits `minimum_normalizer`; do not make an innovation-eigenvalue claim.

## Nonclaims And Next Lane

This result is valid for the frozen approximate SVX-ZC T=10 program only. It
does not establish exact filtering, exact score, source faithfulness,
cross-horizon/cross-data capacity, sampler superiority, or production/default
readiness.

The next action is a new target-specific plan for another unresolved model. Do
not transfer SVX-ZC degree/rank/order, NeuTra recipe, epsilon, L, transport, or
posterior summaries as defaults. They may be warm-start hypotheses only.
