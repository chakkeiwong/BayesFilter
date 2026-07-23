# Zhao-Cui Rung-2 Reset Memo

Date: 2026-07-22

The rung-2 lane moved through three bounded candidate versions under one
scientific contract:

1. v1 used an algebraic map and random TT cores. It failed because the
   defensive reference admitted extreme physical tails and the 4D prefit had
   vanishing products. It is preserved as negative evidence.
2. v2 used a Gaussian-quantile map and deterministic TT-SVD initialization.
   Proposal and conditional-density gates passed, but a uniform fitted
   genealogy failed the d=24 ESS screen (`0.1166`).
3. v3 retained v2 proposal fitting and added a frozen reference-parameter
   weight-aware predictive auxiliary genealogy. CPU and trusted GPU/XLA runs
   passed all declared screens; the GPU artifact is the terminal engineering
   evidence for this scope.

The terminal status is `PASS_ENGINEERING_RUNG2`. The GPU runner set
`TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow import, configured and
verified growth before logical-device initialization, and released its shared
GPU allocation after the run. Memory growth remains mandatory for every future
GPU scope in this lane and is not a hard memory cap.

The route remains `extension_or_invention`. The next agent must not transfer
the selected degree, rank, scale, L1 value, predictive auxiliary law, or random
seed to another model, horizon, particle count, dtype, or block structure
without a new tuning scope. The next discriminating run is multi-seed and
longer-horizon validation with the same memory-growth policy. Before any HMC
claim, define an extended-state or refreshable pseudo-marginal contract; the
current permanently frozen branch only defines a deterministic approximate
posterior target.
