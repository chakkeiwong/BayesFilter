# P4 PP-UKF Target-Specific Training Result

Date: 2026-07-16

Status: `TRAINING_ADMITTED`

`PP-UKF` completed a fresh 5,000-step `wide_lr5e3` dense-IAF training run
under the target-specific affine base and disjoint final seed
`(20260716,10201)`.

## Evidence

- Typed target signature:
  `036948f0faaf028d159d7b70337214f01514d732112c2d10e9f7eea1e13b8e30`.
- Result:
  `docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p4/PP-UKF/training/final/wide_lr5e3/attempt-01/result.json`.
- Result SHA-256:
  `1650d256577f91d54e6c351545e9a7ef0cb208844dc859f19eecc3b496af27c9`.
- Frozen transport SHA-256:
  `86fbb2d30b217f889c95b0228f4192d82527edbbddcb718d7c0eb7ae26b25cbd`.
- Frozen transport semantic hash:
  `18546c2b30a5e2236e001293f9bbfc71babed47f5592d6821cabe0972990beec`.
- Wall time: `470.23` seconds.
- GPU/XLA: RTX 4080 SUPER; memory growth configured before logical-device
  initialization; compiled outputs, trainable variables, and optimizer moments
  recorded on GPU; `jit_compile=true`.
- All recursive hashes match. All 5,000-step records are finite and target
  status valid. Frozen/trainable transport, log determinant, pullback score,
  and log-determinant score parity gaps are all zero.
- Common-heldout target status is valid. Mean reverse-KL `116.8192` remains a
  proxy diagnostic only.

## Decision

The engineering training gate passed. The cell advances to
`TRAINING_ADMITTED`, not `NEUTRA_CONFIRMED`. Fresh transported-kernel tuning,
adaptive warm-up, retained convergence/ESS, and simultaneous same-target
physical-mean agreement remain required in R4.

No claim is made about full-distribution equivalence, filter exactness,
superiority, calibration, robustness, or readiness.
