# P4 PP-SGQF Target-Specific Training Result

Date: 2026-07-16

Status: `TRAINING_ADMITTED`

`PP-SGQF` completed a fresh 5,000-step `wide_lr5e3` dense-IAF training run
under its target-specific level-2 Laplace base and disjoint final seed
`(20260716,11201)`.

## Evidence

- Typed target signature:
  `8e0a9582fd30643b2e77e7615a21c0d44cc6c1827865ea52c841cc6dbfdde1ad`.
- Result:
  `docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p4/PP-SGQF/training/final/wide_lr5e3/attempt-01/result.json`.
- Result SHA-256:
  `de5f7cc35f606fe6d07177d1059d24acc1187e80b4bda42963f9e2823bf64bd4`.
- Frozen transport SHA-256:
  `4f7f94321b73ed5d1a2ed100fb8382265486f0608a97f5dacdb38665ddb9c8ba`.
- Frozen transport semantic hash:
  `603a07c420579788e3981aa44dd67892902dc8c32da6ddf7c171918300da6811`.
- Wall time: `266.50` seconds.
- GPU/XLA: RTX 4080 SUPER; memory growth configured before logical-device
  initialization; compiled outputs, trainable variables, and optimizer moments
  recorded on GPU; `jit_compile=true`.
- All recursive hashes match. All 5,000-step records are finite and target
  status valid. Frozen/trainable transport, log determinant, pullback score,
  and log-determinant score parity gaps are all zero.
- Common-heldout target status is valid. Mean reverse-KL `116.9469` remains a
  proxy diagnostic only.

## Decision

The engineering training gate passed. The cell advances to
`TRAINING_ADMITTED`, not `NEUTRA_CONFIRMED`. Fresh transported-kernel tuning,
adaptive warm-up, retained convergence/ESS, and simultaneous same-target
physical-mean agreement remain required in R4.

No claim is made about full-distribution equivalence, filter exactness,
superiority, calibration, robustness, or readiness.
