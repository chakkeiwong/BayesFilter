# Phase 4 Result

Status: `PASS_HARD_GATES_ROLE_LIMITED_TUNING_UNRESOLVED`

The Phase 4 GPU/XLA screen used the N=100 identity-M0 candidate bank in
`phase4-attempt6`, followed by the predeclared 20-update same-scope repair in
`phase4-attempt7`. A separate propagation screen used the repaired random-walk
bank in `phase4-mutation-revalidation-attempt1`.

## Hard-gate evidence

- Trusted device 0 was an NVIDIA GeForce RTX 4080 SUPER. TensorFlow 2.20.0
  configured memory growth before logical-device initialization; XLA compiled
  the training functions; dtype was float64; batch size was 60; and no HMC was
  launched.
- Both architecture arms preserved a true leading batch dimension through the
  transport, loss, gradient, optimizer update, and target/status calls.
- Forward/inverse round-trip residuals were at most `8.88e-16` in the first
  screen and `4.44e-16` in the longer repair. Log-determinant residuals were at
  most `1.73e-18`.
- Untouched audit rows had finite target values/scores and valid target status;
  transformed target values, scores, and log densities were finite as well.

## Explanatory training evidence

The compact arm was descriptively selected by validation loss in both screens;
this is not a statistical ranking. In the three-update screen its validation
latent `max |mean|` was `1.3370` and largest off-diagonal covariance was
`4.0324`. After 20 updates these were `1.2440` and `3.7309`, respectively.
Gradients were clipped on most updates. The values remain far from an IID
standard normal diagnostic and do not certify transport quality.

The mutation-bank propagation screen also passed hard gates but had compact-arm
diagnostics `3.2681` (latent mean) and `2.3844` (off-diagonal covariance) after
three updates. It cannot be called an improvement.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
|---|---|---|---|---|---|
| Keep NeuTra screen candidate-eligible | GPU/XLA, batch, parity, and target-status gates pass | whitening/tuning is unresolved but explanatory | normalized empirical weights, one seed, short budget | fresh target-specific tuning and unnormalized SMC-U bank before HMC | no IID whitening, posterior, HMC, predictive, or default claim |

The complete manifests and raw receipts are under
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/`.
