# Phase 7 Result: N=300 NeuTra Retuning Screen

Status: `PASS_HARD_GATES_ROLE_LIMITED_MODE_SPLIT_SUPERSEDED`

Phase 7 ran a batch-native NeuTra screen on three independent N=300
M0 banks. The runner now derives a 60/20/20 split from the bank and gathers
weights using the same row indices as the selected particles. A subsequent
audit found that those runs stratified rows by the last coordinate rather than
the pilot's declared signed coordinate `theta[:, 2]`; the transport/status
receipts remain valid, but the mode-stratified split interpretation is
superseded. The runner is repaired to use `MODE_AXIS = 2`, and fresh runs are
required after the upstream Phase 6 receipt repair. No acceptance metric from
the older bank artifacts is used as a promotion criterion.

## Receipts

Each attempt used GPU/XLA with TensorFlow memory growth verified before logical
device creation on two NVIDIA GeForce RTX 4080 SUPER devices. The batch size
was 180, with 60 validation and 60 untouched audit rows. Both architecture
arms (`compact`, `wide_low_lr`) completed 20 batch-native updates, finite
forward/inverse transport parity, and finite target value/score/status on all
untouched audit rows.

| bank | selected arm | validation loss | latent max | max off-diagonal covariance | round-trip residual | logdet residual |
|---|---|---:|---:|---:|---:|---:|
| seed 1701 | compact | 18.1832 | 0.5446 | 1.1234 | 8.88e-16 | 1.39e-17 |
| seed 1801 | compact | 16.8566 | 1.4673 | 1.0693 | 8.88e-16 | 1.39e-17 |
| seed 1901 | compact | 12.7963 | 0.9307 | 1.2009 | 6.66e-16 | 0 |

Across the three banks, selected-arm latent max-mean had descriptive mean
`0.9809`, sample MCSE `0.2675`, and range `0.5446--1.4673`; the off-diagonal
covariance maximum had mean `1.1312`, MCSE `0.0382`, and range
`1.0693--1.2009`. These values are far from an IID standard-normal diagnostic,
but they are not a correctness or promotion criterion.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Retain screen as role-limited candidate | all GPU, batch, split, parity, and audit gates pass | no Phase 7 hard veto | normalized empirical measure; short target-specific tuning; acceptance receipt repair in upstream phase | complete Phase 6 receipt repair, then audit raw measure/ledger before more tuning | no whitening, posterior, mode, HMC, or superiority claim |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | passed for all three banks |
| Statistically supported ranking | none; compact was selected descriptively within each bank only |
| Descriptive-only differences | loss, latent moments, covariance, clipping, runtime |
| Default-readiness | not ready |
| Next evidence needed | corrected upstream mutation receipts, raw proposal/target measure audit, and a reviewed tuning ladder if NeuTra remains viable |

## Red-team note

The strongest alternative explanation is that the target-specific objective is
being optimized against a normalized empirical cloud whose support and mode
weights are not the posterior measure. The corrected row/weight alignment
removes one harness error but cannot fix that measure mismatch. Evidence that
would overturn this result is a hard parity/status failure, a reproducible
weight/hash mismatch, or a raw-measure audit showing that the claimed training
measure is not the declared proposal/target measure.

The three output directories are under
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/`.
HMC was not launched.
