# Phase 9 Result: Corrected Mode-Axis NeuTra Screen

Status: `PASS_HARD_GATES_ROLE_LIMITED_WHITENING_UNRESOLVED`

The screen consumed the metadata-bound, independently audited N=300 M0 bank
from Phase 8. It used the corrected `theta[:, 2]` mode axis, index-aligned
weights, a deterministic 180/60/60 train/validation/audit split, and two
batch-native GPU/XLA architecture arms for 20 updates.

## Hard gates

Both arms passed finite transport values, forward/inverse parity, transformed
target value/score/status checks on all 60 untouched audit rows, and the GPU
memory-growth/XLA/device-provenance policy. Round-trip residuals were at most
`4.44e-16`; logdet residuals were at most `1.39e-17`. No HMC was launched.

## Explanatory diagnostics

| Arm | Validation loss | latent max-mean | max off-diagonal covariance | status |
|---|---:|---:|---:|---|
| compact | 20.2090 | 3.5549 | 2.0046 | `PASS_CANDIDATE` |
| wide_low_lr | 21.3424 | 3.6831 | 2.1162 | `PASS_CANDIDATE` |

The compact arm was selected by validation loss within this single bank; that
selection is descriptive and not a statistical ranking. The latent diagnostics
are far from IID standard normal. They therefore trigger a target-specific
tuning/representation repair and cannot support NeuTra or HMC admission.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Keep corrected screen viable as a role-limited candidate | all hard GPU, batch, parity, target/status, split, and audit gates pass | whitening remains unresolved; this is a repair trigger | 20-step budget, architecture/capacity, empirical measure | run a predeclared longer target-specific tuning ladder | no IID whitening, posterior, mode, HMC, predictive, superiority, or default claim |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | passed |
| Statistically supported ranking | none; one bank and two short arms |
| Descriptive-only differences | validation loss, latent moments, covariance, clipping |
| Default-readiness | not ready |
| Next evidence needed | longer/disjoint target-specific tuning and a downstream measure check on the selected transport |

## Red-team note

The strongest alternative is not simply insufficient optimization: the weighted
empirical measure may be missing geometry or modes, so no transport trained on
it can whiten the intended posterior. A longer ladder can distinguish tuning
failure from representation/input failure. If every predeclared tuning arm
passes hard gates but remains far from the diagnostic target, that is evidence
against this NeuTra training candidate, not a theorem that the particle
authority or the broader research direction is impossible.

Artifact:
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase9-attempt1-bank2401`.
