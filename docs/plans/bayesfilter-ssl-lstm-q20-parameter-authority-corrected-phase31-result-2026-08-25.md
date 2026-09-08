# Corrected Parameter-Authority Phase 31 Result

Date: 2026-08-25  
Status: `PASS_NEUTRA_BOUNDARY_ROLE_LIMITED`

## Receipt

`docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase31-neutra-boundary/`

The corrected M0 bank was split into 40 training, 12 validation, and 12
sign-balanced audit theta rows. The screen ran in the trusted GPU/XLA lane;
memory growth was verified before logical-device initialization on both visible
GPUs.

| Arm | Hard screen | Initial loss | Final loss | Batch | Target audit |
|---|---|---:|---:|---:|---|
| compact | passed | `21.3443` | `21.1888` | 40 | valid |
| wide_low_lr | passed | `21.3443` | `21.2645` | 40 | valid |

Both arms had finite gradients, exact transport round-trip residual below
`5e-16`, finite transformed target/score, and valid target status. The screen
did not launch HMC.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Retain both theta-space NeuTra arms as viable boundary candidates | GPU policy, batch/XLA, finite gradient, transport parity, and target/status gates pass | no hard veto | one seed, three updates, fixed normalized empirical bank; whitening unresolved | close this corrected continuation with explicit non-promotion and require a fresh tuned/replicated plan for any training claim | no IID whitening, posterior correctness, authority, HMC convergence, mode theorem, ranking, or default |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | passed for both arms |
| Statistically supported ranking | none; descriptive loss differences only |
| Descriptive-only differences | loss trajectory, gradient norm, latent moments, runtime |
| Default-readiness | not ready |
| Next evidence needed | target-specific multi-seed training/tuning and downstream posterior gates under a new reviewed plan |

## Red-team note

The screen can pass while the fixed M0 bank is mode-biased or while the learned
transport fails to whiten an unseen target draw. The strongest alternative
explanation is therefore empirical-bank conditioning, not a successful NeuTra
representation. A longer, target-specific, disjoint tuning/claim campaign is
needed before any HMC-facing use.

