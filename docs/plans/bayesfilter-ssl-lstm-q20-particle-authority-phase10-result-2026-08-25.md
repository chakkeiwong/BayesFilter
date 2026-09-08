# Phase 10 Result: NeuTra Tuning Ladder

Status: `PASS_HARD_GATES_ROLE_LIMITED_TUNING_INCOMPLETE`

The three-arm tuning ladder used the same metadata-bound, measure-audited N=300
bank and corrected mode axis as Phase 9. Each arm ran 100 batch-native GPU/XLA
updates with a frozen 180/60/60 split and untouched audit rows.

## Hard-gate evidence

All arms passed GPU memory-growth/device provenance, finite values, batch-native
updates, forward/inverse parity, and transformed target/status checks. No HMC
was launched. The compact arm was descriptively selected by validation loss;
this is not a statistical ranking.

| Arm | Configuration | Validation loss | latent max-mean | max off-diagonal covariance | status |
|---|---|---:|---:|---:|---|
| compact | 16/16, lr 1e-3 | 7.9611 | 0.9922 | 0.6189 | `PASS_CANDIDATE` |
| compact_low_lr | 16/16, lr 2e-4 | 19.3446 | 3.4423 | 1.9422 | `PASS_CANDIDATE` |
| wider_mid_lr | 64/32, lr 5e-4 | 8.6844 | 1.1218 | 0.8366 | `PASS_CANDIDATE` |

The compact arm's diagnostics improved monotonically across the 100-step
trace, so the short Phase 9 failure had a substantial optimization component.
Even the selected arm remains visibly non-IID by the declared diagnostics;
those diagnostics remain explanatory and cannot promote NeuTra.

## Decision and inference tables

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Continue NeuTra candidate investigation | all hard gates pass and compact improves with budget | no hard veto | convergence at longer budget versus missing measure geometry | 300-update convergence check on same audited bank | no IID whitening, posterior, HMC, superiority, or default claim |

| Evidence class | Status |
|---|---|
| Hard veto screen | passed for all three arms |
| Statistically supported ranking | none; one bank and descriptive validation selection |
| Descriptive-only differences | loss, latent moments, covariance, clipping, runtime |
| Default-readiness | not ready |
| Next evidence needed | longer trace and, if residuals plateau, a representation/support diagnosis |

## Red-team note

The improvement with 100 updates could be optimizer progress rather than a
transport that represents the intended posterior. Conversely, a plateau at a
longer budget could reflect the finite weighted cloud rather than model
capacity. The next 300-update trace is the smallest discriminating artifact.
