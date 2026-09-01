# Phase 11 Result: 300-Update NeuTra Trace

Status: `PASS_HARD_GATES_ROLE_LIMITED_RESIDUALS_UNRESOLVED`

The same metadata-bound/audited N=300 bank and corrected mode axis were used for
300 batch-native GPU/XLA updates per arm. All hard device, finite, parity,
target/status, split, and untouched-audit gates passed; no HMC was launched.

| Arm | Validation loss | latent max-mean | max off-diagonal covariance | final gradient norm |
|---|---:|---:|---:|---:|
| compact | 6.3203 | 0.6251 | 0.4779 | 0.728 |
| compact_low_lr | 9.4945 | 1.6916 | 0.9626 | 10.844 |
| wider_mid_lr | 6.3959 | 0.5404 | 0.5134 | 1.020 |

The compact trace improved substantially from 100 updates and appears to be
approaching a finite-budget plateau, but the residual moments remain far from
an IID standard normal diagnostic. The arm choice is descriptive; no ranking
or transport promotion follows.

## Decision and inference tables

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Continue candidate investigation | all hard gates pass; budget-sensitive improvement observed | no hard veto | residuals may be capacity-limited or measure-limited | one high-capacity flow arm on the same audited bank | no IID whitening, posterior, mode, HMC, superiority, or default claim |

| Evidence class | Status |
|---|---|
| Hard veto screen | passed |
| Statistically supported ranking | none |
| Descriptive-only differences | trace loss, latent moments, covariance, gradients |
| Default-readiness | not ready |
| Next evidence needed | capacity arm and, if it fails, support/measure limitation diagnosis |

## Red-team note

Longer optimization can fit the finite weighted cloud without making it the
intended posterior. A high-capacity arm is a discriminating test of flow
representation, not a license to claim correctness.
