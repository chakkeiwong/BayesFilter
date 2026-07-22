# P4 Result: PP-UKF Same-Target Plain HMC

Date: 2026-07-16

Status: `COMPARATOR_ADMITTED`

## Decision

The exact admitted `PP-UKF` filter posterior now has a same-target plain-HMC
comparator. The identity-mass baseline failed the warm-up convergence gate, but
a reviewed target-bound affine-mass repair passed the unchanged warm-up,
retained, health, status, and modern diagnostic gates.

## Attempt History

| Attempt | Classification | Result |
| --- | --- | --- |
| identity mass attempt 01 | supervisor interruption, infrastructure only | preserved 1,000-draw partial warm-up; no scientific verdict |
| identity mass attempt 02 | completed healthy kernel, geometry failure | warm-up cap hit at 10,000 per chain; no retained samples; `COMPARATOR_BLOCKED` for this geometry |
| affine mass attempt 01 | completed reviewed repair | `COMPARATOR_ADMITTED` |

Identity-mass attempt 02 selected step `0.032` with eight leapfrog steps. All
ten chunks were finite and health-valid, acceptance was approximately
`0.993-0.996`, but recent-window rank R-hat ranged from `1.1505` to `1.7478`
after checks began and ended at `1.1894`; folded R-hat ended at `1.0604`.
This is sampler geometry/nonconvergence, not a target failure.

The tuning-only within-chain covariance from the failed warm-up had condition
number `117.8`. The affine repair used the exact map
`theta = center + z @ factor.T`, where `factor` is the regularized Cholesky
factor of the target-bound within-chain covariance. Failed warm-up was used
only to construct this mass artifact and was never pooled with fresh warm-up or
retained inference.

## Admitted Evidence

- Artifact root:
  `docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p4/PP-UKF/plain-hmc-affine/attempt-01-20260715T152500Z`
- Result SHA-256:
  `4c7e001b181033f4191acf5a6dd841c2dc507c4b25c015ce69817976eec345d5`
- Typed target signature:
  `036948f0faaf028d159d7b70337214f01514d732112c2d10e9f7eea1e13b8e30`
- Selected fixed kernel: step `0.30`, eight leapfrog steps.
- Probe minimum bulk ESS: `1024.34`; acceptance `0.97656`.
- Fresh warm-up: `2,000` draws per chain, four chains.
- Warm-up max rank R-hat: `0.99922`.
- Warm-up max folded R-hat: `1.01249`.
- Fresh retained: `4,000` draws per chain, four chains.
- Retained max modern R-hat: `1.00205`.
- Retained minimum bulk ESS: `28,976.14`.
- Retained minimum tail ESS: `8,013.68`.
- Hard vetoes: none.
- Trusted GPU/XLA wall time: `4,356.09` seconds.
- Artifact hash ledger: 26 files verified.
- Focused post-run regression: 15 tests passed.
- Bounded Claude review of the affine repair subplan: `VERDICT: AGREE`.

## Decision Table

| Decision field | Status |
| --- | --- |
| Primary comparator criterion | passed |
| Target/status/energy vetoes | clear |
| Main uncertainty | one synthetic trajectory and one tuned comparator run |
| Next justified action | run independent PP-SGQF comparator; then target-specific PP-UKF training protocol |
| Not concluded | NeuTra quality, UKF exactness, calibration, superiority, robustness, or readiness |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | passed for admitted affine comparator |
| Statistically supported ranking | none; no sampler/filter ranking was tested |
| Descriptive-only differences | identity versus affine runtime, acceptance, and probe diagnostics |
| Default readiness | not established |
| Next evidence needed | target-specific dense-IAF training and same-target NeuTra agreement |

## Post-Run Red Team

The strongest alternative explanation is that the mass artifact is especially
well matched to this one synthetic trajectory because it used a long failed
warm-up from the same target. That is permitted tuning, but it does not prove
robust geometry on another dataset or seed. A fresh target or replicated
campaign with poor diagnostics would overturn any broader robustness claim.
The weakest evidence is therefore cross-fixture generality, not same-target
correctness for this comparator.

