# Phase 12 Result: High-Capacity NeuTra Arm

Status: `PASS_HARD_GATES_ROLE_LIMITED_CAPACITY_NOT_RESOLVING_RESIDUALS`

The high-capacity profile used the same metadata-bound/audited N=300 bank,
corrected mode axis, split, target, and 300-update budget as Phase 11. Both
arms passed GPU memory-growth, XLA/batch, finite, parity, target/status, and
untouched-audit gates. No HMC was launched.

| Arm | Validation loss | latent max-mean | max off-diagonal covariance | status |
|---|---:|---:|---:|---|
| compact | 6.2768 | 0.6614 | 0.5038 | `PASS_CANDIDATE` |
| high_capacity (128/64/32, 3 stages) | 4.7291 | 1.2591 | 0.4952 | `PASS_CANDIDATE` |

The high-capacity arm has lower validation loss but does not improve the
moment diagnostic; arm selection is therefore descriptive and no ranking is
supported. The result weakens a simple capacity-only explanation for the
remaining non-IID residuals, while leaving optimization and finite-measure
effects unresolved.

## Decision and inference tables

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Do not promote a learned transport | hard gates pass but IID diagnostic remains poor | no engineering veto | objective/measure versus optimization | run an exact affine weighted-whitening oracle on the same bank | no posterior, HMC, mode, or default claim |

| Evidence class | Status |
|---|---|
| Hard veto screen | passed |
| Statistically supported ranking | none |
| Descriptive-only differences | loss, latent moments, covariance, gradients |
| Default-readiness | not ready |
| Next evidence needed | affine moment oracle and density/measure comparison |

## Red-team note

A lower forward-KL loss is not equivalent to Gaussian moments or target
correctness. The high-capacity arm can fit the finite weighted objective while
retaining mode/measure distortions. An affine oracle provides the smallest
independent test of whether the weighted cloud's first two moments themselves
are well-conditioned.
