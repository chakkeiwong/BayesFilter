# Phase 13 Result: Affine Whitening Oracle

Status: `PASS_AFFINE_ORACLE_ROLE_LIMITED`

The exact TensorFlow weighted affine oracle ran on the Phase 8 audited N=300
bank. The input weighted mean was
`[2.4181, -1.2864, 0.4199, 1.6776]`; the covariance was finite and positive
definite. Applying the Cholesky whitening map produced weighted mean residual
below `2.4e-16` and maximum covariance residual `1.11e-15` (maximum off-diagonal
also below `1.2e-15`).

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Retain affine map as diagnostic comparator | exact finite moment gates pass | no oracle veto | affine moments do not identify density or modes | test whether affine preconditioning helps the learned representation | no target density, posterior, IID law, HMC, or default claim |

The result is an independent numerical check: the poor learned-flow moments are
not explained by a singular weighted covariance alone. The affine map remains
diagnostic-only.
