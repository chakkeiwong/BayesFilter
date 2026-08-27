# Phase 35 Repair and Refresh Note

| Attempt | Failure class | Repair | Result |
|---|---|---|---|
| MathDevMCP affine expression | parser/domain limitation | preserve raw output and use explicit finite checks | `inconclusive`, non-blocking |
| affine GPU/XLA run, attempt1 | none | none | `PASS_NEUTRA_BOUNDARY_ROLE_LIMITED` hard gates |
| weighted affine oracle | none | none | `PASS_AFFINE_ORACLE` |

Record the affine covariance eigenvalues/condition, determinant composition,
transport parity, target/status gates, full trace, device policy, and remaining
budget. The weighted Cholesky condition estimate was `49.72885`, theta/chart
round-trip was `8.88e-16`, and all target/status, transport, batch, XLA, and
memory-growth gates passed. The oracle's maximum finite-bank covariance
residual was `7.77e-16`.

The learned affine traces did not retain the oracle's whitening moments by
step 200. This is a candidate training/validation limitation and a promotion
veto for whitening, not a target/measure continuation veto. Phase 36 is
refreshed to adjudicate the bounded continuation and specify a fresh
scope-specific held-out tuning plan before any claim-bearing NeuTra or HMC
work.
