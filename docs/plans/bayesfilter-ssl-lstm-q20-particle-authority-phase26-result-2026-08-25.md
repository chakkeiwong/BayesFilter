# Phase 26 Result: Reduced-Coordinate LEDH Boundary

Status: `REDUCED_MECHANICS_PASS_TARGET_BINDING_VETO`

The command completed and wrote:

`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase26-attempt1/`

The reduced map `x = L u`, with `L` the Cholesky factor of the actual q=20
innovation covariance, passed all mechanics checks. The minimum covariance
eigenvalue was `1.0112162388`, the inverse round-trip residual was
`5.55e-17`, and the change-of-variables density residual was `0`. Target
values, scores, and statuses were finite.

The result does not bind to the particle authority. The reduced coordinates
have dimension `20`; the declared particle target has dimension `4` and is an
aggregate UKF marginal value/score. No target-to-innovation proposal callback,
common measure identity, or parameter-space LEDH determinant was available.

## Terminal decision

| Ledger | Finding | Decision |
|---|---|---|
| Engineering | source/hash/device/finite checks pass | mechanics artifact valid |
| Numerical | inverse and density identities pass | reduced fixture valid |
| Scientific | target-binding veto | direct q=20 LEDH closed relative to stated target |
| Wider direction | ETPF/SMC/GenUT/NeuTra arms retain their role-limited evidence | do not generalize LEDH failure to all methods |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | reduced mechanics pass; target-binding veto fires |
| Statistically supported ranking | not applicable |
| Descriptive-only differences | covariance eigenvalues and residuals |
| Default-readiness | no LEDH default or HMC admission |
| Next evidence needed | a separately reviewed target/measure change, if LEDH is revisited |

## Real-blocker determination

This is a real blocker for the requested direct q=20 LEDH arm, not a harness
failure: the computed density is on a different measure and dimension from
the declared parameter target. Continuing would require changing the target
or claiming an unproved measure transformation, both outside this program's
scope. The broader particle-authority and modular-method results remain
available for a future decision or separate campaign.

Strongest overturning evidence would be a new, hash-bound q=20 target whose
proposal, transition/observation terms, support, and determinant all live in
the same four-dimensional parameter measure. That evidence was not present in
this campaign.
