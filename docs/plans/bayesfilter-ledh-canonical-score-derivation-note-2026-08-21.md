# Canonical LEDH Analytical Recursive Score: Derivation Note (P4)

Date: 2026-08-21. Status: STAGE LEDGER — each stage closes only with a
derivation and a passing oracle-parity gate. Owner fallback #4 governs:
blocked stages ship value-only; no autodiff ever ships.

## Score identity (target)

d/dtheta log p(z_{1:T} | theta) = sum_t E_{w_t}[ s_t ] where s_t are
per-particle score marks accumulated through the recursion. Each pipeline
stage contributes its analytical parameter derivative to the marks.

## Stage ledger

| Stage | Derivative required | Derivation status | Oracle gate |
|---|---|---|---|
| S1 UKF predict | d(predicted mean, P^i)/dtheta through sigma points (chain rule over Cholesky + dynamics Jacobian; Cholesky differential via Phi operator, existing `_cholesky_jvp` pattern) | OPEN | pending |
| S2 Anchor + pre-flow | d(anchor)/dtheta = transition-mean parameter Jacobian (EXISTS per model: `_transition_mean_and_parameter_tangent` for Austria; analytic for LGSSM) | PARTIAL (per-model parts exist) | pending |
| S3 Flow map | d(A^i, b^i)/dtheta through P^i, H_i, R(theta); substep chain product d(total affine)/dtheta; d(log det)/dtheta = sum_j eps * tr[(I+eps A_j)^{-1} dA_j] | OPEN | pending |
| S4 PF-PF weight | d(log w) = d(transition log density at post-flow) + d(observation log density) + d(log-det) - d(proposal log density); first and last need S2/S3 outputs | OPEN (mechanical given S3) | pending |
| S5 UKF update | as S1 with observation stage | OPEN | pending |
| S6 OT reset | Sinkhorn fixed-point implicit derivative OR unrolled-iteration analytical recursion (repo precedent: the reset JVP `_restore_cloud_batch_jvp` is hand-analytical, reusable pattern) | PARTIAL (pattern exists) | pending |
| S7 Dual-cap correction | analytical JVPs EXIST (`higher_moment_shape_jvp` carries hand-derived tangents for every sub-stage) | EXISTS | pending wiring |
| S8 Likelihood accumulation | weighted-mark recursion (all-parent backward marks pattern EXISTS in `ledh_pfpf_genut_initial_rqmc_tf.py`) | EXISTS (pattern) | pending |

## Honest assessment (recorded at P4 open)

S3 (flow-map parameter derivative with the substep chain product) is the
genuinely novel derivation; S1/S5 are careful but standard unscented
differentials; S4/S8 are mechanical once S3 lands; S6/S7 reuse existing
hand-derived tangent machinery. Estimated as the dominant remaining
research cost of the rebuild. Per fallback #4 the value path (P3, closed)
is shippable independently.

## Oracle discipline

`ledh_canonical_autodiff_oracle_tf.py` provides the forward-autodiff judge.
It is namespaced `*_oracle_*`; conformance C-9 statically forbids autodiff
symbols outside oracle namespaces. Every stage derivation must match the
oracle at rtol 1e-4 (float64 fixtures) before entering the score assembly.
