# Phase 8 Result: Rung 1 Preparation And Telemetry

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status:
`PREPARATION_AND_TELEMETRY_INFRASTRUCTURE_CLOSED_TARGET_RIDGE_AND_FORMAL_FD_BLOCKED`

## Outcome

The repository now has an owned TensorFlow preparation module for canonical
LGSSM target-prefix work. It freezes PHILOX algorithm/key/domain/axis/dtype
semantics, constructs the Phase 1 centered population-scaled residual design,
requires all reset/transport/ridge inputs explicitly, and hashes every realized
tensor. It contains no ridge generator or selection policy.

The canonical primal now returns the pre-mask, per-time quotient and Contract E
telemetry required to interpret future rungs. These outputs reuse exact reset
core tensors and do not recompute alternative moments. Fresh certificates prove
that adding telemetry did not alter the frozen eager derivative or CPU-XLA
objective/score/branch identities.

## Evidence

- Plan review converged after removing an unjustified ridge ladder, freezing
  PHILOX identity, defining telemetry formulas, and binding moments to exact
  reset-core population semantics.
- Preparation plus canonical union: `19 passed, 2 warnings in 102.86s`.
- Fresh exact derivative: zero ULP for per-batch and aggregate score.
- Fresh float64 CPU-XLA: objective, all five score hex values, and branch hash
  reproduce Phase 5 v2 exactly.
- Fresh float32 CPU-XLA: finite, repeatable, branch-identical, and one concrete
  value-and-score callable.
- Python compilation and `git diff --check`: passed.

The closeout manifest and focused-check record are stored under
`docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase8/rung1-preparation-telemetry/`.
They bind the exact commands, CPU-hidden environment, source/artifact hashes,
and the result's nonclaims.

Telemetry includes full quotient mass, row residual, target/output/injected
moments, affine, ridged identity residual/absolute scale, raw covariance
residual/prediction/error, mean residual, residual-design sums/scales, all three
Cholesky diagonals and condition proxies, realized ridge, and active mask.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept preparation builder | exact repeatability, domain separation, centering, explicit input validation and hashes | Passed | Target hypothesis choices remain open | Use only under a reviewed execution plan | Target scientific validity |
| Accept telemetry | exact formulas and unchanged scalar/score certificates | Passed | No adequacy thresholds | Run a labeled harness smoke | Numerical adequacy |
| Select target ridge/default | Forbidden in this phase | Blocked | covariance error and raw-bias budget | Separate pre-result decision plan | Ridge readiness |

## Inference Status

| Inference | Status |
| --- | --- |
| Hard veto screen | Preparation identity, telemetry, exact derivative, XLA preservation pass |
| Statistically supported ranking | None |
| Descriptive-only differences | None from target filter output; no target run occurred |
| Default-readiness | Not established |
| Next evidence needed | Target-prefix harness smoke, then reviewed ridge/transport/reset and formal FD decisions |

## Close Record

The preparation/telemetry subplan is closed because all infrastructure pass
criteria and hard-veto checks passed. The next phase is permitted to execute
only the separately reviewed `T=1,N=4` wiring smoke. No target numerical
setting, Kalman-equivalence margin, formal FD bound, or primary-shape gate is
closed by this record.

## Nonclaims

This result does not establish ridge, reset, Sinkhorn, chunk, or residual
adequacy; target-prefix chart validity; Kalman equivalence; formal FD
certification; GPU feasibility; HMC, admission, leaderboard, release, or
integrity readiness.
