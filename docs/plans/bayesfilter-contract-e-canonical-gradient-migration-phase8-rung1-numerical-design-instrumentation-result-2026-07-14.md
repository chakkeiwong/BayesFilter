# Phase 8 Result: Target Numerical-Design Instrumentation

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status: `NUMERICAL_DESIGN_INSTRUMENTATION_PASSED_TARGET_NUMERICAL_REQUIREMENTS_BLOCKED`

## Outcome

The instrumentation-only slice now exposes the pre-ridge reset mean/covariance
and symmetric gap eigenvalues, plus full final-coupling row and column masses,
targets, signed residuals, maximum absolute residuals, and fixed reporting
scales. The streaming reporter retains O(N) vectors plus fixed row/column
blocks and does not materialize an N-by-N production coupling.

No new target-prefix Contract E value or score was evaluated. The completed
transferred smoke remains explanatory only.

## Evidence

- instrumentation/compatibility bundle: `19 passed, 2 warnings in 53.82s`;
- dense tiny row/column mass and signed residual definitions agree with the
  streaming reporter within the predeclared tiny floating-point check;
- gap telemetry equals `target_cov - plus_cov`; eigenvalue and trace checks pass;
- production source/AST audit passes with retained state
  `O(B*N*d+B*N+B*row_chunk*col_chunk)` and dense reference restricted to tests;
- fresh float64 aggregate/per-batch manual JVP agrees with forward autodiff at
  zero ULP on the frozen fixture;
- fresh float64 CPU-XLA reproduces the frozen Phase 5 v2 objective/score hex,
  branch hash, center replay, endpoint charts, and one-callable identity.

The first focused attempt exposed only three local test/reporting defects: an
overstrong exact comparison after `exp(log(weight))`, an overstrong exact
comparison across a repeated eigensolver call, and a report-only `tf.maximum`
that violated an inherited no-hidden-floor source guard. They were repaired
without changing the canonical transport, quotient, reset, derivative, or
target configuration; the unchanged bundle then passed.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept instrumentation | definitions, dense tiny parity, scalar preservation, O(N) audit | Passed | target numerical requirements | Draft owner-dependent amendment | Numerical adequacy |
| Select ridge/transport settings | Forbidden in this slice | Blocked | parameter domain, raw-bias budget, convergence rule | Require reviewed pre-result amendment | Candidate/default |
| Advance to target arm | Requires owner amendment; the formal FD certificate is separately retired as unsupported | Blocked | owner scientific/error budgets | Ask owner to freeze requirements or stop | Kalman/HMC/leaderboard |

## Inference Status

| Inference | Status |
| --- | --- |
| Hard veto screen | Instrumentation and frozen-certificate checks pass |
| Statistically supported ranking | None; no target arm or candidate comparison |
| Descriptive-only differences | Existing transferred-smoke telemetry and oracle differences |
| Default-readiness | Not established |
| Next evidence needed | Owner-approved parameter domain, fixed-ridge/raw-bias and conditioning budgets, transport/chunk rule, lower-rung design, and complete primary-shape statistical amendment |

## Artifacts

- Subplan: `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase8-rung1-numerical-design-subplan-2026-07-14.md`
- Result JSON: `docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase8/rung1-numerical-design-instrumentation-attempt1/result.json`
- Allocation audit: `docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase8/rung1-numerical-design-instrumentation-attempt1/allocation-audit.md`
- Focused checks and manifest: same attempt directory

## Handoff

No numerical arm is authorized. Before another target value or gradient is
observed, the owner-dependent amendment must freeze the parameter domain, fixed
ridge candidates, Cholesky safety rule, raw covariance bias budget, transport
convergence and chunk rule, seeds, shape, attempt budget, and deterministic
selection/no-selection behavior. The later representable-step repair passed all
35 cases under the owner-directed FD-only heuristic. A rigorous callable-error-
bound FD certificate is retired as unsupported because the required
TensorFlow/XLA absolute callable error bounds are absent; it is not a pending
prerequisite for another numerical arm.

## Nonclaims

This result does not establish a target ridge, transport, reset, residual,
conditioning, or chunk setting; Kalman value/gradient equivalence; formal FD
certification; GPU or full-shape feasibility; HMC readiness; route admission;
leaderboard completeness; release; or integrity readiness.
