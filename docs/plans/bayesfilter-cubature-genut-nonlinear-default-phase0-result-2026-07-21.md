# Cubature/GenUT Nonlinear Default Program: Phase 0 Result

Date: 2026-07-21

Status: `PASS_PHASE0_BOUNDARY_FREEZE_PHASE1_AUTHORIZED`

## Scope

Phase 0 froze Cubature/GenUT as an experimental candidate only. Contract
E--Chol remains the sole route eligible for canonical, leaderboard,
default-readiness, or HMC-facing status.

## Boundary Findings

| Item | Result |
|---|---|
| LGSSM candidate | Diagnostic only; hard-coded runner and Kalman oracle |
| Nonlinear reusable route | Not implemented before this program |
| GenUT in Gaussian scope | Bitwise Cubature alias; not independent evidence |
| GenUT positive OT eligibility | Requires positive central weight and representable masses |
| Existing nonlinear evidence | Short-prefix/experimental or blocked; not default evidence |
| Leaderboard state | Existing Phase 2 LEDH result has zero eligible rows; completion remains blocked |
| XLA status | LGSSM comparison was no-XLA diagnostic; default route requires XLA |

## Skeptical Audit

The program was revised before execution to prevent four invalid shortcuts:

- promoting the LGSSM result to nonlinear default evidence;
- treating CI inclusion or finite reset residuals as a precision certificate;
- using caller-stamped identity or stale tuning artifacts; and
- treating GenUT signed weights as positive OT masses.

The audit passed with phased scope. No canonical/default policy was changed.

## Entry Decision

Phase 1 is authorized to add a candidate-only generic design/identity layer and
focused tests. Phase 2 remains gated on those tests and on preserving all
existing tests.

## Evidence Sources

- `AGENTS.md:86-126,128-160,394-410`;
- `docs/plans/bayesfilter-contract-e-tp-phase10-terminal-synthesis-result-2026-07-15.md`;
- `docs/plans/bayesfilter-complete-highdim-leaderboard-phase2-ledh-fulltime-seed81120-result-2026-07-11.md`;
- `docs/plans/bayesfilter-contract-e-cubature-genut-mathdevmcp-audit-result-2026-07-20.md`;
- `docs/plans/bayesfilter-lgssm-recursive-score-matched-three-horizon-comparison-result-2026-07-21.md`.
