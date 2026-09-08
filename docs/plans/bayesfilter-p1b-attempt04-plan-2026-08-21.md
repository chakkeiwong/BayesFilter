# P1B Ladder attempt04 Plan — Triangular Adapted Maps (2026-08-21)

Status: `DECLARED_BEFORE_EXECUTION`. Supersedes the attempt03 scope for
the n=4 question; n=2 rows retained as regression guard.
Evidence basis: adapted-maps design note Sections 9-13 (triangular map +
truncation correction: n=2 at 5.4e-5/step; n=4 residual isolated to fit
resolution at r=6/deg12).

## Question

r*(4): the smallest rank at which the branch-axis program under the
VALID design (triangular_affine maps, truncation correction, Sobol rows)
meets the declared per-step tolerance at n=4; n=2 rung re-measured under
the same design as the regression guard.

## Design (all declared)

- Engine: adapted triangular + truncation correction; XLA step port
  (parity-gated vs the eager adapted engine at 1e-12 on the n=2 fixture
  before any ladder cell runs — same measured-gate discipline as P3.3).
- Maps: M2-joint companion-Kalman hints (exact for the LGSSM family;
  the benchmark owns the model); kappa_c=4, kappa_p=3 (design-note
  defaults; kappas join scope identity).
- Grid: n in {2, 4}; ranks {6, 8, 10}; seeds {42, 142, 242} standard
  arm; rows 8192 (n=2 evidence: budget no longer binding; 5.4e-5 at
  32768 vs 8.6e-5 at 8192).
- Tolerance: 2.5e-3/step (unchanged). Resource stop: 45 min/cell.
- r*(n) = smallest rank with all standard seeds passing (unchanged).
- Artifact: docs/benchmarks/artifacts/p1b_lgssm_value_ladder_20260817/attempt04/
  (fresh dir; schema v3 with map/kappa/correction fields).

## Pre-mortem

- Degree could bind before rank at n=4 (deg12 fixed): if r=10 fails
  with fit rms plateauing across ranks, declare rank-saturated and add
  ONE degree arm (deg16, best rank) before concluding — do not
  conclude "rank insufficient" from a degree-limited grid.
- XLA parity gate failure blocks the ladder (fall back to eager r=6/8
  only, drop r=10, note the resource limit).
- Truncation-correction MC noise: correction shares the row budget;
  its per-step ratio telemetry is recorded — if ratio > 0.5 anywhere
  (correction dominating), the cell is flagged (map/kappa mis-set),
  not counted as rank evidence.

## Non-claims

Single-family LGSSM fixtures; no cross-model transfer of r*(4); no n=8
claim; kappas untuned (defaults) — tuning procedure v1.1 owns them for
claim-bearing scopes.
