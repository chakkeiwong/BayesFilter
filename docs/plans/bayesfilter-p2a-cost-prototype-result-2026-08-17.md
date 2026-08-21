# Result: P2A Cost Prototype — Mode Decision (2026-08-17)

Plan: `bayesfilter-zhao-cui-generic-highdim-analytic-score-program-plan-2026-08-15.md` (rev 3, P2A).
Artifact: `docs/benchmarks/artifacts/p2a_cost_prototype_20260817/attempt01/result.json`
Command: `python docs/benchmarks/run_p2a_cost_prototype_20260817.py --output ...`
Environment: eager CPU float64, commit `18cfe609` (dirty tree), TF 2.19.1.
Scope: ONE fitted branch-axis step (n=2, deg 10, B=5, r=4, s=2, N=2048),
synthetic smooth targets/tangents. Engineering measurement only (V12);
concurrent P1B ladder on the same host may inflate absolute walls — ratios
are the decision quantity and share the contention.

## Measurements

Value pass: 0.41 s.

| p | mode | wall | gradient/value ratio | tracemalloc peak |
|---|---|---|---|---|
| 3 | forward + exact dot_A | 1.6 s | 3.9 | ~0 MB |
| 30 | forward + exact dot_A | 13.0 s | 31.9 | 1 MB |
| 300 | forward + exact dot_A | 132.5 s | **325.8** | 1 MB |
| 300 | chunked (32) forward | 132.2 s | 325.1 | 1 MB |

Ratio ≈ 1.08 p: the per-parameter environment-tangent (`dot_A`)
construction dominates, exactly as the Codex audit's F3 predicted; the
memo-era "<= ~6x by batched forward" expectation is refuted at p=300.
Chunking changes memory layout, not flops.

## Solver-reuse obligations (plan-bound): PASSED

- scaled-primal vs normal-equation solution agreement: 8.2e-17 relative;
- derivative consistency vs the ACTUAL scaled augmented solver (centered
  FD of `_solve_scaled_augmented_ridge` along a joint core+target
  perturbation): 4.7e-11 relative;
- factorization reuse implemented and exercised (one LU per update, all
  tangent RHS via `lu_solve` multi-RHS).

## Decision

| Item | Status |
|---|---|
| Decision | **Adjoint (reverse) replay selected as the P2 score mode.** Forward and chunked-forward are DISQUALIFIED at p=300 by the declared <=6x gate (measured ~326x) |
| Primary criterion | <=6x at p=300: failed by both forward modes; adjoint expected O(1)x in flops (transposed chain), to be MEASURED at P2 before the gate is called passed |
| Veto status | none fired; solver-reuse checks all green |
| Precondition created | UB-1 adjoint addendum (manual reverse-sweep derivation) must land before P2 implementation — Method A requires a manual/analytical backend, so TF GradientTape is not an admissible route for the claim-bearing score |
| Remaining P2A obligation | full-horizon T=120 tangent/adjoint-state stress runs on the CHOSEN mode after the addendum + P2 skeleton exist (running it on a disqualified mode would measure the wrong object) |
| Not concluded | no feasibility claim; forward mode remains valid for small p (<= ~5, ratio <= ~5) and stays available as an FD-independent cross-check of the adjoint at p=3 |

## Pre-mortem note for P2

The adjoint sweep must checkpoint per-update (design row-set id, LU
factors or recomputation seeds, solution, shift branch); memory of the
checkpoint trace is the quantity the T=120 stress must bound. If
checkpoint memory is the binding constraint, recompute-from-frozen-seeds
is admissible (the program is deterministic) and its recompute/store
trade is part of the stress measurement.
