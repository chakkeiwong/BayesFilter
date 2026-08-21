# P1B LGSSM Value Ladder attempt02 Result (v0.3 program) — 2026-08-18

Plan: `bayesfilter-p1b-lgssm-value-ladder-plan-2026-08-17.md`
Artifact: `docs/benchmarks/artifacts/p1b_lgssm_value_ladder_20260817/attempt02/result.json`
Command: `run_p1b_lgssm_value_ladder_20260817.py --dims 2,4,8 --output .../attempt02/result.json`
Manifest: main @ dae37183 (working tree), tf-gpu python (TF 2.19.1),
CPU-only (`CUDA_VISIBLE_DEVICES=-1`, GPU intentionally hidden), wall
16622 s, resource stop at n=4 r=8 (5332 s > 45 min/cell), 25 cells.

## Result

Every cell fails the declared 2.5e-3/step tolerance; `r_star` is null at
all n. Per-step gaps: n=2 ~5e-2..1.2e-1 across r in {2,4,6,8}; n=4
~0.6..2.5. The adversarial arm is indistinguishable from standard
(descriptive).

## The informative signature (explanatory diagnostics)

1. RANK-INSENSITIVE: at n=2 the gap does not improve from r=2 to r=8
   while max weighted fit rms improves monotonically (3.9e-1 -> 4.1e-2).
2. PROGRAM-VERSION-INSENSITIVE: attempt01 (v0.1) and attempt02 (v0.3)
   gaps agree cell-by-cell within seed noise (e.g. n=2 r=6 s=142:
   5.34e-2 vs 5.84e-2). The memo's expectation that the v0.3 smooth
   shift would remove the attempt01 noise floor is REFUTED for the
   ladder: the dominant error mechanism is shared by both programs.
   (The v0.3 repairs remain justified by the score-path C0/FD evidence
   in Addenda 3-5; they were never ladder-motivated alone.)
3. ROW-TYPE/BUDGET-SENSITIVE (focused diagnostic, n=2 r=4 seed 42,
   `/tmp/p1b_rowscale_diag.log`): scattered rows 4096 -> 16384 -> 65536
   give per-step 5.18e-2 -> 1.40e-2 -> 9.7e-3 (28 s / 192 s / 971 s).
   The declared ROWS_BY_N budget (4096 at n=2) is under-resolved by
   >= 16x; improvement flattens near ~1e-2 at r=4, consistent with a
   rank/resolution floor beneath the sampling bias (same-fixture
   quadrature arm: see addendum below).
4. The same engine passes the n=2 smoke gate with tensor-quadrature
   rows at 1.5e-3/step (r=6): harness, oracle, and engine are NOT
   invalidated.

## Mechanism (hypothesis, supported not proven)

The value increment uses the EXACT Gram normalizer Z_h = int h_fit^2 of
the fitted TT, which includes the fit's true L2 error mass ||e||^2 as a
positive bias; scattered-row fits generalize worse than their in-sample
weighted rms indicates, so the per-step log increment carries a
row-sampling-driven positive bias that rank cannot remove. Consistent
with signatures 1-3; not yet isolated by a dedicated holdout-mass
diagnostic (recorded as the natural follow-up instrument).

## Decision table

| item | status |
|---|---|
| decision | ladder attempt02 INVALID AS RANK EVIDENCE: declared row budgets under-resolved; r*(n) not measurable at this design |
| primary criterion (2.5e-3/step) | failed in all cells (hard screen; but attributable to fixture budget, not candidate structure) |
| veto diagnostics | no crash/non-finite/condition veto fired; resource stop at n=4 r=8 as declared |
| main uncertainty | how much of the residual ~1e-2 floor at r=4 is rank vs remaining sampling bias (quadrature arm addendum) |
| next justified action | amend the ladder plan's row budgets / row design (QMC or importance rows are V2-compliant candidates), then a bounded attempt03; row-budget scaling is now a measured input to tuning v1.1 |
| not concluded | rank-structure failure of the branch-axis program; any n>=4 rank statement; any superiority/ranking claim (3 seeds, descriptive only) |

## Inference status (stochastic-comparison discipline)

| row | status |
|---|---|
| hard veto screen | passed (no non-finite/crash/condition vetoes) |
| statistically supported ranking | none claimed |
| descriptive-only differences | all per-cell gaps, walls, rms, adversarial-vs-standard |
| default-readiness | none; no configuration promoted |
| next evidence needed | attempt03 under amended row budgets/design for r*(n); holdout-mass diagnostic for the mechanism |

## Research-question ledger

Candidate under test: scattered-frozen-row branch-axis value program at
declared budgets. Outcome: promotion veto on THIS fixture design (row
budget), explicitly NOT a continuation veto: the planned pivot branch
(structure exploitation / row design) is exactly the repair this failure
motivates. The harness (exact Kalman oracle, per-step tolerance,
telemetry) remains valid.

## Addendum (same session): same-fixture quadrature arm

Same ladder fixture (n=2, seed 42, deg 12, hw 3.0), tensor-GL rows
qo=14 (`/tmp/p1b_quadrow_diag.log`):
- r=4: per_step 8.858e-3 (wall 543 s) — descriptively equal to
  scattered-65536 (9.7e-3; single runs, no uncertainty claim): the
  ~1e-2 floor at r=4 is RANK/RESOLUTION-limited, not row-limited.
- r=6: per_step 2.314e-3 (wall 3372 s) — UNDER the declared 2.5e-3/step
  tolerance on the exact ladder fixture (single run, diagnostic-only
  rows): under resolved rows, r=6 clears the gate at n=2 and r=4 does
  not. This is the first fixture-valid rank signal of the program; it is
  NOT ladder evidence (quadrature rows are the V2 diagnostic exception,
  and one seed is descriptive).

Combined reading (descriptive): per-step error = row-sampling bias
(dominant at the declared 4096-row budget, decays with row count) +
a rank floor (~9e-3 at r=4; smoke evidence puts r=6 near 1.5e-3 under
resolved rows). The declared 2.5e-3/step tolerance therefore needs BOTH
r >= 6 and a row design well beyond 4096 MC rows at n=2. Row budgets
for a valid attempt03 must be set from this measured scaling, and
alternative V2-compliant row designs (QMC/scrambled-Sobol, importance
rows) are the identified lever for larger n where brute-force row
scaling is unaffordable.

## Addendum 2 (same session): row-design diagnostic and attempt03 launch

Sobol-vs-MC discriminating diagnostic (n=2, r=6, seed 42 fixture,
`/tmp/p1b_sobol_diag.log`; rotated Sobol = Sobol + seed-frozen
Cranley-Patterson shift):

| rows | MC per-step | Sobol per-step |
|---|---|---|
| 4096 | 9.82e-2 | 4.94e-2 |
| 16384 | 3.90e-3 | 2.55e-3 |

Sobol-16384 reaches the fixture's quadrature resolution floor
(2.31e-3) where MC needs >= 65536: the row-design lever is validated
descriptively (single runs). Consequences implemented:
- `EngineConfig.row_design = "mc" | "sobol"` (frozen randomized-QMC
  scattered rows; V2-compliant; BOTH engines routed through the same
  selector per V5). Default "mc" — no behavior change for existing
  claims; adjoint/FD/P1A regression suite green post-change.
- Ladder plan Amendment A1 (declared before execution): attempt03 =
  n=2 only, Sobol 32768 rows, ranks {4,6} (+r=8 contingency), all
  other contract fields unchanged; pre-declared honesty bound on the
  thin r=6 margin (floor 2.31e-3 vs tolerance 2.5e-3).
- attempt03 launched to
  `docs/benchmarks/artifacts/p1b_lgssm_value_ladder_20260817/attempt03/`.
