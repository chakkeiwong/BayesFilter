# P1B LGSSM Value Ladder attempt03 Result (Sobol rows, n=2) — 2026-08-18

Plan: `bayesfilter-p1b-lgssm-value-ladder-plan-2026-08-17.md` + Amendment
A1 (declared before execution).
Artifact: `docs/benchmarks/artifacts/p1b_lgssm_value_ladder_20260817/attempt03/result.json`
Command: `run_p1b_lgssm_value_ladder_20260817.py --dims 2 --ranks 4,6
--rows 32768 --row-design sobol --output .../attempt03/result.json`
Manifest: main @ dae37183 (working tree), tf-gpu python (TF 2.19.1),
CPU-only (`CUDA_VISIBLE_DEVICES=-1`, GPU intentionally hidden), wall
7468 s, no resource stop, 6 cells, schema v2 (row_design recorded).

## Result: r*(2) = 6 under the declared contract

| rank | seed | per-step gap | pass (2.5e-3) | margin | max fit rms | wall |
|---|---|---|---|---|---|---|
| 4 | 42  | 1.048e-2 | no | -319% | 5.8e-2 | 355 s |
| 4 | 142 | 7.012e-3 | no | -180% | 4.3e-2 | 379 s |
| 4 | 242 | 5.601e-3 | no | -124% | 4.3e-2 | 368 s |
| 6 | 42  | 2.307e-3 | YES | +8%  | 2.1e-2 | 2282 s |
| 6 | 142 | 1.575e-3 | YES | +37% | 1.6e-2 | 2102 s |
| 6 | 242 | 1.024e-3 | YES | +59% | 1.6e-2 | 1984 s |

r=6 passes on all three standard seeds; r=4 fails on all three. The
seed-42 pass sits AT the fixture's measured quadrature resolution floor
(2.31e-3, attempt02 addendum) — per the pre-declared honesty bound this
is a SCOPE-MARGINAL pass: the declared tolerance and the r=6 resolution
floor nearly coincide on that data realization. No robustness claim.

## Decision table

| item | status |
|---|---|
| decision | r*(2) = 6 ACCEPTED as the n=2 ladder outcome under Amendment A1 (Sobol 32768 rows) |
| primary criterion | r=6: all standard seeds pass; r=4: all fail (clean separation, one rank apart) |
| veto diagnostics | none fired (no non-finite/crash/condition veto; no resource stop) |
| main uncertainty | seed-42 margin +8% is within plausible seed/row noise of the tolerance; 3 seeds are descriptive |
| next justified action | n>=4 row-design/feasibility note (brute-force rows unaffordable there), then P3 XLA port per plan |
| not concluded | any n>=4 rank statement; robustness of the +8% margin; MC-vs-Sobol superiority beyond descriptive single-run evidence; any cross-model transfer of r*=6 |

## Inference status

| row | status |
|---|---|
| hard veto screen | passed |
| statistically supported ranking | none claimed (3 seeds; pass/fail against a predeclared hard screen only) |
| descriptive-only differences | all continuous gaps/margins/walls; Sobol-vs-MC row-design comparison |
| default-readiness | row_design="sobol" remains an OPTION, not a new default (default-change bar not met: one fixture, one n, single-run design comparison) |
| next evidence needed | n>=4 design note + budgeted cells; multi-fixture Sobol evidence before any default change |

## Post-run red team

Strongest alternative explanation: the r=4/r=6 separation could partly
reflect the deg-12 basis resolution rather than rank per se (rank and
degree were not varied independently); the attempt02 diagnostics
support rank as the binding constraint at deg 12, but a degree arm was
not run. What would overturn r*(2)=6: a seed or row-shift replication
pushing a r=6 cell above tolerance (the +8% cell is the candidate);
that would move the verdict to "r=6 marginal, tolerance and floor
coincide at n=2". Weakest evidence: single row-design realization per
cell (one Sobol shift per engine seed).
