# P1B n>=4 Row-Design / Feasibility Note (draft under execution) — 2026-08-18

Status: `IN_PROGRESS` — Section 3 awaits the running/queued measured
diagnostic; everything else is evidence-in-hand.
Follows: `bayesfilter-p1b-lgssm-value-ladder-attempt03-result-2026-08-18.md`
(r*(2)=6 under Sobol 32768 rows).

## 1. Question

Under what row design and budget can the branch-axis value ladder
produce a VALID r*(n) at n in {4, 8} within the declared per-cell
resource bound (45 min), given that attempt02 established row-sampling
bias as the dominant error at the original MC budgets?

## 2. Evidence in hand (measured, n=2 unless noted)

- MC row bias decays slowly: 4096 -> 65536 rows gives per-step
  5.2e-2 -> 9.7e-3 at r=4 (rank floor ~9e-3 reached only at 16x the
  declared budget).
- Rotated Sobol reaches the r=6 resolution floor at 16384 rows
  (2.55e-3 vs quadrature 2.31e-3); MC needs >= 65536.
- attempt03 (Sobol 32768): r=6 passes all seeds; walls ~2000-2300 s per
  r=6 cell at n=2 (fit variables 2n+1 = 5 axes).
- attempt02 n=4 walls (MC 8192 rows): r=6 ~1470 s, r=8 ~5330 s (over
  the 45-min stop). Fit variables at n=4: 9 axes; rows enter the ALS
  design assembly linearly, ranks quadratically in the environment
  contraction.
- Dimension scaling caveat: QMC advantage degrades with effective
  dimension; the n=2 gain (4x rows) is NOT assumed to transfer to
  2n=8/16-dimensional row spaces (plan discipline: cross-task transfer
  is a hypothesis, not a default).

## 3. Discriminating diagnostic (single seed, descriptive, scoping-only)

n=4, r=6 warm-start hypothesis, Sobol rows in {8192, 16384, 32768}:
per-step gap and wall per cell. Decision rule declared in advance:
- If per-step approaches a flat floor above 2.5e-3 across the row
  sweep, the binding constraint at n=4 is rank/resolution, not rows ->
  next arm varies rank at the best affordable row budget.
- If per-step is still falling at 32768 rows, rows remain binding ->
  the ladder needs either a larger budget with a longer per-cell stop
  (owner resource decision) or a better row design (importance rows).
- If walls exceed ~40 min at the row budget needed, the n=4 ladder is
  NOT feasible pre-P3; defer the n>=4 ladder until the XLA port lands
  and re-measure (V12: no feasibility language without measured walls).

RESULTS (2026-08-18, `/tmp/p1b_n4_rowdesign_diag.log`, script
`docs/benchmarks/run_p1b_n4_rowdesign_diag_20260818.py`, single seed 42,
n=4, r=6, deg 12, hw 3.0):

| rows (Sobol) | per-step gap | max fit rms | wall |
|---|---|---|---|
| 8192  | 2.291 | 7.1e-2 | 1386 s |
| 16384 | 2.007 | 7.9e-2 | 2745 s |
| 32768 | 1.800 | 8.7e-2 | 5508 s |

Reading (complete, single seed, descriptive): FLAT-FLOOR branch of the
declared decision rule. 4x rows buys 21% total where the same lever at
n=2 bought an order of magnitude; Sobol-8192 is descriptively equal to
attempt02's MC cells (2.05-2.46) at the same fixture. The n=4 error at
this fixture is NOT row-dominated; the ~2/step magnitude (~1000x
tolerance) indicates a rank/resolution/fixture constraint. Candidate
mechanisms for the next discriminating arm (one variable at a time):
r=6 far below r*(4); basis degree 12 under-resolving the 4-D marginal
at half-width 3.0; retained boundary-rank/branch-count growth outpacing
the fit. Walls independently settle feasibility: 16384-row cells cost
~46 min and 32768-row cells ~92 min against the ladder's declared
45-min stop, so ANY n=4 resolution/rank upgrade requires the P3 XLA
port first (V12: measured statement for this CPU route only).

DECISION (per the pre-declared rule, third branch): defer the n>=4
ladder until after P3; run the mechanism-discriminating arms (rank,
degree, half-width) at XLA speed. P3 is now the program's critical
path.

## 4. Non-claims

No r*(4) or r*(8) statement exists yet; r=6 at n=4 is a warm-start
hypothesis only; single-seed diagnostics nominate, they do not promote.
