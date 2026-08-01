# Actual-SV Overcomplete Analytical Chart Phase 4 Result

Date: 2026-07-17

Status: `PASS_PHASE_4_HELD_OUT_AUDIT`

The frozen selected `K=23,T=1000` candidate was evaluated exactly once on the
eight Phase 1 held-out points without retuning.  All eight points pass the
finite-program rank, condition-roundoff, equality-residual, finiteness, and
strict computed-positivity gates under CPU XLA.  Warm replay is bitwise
identical.

The weakest held-out point is ordered point 2, normalized coordinate
`(0.5,-0.5)`, with minimum computed weight `5.396877738726166e-165`.
Reconstruction identifies zero-based time 0 and anchor 22 as the weakest
solve.  Independent mpmath recomputation at 50, 100, and 200 decimal digits
retains the same minimum anchor and strict positive sign; the 200-digit value
begins `5.396877738726627672483040726295261e-165`.

The high-precision calculation is corroboration of the saved float64 system,
not interval proof and not a new threshold.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Pass Phase 4 | Every held-out point and the weakest sign-stability audit pass | No held-out veto fired | Full-horizon manual-score/FD behavior remains untested | Run `T=2,10,100,1000` own-scalar derivative regression | No scientific score equivalence, GPU, HMC, canonical, or leaderboard claim |
