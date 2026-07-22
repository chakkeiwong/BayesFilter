# Actual-SV Overcomplete Analytical Chart Phase 3 Result

Date: 2026-07-17

Status: `PASS_PHASE_3_SELECTED_K23`

## Selection Result

The frozen smallest-pass rule selects global constant capacity `K=23`.

- `K=5,6` fail center preparation by `T=10` from a negative Pearson
  reference.
- `K=7,...,22` pass the `T=10` center screen but fail center preparation by
  `T=100`; failures occur at times 7, 10, or 90 depending on capacity and are
  negative-reference failures.
- `K=23` passes center preparation and all nine design points at `T=10`,
  `T=100`, and `T=1000`.
- `K=24,25` fail strict positive Voronoi/reference mass at time 0.

The selected full-horizon preparation contains all 999 projections.  The
`T=1000` CPU-XLA design result passes every finite-program gate at every design
point.  The weakest design output is ordered design point 7, with minimum
computed weight `5.395467979032687e-165` and bitwise-identical warm replay.

## Numerical Corroboration

The predeclared weakest design solve was reconstructed at zero-based time 0.
Using its saved float64 `4 x 23` matrix, target, and reference, mpmath solves at
50, 100, and 200 decimal digits retain the same minimum anchor and strict
positive sign.  The 200-digit value begins
`5.395467979023440381971846603610243e-165`.

This is corroborating numerical evidence, not interval proof and not an added
selection threshold.

## Engineering Notes

The frozen `K=5/25` timing endpoints were mathematically ineligible before
runtime.  A visible plan amendment substituted executable timing endpoints
`K=7/23` and the `K=23,T=100` center probe.  The conservative full-ladder
projection was 2.20 CPU core-hours against the five-core-hour Phase 3 cap.

One attempted `T=100,K=23` design command named a nonexistent capacity-copy
path.  No model execution occurred.  The immutable pilot preparation with
SHA-256 `476e4c117b4855358c9b2179d5e5c45623408be5f583f1df04a491ff5e8ace03`
was used instead, and the fresh retry passed.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Select `K=23` | Smallest capacity passing all design points at `T=10,100,1000` | No Phase 3 chart, budget, timeout, or high-precision veto fired | Held-out interiority is untested | Run the single frozen held-out audit without retuning | No held-out, derivative, score-equivalence, GPU, HMC, canonical, or leaderboard claim |

## Handoff

Capacity is frozen.  Phase 4 may inspect the predeclared held-out set exactly
once.  A held-out failure rejects this candidate and cannot select another
capacity from the same audit points.
