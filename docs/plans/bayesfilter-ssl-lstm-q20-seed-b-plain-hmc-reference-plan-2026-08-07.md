# SSL-LSTM q=20 seed-B plain-HMC reference plan (2026-08-07)

## Purpose

The target-only quadrature diagnostic found two distinct stationary modes, so a
single-center deterministic quadrature reference is invalid. This plan executes
the predeclared repair: freshly tuned plain fixed-metric HMC on the exact q=20
target, followed by the repository sequential warm-up/retained screen. Its
posterior is the independent authority for comparing seed-B NeuTra draws.

## Research intent ledger

| Item | Frozen statement |
|---|---|
| Main question | Does seed-B NeuTra/fixed-HMC agree with an independently tuned plain-HMC authority that can traverse both target-only modes? |
| Candidate | Seed-B terminal NeuTra draws already archived in the r2 sequential artifact. |
| Comparator | Plain q=20 target, identity mass, no transport, current target signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`, freshly tuned `L` grid, and a fresh sequential screen. |
| Expected failure mode | Identity-metric plain HMC may not mix the two near-equal modes, or the CPU cost may exceed the authorized campaign cap. |
| Promotion criterion | Complete target-specific tuning yields an admitted frozen kernel, then four-chain sequential HMC passes warm-up/retained R-hat and ESS gates with all hard vetoes clear; only then run the posterior comparison. |
| Promotion veto | Target/adapter/signature mismatch, stale source binding, invalid or nonfinite target/status, exposed positive divergence, acceptance outside `[0.35,0.95]`, no chain movement, failed R-hat/ESS, incomplete grid, or wall cap. |
| Continuation veto | Corrupted artifacts, CPU/GPU/XLA policy failure, missing required API, worker crash, or resource projection beyond the 20,000-second campaign cap. A plain-HMC failure does not reject NeuTra or the research direction. |
| Repair trigger | No viable kernel triggers a tuning repair only if budget remains; a tuned kernel that fails sequential mixing triggers a plain-HMC validity failure and preserves NeuTra as unresolved. |
| Explanatory diagnostics | Acceptance, finite energy tails, mode occupancy, per-chain movement, target status, runtime, and tuning step size. |
| Must not be concluded | Plain HMC superiority, model adequacy, broad posterior correctness, cross-seed robustness, native zero divergences, or default readiness. |

## Evidence contract

- Exact baseline: same signed q=20 target value/score/status program as the
  candidate, but with no learned transport and no NeuTra checkpoint.
- Tuning: the complete reviewed `L=(3,5,9,13,18,25)` grid, three independent
  64-draw screens per candidate, target acceptance `0.70`, fixed identity mass,
  XLA enabled, all tuning draws discarded. `L=1` is forbidden.
- Fresh mode starts: the two target-only stationary endpoints from
  `r3/map-progress.json`, rounded only for serialization and re-evaluated by the
  target before launch:

  ```text
  mode_plus  = (0.73311370, 0.17273238, 0.58942510, 0.15892059)
  mode_minus = (0.44667563, -0.24131804, -0.58769660, 0.11989041)
  ```

  Chains 0/1 start at `mode_plus`; chains 2/3 start at `mode_minus`. These are
  target-derived starts, not NeuTra draws, and the sequential gate must still
  demonstrate mixing rather than assuming it.
- Sequential gate: at least 2,000 warm-up transitions per chain, latest 1,000
  warm-up window maximum rank-normalized split/folded R-hat `<=1.05`; then grow
  retained draws cumulatively to at least 1,000 per chain with maximum R-hat
  `<=1.01`, minimum bulk ESS `>=400`, and minimum tail ESS `>=400`. Acceptance
  must stay in `[0.35,0.95]`; hard vetoes are nonfinite values/status, exposed
  positive divergence, invalid target status, and no chain movement.
- Comparison criterion: after both authorities pass, reuse the predeclared
  chain-aware moving-block bootstrap and margins from the posterior-reference
  plan. No comparison is run from an incomplete or failed plain-HMC authority.
- Artifact: versioned output under
  `docs/plans/artifacts/ssl-lstm-q20-seed-b-plain-hmc-reference-2026-08-07/r1/`.

## Default and assumption audit

| Choice | Provenance/status | Justification | Failure mode | Early diagnostic |
|---|---|---|---|---|
| Plain target / identity mass | Required baseline hypothesis; current API | Independent of learned transport and exact target-bound | Poor geometry / slow mixing | Complete tuning grid and sequential R-hat/ESS |
| `L=(3,5,9,13,18,25)` | Reviewed historical grid; reused as baseline only | Covers short through long trajectories without `L=1` | Cost or unstable long trajectories | CPU/XLA rate canary and cap projection |
| Three 64-draw screens | Existing fixed-grid mechanics policy | Candidate nomination only, not convergence evidence | Noisy acceptance | Fresh sequential screen is primary gate |
| Two target-only modes, two chains each | Derived from independent MAP diagnostic | Prevents all chains starting in one basin | HMC cannot cross modes | Mode occupancy and between-chain R-hat |
| 2,000 warm-up + 1,000 retained | Owner NeuTra sequential default reused for comparator | Necessary minimum for complex multimodal target | Exceeds budget | Preflight projection and hard cap |
| 25 CPU workers x 4 rows | Measured q=20 batch target throughput | Uses the existing CPU/XLA lane for independent candidate generation | Process overhead / memory | Worker topology, target signatures, RSS |
| 20,000 s cap | User-authorized remaining headroom | Bounded campaign | Incomplete authority | Projection before grid and stop at cap |

## Resource ladder and stop conditions

1. Run focused tests and a CPU/XLA target/worker canary; no claim-bearing
   sampling.
2. Measure one 25x4 tuning-screen workload. Project the complete grid and the
   minimum sequential authority using the measured current-source rate.
3. Do not launch the grid if its worst-case projection plus the mandatory
   sequential minimum exceeds the 20,000-second cap. Record a resource veto.
4. If feasible, run the complete grid with fresh seeds and preserve only
   structured candidate evidence; discard tuning draws.
5. Select a viable kernel by smallest `L` only. Acceptance is a nomination gate,
   not a convergence ranking.
6. Run four persistent CPU/XLA chain workers with the mode-derived starts and
   sequential chunks. Stop on any hard veto or cap; do not shorten the declared
   warm-up/retained minima.
7. Compare NeuTra against plain HMC only when the plain authority passes.

## Skeptical audit

Audit status: **passed after revision**.

- The quadrature baseline was invalidated by two target-only modes; this plan
  does not reuse it or silently claim quadrature authority.
- Historical plain-HMC artifacts bind stale target signature `302d...` and are
  not reused. The new worker binds current signature `9a86...`.
- Acceptance and runtime are nomination/explanatory diagnostics. Promotion is
  sequential R-hat/ESS plus hard veto status.
- The two modes are target-derived, not candidate-derived. No NeuTra samples,
  checkpoint parameters, or HMC traces design tuning or initialization.
- The cap is a continuation veto. The plan cannot become valid by shrinking
  samples, omitting long `L`, or treating a partial grid as an authority.
- CPU-only execution is explicit and records GPU hiding; XLA remains required.
- A failed plain authority rejects this comparator attempt only. It does not
  reject the NeuTra candidate or the scientific direction.

## Execution note

The target-only MAP diagnostic completed with two stationary basins and therefore
fired the single-center quadrature veto. The plain-HMC preflight and CPU/XLA
timing canaries passed, including a four-process one-chain canary. The final
resource gate used the measured four-chain rate for the implemented six-worker
grid and one four-chain sequential process: complete grid `145,853.9 s`,
minimum sequential authority `81,155.2 s`, total `227,009.1 s` against cap
`20,000 s`. It fired `RESOURCE_VETO_BEFORE_TUNING`; no partial plain-HMC tuning
or posterior result was launched. The one-chain parallel canary remains
explanatory only and was not substituted into this projection.

## Pre-mortem

- Plain chains remain trapped in one mode: mode occupancy and between-chain R-hat
  expose this; no posterior comparison is made.
- Tuning appears viable but long-run step is unstable: sequential target/status,
  acceptance and finite-energy vetoes catch it.
- CPU timing projection is optimistic: a measured 25x4 canary precedes the grid;
  cap stop prevents partial evidence being promoted.
- Reference implementation accidentally changes the target: exact signatures,
  source hashes, target status, and independent endpoint checks fail closed.
