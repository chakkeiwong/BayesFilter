# NeuTra Banana HMC Repair Reset Memo (2026-08-16)

## Terminal State

- Plan: `docs/plans/bayesfilter-neutra-banana-hmc-repair-plan-2026-08-16.md`.
- Runner: `docs/benchmarks/run_neutra_banana_hmc_repair_2026_08_16.py`.
- Terminal artifacts: `docs/plans/artifacts/neutra-banana-hmc-repair-2026-08-16-r3/`.
- Campaign wall time: `534.26 s`; GPU 0; float64; XLA; TF32 disabled; memory
  growth verified before TensorFlow initialization.
- The learned transport was replayed from seed 15, 6,000 updates, and passed
  the exact r3 proposal audit partition `59015`.

## Findings

1. Learned transport plus the original iid-normal start bank selected `L=5`
   and failed retained exact-law adjacent cross moments 4 and 6. HMC health,
   warm-up, and retained convergence passed.
2. The same learned transport plus a central deterministic start bank selected
   `L=10` and passed warm-up, retained convergence, health, and exact-law
   screens.
3. The exact analytic banana transport plus the original iid-normal bank
   selected `L=10` and passed all screens. This is a mechanics/geometry
   positive control, not learned-transport evidence.
4. The pattern rules out a controller-wide or analytic-target failure. It does
   not prove a pure start-bank cause because Arm A and Arm B were independently
   tuned and selected different kernels.

## Required Next Step

- The matched-kernel cross-over is complete under
  `docs/plans/artifacts/neutra-banana-hmc-matched-kernel-2026-08-16-r1/`.
- Both banks pass with frozen `L=10`; both fail with frozen `L=5`. Classify the
  failure as kernel-sensitive learned-transport dynamics under the tested
  identity-z mass, not initial-state sensitivity.
- The frozen `L=10` confirmation is complete under
  `docs/plans/artifacts/neutra-banana-hmc-l10-confirmation-2026-08-16-r1/`:
  both original and central banks passed at 5,000 retained draws per chain
  under unchanged exact-law and sequential gates.
- The frozen `L=10` candidate is viable for this banana control. The next
  scientific check is predictive/output-distribution equivalence, not a new
  kernel or start-bank search.

## Constraints

- Do not promote the central bank as a repository default.
- Do not transfer the banana kernel, start bank, or 6,000-update training
  setting to SSL-LSTM.
- Do not relax exact-law, finite/log-acceptance, R-hat, ESS, movement, energy,
  or divergence gates.
- Preserve the analytic arm as a bounded mechanics control and the Gaussian
  candidate as the existing exact-law HMC authority.
