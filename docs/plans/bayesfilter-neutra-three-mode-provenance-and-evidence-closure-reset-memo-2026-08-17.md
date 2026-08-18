# NeuTra Three-Mode Provenance And Evidence Closure Reset Memo (2026-08-17)

## State

Checkpoint provenance is repaired and the component-aware three-mode result now
has two fresh successful training/HMC replications in addition to the original
seed. The naive zero-centered iid Student-`t(3)` mode-blind proposal failed
importance support and stopped before training/HMC.

## Canonical Entry Points

- Plan: `docs/plans/bayesfilter-neutra-three-mode-provenance-and-evidence-closure-plan-2026-08-17.md`
- Result: `docs/plans/bayesfilter-neutra-three-mode-provenance-and-evidence-closure-result-2026-08-17.md`
- Active HMC runner: `docs/benchmarks/run_weighted_neutra_three_mode_hmc_2026_08_12.py`
- Mode-blind preflight: `docs/benchmarks/preflight_neutra_three_mode_blind_student_t_2026_08_17.py`
- Artifact root: `docs/plans/artifacts/neutra-three-mode-provenance-evidence-closure-2026-08-17/`

## Frozen Decisions

- The active reviewed baseline is checkpoint SHA
  `b39c682030fb3ba8bafe863c747674db40b5d7c13e164c8445ddfab649ad93f6`,
  not obsolete SHA `57b21cc...`.
- Component-aware viability is replicated at a bounded two-fresh-seed level.
- Do not select a universal HMC kernel: fresh seeds selected different viable
  `(L, epsilon)` pairs.
- Do not claim mode discovery. The tested centered Student-t family is rejected
  for inadequate ESS; it does not reject target-query-driven discovery methods.
- Native divergence remains `not_exposed_by_kernel`; unavailable is not zero.
- Cross-target geometry, KSC/German, and SSL-LSTM evidence remain separate.

## Next Scientific Step

Design a mode-discovery proposal using only target value/score queries and a
declared exploration budget. Plausible hypotheses are tempered exploration,
multi-start local exploration followed by clustering, or sequential defensive
mixture enrichment. The earliest gate is proposal support on disjoint rows;
training and HMC remain blocked until global and median 4,096-row batch ESS
fractions meet their reviewed threshold.

After the proposal/discovery lane, continue the already planned varying-Hessian
and application targets under target-specific protocols. Do not infer SSL-LSTM
readiness from the analytic mixture.

## Verification

- Focused suite: `80 passed`.
- `git diff --check`: passed.
- Top-level artifact hash receipts: passed for both fresh trainings, both full
  HMC results, and the mode-blind preflight.
- No campaign process remains running.
