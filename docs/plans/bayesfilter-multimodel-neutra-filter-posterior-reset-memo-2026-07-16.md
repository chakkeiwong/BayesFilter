# Multi-Model NeuTra Filter-Posterior Reset Memo

Date: 2026-07-17

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `FINAL_CELL_COMPLETE_WITH_BLOCKERS`

## Current Ground Truth

- The P0-P7 runbook and the bounded P4 tuning-admission repair are complete.
- `PP-UKF` and `PP-SGQF` are `NEUTRA_CONFIRMED` only for six same-target
  physical posterior means on one T=20 fixture.
- `SIR-SGQF` is `NEUTRA_CONFIRMED` only for three same-target physical
  posterior means on one T=20 fixture.
- The other eight cells retain the exact blocker states in P7 attempt 02.
- P7 attempt 01 and the historical P4 correction remain valid evidence of the
  defect that triggered repair, but they are not the active terminal matrix.
- Across BayesFilter and `/home/chakwong/python`, learned NeuTra has reached
  transformed HMC on `9` distinct model families and `12` materially distinct
  posterior-target configurations. Nine have clean/strong historical
  diagnostic passes and three are qualified/marginal under their original
  contracts.

Active structured ledger:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p7/attempt-02/cell_ledger.json`,
SHA-256
`7e078b56a4fa71ac86d8dd7171825ff4d53d845e746d3276ac577b4640ebaf5d`.

Cross-repository count and prospective diagnostic policy:
`docs/plans/bayesfilter-neutra-cross-repository-model-evidence-ledger-2026-07-17.md`.

## Optional Repair Lanes

The runbook is closed; these are independent prospective research programs.

1. `SIR-UKF`: localize the GPU/CPU score gap above `1e-7`, preserve the target
   and eager correctness evidence, and replay R1 before issuing an identity.
2. `STR-UKF`: prospectively improve comparator geometry, require the score
   gate before HMC, and retain the source energy-error attempt as tuning-only
   evidence.
3. `SIR-ZC`, `SVX-ZC`, `PP-ZC`, and `STR-ZC`: design the missing observed-data
   source route or extension target; sampler tuning cannot repair a missing or
   production-ineligible route.
4. `SVX-SGQF` and `KSC-UKF`: define a new prospective filter-candidate ladder;
   do not retroactively change the failed frozen margins.
5. Future runnable cells: start with one central-truth dataset seed. Stop on a
   clean `p_truth >= 0.05` pass; run one additional seed only for a marginal
   `0.003 <= p_truth < 0.05` miss; investigate immediately for
   `p_truth < 0.003` or invalid sampler diagnostics.

## Exact Restart Context

Read in this order:

1. `docs/plans/bayesfilter-multimodel-neutra-filter-posterior-terminal-result-2026-07-16.md`;
2. `docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p7/attempt-02/result.json`;
3. `docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p7/attempt-02/cell_ledger.json`;
4. the selected cell's original subplan/result and the earliest re-entry rung.

The active sampler standards are disjoint modern-R-hat tuning admission,
retained but separately excluded warm-up, adaptive collection up to 10,000
draws per chain, rank/folded R-hat plus bulk/tail ESS, health/status vetoes,
target-specific GPU/XLA training, memory growth, and no NumPy or Python sample-
axis loop in active training/HMC paths. For new synthetic scientific screens,
posterior truth-tail probability is primary; plain-HMC agreement is optional
debugging evidence rather than a mandatory comparator.

## Forbidden Carry-Forward Claims

Do not claim universal NeuTra success, all cells confirmed, full-distribution
equivalence, filter exactness or ranking, calibration, cross-fixture
robustness, production readiness, or default readiness. Do not treat a target,
filter, comparator, source-route, or implementation blocker as evidence that
NeuTra or the scientific direction is invalid.
