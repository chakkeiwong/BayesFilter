# Phase 0 Complete — 2026-09-07

**Status:** PASS

## What was done

- Implemented `derive_parity_tolerance` / `derive_golden_master_tolerance` /
  `condition_number_from_matrix` in `bayesfilter/inference/tolerance_derivation.py`,
  giving one derived tolerance for all parity and golden-master checks instead of
  the two incompatible regimes (1e-12 vs 5e-4) the earlier programs carried.
- Wrote the seed-policy memo binding one frozen master ω to Corollary 5.2 and
  Remark 5.3, with the branch-discontinuity proposition as the reason θ-hashed
  seeding is inadmissible rather than merely inconvenient.
- Implemented `joint_mahalanobis_coverage` and `marginal_coverage_diagnostic` in
  `bayesfilter/inference/coverage.py`, replacing the conjunction-of-marginals
  criterion that fails a correct sampler ~23% of the time at P=5.

## Tests passed

21 of 21, CPU-only.

- `tests/inference/test_tolerance_derivation.py` — 10 passed (2.01 s)
- `tests/inference/test_coverage.py` — 11 passed (5.52 s)

The load-bearing one is `test_joint_region_is_calibrated_across_replications`:
200 independent replications of a correct sampler, joint region covers at ~0.95.
That is the check that justifies the Option A coverage decision rather than
asserting it. `test_marginal_conjunction_is_weaker_than_joint_region` pins the
0.95**P defect so it cannot be reintroduced.

## Repairs made (2 attempts, no approval needed per execution contract)

1. **NumPy in a runtime path.** The first draft of `tolerance_derivation.py`
   used `np.linalg.svd`. Tolerance derivation is admission logic on the runtime
   path, not diagnostic code, so the CLAUDE.md Backend Rule forbids NumPy there.
   Replaced with `tf.linalg.svd`; renamed `derive_tolerance_from_svd` to
   `derive_tolerance_from_matrix`.

2. **Wrong citation in my own handoff note.** The Phase 0 handoff pointed at
   `contract-e-jvp-serial-bottleneck-analysis-2026-09-03.md` for Corollary 5.2.
   The actual source is
   `docs/bayesfilter-genut-score-variance-problem-and-repair-note-2026-07-31.tex`
   (corollary at line 953, requirements remark at 980, branch proposition at
   1166). The memo cites the verified source with line numbers.

## Artifacts

- `bayesfilter/inference/tolerance_derivation.py`
- `bayesfilter/inference/coverage.py`
- `docs/memos/ledh-surrogate-hmc-seed-policy-2026-09-07.md`
- `tests/inference/test_tolerance_derivation.py`
- `tests/inference/test_coverage.py`
- `results/phase0-summary.json`

## Budget consumed

GPU-hours: 0.0 of 44. Repair attempts: 2 of 20.

## Next

Phase 1 — JVP parity, Sinkhorn convergence, GPU memory-growth verification.
See `docs/plans/phase1-execution-handoff.md`.
