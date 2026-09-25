# Canonical LEDH Leaderboard Rerun Plan (Part 4)

Date: 2026-08-22. Status: slice 1 executed this session; full six-model
execution gated on remaining model onboardings (ledger).

## Contract

- Question: how do the canonical lane's VALUE and SCORE compare, per model,
  against (i) exact references where they exist, (ii) the repository
  comparator algorithms (fixed_sgqf, ukf, zhao_cui reference), and (iii)
  the bootstrap comparator for proposal-quality context?
- All prior LEDH leaderboard rows are invalidated (2026-08-21 notice);
  this rerun starts a fresh artifact lineage with the conformance stamp.
- Statistical discipline: hard vetoes first (nonfinite/invalid rows);
  descriptive tables with per-seed spread; NO ranking language without
  uncertainty support; exact-reference deltas are the only absolute
  correctness claims, and only where the reference is exact (LGSSM).
- Value comparisons: absolute log-likelihood error vs exact reference
  (LGSSM); cross-algorithm descriptive tables elsewhere.
- Score comparisons: canonical analytical score vs autodiff oracle
  (self-consistency, rtol declared 1e-4) and vs exact Kalman score (LGSSM,
  derivable); descriptive cross-model tables elsewhere.
- Arms per model: canonical (CPU float64 reference semantics now; GPU
  float32/TF32 arms enter with the P6 calibrated lane), bootstrap
  comparator, and the repo comparator algorithms as onboarded.
- Seeds: >= 3 per cell for slice 1 (descriptive); the full campaign uses
  16 paired seeds per the SQMC precedent.
- Artifacts: `docs/benchmarks/artifacts/ledh_canonical_leaderboard_2026-08/`
  fresh root, manifest with git commit, environment, seeds, and the
  conformance suite version.

## Slice 1 (this session): LGSSM + Austria, CPU float64

Executed by `docs/benchmarks/run_ledh_canonical_leaderboard_slice1.py`.
Cells: LGSSM value-vs-Kalman (3 seeds), LGSSM score-vs-oracle, Austria
canonical-vs-bootstrap ESS and value spread (3 seeds). Results table in
the session report and artifact JSON.

## Full campaign (pending): remaining four models

Blocked on: predator-prey + 3 SV model onboardings (canonical callbacks +
per-model S-1-class gates), comparator-algorithm harness hookup, P6
calibration for the GPU arms. Execution order per the master plan ledger.
