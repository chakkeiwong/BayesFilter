# Completed nonlinear screening — 2026-10-02

Question: T20 PP/SIR-d18 likelihood/full-score stability and accuracy.
Plan: docs/plans/ledh-nonlinear-execution-20261002.md.
Results: docs/benchmarks/ledh-nonlinear-execution-results-20261002.md.
Artifacts: docs/plans/artifacts/ledh-nonlinear-execution-20261002/.
Branch sqmc-development; base 74bdc7777. Scientific screening finished.
GPU wall 1289.335/3600 seconds, 6/8 launches; no jobs running.
CPU focused checks 25 passed; independent reference 3 passed (overlapping).
PP 70/70 finite, 69/70 valid. SIR FP32 0/10 valid. Same-input FP64 4/4 finite,
3/4 valid, but logL -969..-1042 versus approximate reference -678.0775 (MCSE .0149).
SIR first failure: observation4 Contract E reset; upstream UKF/flow finite,
covariance-only reset states finite but tangents nonfinite; guarded reset input invalid.
Reference particle ladder N8192/32768/131072/524288, 4 reps each; no exact oracle,
no scope tuning, no supported ranking or admission. All raw values/scores retained.
Monograph updated with reference derivation and actual findings: 606 pages,
no unresolved references/citations, changed pages rendered and inspected.
Scientific work and documentation complete. Git delivery uses the live refs.
Next scientific work: reset factorization/tangent localization and protection
evaluation, then fresh calibration/holdout partitions.
Do not rerun completed screens or relax trace tolerances. Live Git refs and
execution-manifest.json are authoritative for delivery and compute accounting.
