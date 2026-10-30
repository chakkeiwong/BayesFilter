# Observation-weighted parents: SIR comparison, 28 October 2026

Plan: `../../observation-parent-default-and-sir-accuracy-20261028.md`.
Result: `../../observation-parent-sir-accuracy-result-20261028.md`.

- `pilot02/`: numerical smoke, prefix checks, first nominal stream (18 rows).
- `campaign01/`: remaining paired filter measurements; combine with pilot02.
- `reference01/`: two saved Zhao-Cui fits at the original seven parameter points.
- `reference-local01/`: four local rate perturbations added after overlap failure.
- `analysis01/`: final deduplicated grid, paired inference, conditional heuristic checks, figures.
- `attempts.json`: failed/repaired attempts, tests, and explicit budget accounting.
- `logs/`: complete command logs.
- `source-snapshot-final/`: captured numerical/reporting sources at that stage; each launch manifest remains authoritative for its own sources.
- `compatibility-before/`, `compatibility-after/`: narrowly changed historical consumers, preserving prior and final versions.
- `manuscript-baseline/`: protected manuscript before this task.
- `delivery-final/`: final document/reporting snapshots, hashes and scientific-source verification.
- `budget-final.json`: final conservative accounting within the 7200-second campaign cap.

Reference validity and parameter cases must remain attached to accuracy tables.
The 13 failed-reference cells are completed measurements with an unusable
comparator, not missing runs and not evidence against the particle filters.
This is a fixed-data, untuned FP64 GPU/XLA diagnostic; canonical admission,
HMC readiness, and production TF32 accuracy are separate questions.
