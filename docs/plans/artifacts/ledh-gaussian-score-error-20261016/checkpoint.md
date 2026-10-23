# Active checkpoint

- Question: why is the Gaussian LEDH baseline score inaccurate before adding
  more learned guidance? Stage: partial diagnostic closed at a validity veto;
  CPU localization and observability repair complete. See `result.md`.
- Branch: `ledh-refactor-with-policy-fix`; checkout
  `/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922`.
  Base HEAD `ea1018de1da55ed02a29a8d60f08a6827fce8a42`; preserve prior dirty work.
- Master: `../../younis-kdm-score-master-program-2026-09-14.md` now resumes LEDH
  improvement; bounded replication remains closed. Previous stage evidence:
  `../ledh-fitted-guide-20261016/result.md` (624 rows; no promotion).
- Plan: `../../ledh-gaussian-score-error-20261016.md`; attempt02 stops at
  255/280 rows (102 LEDH). Both six-coordinate derivative fixtures pass,
  maximum scaled error 2.459e-9. Persistent model-score errors remain large;
  all conditional heuristic promotion screens fail. No statistical ranking.
- Failed cell: persistent/dataset3/stream1/N128/flow48. CPU XLA and non-XLA
  replay identify time5 column-TV residual 0.00027054666504 > 0.0001. The
  guard correctly rejects; all other validity flags pass. Failed GPU tensors
  were not archived, so their equality to the CPU reconstruction is unproved.
- Observability repair: preserve shared validity flags and column-TV residual;
  archive rejected evaluations before raising. No guard/control changed.
  CPU prior diagnostics have zero difference, eight focused and seven
  shared-consumer tests pass. Evidence: `cpu-veto-observability/manifest.json`,
  `observability-repair.patch`, and the three test XML files.
- Reset weighted moments restore to machine precision, but concentration and
  covariance error already occur before the first reset. Do not blame reset
  alone. Flow48-to192 barely changes persistent errors.
- Budget: GPU 769.056/1,200s used, both allowed launches consumed; no third
  launch. Timed CPU work including document builds is 228.703/600s;
  unmetered inspections mean remaining CPU is less than 371.297s. Attempt01's
  `running_grid` status is stale after exit130. No campaign process remains.
- Exact next action: a separately bounded terminal-balancing calibration,
  keeping epsilon1/Sinkhorn5 and the validity threshold unchanged; assess
  numerical non-harm, not score-MSE selection. Then diagnose first-step
  physical-prior/proposal weighting against the exact Gaussian conditional
  before a fresh guide comparison. Nonlinear/KDM training remains later.
- `postrun-audit.json` verified sources before observability edits. Its
  source-match result must not be attributed to the newer working tree.
  `run-source.patch` reconstructs the run-time sources from the base commit;
  `run-source.json` records all 253 verified source hashes.
- Manuscript updated and 66-page PDF built, without final-pass warnings;
  changed pages 60/63 visually inspected. See `latex/`. MathDev audit remains
  inconclusive; no new independent mathematical audit is claimed.
- Final verification: `final-validation.json` passes; branch, tests, rejected
  non-finite serialization, source preservation and PDF checks are recorded.
- No tuning/default, nonlinear, trained-KDM, HMC or superiority claims; no new
  commit, push, branch switch or synchronization requested.
