# Active checkpoint — 16 October 2026

- Question: do fitted future messages improve exact Gaussian model scores
  through the existing three-proposal mixture, not the previous single flow?
- Branch `ledh-refactor-with-policy-fix`, worktree
  `/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922`,
  HEAD `ea1018de1da55ed02a29a8d60f08a6827fce8a42`; preserve dirty work.
- Current plan: `../../ledh-guided-mixture-score-comparison-20261016.md`;
  terminal audit/result `proportionality-result.md`, machine decision
  `decision-proportionality.json`. Earlier `result.md` is historical.
- Owner correction: stopping for about 1e-13 roundoff was disproportionate.
  The active plan amendment replaces the unsupported machine-epsilon cutoff
  with an explicit half-significand relative moment budget; old cutoffs become
  recorded warnings. This changes the validity screen, not the value/score
  algorithm. No reset-stabilization campaign or score tuning is justified.
- Engineering:12 focused tests pass (`cpu07/`), saved stress replay completes
  (`cpu09/`); cpu08 preserves a missing-PYTHONPATH invocation. No numerical
  update/derivative change. Max covariance residual across GPU rows2.138e-13.
- Comparison complete: attempt02 has392 valid rows; all336 prior accepted
  scores, likelihood errors and betas are bitwise unchanged. Only one old-
  cutoff warning. Source closure unchanged; epsilon1/iterations5 retained.
- Scientific result: fitted guidance fails descriptive heuristic promotion;
  intervals versus current/global include zero in all four conditions.
  Fitted and exact-future scores agree within1.24e-11. No demonstrated score
  improvement, overall inferiority, canonical admission or direction rejection.
- Budget charged156/180 CPU and664/900 GPU seconds conservatively; attempt02
  took411.314 seconds on the shared RTX4080 SUPER with memory growth/XLA.
  All processes finished. Preserve old and new evidence; no active job.
- Exact next action: design a fresh paired current/exact-future comparison
  with beta held fixed versus calibrated beta, retaining the Kalman oracle and
  heuristic set. This isolates guidance from mixture selection before adding
  KDM capacity. No reset-precision campaign is justified by1e-13 roundoff.
