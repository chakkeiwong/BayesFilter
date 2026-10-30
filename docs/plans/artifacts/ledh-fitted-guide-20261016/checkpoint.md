# Active checkpoint

- Question: does replicated scalar controlled-SMC fitting transfer correctly
  as future-only guidance, and does it improve the model-score estimate?
- Checkout: `ledh-refactor-with-policy-fix` at
  `/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922`;
  HEAD `ea1018de1da55ed02a29a8d60f08a6827fce8a42`.
- Stage complete: actual scalar fit/export and dynamic guide use the full
  shared analytical LEDH executor. Seven new tests and 37 regressions pass.
  Attempt02 completed 624 GPU/XLA FP64 reference rows after the bounded
  24/48-step numerical repair. Post-run consistency and source-hash checks pass.
- Scientific decision: every condition fails heuristic promotion. Informative
  guidance has a favorable exploratory comparison with plain LEDH, but loses
  to simple particle comparators. Validation variation is large. No nonlinear,
  trained-KDM, canonical, default, HMC or superiority claim is supported.
- Coordinate correction: primary score is log process-noise SD; original
  `log_q` JSON keys are preserved legacy aliases. See `postrun-audit.json`.
- Documentation: 65-page PDF built and changed proof pages inspected.
  MathDev derivation audit is inconclusive; structural-tool limitations and
  local executable equation/call-chain audit are in `math-code-audit.md`.
- Evidence: `result.md`, `attempt02/manifest.json`, `postrun-audit.json`;
  plan `../../ledh-fitted-guide-pilot-20261016.md`.
- Budget: GPU used 413.140/1,800 seconds, 1,386.860 unused. Recorded CPU tests
  used 165.354 seconds and post-run inspection 1.943 seconds; short LaTeX
  builds were not separately timed, so CPU remainder is at most 1,032.703
  seconds. Two of three repair allowances used. No campaign process remains.
- This stage's next diagnostic has now run: see
  `../ledh-gaussian-score-error-20261016/result.md` and its active checkpoint.
  It stopped at 255/280 rows on a transport-balancing validity failure;
  finite-program derivatives pass but persistent model-score errors remain.
  Follow that checkpoint for balancing repair and first-step weighting
  diagnosis. Preserve this pilot's untouched results and frozen controls.
- Preserve all uncommitted work, including the preceding Jacobian proof/tests,
  and unrelated dirty surrogate-hmc work. No new commit or push was requested.
