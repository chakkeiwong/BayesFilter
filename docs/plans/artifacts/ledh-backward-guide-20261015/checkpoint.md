# Backward-guide integration checkpoint

- Active checkout: `ledh-refactor-with-policy-fix`,
  `/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922`.
  HEAD `ea1018de1da55ed02a29a8d60f08a6827fce8a42`.
- The earlier scoped surrogate commit `ad5de4a6d` was integrated into main
  and synchronized to LEDH. That synchronization is finished.
- Historical integration evidence remains in `result.md` and
  `run-manifest.json` here. Subsequent actual nonlinear flow-Jacobian proof
  and tests are under `../ledh-backward-guide-jacobian-20261016/`.
- Latest stage: replicated scalar fitted-guide export, shared consumer and
  bounded GPU/XLA reference comparison complete; 44 tests pass, 624 rows
  audited. All conditions fail heuristic promotion; no trained KDM or
  nonlinear score-improvement evidence. The manuscript now builds to 65 pages.
- **Current question, findings, budget and exact next action:** read
  `../ledh-fitted-guide-20261016/checkpoint.md`; full evidence and decisions
  are in `result.md` in that directory. The next question concerns the
  Gaussian baseline's model-score error before broader nonlinear fitting.
- Preserve earlier uncommitted proof/tests and unrelated surrogate-hmc edits.
  No new commit, synchronization or push is pending from this stage.
