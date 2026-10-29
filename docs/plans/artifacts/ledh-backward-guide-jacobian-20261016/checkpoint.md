# Active checkpoint

- Question: is the guided shared flow's recorded Jacobian the actual innovation
  Jacobian, with its full analytical parameter derivative?
- Checkout: `ledh-refactor-with-policy-fix`, initial `ea1018de1da55ed02a29a8d60f08a6827fce8a42`;
  `/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922`.
- Checked: nonlinear endpoint innovation Jacobian, positive determinant and total
  parameter derivative pass finite differences at both observations. The full
  focused regression passes 37 tests. No runtime algorithm was changed.
- Stage: complete. Proposition/proof added; MathDevMCP label audit is inconclusive
  for all ten obligations (seven not encodable, three not extracted). The
  64-page manuscript builds cleanly; proof pages 57 and 58 were visually checked.
- Evidence: `docs/plans/ledh-backward-guide-jacobian-validation-20261016.md`;
  prior integration evidence remains in `../ledh-backward-guide-20261015/`.
- Evidence for this stage: `result.md`, `run-manifest.json`,
  `integration-tests.xml`, `mathdev-label-audit.json` and the built PDF here.
- Budget: 193.17 timed test/build seconds against 1,200 seconds; untimed
  bibliography, rendering and review recorded separately. Two of three local
  repairs used: executed-input capture and pdflatex/bibtex for absent latexmk.
- Next: design the separate scope-specific guide-fitting and untouched
  target-score experiment, including tuning, conditional heuristic comparisons
  and uncertainty. No such campaign is authorized by this bounded plan.
- Limits: no model-score improvement, GPU/XLA, default, canonical or HMC claim.
  No commit, push or branch change in this continuation. Preserve unrelated
  surrogate-hmc work and continue development in this LEDH checkout.
