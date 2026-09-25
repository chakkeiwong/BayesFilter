# Master-program amendment checkpoint

- Active request: apply the agreed Claude/Codex corrections to the master
  program and review the amended specification thoroughly.
- Checkout: `/home/chakwong/BayesFilter`, branch `surrogate-hmc`, HEAD
  `5836f0344293f1c4af85abba23689ba83d34d9af`. Unrelated dirty code, policy,
  tests, and other agents' documents are preserved.
- Stage: master amended and reviewed; baseline saved as `master-before.md`.
- Findings: separate deterministic stencil checks from stochastic MSE;
  include the bias cross-term and measured coupling covariance; correct
  smoothness/direction solve; move tuning and baseline preparation earlier;
  specialize identity tests by target; preserve active KDM/proposal/FD scope.
  Claude's bounded review did not approve its explicitly unchecked sections.
- Validation: `validation.json` and `verification-summary.json` record passing
  exact-arithmetic, document-anchor, and all nine local-link checks; the complete
  protected diff is `master-amendment.diff`. The 22-page `master-preview.pdf`
  built with PDFLaTeX; amended mathematical/OT pages 12, 16, and 17 were viewed.
- Review: `docs/reviews/younis-score-master-program-amendment-review-2026-09-14.md`.
  Claude's scope clarification is accepted. Stencil order is separated from
  stochastic MSE, tuning and target-specific tests precede claims, and KDM
  centering/SGQF/twist prerequisites are explicit.
- Remaining implementation obligations: repair/recheck KDM trace/callback
  compatibility; verify parameter-dependent initialization and integrated SGQF
  moment/sensitivity lifecycle; derive selected twist corrections. The companion
  LaTeX still needs the named localized corrections before serving as the revised
  implementation specification.
- Scope used: local document work, source inspection, and small independent
  reference checks. No particle campaign, GPU execution, runtime changes,
  external review, or MathDevMCP audit was performed.
- Exact next action: deliver the amended master, PDF preview, and review. On
  future execution, recheck this source snapshot and start with Phase 0 exact
  Gaussian mechanics and the per-row prerequisites, not deferred smoothing/DSGE.
