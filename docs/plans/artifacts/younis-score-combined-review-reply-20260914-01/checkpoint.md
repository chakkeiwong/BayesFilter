# Combined score-program review checkpoint

- Active request: assess the new Claude proposal/finite-difference review,
  combine it with the previous assessment, propose master-program changes,
  and write a Markdown reply to Claude.
- Stage: reply and proposed amendment sequence complete; document and
  exact-arithmetic reference checks passed.
- Checkout: `/home/chakwong/BayesFilter`, branch `surrogate-hmc`, HEAD
  `5836f0344293f1c4af85abba23689ba83d34d9af`; unrelated dirty implementation,
  governance, and oracle-test files are preserved.
- Findings: accept deterministic/stochastic FD test separation; correct the
  review's MSE cross-term omission, universal U-curve claim, rectangular
  direction solve, fixed roundoff/conditioning thresholds, and selection
  criteria. Carry forward ratio-bias, OT, initial-law and covariance findings.
  Phase dependencies need target-specific tests, early tuning/baselines,
  explicit 4B/4C entries, and exclusion of separately assigned smoothing/DSGE
  programs. Current KDM trace/callback conflict is still visible statically.
- Evidence: new review under `docs/reviews`, existing assessment under
  `docs/plans`, current master, current handoff, and selected TeX/code anchors.
- Scope: document assessment and change proposal only. No master, manuscript,
  or runtime edits; no experiments or external reviewer calls are scheduled.
- Budget: local reads and a small exact-arithmetic reference check only;
  no research campaign, GPU use, or mathematical-audit claim.
- Deliverable:
  [reply to Claude](../../../reviews/younis-score-proposal-fd-codex-reply-2026-09-14.md).
- Validation: `derivation-checks.json` and its saved command log record exact
  stencil moments/coefficients, MSE and ratio-bias counterexamples, shared-noise
  cancellation, and rectangular reconstruction. `validation.json` records
  six resolved reply links, nine source hashes, and the unchanged master hash.
  These checks do not establish filter runtime correctness or a math-audit pass.
- Exact next action: deliver the reply and proposed changes. Applying the
  master/manuscript amendments is the next substantive task; no experiment or
  external reviewer invocation is pending in this writing task.
