# KSC information diagnosis checkpoint — 18 October 2026

- Checkout: ledh-iapf-execution-20260922, branch ledh-refactor-with-policy-fix,
  HEAD ea1018de1da55ed02a29a8d60f08a6827fce8a42. Preserve unrelated dirty work.
- Research question: can future-message guidance improve the finite-particle
  likelihood score? The fitted-guide campaign failed promotion; the present
  stage diagnoses its consumer, not fitting capacity or canonical admission.
- Completed: 48 saved-fit/control rows and four independent CPU tests pass,
  including actual build_maps global/local wiring. Positive mixture information
  J exceeds the negative log-message Hessian by the component-score covariance.
  For homoscedastic kernels the global consumer uses C0, whereas the exact
  product with the same Gaussian has variance C0+Var_alpha(b_k).
- Evidence: result.md, attempt-01/{manifest,result}.json and rows.json.
  Product/consumer variance ranges are 4.73–6.18 ordinary, 4.06–6.04 negative
  tail and 2.45–4.77 positive tail. Negative negative-Hessians at 252/792 fitted
  anchors rule out unexamined direct Hessian substitution. Maximum relative
  consumer-identity error 1.43e-15; homoscedastic gap error 2.22e-16.
- Interpretation: omitted local mixture uncertainty is confirmed. Its causal
  contribution to score error, or improvement from restoring it, is unproved.
  The exact product is with an approximate current-observation Gaussian, not
  the exact filtering/smoothing distribution. No stochastic ranking attempted.
- Budget CLOSED: one attempt, 6.013315/180 CPU diagnostic seconds; GPU hidden
  before TensorFlow import. The prior fitted campaign is also closed at
  236.915905972/900 GPU-process seconds and 2/2 launches. No running process;
  unused stage budgets do not transfer to a new experiment.
- Math audit: explicit proposition/proof added to ledh_backward_guidance.tex.
  MathDev/SymPy checks scalar algebra. Implementation audit remains unverified
  because its proof parser is inconclusive; do not call it a full certificate.
  Preserve mathdev-audit.json and mathdev-scoped-audit.json.
- Manuscript and masters refreshed. The final PDF builds cleanly at 73 pages;
  proof/result/next-study pages inspected. Build evidence and hashes are in
  document-build/review.md. Documentation closeout is complete.
- Exact next action: derive general Gaussian-product mixture moments and
  analytical derivatives for an optional shared global-map consumer. Audit
  source/scoping and the actual call chain, write a bounded implementation and
  fresh downstream-comparison plan, then execute. Keep existing local maps
  explicit; do not silently introduce indefinite information. Freeze fits for
  each declared derivative, use independent fit/cloud seeds for confirmation,
  and retain the same unguided/heuristic comparators. Saved final fits may
  motivate the repair but cannot confirm it.
- Preserve OT epsilon 1/five iterations as the owner's untuned comparator.
  No original KDM training, full twisted SMC, canonical UKF/dual-cap/Contract-E
  admission, default change or HMC conclusion follows. No commit/push requested.
