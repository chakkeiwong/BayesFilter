# Executed observation-guided TT implementation audit

2026-09-14. The master is `docs/benchmarks/run_observation_aware_tt_complete.py`; numerical code is `bayesfilter/highdim/observation_guided_tt_tf.py`, using shared Hermite density/inversion in `c2_gaussian_hermite_proposal_tf.py`. Classification: **extension_or_invention**. SGQF coordinates, fixed Hermite/L1 fitting and the defensive mixture are proposal extensions, not the author's TT-cross implementation.

The old scalar driver and run-01 through run-20 PASS labels are **wrong as evidence of a complete, exactly corrected remedy**: wrong initial prior, bounded support, dependent deterministic innovations and an incomplete retained-TT lifecycle. The prior audit is preserved as `implementation-status-audit-before-complete-20260914.md`; its completion claims are withdrawn.

| Operation | Actual call chain | Executable evidence |
|---|---|---|
| Recursive observation guide | `execute` → `build_guide_path` → `sgqf_update` → repository `tf_fixed_sgqf_cloud` | Observation-magnitude and history-response tests; both complete fixtures pass rough-reference screen |
| Signed quadrature validity | `sgqf_update` positive mass; `Chart.from_moments` finite covariance and relative SPD margin | SPD rejection test; all attempted rules retained |
| Affine pullback | `build_tt_path.log_target`: prior at zero; then previous retained TT × transition × likelihood × both Jacobians / reference density | Two-step wiring test, explicit formula review, completed T20 fits |
| L1-tuned TT regression | `fit_amplitude` → cached `compiled_fitter` | Separate training/validation/audit rows, core objective checks, per-time fit records |
| Full joint and retained marginal | `TTStep.retained` contracts suffix Gram → `GaussianHermiteRetainedProposal` → next target's physical density | Independent quadrature of nonuniform joint agrees with retained marginal |
| Upper conditional KR | `particle_filter` → `sample_tt_step` → `compiled_conditional_sampler` → shared `inverse_hermite_polynomial_kr(reverse=True)` | Nonseparable multivariate normalization, suffix response, CDF derivative/order and explicit monkeypatch wiring tests |
| Actual mixture density | Conditional polynomial/Gaussian mixture plus inverse affine determinant | Integration and pointwise density checks; bracket/CDF checks on each draw |
| Original-model correction | `filter_kernel`: p0g/q at zero; carried weight × fg/q later; log normalization; random systematic resampling after correction | T20 mean/evidence screens against scalar grid and larger particle reference |
| Frozen analytical score | `frozen_score_check`: stationary prior derivative, three controls, recursive weight derivative/reset | Same-scalar centered differences: relative errors 5.16e-10 and 4.22e-10 |
| Conditional comparisons | `summarize` constructs three simple proposals and two TT arms by observation regime | Both `decision.json` files; no ranking established; promotion veto recorded |
| GPU/XLA | Stable-signature fitter, conditional sampler and particle-update kernels; memory growth before device initialization | CPU/GPU parity within 1.73e-15; RTX5080 full campaign 99.574 seconds |

Review traced claim-bearing consumers; function existence alone was not counted. Ten focused tests passed. All three executed Python source hashes and the frozen plan match `campaign-01/run_manifest.json`. No NumPy or pfor computation was introduced. Setup and independent references outside compiled kernels are explicit exceptions; the entire driver is not one XLA graph.

Limits: Gaussian closure propagates mean/covariance, despite third/fourth quadrature diagnostics; fitting error remains; full support does not bound weights; bisection is a checked numerical inverse, and zero/invalid polynomial conditional mass fails closed. The score differentiates a frozen finite recursion, not retraining or exact likelihood. Only d=1,d=4 on one observation sequence were evaluated. Comparative benefit, higher-dimensional scaling and HMC remain open.

Result: `docs/plans/observation-aware-tt-repair-complete-program-20260913-result.md`. The integrated derivation/proofs are in `sec:executed-sgqf-tt` of the attempt05 LaTeX/PDF. MathDevMCP review files are in `mathdev-complete-20260914/`; their coverage and limitations are part of the review.
