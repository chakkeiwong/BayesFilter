# BayesFilter v1 Linear QR HMC Readiness Smoke

Purpose: record a target-specific finite-diagnostic and tiny HMC smoke for `linear_qr_score_hessian_static_lgssm`.

## Claim Scope

Target-specific finite-diagnostic and tiny HMC smoke only; no convergence, production sampler, GPU/XLA, MacroFinance, or DSGE claim.

## Result

JSON artifact: `docs/benchmarks/bayesfilter-v1-linear-qr-hmc-readiness-smoke-2026-05-11.json`.

```text
initial_target_log_prob = -1.3568111688
initial_gradient = [-0.40893740245060484, -1.0379645721254034]
value_residual = 0
score_residual = 4.441e-16
hessian_residual = 6.661e-16
hessian_symmetry_residual = 0
negative_hessian_eigenvalues = [2.3467420404247332, 2.998234138312041]
acceptance_rate = 1.0000
finite_sample_count = 12
nonfinite_sample_count = 0
sample_mean = [0.25445034129324, -0.8762790772857921]
sample_stddev = [0.16146046675518105, 0.0957829416423149]
```

## Interpretation

The fixed CPU-only smoke finished with finite target, gradient, curvature, and sample diagnostics.  This supports the first target-specific HMC readiness smoke only; it is not a convergence claim and does not promote SVD-CUT, DSGE, MacroFinance, GPU, or XLA readiness.
