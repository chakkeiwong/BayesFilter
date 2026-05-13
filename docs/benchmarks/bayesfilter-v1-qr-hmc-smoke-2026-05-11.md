# BayesFilter v1 QR HMC Smoke

Purpose: record a narrow CPU-only TFP HMC smoke for `linear_qr_score_hessian_static_lgssm` using QR value autodiff for the sampler gradient.

## Claim Scope

This artifact is a first target-specific HMC smoke.  It is not a general HMC readiness claim for all BayesFilter filters.

## Result

The JSON file is authoritative: `docs/benchmarks/bayesfilter-v1-qr-hmc-smoke-2026-05-11.json`.

| Status | Acceptance | Nonfinite Samples | Initial Grad Finite |
| --- | ---: | ---: | --- |
| ok | 1.0000 | 0 | True |

## Interpretation

Passing means this one small QR target can run a fixed-seed HMC smoke with finite samples and diagnostics.  Broader posterior recovery, tuning, GPU, DSGE, MacroFinance switch-over, and SVD-CUT HMC remain separate gates.
