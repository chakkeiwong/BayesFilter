# BayesFilter v1 SVD-CUT Branch Frequency

Purpose: quantify SVD-CUT derivative branch labels over a small parameter box without promoting SVD-CUT to HMC readiness.

## Claim Scope

Diagnostic artifact only.  A smooth fraction of one on this tiny box is not a general SVD-CUT HMC claim.

## Result

The JSON file is authoritative: `docs/benchmarks/bayesfilter-v1-svd-cut-branch-frequency-2026-05-11.json`.

| Status | Smooth | Active Floor | Weak Gap | Smooth Fraction |
| --- | ---: | ---: | ---: | ---: |
| diagnostic_smooth_box | 3 / 3 | 0 | 0 | 1.0000 |

## Interpretation

The result quantifies this parameter box only.  SVD-CUT HMC remains blocked pending target-specific sampler evidence and broader branch coverage.
