# BayesFilter v1 SVD-CUT Branch-frequency Diagnostic

Purpose: record branch frequencies for a tiny SVD-CUT parameter box and a weak-gap control.

## Claim Scope

Diagnostic branch-frequency artifact only. SVD-CUT HMC remains blocked pending a separate target-specific HMC plan even if the tiny smooth box is all smooth.

## Result

JSON artifact: `docs/benchmarks/bayesfilter-v1-svd-cut-branch-frequency-diagnostic-2026-05-11.json`.

```text
smooth_box_total = 3
smooth_box_smooth = 3
smooth_box_weak_gap = 0
smooth_box_active_floor = 0
smooth_box_min_placement_gap = 0.0218990051258
smooth_box_min_innovation_gap = inf
smooth_box_max_support_residual = 0
smooth_box_max_deterministic_residual = 0
weak_gap_control_total = 2
weak_gap_control_weak_gap = 2
```

## Interpretation

The tiny smooth box is smooth under the chosen tolerance, while the repeated-spectrum control is blocked by weak spectral-gap diagnostics.  This closes branch-frequency telemetry for this phase but does not promote SVD-CUT HMC; it only supports writing a later target-specific SVD-CUT HMC plan if desired.
