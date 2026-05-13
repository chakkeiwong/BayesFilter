# Advanced particle filter kernel PFF debug-gate result

## Date

2026-05-11

## Scope

This report covers MP3 of the quarantined student DPF experimental-baseline
stream.  It narrows the prior kernel PFF timeout/failure classification
using reduced bounded diagnostics.  It does not promote kernel PFF into
routine comparison panels.

## Overall

- runs: 16
- ok: 16
- failed: 0
- readiness decision: `excluded_pending_debug`

## Kernel Summary

| Kernel | Runs | Median runtime seconds | Median avg iterations | Max hit-max fraction | Median RMSE vs KF |
| --- | ---: | ---: | ---: | ---: | ---: |
| matrix | 8 | 0.12948 | 40 | 1 | 0.129901 |
| scalar | 8 | 0.147597 | 40 | 1 | 0.12211 |

## Tolerance Summary

| Tolerance | Runs | Median runtime seconds | Median avg iterations | Max hit-max fraction | Median final flow magnitude |
| --- | ---: | ---: | ---: | ---: | ---: |
| loose | 8 | 0.149057 | 40 | 1 | 0.0442958 |
| strict | 8 | 0.128584 | 40 | 1 | 0.0442958 |

## Hypothesis Results

### k1_reduced_fixtures_runnable

Supported.  Reduced scalar and matrix kernel PFF runs completed on both reduced fixture families.

### k2_tolerance_sensitivity

Not supported for this bounded panel.  Loose tolerance did not reduce median iterations or runtime.

### k3_max_iteration_failure_mode

Supported.  Non-converged behavior appears as finite completed runs with `hit_max_iter` diagnostics, not missing dependencies.

### k4_routine_panel_readiness

Supported.  Kernel PFF should remain excluded from routine panels pending further debug.

## Interpretation

A completed filter run is not the same as converged flow iterations.
Runs that complete while every step hits `max_iterations` remain
debug evidence only.  Kernel PFF should not enter routine panels
unless bounded convergence is consistent across reduced cases.
