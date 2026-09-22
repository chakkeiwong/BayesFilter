# A10 calibration and pre-confirmation review

attempt-calibration-01 COMPLETE in 581.310050 seconds on RTX 5080, float64,
XLA and verified memory growth. All six sequence references pass. Every one
of the 15 arms per sequence completes, with no recorded arm failure.

Frozen controls, written by the driver before confirmation:

| Dimension | Repaired-guide TT L1 | Stable-chart TT L1 | Physical defense epsilon |
|---|---:|---:|---:|
| 1 | .00001 | .001 | .05 |
| 4 | .001 | .001 | .05 |

All selected controls are calibration-admissible. Epsilon is the smallest
passing weight, not the empirical MSE optimizer. Its conditional second-moment
multiplier bound relative to transition sampling is 20. On the d4 calibration
set epsilon=.5 fails the 10% non-harm screen; .05 and .2 pass. This is a bounded
calibration curve, not statistical superiority evidence or a universal default.

Pre-confirmation skeptical self-review: PASS. Target and exact-weight consumer
are unchanged; all three sequences per dimension remain in selection; controls
are frozen; confirmation blocks 24--47 are disjoint from calibration blocks
0--5; reference precision, log-evidence and numerical checks remain active.
The plan's unchanged comparison and stop conditions govern the next run.
No confirmation results have been inspected or used for selection.

Evidence: docs/benchmarks/artifacts/observation_tt_robust_guide_20260916/
attempt-calibration-01/{run_manifest.json,selected-controls.json,result.json}.
