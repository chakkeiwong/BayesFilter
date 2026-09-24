# A10 smoke result

GPU/XLA smoke-02 COMPLETE, 169.253896 seconds, RTX 5080 with verified memory
growth. Reference-free mechanism checks only. All protected arms completed
healthy d1/d4 and exposed A09 d4-s04/s10, with zero nonfinite consumer steps,
zero inverse-CDF bracket failures, maximum CDF residual 1.11e-15. Original
guide and baseline fail on s10 and are preserved as failures.

On s04 the original minimum chart eigenvalue is 3.7164e-32, repaired-guide
minimum .27047, stable-chart minimum .80365. Positive Laplace quadrature was
used at 4/20 s04 and 9/20 s10 steps. These are mechanism diagnostics, not
accuracy or superiority evidence. Stable coordinates affect TT fitting after
the initial analytic Gaussian step.

Validation: tests-03.log records 37 passing focused tests. Smoke-01 was a
harness/reporting failure: undefined single-repetition MCSE rejected by strict
JSON. Four repetitions repair it. Preserve that attempt; reserve 150 seconds
conservatively because its original manifest finalizer also failed.

Decision: proceed to planned calibration. Safety mechanics pass; filtering
accuracy, statistical comparison, high-dimensional scaling and HMC remain
unestablished. See attempt-smoke-02/run_manifest.json and per-sequence summary
files under docs/benchmarks/artifacts/observation_tt_robust_guide_20260916/.
