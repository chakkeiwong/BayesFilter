# BayesFilter V1 Model B/C Final Testing Summary

## Date

2026-05-15

## Controlling Documents

Master program:

```text
docs/plans/bayesfilter-v1-model-bc-thorough-testing-master-program-2026-05-14.md
```

BC8 plan:

```text
docs/plans/bayesfilter-v1-model-bc-bc8-consolidation-release-gate-plan-2026-05-14.md
```

## Claim Scope

This summary consolidates BC0-BC7 evidence for Models B and C.  It supports
Model B/C value and analytic-score readiness over the tested boxes and
diagnostic readiness classifications for HMC and GPU/XLA.  It does not certify
exact full nonlinear likelihood, HMC convergence, broad GPU speedup, nonlinear
Hessian production support, external client switch-over, or production GPU
policy.

## Validation Gates

Fast public API:

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_v1_public_api.py -p no:cacheprovider
```

Result: `2 passed, 2 warnings`.

Focused nonlinear B/C:

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_nonlinear_benchmark_models_tf.py tests/test_nonlinear_reference_oracles.py tests/test_nonlinear_sigma_point_values_tf.py tests/test_nonlinear_sigma_point_scores_tf.py tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py tests/test_compiled_filter_parity_tf.py -p no:cacheprovider
```

Result: `51 passed, 2 warnings`.

Full default CPU:

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q -p no:cacheprovider
```

Result: `204 passed, 8 skipped, 2 warnings`.

## Evidence Matrix

| Model | Filter | Value status | Score status | Reference status | HMC status | GPU/XLA status | Hessian status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model B nonlinear accumulation | SVD cubature | `certified_tested_box`; BC1 60-row branch matrix stable, BC3 horizon/noise envelope stable | `certified_tested_box`; BC2 finite-difference residuals passed | `deferred_exact`; dense one-step projection diagnostic only | `candidate_not_convergence`; BC5 finite CPU samples and branch gates | `diagnostic_shape_specific`; BC6 `T=8,16` CPU/GPU graph/XLA rows ok | `deferred_no_consumer` |
| Model B nonlinear accumulation | SVD-UKF | `certified_tested_box`; BC1/BC3 stable | `certified_tested_box`; BC2 passed | `deferred_exact`; dense one-step projection diagnostic only | `candidate_not_convergence`; BC5 finite CPU samples and branch gates | `diagnostic_shape_specific`; BC6 rows ok | `deferred_no_consumer` |
| Model B nonlinear accumulation | SVD-CUT4 | `certified_tested_box`; BC1/BC3 stable | `certified_tested_box`; BC2 passed | `deferred_exact`; dense one-step projection diagnostic only | `candidate_not_convergence`; BC5 plus existing opt-in HMC smoke passed | `diagnostic_shape_specific`; BC6 rows ok | `deferred_no_consumer` |
| Model C autonomous nonlinear growth | SVD cubature | `certified_tested_box`; structural fixed support used | `certified_structural_fixed_support`; BC2 passed with `allow_fixed_null_support=True` | `deferred_exact`; dense one-step projection diagnostic only | `candidate_not_convergence`; BC5 fixed-support target finite | `diagnostic_shape_specific`; BC6 fixed-support rows ok | `deferred_no_consumer` |
| Model C autonomous nonlinear growth | SVD-UKF | `certified_with_boundary`; BC3 stable except selected `T=32` score rows | `certified_structural_fixed_support_with_boundary`; selected `T=32` rows blocked as `blocked_moving_structural_null` | `deferred_exact`; dense one-step projection diagnostic only | `candidate_not_convergence` for short default target only; not promoted to blocked `T=32` rows | `diagnostic_shape_specific`; BC6 `T=8,16` fixed-support rows ok | `deferred_no_consumer` |
| Model C autonomous nonlinear growth | SVD-CUT4 | `certified_tested_box`; structural fixed support used | `certified_structural_fixed_support`; BC2 passed with `allow_fixed_null_support=True` | `deferred_exact`; dense one-step projection diagnostic only | `candidate_not_convergence`; BC5 fixed-support target finite | `diagnostic_shape_specific`; BC6 fixed-support rows ok | `deferred_no_consumer` |

## Phase Results

BC0:
- produced the baseline matrix;
- every Model B/C/filter cell had known status.

BC1:
- all 60 branch-box rows were stable;
- no hidden active score floors, weak gaps, nonfinite rows, or unlabeled
  failures.

BC2:
- all 60 analytic score rows passed finite-difference tolerances;
- maximum absolute residuals were about `1.44e-08` for Model B and `3.33e-08`
  for Model C.

BC3:
- 141 of 144 horizon/noise rows were stable;
- Model C + SVD-UKF has an explicit selected `T=32`
  `blocked_moving_structural_null` boundary.

BC4:
- no stronger exact nonlinear reference is required for current claims;
- exact full nonlinear likelihood remains deferred.

BC5:
- all six default short Model B/C targets are HMC-readiness candidates;
- this is not convergence because the draw count is tiny and R-hat is about 2.

BC6:
- escalated GPU probes succeeded;
- 48 GPU/XLA scaling rows were `ok`;
- GPU graph was slower than CPU graph on these small scalar shapes;
- GPU XLA reduced GPU overhead and was competitive in selected Model C rows;
- no broad speedup claim is made.

BC7:
- no concrete nonlinear Hessian consumer was named;
- nonlinear Hessians remain deferred.

## Remaining Gaps And Hypotheses

Exact nonlinear references:
- gap: no exact full nonlinear likelihood for Models B/C;
- hypothesis: low-dimensional deterministic quadrature or seeded high-particle
  SMC can provide a labeled reference for approximation-quality claims if a
  future claim needs it.

HMC convergence:
- gap: BC5 classified readiness candidates only;
- hypothesis: longer chains with predeclared R-hat, ESS, divergence, MCSE, and
  posterior-recovery gates will identify which filters are usable inference
  targets rather than just finite targets.

GPU payoff:
- gap: BC6 small scalar shapes do not justify a broad speedup claim;
- hypothesis: larger horizons, batched parameter grids, or batched independent
  panels are needed before GPU/XLA has a practical advantage.

Model C SVD-UKF structural boundary:
- gap: selected `T=32` score rows are blocked by moving structural-null support;
- hypothesis: either a smoother support contract or an explicit blocked-branch
  policy is needed before promoting long-horizon SVD-UKF Model C score claims.

Nonlinear Hessians:
- gap: no production nonlinear Hessian implementation;
- hypothesis: Hessians should remain deferred until a named consumer appears,
  because no current BC claim needs second-order derivatives.

## Recommended Next Phases

1. HMC convergence ladder for candidate Model B/C targets, with predeclared
   convergence and posterior-recovery criteria.
2. Batched GPU/XLA scaling ladder over larger horizons and batch axes.
3. Optional exact-reference plan only if a stronger approximation-quality claim
   is needed.
4. Model C long-horizon structural-support decision for SVD-UKF.
5. Keep nonlinear Hessians deferred unless a Newton, Laplace, observed
   information, or curvature-aware HMC consumer is explicitly named.

## Veto Audit

No BC8 veto diagnostic fired:

- all claims are tied to artifacts;
- no out-of-lane files were edited;
- default CI does not depend on GPU, HMC, external projects, or long
  experiments;
- Model C language uses structural fixed support, not stale fixed-null
  terminology as a claim substitute.
