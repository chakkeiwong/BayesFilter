# BayesFilter V1 P3 Benchmark Refresh Result

## Date

2026-05-14

## Scope

This result executes P3/R4 from:

```text
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
docs/plans/bayesfilter-v1-p3-benchmark-refresh-plan-2026-05-14.md
```

P3 stays inside the BayesFilter V1 lane.  It refreshes the nonlinear benchmark
artifact with value and score branch metadata.  It does not claim exact full
nonlinear likelihoods for Models B-C, does not run GPU/XLA, does not run HMC,
does not implement Hessians, and does not touch MacroFinance, DSGE, Chapter
18b, or structural-lane plans.

## Phase Plan Audit

Pass with one benchmark-harness correction.

The existing benchmark was value-centric and still listed
`nonlinear_models_b_c_analytic_score` as a blocked claim.  That statement was
stale after P1/P2.  The benchmark harness was updated to include score branch
metadata and to keep the remaining blocked claims precise:

- exact full nonlinear likelihood for Models B-C remains blocked;
- nonlinear HMC readiness remains blocked until P4;
- GPU/XLA speedup remains blocked until P6;
- nonlinear Hessian readiness remains blocked until P5 names a consumer.

During the first benchmark run, the Model A score branch grid exposed a weak
spectral gap because the benchmark's parameterized affine branch-grid builder
used repeated initial covariance eigenvalues.  The harness was corrected to
use a non-repeated affine covariance for score-branch metadata, and score
status is now computed from the actual branch summary rather than assumed.

## Implementation

Updated:

```text
docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_filters.py
```

Added row metadata:
- value status;
- score status;
- score branch label;
- finite score status;
- value branch failure labels;
- value branch structural-null diagnostics;
- score branch ok counts;
- score branch active-floor, weak-gap, and nonfinite counts;
- score branch failure labels;
- score branch structural-null diagnostics;
- `score_allow_fixed_null_support`.

Generated:

```text
docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-14.json
docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-14.md
```

## Commands

Syntax check:

```bash
python -m py_compile docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_filters.py
```

Focused regression:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  tests/test_nonlinear_sigma_point_values_tf.py \
  -p no:cacheprovider
```

Result:
- `23 passed, 2 warnings`.

Benchmark:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
MPLCONFIGDIR=/tmp/matplotlib-bayesfilter \
python docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_filters.py \
  --repeats 1 \
  --output docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-14.json \
  --markdown-output docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-14.md
```

Result:
- benchmark completed;
- JSON environment recorded `cuda_visible_devices=-1`;
- JSON logical devices recorded CPU only.

TensorFlow still printed CUDA plugin/cuInit messages during startup.  Because
the benchmark environment and logical devices show CPU-only execution, these
messages are treated as framework startup noise, not GPU evidence.

## Benchmark Summary

All nine model/backend rows completed:

| Model | Backends | Value branch | Score branch |
| --- | --- | --- | --- |
| Model A affine Gaussian oracle | SVD cubature, SVD-UKF, SVD-CUT4 | `3/3` for each backend | `3/3`, smooth simple spectrum |
| Model B nonlinear accumulation | SVD cubature, SVD-UKF, SVD-CUT4 | `3/3` for each backend | `3/3`, smooth simple spectrum |
| Model C autonomous nonlinear growth | SVD cubature, SVD-UKF, SVD-CUT4 | `3/3` for each backend | `3/3`, structural fixed support |

Reference scope:
- Model A: exact linear-Gaussian Kalman reference;
- Models B-C: dense one-step Gaussian projection diagnostics only.

Blocked claims retained:
- exact full nonlinear likelihood for Models B-C;
- nonlinear HMC readiness;
- GPU/XLA speedup;
- nonlinear Hessian readiness.

## Interpretation

H-P3 is supported.  The nonlinear benchmark now exposes value evidence,
score-branch stability, point counts, polynomial degree, deterministic
residuals, support residuals, structural-null score diagnostics, failure
labels, and reference type in one artifact.

CUT4 retains the larger point count.  In this small CPU artifact, CUT4 improves
the first-step dense-projection error for Model B and Model C relative to the
other listed sigma-point rules, but this remains diagnostic and shape-specific.
It is not a GPU/XLA speedup claim and not an exact likelihood claim.

## Gate Decision

P3 passes.

Primary gate:
- benchmark artifacts reproduce from documented commands;
- every score row carries branch metadata;
- exactness claims are limited to the actual reference type.

Continuation:
- P4 is justified.  Model B is the first nonlinear HMC target candidate because
  it has finite implemented values, certified analytic scores, smooth branch
  stability, and benchmark metadata.  Default Model C remains a possible later
  target only under structural fixed support, but it should not be the first
  HMC target.
