# BayesFilter V1 Model B/C BC1 Branch Boxes Result

## Date

2026-05-15

## Governing Plan

```text
docs/plans/bayesfilter-v1-model-bc-bc1-wider-branch-boxes-plan-2026-05-14.md
```

## Phase Intent

BC1 tests whether Model B/C value and score branch claims survive bounded
parameter boxes wider than the first-rung P2/P3 evidence.

## Plan Tightening

No master-plan ambiguity required a rewrite.  During execution, the artifact
harness was tightened so expected structural zero support in default Model C
value diagnostics is not mislabeled as a score-branch blocker.  BC1 blocker
classification is based on actual value/score failures, active score floors,
weak gaps, nonfinite rows, support residuals, and structural fixed-support
residuals.

## Independent Audit

As a second-developer audit, BC1 stays in the V1 lane and uses only
BayesFilter-local testing infrastructure.  It records deterministic and seeded
row metadata before interpreting stability.  It does not make HMC, GPU,
reference-likelihood, or Hessian claims.

## Artifact

Authoritative JSON:

```text
docs/benchmarks/bayesfilter-v1-model-bc-branch-boxes-2026-05-15.json
```

Readable summary:

```text
docs/benchmarks/bayesfilter-v1-model-bc-branch-boxes-2026-05-15.md
```

Harness:

```text
docs/benchmarks/benchmark_bayesfilter_v1_model_bc_testing.py
```

## Predeclared Rows

Each model/filter cell used:

- 5 deterministic rows, including endpoints and mixed-corner rows;
- 5 seeded random rows with seed `20260515`.

Model B box:

```text
rho in [0.55, 0.85], sigma in [0.15, 0.40], beta in [0.45, 1.10]
```

Model C box:

```text
sigma_u in [0.60, 1.40], sigma_y in [0.60, 1.40], P0x in [0.10, 0.50]
```

## Results

| Model | Filter | Rows | Stable rows | Min score placement gap | Max support residual | Max deterministic residual | Max structural-null covariance residual | Max fixed-null derivative residual | Branch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Model B | SVD cubature | 10 | 10 | `4.269591596540022e-03` | `0.0` | `0.0` | `0.0` | `0.0` | smooth simple spectrum |
| Model B | SVD-UKF | 10 | 10 | `4.269591596540022e-03` | `0.0` | `0.0` | `0.0` | `0.0` | smooth simple spectrum |
| Model B | SVD-CUT4 | 10 | 10 | `4.408267865176924e-03` | `0.0` | `0.0` | `0.0` | `0.0` | smooth simple spectrum |
| Model C | SVD cubature | 10 | 10 | `1.0e-01` | `3.8459253727671276e-16` | `0.0` | `4.930380657631324e-32` | `1.1599889460935306e-31` | structural fixed support |
| Model C | SVD-UKF | 10 | 10 | `1.0e-01` | `5.681991748178211e-16` | `0.0` | `1.0761676742121763e-31` | `5.78944864603357e-16` | structural fixed support |
| Model C | SVD-CUT4 | 10 | 10 | `1.0e-01` | `0.0` | `0.0` | `0.0` | `0.0` | structural fixed support |

## Gate Result

BC1 primary gate passes.  All 60 predeclared rows were stable:

- Model B value and score rows stayed on the smooth simple-spectrum branch.
- Model C score rows used structural fixed support with
  `allow_fixed_null_support=True`.
- No row reported active score floors, weak spectral gaps, nonfinite values,
  nonfinite scores, or unlabeled failures.

## Veto Diagnostics

| Veto | Status |
| --- | --- |
| Active floors hidden by aggregation | Clear; row-level JSON records placement and innovation floor counts |
| Default Model C evaluated without `allow_fixed_null_support=True` | Clear; all Model C rows set it to `true` |
| Failures lack row-level metadata | Clear; no failures, and every row records model, filter, row family, index, seed, parameters, branch, and diagnostics |
| Branch-grid pass described as HMC readiness | Clear; no HMC claim is made |

## Interpretation

The full BC1 boxes are stable for all Model B/C/filter cells at row-level
branch-diagnostic scope.  This does not certify score finite-difference
accuracy, horizon/noise robustness, HMC readiness, exact likelihood quality,
GPU/XLA scaling, or Hessians.

## Continuation Decision

BC2 is justified for all six model/filter cells.  BC2 should use the stable BC1
rows and must exclude or label any finite-difference row that crosses a branch
change.
