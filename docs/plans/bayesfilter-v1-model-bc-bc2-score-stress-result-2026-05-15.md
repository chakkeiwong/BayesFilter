# BayesFilter V1 Model B/C BC2 Score Stress Result

## Date

2026-05-15

## Governing Plan

```text
docs/plans/bayesfilter-v1-model-bc-bc2-score-accuracy-stress-plan-2026-05-14.md
```

## Phase Intent

BC2 quantifies analytic score accuracy for Models B-C over the BC1 stable
branch boxes by comparing analytic scores to centered finite differences of
the same implemented value filter.

## Plan Tightening

No plan rewrite was needed.  The predeclared tolerance table was encoded before
interpreting residuals in:

```text
docs/benchmarks/benchmark_bayesfilter_v1_model_bc_testing.py
```

Tolerance table:

| Model | Absolute tolerance | Relative tolerance | Primary step | Step ladder | Source |
| --- | ---: | ---: | ---: | --- | --- |
| Model B | `2.0e-03` | `2.0e-03` | `2.0e-05` | `5.0e-05, 2.0e-05, 1.0e-05` | P1 baseline used `5e-4` at the center; widened box gets a 4x finite-difference margin |
| Model C | `1.0e-02` | `1.0e-02` | `1.0e-05` | `3.0e-05, 1.0e-05, 3.0e-06` | P1 baseline used `1e-3` at the center; structural fixed-support box gets a conservative 10x finite-difference margin |

## Independent Audit

As a second-developer audit, BC2 stayed inside the V1 lane and did not promote
finite differences to exact likelihood evidence.  Every row records the branch
label and compiled/eager residual.  Model C rows use structural fixed support
with `allow_fixed_null_support=True`.

## Artifact

Authoritative JSON:

```text
docs/benchmarks/bayesfilter-v1-model-bc-score-stress-2026-05-15.json
```

Readable summary:

```text
docs/benchmarks/bayesfilter-v1-model-bc-score-stress-2026-05-15.md
```

## Results

| Model | Filter | Rows | Passing rows | Max abs residual | Max relative residual | Max compiled value residual | Max compiled score residual | Branch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Model B | SVD cubature | 10 | 10 | `1.4409583748431487e-08` | `3.0205270442437916e-09` | `4.440892098500626e-16` | `1.7763568394002505e-15` | smooth simple spectrum |
| Model B | SVD-UKF | 10 | 10 | `1.440812180675266e-08` | `3.020212674748106e-09` | `4.440892098500626e-16` | `1.7763568394002505e-15` | smooth simple spectrum |
| Model B | SVD-CUT4 | 10 | 10 | `1.4010050009005681e-08` | `2.9785237152763646e-09` | `2.220446049250313e-16` | `1.7763568394002505e-15` | smooth simple spectrum |
| Model C | SVD cubature | 10 | 10 | `2.944951216221625e-08` | `6.2646316329823514e-09` | `1.7763568394002505e-15` | `8.881784197001252e-16` | structural fixed support |
| Model C | SVD-UKF | 10 | 10 | `1.8551769009889085e-08` | `3.1872726572175573e-09` | `8.881784197001252e-16` | `6.661338147750939e-16` | structural fixed support |
| Model C | SVD-CUT4 | 10 | 10 | `3.328899023102849e-08` | `8.331740510171525e-09` | `1.7763568394002505e-15` | `8.881784197001252e-16` | structural fixed support |

## Gate Result

BC2 primary gate passes:

- all 60 BC1 rows passed the predeclared absolute and relative tolerances;
- no finite-difference row crossed a branch change;
- compiled/eager value and score parity residuals stayed at numerical noise
  scale;
- Model C used the structural fixed-support branch with
  `allow_fixed_null_support=True`.

## Veto Diagnostics

| Veto | Status |
| --- | --- |
| Finite differences cross active branch changes | Clear; no row reported `blocked_fd_branch_change` |
| Model C structural fixed-support diagnostics omitted | Clear |
| Compiled/eager mismatch ignored | Clear; residuals are reported and at numerical noise scale |
| Tolerances loosened after failures | Clear; tolerances were encoded before interpretation |

## Interpretation

Analytic scores for Models B-C agree with centered finite differences across
the full BC1 stable boxes for all three filters.  This is a score-accuracy
claim for the implemented filter values, not an exact nonlinear likelihood
claim and not a Hessian claim.

## Continuation Decision

BC3 is justified for all six model/filter cells.  BC3 should test horizon and
observation-noise robustness using the same branch vocabulary and structural
fixed-support rule for default Model C.
