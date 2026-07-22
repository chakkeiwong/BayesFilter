# GenUT SIR Feasibility Result

Date: 2026-07-22  
Plan: `bayesfilter-genut-sir-feasibility-plan-2026-07-22.md`  
Artifact: `docs/benchmarks/artifacts/genut_sir_feasibility_20260722/attempt06/`

## Outcome

GenUT executes successfully on the reduced continuous preclip SIR diagnostic
target in FP32 with TF32 enabled, GPU/XLA, and `N=96`. All 24 requested runs
(`T=2,5,10`, eight seeds per horizon) returned finite value and recursive
score. The staged candidate core compiled on the RTX 4080 SUPER and used the
GPU device.

| T | runs | value mean | value SD | score mean `(kappa,nu,obs)` | score SD `(kappa,nu,obs)` | FD audit |
|---:|---:|---:|---:|---|---|---|
| 2 | 8 | -2.77072 | 2.25484 | `(0.000299,-0.003751,1.00122)` | `(0.001024,0.015570,2.72240)` | 8/8 pass |
| 5 | 8 | -5.84962 | 2.57444 | `(0.000059,-0.003545,1.11876)` | `(0.001154,0.022865,3.21248)` | 8/8 pass |
| 10 | 8 | -10.70939 | 1.48217 | `(0.000216,-0.001672,1.51101)` | `(0.001411,0.032234,2.31495)` | 8/8 pass |

The maximum mean-restoration residual was `8.58e-5`; maximum Sinkhorn row and
column marginal residuals were `1.56e-7` and `6.44e-6`. The largest
recursive-score/central-FD relative discrepancy was `4.00%`, using an
`h=8e-3` central difference. FD is an explanatory audit only; the runtime
score is the hand-written recursive forward sensitivity.

## Decision table

| Decision | Status | Evidence |
|---|---|---|
| Hard finite/XLA/reset screen | PASS | 24/24 finite; all residuals below `5e-4`; XLA GPU compilation logged |
| Recursive score path | PASS as an implementation feasibility check | score increments sum exactly to returned score in every row |
| FD consistency | PASS as diagnostic | 24/24 rows pass at the declared FP32 audit step; max relative discrepancy 4.00% |
| Statistical ranking or accuracy | NOT TESTED | no exact SIR observed-data oracle was used |
| Default/leaderboard readiness | BLOCKED | reduced target is not the canonical clipped Austria SIR measure |

## What this establishes

The existing positive Gaussian-GenUT design and staged OT/Contract-E candidate
core can carry a nonlinear two-state SIR transition, observation update, and
three-parameter recursive score through TensorFlow/XLA without a Python or
NumPy numerical loop inside the compiled candidate. The hand-derived RK4
state/parameter tangent is numerically consistent with the same scalar value
program at the FP32 diagnostic resolution.

## What this does not establish

This is `SIR_GENUT_FEASIBILITY_DIAGNOSTIC_ONLY`. It does not establish exact
likelihood or exact score accuracy, posterior validity, clipping-measure
equivalence, source-faithful Zhao-Cui inference, HMC readiness, scalability to
the `d=18` Austria row, or GenUT promotion as a leaderboard/default route.
The repository's documented clipped-simulator/Gaussian-density measure
mismatch remains unresolved for the canonical Austria row.

## Attempts and repairs

- `attempt01`: CPU-hidden smoke was finite; GPU campaign was not run.
- `attempt02`: stopped before evaluation because memory growth was configured
  after TensorFlow initialization.
- `attempt03`: same ordering issue remained due to a module-level TensorFlow
  constant.
- `attempt04`: GPU/XLA compiled; a positional `stateless_normal` dtype bug
  stopped before the campaign.
- `attempt05`: full path was finite, but a too-small FP32 FD step produced a
  marginal explanatory veto at `T=5`; a step calibration showed this was
  cancellation, not a runtime score failure.
- `attempt06`: moved FD to the calibrated `h=8e-3` diagnostic and removed FD
  from the hard stop in accordance with the plan's evidence contract.

## Reproduction

```text
python docs/benchmarks/run_genut_sir_feasibility.py
pytest -q tests/highdim/test_cubature_genut_candidate.py
```

The run manifest records the command, plan, commit, TensorFlow environment,
GPU/memory policy, and SHA-256 of `result.json`.
