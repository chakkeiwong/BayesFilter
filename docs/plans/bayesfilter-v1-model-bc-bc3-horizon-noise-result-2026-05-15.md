# BayesFilter V1 Model B/C BC3 Horizon/Noise Result

## Date

2026-05-15

## Governing Plan

```text
docs/plans/bayesfilter-v1-model-bc-bc3-horizon-noise-robustness-plan-2026-05-14.md
```

## Phase Intent

BC3 tests whether Model B/C value and score behavior remains stable across
longer panels and lower observation-noise cases after BC1 branch boxes and BC2
score residuals have passed.

## Plan Tightening

No plan rewrite was needed.  During audit, the blocker parser in the V1
testing harness was tightened to preserve the exact
`blocked_moving_structural_null` label instead of reporting it as a generic
blocker.  This keeps BC3 aligned with the plan requirement for exact row-level
failure labels.

## Independent Audit

As a second-developer audit, BC3 stayed in the BayesFilter V1 lane and did not
change production behavior.  It used CPU-only execution with
`CUDA_VISIBLE_DEVICES=-1`, recorded deterministic and seeded panel families,
and treated runtime as explanatory rather than a correctness veto.

## Artifact

Authoritative JSON:

```text
docs/benchmarks/bayesfilter-v1-model-bc-horizon-noise-2026-05-15.json
```

Readable summary:

```text
docs/benchmarks/bayesfilter-v1-model-bc-horizon-noise-2026-05-15.md
```

## Predeclared Ladder

- Horizons: `T in {3, 8, 16, 32}`.
- Observation-noise scales: `{1.0, 0.5, 0.25}`.
- Panel families:
  - deterministic tiled fixture panels;
  - seeded stochastic panels with seed `20260515`.

## Results

| Model | Filter | Rows | Stable rows | Blocked rows | Min placement gap | Max support residual | Max structural-null covariance residual | Max fixed-null derivative residual | Max runtime seconds | Envelope |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Model B | SVD cubature | 24 | 24 | 0 | `1.2221927672460627e-03` | `0.0` | `0.0` | `0.0` | `0.5243916539475322` | full ladder |
| Model B | SVD-UKF | 24 | 24 | 0 | `1.2246066436137422e-03` | `0.0` | `0.0` | `0.0` | `0.6397262059617788` | full ladder |
| Model B | SVD-CUT4 | 24 | 24 | 0 | `1.6517303330128587e-03` | `0.0` | `0.0` | `0.0` | `0.7429029310587794` | full ladder |
| Model C | SVD cubature | 24 | 24 | 0 | `3.7043411126487236e-03` | `1.2306961192854808e-14` | `1.262177448353619e-29` | `1.1330041915241575e-23` | `0.5402254801010713` | full ladder |
| Model C | SVD-UKF | 24 | 21 | 3 | `2.3909279571563413e-03` | `4.109386484554766e-14` | `1.444917227494806e-28` | `2.5617654088287485e-11` | `0.722257953020744` | blocked at selected `T=32` structural fixed-support score rows |
| Model C | SVD-CUT4 | 24 | 24 | 0 | `2.9410067044792765e-03` | `6.925512794457508e-14` | `2.0194839173657902e-28` | `1.0964308996012998e-20` | `0.7224783001001924` | full ladder |

The three blocked rows are:

| Model | Filter | Panel | Horizon | Noise scale | Blocker |
| --- | --- | --- | ---: | ---: | --- |
| Model C | SVD-UKF | deterministic | 32 | 1.0 | `blocked_moving_structural_null` |
| Model C | SVD-UKF | seeded stochastic | 32 | 1.0 | `blocked_moving_structural_null` |
| Model C | SVD-UKF | seeded stochastic | 32 | 0.5 | `blocked_moving_structural_null` |

All Model C score rows used structural fixed support with
`allow_fixed_null_support=True`.  The blocked SVD-UKF rows are value-finite,
but the analytic score correctly refuses promotion because the structural null
support becomes parameter-dependent at those rows.

## Gate Result

BC3 primary gate passes:

- every model/filter cell has either a full documented envelope or an exact
  blocker label;
- five of six model/filter cells pass the full `T <= 32`, noise-scale
  `{1.0, 0.5, 0.25}` ladder for both deterministic and seeded panels;
- Model C + SVD-UKF has an explicit score envelope boundary at selected
  `T=32` rows with `blocked_moving_structural_null`.

## Veto Diagnostics

| Veto | Status |
| --- | --- |
| Stochastic panels omit seeds | Clear; seeded rows use seed `20260515` |
| Model C structural fixed-support metadata absent | Clear |
| Runtime treated as correctness failure | Clear |
| Low-noise failures generalized beyond tested envelope | Clear |

## Interpretation

The horizon/noise results strengthen Models B-C as nonlinear value/score
fixtures while also exposing a useful Model C + SVD-UKF boundary.  That
boundary is not a production failure: it is a branch-contract refusal for a
parameter-dependent structural null support at specific long-panel rows.

## Continuation Decision

BC4 is justified.  HMC target selection in BC5 should treat Model C + SVD-UKF
as blocked for the `T=32` rows above unless a new structural-support decision
is made.  Model B all filters, Model C SVD cubature, and Model C SVD-CUT4 have
full BC3 envelopes at the tested scope.
