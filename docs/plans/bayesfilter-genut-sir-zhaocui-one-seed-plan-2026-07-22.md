# One-Seed GenUT/Zhao-Cui Reduced SIR Comparison

Date: 2026-07-22

## Question

For one identical reduced continuous preclip SIR DGP seed, what value and
recursive score do the positive GenUT candidate and the local fixed-design
Zhao-Cui diagnostic route produce?

## Scope

Use the reduced `J=1`, state `(S,I)`, three log-scale parameters, and the same
FP32 DGP observations for `T=2,5,10`. Both routes use initial-observation-first
timing: `y_0` is observed from the unprojected initial latent state and later
observations follow preclip transitions. GenUT uses an explicit initial-observation
mode in `finite_value_score`. Zhao-Cui uses
`multistate_nonlinear_fixed_design_tt_score_path` with a small rank-2 product
Legendre design in float64 on CPU.

## Comparator limitation

The two routes evaluate the same target and timing, but use different finite
approximations and precision. The Zhao-Cui route is the local fixed-design
retained-grid diagnostic, not the production fixed-variant source route and
not an oracle. A reduced dense-grid value/manual-score reference is included
as the accuracy anchor. One seed remains descriptive only.

## Hard checks

- both routes return finite value and score;
- GenUT reset residuals remain below `5e-4`;
- Zhao-Cui route reports `HighDimStatus.OK` and finite score;
- same DGP seed, numeric observation rows, truth theta, and horizons are
  recorded in the artifact.

## Nonclaims

No exact likelihood claim, no Zhao-Cui oracle claim, no source-faithfulness
claim for the reduced route, no timing-equivalent comparison, and no
leaderboard/default promotion.

## Artifact

`docs/benchmarks/artifacts/genut_sir_zhaocui_one_seed_20260722/attempt01/`
