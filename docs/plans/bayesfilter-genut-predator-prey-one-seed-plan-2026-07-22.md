# GenUT Predator-Prey T20 One-Seed Feasibility Plan

Date: 2026-07-22

## Question

Can the positive GenUT candidate execute the canonical additive-Gaussian
predator-prey T20 target with six-parameter recursive score in FP32/TF32/XLA
for one seed, and how does its finite output compare descriptively with the
existing local Zhao-Cui fixed-design T20 diagnostic?

## Target

Use `zhao_cui_predator_prey_T20`, DGP seed `81104`, truth theta
`(r,K,a,s,u,v)=(0.6,114,25,0.3,0.5,0.5)`, state `(prey,predator)`, 20
observations (`x_0` observation first, then 19 transitions), RK4 delta `2.0`
with internal step `0.1`, process covariance `4I`, observation covariance
`4I`, and initial covariance `I`.

## Evidence contract

- Hard screen: finite value/score, finite XLA output on GPU, and GenUT mean /
  Sinkhorn residuals below `5e-4`.
- Comparator: existing
  `multistate_nonlinear_fixed_design_tt_score_path` on the same observations,
  theta, and timing, recorded as diagnostic/historical.
- Explanatory diagnostics: score increment sum, runtime, route identities,
  and per-coordinate GenUT/Zhao-Cui differences.
- Nonclaims: no exact likelihood or score claim, no Zhao-Cui oracle claim, no
  source-faithfulness claim for the local fixed-design route, no superiority,
  HMC readiness, leaderboard admission, or default promotion.

## Skeptical audit

The existing canonical row has no exact analytic oracle and the local Zhao-Cui
route is explicitly not the production fixed-variant source route. A successful
run therefore establishes only executable feasibility and a descriptive
same-target route comparison. The first cheap failure diagnostic is the XLA
T20 compilation itself; no larger particle ladder is launched in this phase.

## Artifact

`docs/benchmarks/artifacts/genut_predator_prey_one_seed_20260722/attempt01/`
