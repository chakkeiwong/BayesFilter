# GenUT Predator-Prey T20 One-Seed Result

Date: 2026-07-22  
Plan: `bayesfilter-genut-predator-prey-one-seed-plan-2026-07-22.md`  
Artifact: `docs/benchmarks/artifacts/genut_predator_prey_one_seed_20260722/attempt02/`

## Setup

Canonical row: `zhao_cui_predator_prey_T20`, DGP seed `81104`, `T=20`,
`N=96`, FP32 tensors, TF32 enabled, GPU/XLA. Parameter order is
`(r,K,a,s,u,v)` with truth `(0.6,114,25,0.3,0.5,0.5)`. The model uses RK4
delta `2.0`, internal step `0.1`, process covariance `4I`, observation
covariance `4I`, and initial covariance `I`.

GenUT uses the positive replicated Gaussian GenUT dimension-2 design, epsilon
`2.0`, 8 Sinkhorn iterations, and ridge `1e-5`. Its score is a recursive
forward sensitivity of the same finite value program. The comparator is the
existing float64 `multistate_nonlinear_fixed_design_tt_score_path`; it is a
retained-grid diagnostic/historical route and is not the production fixed-
variant Zhao-Cui route or an oracle.

## Results

| Route | Value | Score `(r,K,a,s,u,v)` | Runtime | Finite |
|---|---:|---|---:|---|
| GenUT | `-103.79195` | `(-23.52321, 1.37469, 0.009407, -1.67151, -3.93866, 4.38100)` | `5.32 s` | yes |
| Zhao-Cui diagnostic | `-179.92342` | `(136.61432, 6.54013, 0.161710, -59.53580, -5.71829, 6.67239)` | `230.10 s` | yes |

GenUT minus Zhao-Cui is `+76.13147` in value, with score difference
`(-160.13753, -5.16544, -0.152303, +57.86430, +1.77963, -2.29139)`.

## Hard diagnostics

- GPU/XLA compilation: pass; output device `/GPU:0`.
- GPU memory policy: memory growth verified before logical GPU creation.
- GenUT maximum mean-restoration residual: `8.54e-5`.
- GenUT maximum Sinkhorn row residual: `1.36e-7`.
- GenUT maximum Sinkhorn column residual: `4.32e-6`.
- GenUT score increment-sum residual: `1.91e-6`.
- GenUT same-scalar central-FD audit maximum relative error: `1.43%`.

## Interpretation

The candidate is computationally feasible on the canonical predator-prey T20
shape for one seed. The finite Zhao-Cui diagnostic is substantially slower and
returns a materially different value and score. That difference is
descriptive only: the Zhao-Cui retained-grid route has its own approximation,
rank, coordinate chart, and float64 configuration, and there is no exact
predator-prey observed-data oracle in this run.

This result therefore does not establish that GenUT is statistically more
accurate, does not certify the Zhao-Cui route, and does not support leaderboard
or default promotion. A multi-seed run and an independently refined
predator-prey reference are required for an accuracy claim.

## Reproduction

```text
python docs/benchmarks/run_genut_predator_prey_one_seed.py
```

The result JSON embeds the canonical observation hash, route identities,
device/memory provenance, values, scores, and nonclaims.
