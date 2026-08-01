# One-Seed GenUT/Zhao-Cui Reduced SIR Result

Date: 2026-07-22  
Plan: `bayesfilter-genut-sir-zhaocui-one-seed-plan-2026-07-22.md`  
Artifact: `docs/benchmarks/artifacts/genut_sir_zhaocui_one_seed_20260722/attempt02/`

## Target and routes

This is the reduced continuous preclip `J=1` SIR target with state `(S,I)`,
truth theta `(0,0,0)`, and one shared DGP seed `97001`. Both methods use the
same initial-observation-first timing, the same observations, and the same
parameter point.

- GenUT: positive replicated Gaussian GenUT, `N=96`, FP32, recursive score.
- Zhao-Cui: local `multistate_nonlinear_fixed_design_tt_score_path`, rank-2
  retained-grid diagnostic, float64.
- Dense reference: split Gauss-Legendre order 29, radius 7, manual filtering
  score. It is the accuracy anchor for this reduced target, not an exact error
  bound.

## Results

| T | dense value | GenUT value | Zhao-Cui value | GenUT value error | Zhao-Cui value error | dense score `(kappa,nu,obs)` | GenUT score | Zhao-Cui score |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 2 | -1.20166 | -1.19887 | -1.26059 | +0.00278 | -0.05894 | `(-0.000082, 0.002742, -0.776853)` | `(-0.000014, 0.000596, -0.701556)` | `(0.000033, 0.001062, -0.086477)` |
| 5 | -3.24782 | -3.34517 | -3.44482 | -0.09734 | -0.19699 | `(-0.001149, 0.023061, -1.481291)` | `(-0.000264, 0.006413, -1.305150)` | `(-0.001786, 0.018943, 0.833659)` |
| 10 | -8.57691 | -8.73720 | -9.27574 | -0.16029 | -0.69883 | `(-0.003573, 0.050177, -0.013847)` | `(-0.001346, 0.019030, 0.320773)` | `(-0.012739, 0.062177, 4.760340)` |

## Interpretation

For this one seed, GenUT is descriptively closer to the dense reference in
value at all three horizons. The value absolute errors are approximately
`0.0028, 0.0973, 0.1603` for GenUT versus `0.0589, 0.1970, 0.6988` for
Zhao-Cui.

The largest difference is the observation-noise score component. GenUT's
absolute errors are approximately `0.0753, 0.1761, 0.3346`; Zhao-Cui's are
`0.6904, 2.3149, 4.7742`. The kappa and nu score differences are smaller and
mixed across horizons; no ranking should be inferred from them with one seed.

## Decision table

| Decision | Status | Evidence |
|---|---|---|
| Same target/timing/seed replay | PASS | identical observations and theta are embedded in the JSON artifact |
| Finite GenUT value/score/reset | PASS | all three GenUT rows finite; reset residuals below `8e-7` |
| Finite Zhao-Cui value/score | PASS | all three rows return `HighDimStatus.OK` and finite values |
| Accuracy relative to reduced dense anchor | Descriptively favorable to GenUT | one seed; no statistical uncertainty |
| Zhao-Cui as oracle | REJECTED | route is `diagnostic_historical_retained_grid`, not an oracle |
| Leaderboard/default conclusion | BLOCKED | reduced target and one-seed evidence only |

This is not statistical evidence of superiority. It does show that, on this
paired reduced target, the fixed-design Zhao-Cui score can be substantially
more distorted than the GenUT recursive score, especially for the observation
noise scale as horizon grows.

## Reproduction

```text
CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/run_genut_sir_zhaocui_one_seed.py
```

The JSON artifact preserves the observations, values, scores, dense reference,
route identities, and nonclaims.
