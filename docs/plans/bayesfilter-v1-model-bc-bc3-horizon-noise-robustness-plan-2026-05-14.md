# BayesFilter V1 Model B/C BC3 Horizon And Noise Robustness Plan

## Date

2026-05-14

## Governing Master Program

This plan executes Phase BC3 in:

```text
docs/plans/bayesfilter-v1-model-bc-thorough-testing-master-program-2026-05-14.md
```

## Purpose

Test whether Model B/C value and score behavior remains stable as observation
panels become longer and observation noise becomes more demanding.

## Entry Gate

BC3 may start only after BC1 and BC2 pass or explicitly block the relevant
model/filter cells.

## Evidence Contract

Question:
- What horizon/noise envelope can each Model B/C/filter cell support without
  nonfinite values, unstable score diagnostics, or branch-contract violations?

Baseline:
- BC1 stable boxes and BC2 score residual tables.

Primary criterion:
- Each model/filter has a documented robustness envelope or a declared blocker.

Veto diagnostics:
- stochastic panels omit seeds;
- low-noise failures are interpreted as global model failure without branch
  diagnostics;
- runtime regressions are treated as correctness failures.

What will not be concluded:
- HMC convergence, exact likelihood quality, or broad performance scaling.

Artifact:
- Robustness benchmark artifact and BC3 result file.

## Execution Steps

1. Create deterministic panels for horizons \(T\in\{3,8,16,32\}\).
2. Create seeded stochastic panels with recorded seeds.
3. Sweep observation-noise scales around defaults and low-noise stress cases.
4. Record finite log likelihood, score finiteness, branch diagnostics, support
   residuals, structural-null residuals, and runtime.
5. Separate deterministic panel results from seeded stochastic panel results.
6. Write the BC3 result artifact and update the V1 reset memo.

## Stop Rules

- Run the planned horizon ladder \(T\in\{3,8,16,32\}\) and predeclared noise
  scales once per deterministic/seeded panel family.
- If low-noise rows fail, report the largest passing envelope and exact
  blocker labels instead of repeatedly changing the ladder.
- Do not add new stochastic seeds after seeing failures unless the result is
  labeled exploratory and not used for promotion.
- If a model/filter cannot pass \(T=3\) at the declared default-noise row,
  classify that row as `blocked` and stop robustness promotion for that cell.

## Primary Gate

BC3 passes only if every model/filter cell has a documented horizon/noise
envelope or a blocker with exact failure labels.

## Veto Diagnostics

Stop and ask for direction if:
- stochastic rows lack seeds;
- Model C structural fixed-support metadata is absent;
- runtime is used as a correctness veto without a correctness failure;
- low-noise stress failures are generalized beyond the tested envelope.

## Expected Artifacts

Use the execution date in result filenames.  The plan date remains
2026-05-14, but future result artifacts should use `YYYY-MM-DD`.

```text
docs/plans/bayesfilter-v1-model-bc-bc3-horizon-noise-result-YYYY-MM-DD.md
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

Optional, only if needed:

```text
tests/test_nonlinear_sigma_point_values_tf.py
tests/test_nonlinear_sigma_point_scores_tf.py
docs/benchmarks/bayesfilter-v1-model-bc-horizon-noise-*.json
```

## Continuation Rule

Continue to BC4 for reference decisions after BC3 records the robustness
envelopes.  Continue to BC5 only for target cells whose value, score, branch,
and robustness gates pass.
