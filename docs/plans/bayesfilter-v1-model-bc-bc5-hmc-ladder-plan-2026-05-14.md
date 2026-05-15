# BayesFilter V1 Model B/C BC5 HMC Readiness Ladder Plan

## Date

2026-05-14

## Governing Master Program

This plan executes Phase BC5 in:

```text
docs/plans/bayesfilter-v1-model-bc-thorough-testing-master-program-2026-05-14.md
```

## Purpose

Move beyond the tiny Model B SVD-CUT4 HMC smoke only where value, score,
branch, robustness, and reference-decision gates allow it.

## Entry Gate

BC5 may start only for a target model/filter after:
- BC1 stable/narrowed branch box passes;
- BC2 score accuracy passes for the same box;
- BC3 robustness envelope is documented;
- BC4 does not require unresolved stronger reference evidence for the claim.

GPU HMC is out of scope unless separately authorized with escalated GPU
permissions and its own evidence contract.

## Evidence Contract

Question:
- Which Model B/C/filter targets can be classified as `diagnostic`, `blocked`,
  or `candidate` HMC targets, and what additional convergence/posterior checks
  would be required before any stronger claim?

Baseline:
- Tiny Model B + SVD-CUT4 CPU smoke from P4 and BC1-BC4 gates.

Primary criterion:
- Every attempted target reports finite value/score gates, chain diagnostics,
  and a non-promotional classification.

Veto diagnostics:
- HMC starts before target gates pass;
- tiny smoke is promoted to convergence;
- Model C is sampled without structural fixed-support score diagnostics;
- GPU HMC is run without escalated permissions.

What will not be concluded:
- Production sampler readiness unless convergence diagnostics and posterior
  checks actually pass at the stated scope.

Convergence wording rule:
- The default BC5 claim is HMC readiness classification only.
- A convergence claim requires predeclared R-hat/ESS/divergence/MCSE criteria,
  posterior recovery or known-reference coverage checks, and a result artifact
  that explicitly says the convergence gate was run.
- If those criteria are not predeclared before execution, BC5 must use only
  `diagnostic`, `blocked`, or `candidate`.

Artifact:
- HMC JSON summaries, optional NPZ chains, and BC5 result file.

## Target Order

1. Model B + SVD-CUT4.
2. Model B + SVD cubature and Model B + SVD-UKF.
3. Model C targets only after structural fixed-support branch diagnostics and
   score residuals pass on the intended target box.

## Execution Steps

1. Define prior and target parameter box for each attempted target.
2. Run finite value/score and compiled/eager parity before sampling.
3. Run opt-in CPU HMC ladders with multiple seeds.
4. Record warmup/draw counts, chains, step size, acceptance rate, divergence
   count, R-hat, ESS, MCSE/SD where available, finite-gradient failures, and
   runtime.
5. Save JSON summaries and optional NPZ chains for audit.
6. Write the BC5 result artifact and update the V1 reset memo.

## Primary Gate

BC5 passes if each target can be classified as `diagnostic`, `blocked`, or
`candidate` without overstating convergence.

## Veto Diagnostics

Stop and ask for direction if:
- any target lacks value/score/branch/robustness gate evidence;
- convergence language appears without convergence diagnostics;
- Model C structural fixed-support metadata is missing;
- GPU/CUDA commands are needed but escalation is unavailable.

## Expected Artifacts

Use the execution date in result filenames.  The plan date remains
2026-05-14, but future result artifacts should use `YYYY-MM-DD`.

```text
docs/plans/bayesfilter-v1-model-bc-bc5-hmc-ladder-result-YYYY-MM-DD.md
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

Optional and opt-in only:

```text
tests/test_hmc_nonlinear_model_b_readiness_tf.py
docs/benchmarks/bayesfilter-v1-model-bc-hmc-*.json
docs/benchmarks/bayesfilter-v1-model-bc-hmc-*.npz
```

## Continuation Rule

BC5 does not gate BC6.  Continue to BC7/BC8 only after BC5 records target
classifications or is explicitly deferred.  Do not let HMC classification
determine GPU/XLA performance claims.
