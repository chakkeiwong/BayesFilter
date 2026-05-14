# BayesFilter V1 Model B/C BC5 HMC Ladder Result

## Date

2026-05-15

## Controlling Documents

Master program:

```text
docs/plans/bayesfilter-v1-model-bc-thorough-testing-master-program-2026-05-14.md
```

Phase plan:

```text
docs/plans/bayesfilter-v1-model-bc-bc5-hmc-ladder-plan-2026-05-14.md
```

## Plan Tightening And Drift Check

The phase plan was already scoped to HMC readiness classification, not
convergence.  One ambiguity was tightened before execution: the benchmark
harness now records compiled/eager value and gradient parity for every target
before sampling.  No out-of-lane edits were needed.

No drift found:

- HMC remains CPU-only for BC5;
- HMC results are classified only as `candidate` or `blocked`;
- Model C uses the structural fixed-support branch with
  `allow_fixed_null_support=True`;
- no convergence or posterior-recovery claim is made.

## Independent Plan Audit

The plan is safe to execute after BC1-BC4 because:

- BC1 established branch boxes for the default short targets;
- BC2 established analytic score residuals on those boxes;
- BC3 documented the horizon/noise boundary, including the selected Model C +
  SVD-UKF `T=32` blocker;
- BC4 decided that no stronger exact nonlinear reference is required for the
  current readiness-classification claim.

BC5 does not use the blocked Model C + SVD-UKF `T=32` rows as an HMC promotion
target.  It uses only the short default target and labels the evidence as a
readiness diagnostic.

## Execution

Benchmark harness:

```text
docs/benchmarks/benchmark_bayesfilter_v1_model_bc_hmc_ladder.py
```

Artifacts:

```text
docs/benchmarks/bayesfilter-v1-model-bc-hmc-ladder-2026-05-15.json
docs/benchmarks/bayesfilter-v1-model-bc-hmc-ladder-2026-05-15.md
```

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/benchmark_bayesfilter_v1_model_bc_hmc_ladder.py --output docs/benchmarks/bayesfilter-v1-model-bc-hmc-ladder-2026-05-15.json --markdown-output docs/benchmarks/bayesfilter-v1-model-bc-hmc-ladder-2026-05-15.md
```

Focused opt-in test:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 BAYESFILTER_RUN_HMC_READINESS=1 pytest -q tests/test_hmc_nonlinear_model_b_readiness_tf.py -p no:cacheprovider
```

Result: `3 passed, 2 warnings`.

## Results

All six default short Model B/C targets are HMC-readiness candidates:

| Target | Classification | Fixed support | Parity | Nonfinite samples | Acceptance | Max R-hat |
| --- | --- | --- | --- | ---: | --- | ---: |
| Model B + SVD cubature | `candidate` | false | pass | 0 | 1.000, 1.000, 1.000 | 1.995 |
| Model B + SVD-UKF | `candidate` | false | pass | 0 | 1.000, 1.000, 1.000 | 1.995 |
| Model B + SVD-CUT4 | `candidate` | false | pass | 0 | 1.000, 1.000, 1.000 | 1.995 |
| Model C + SVD cubature | `candidate` | true | pass | 0 | 1.000, 1.000, 1.000 | 2.024 |
| Model C + SVD-UKF | `candidate` | true | pass | 0 | 1.000, 1.000, 1.000 | 2.024 |
| Model C + SVD-CUT4 | `candidate` | true | pass | 0 | 1.000, 1.000, 1.000 | 2.024 |

## Interpretation

The primary gate passed.  Every attempted target reported finite branch-gated
value/score diagnostics, compiled/eager parity, finite initial gradients, and
finite CPU HMC samples.

This is not a convergence claim.  The draw count is intentionally tiny, all
acceptance rates are 1.0, and the maximum R-hat values are around 2.0.  The
right interpretation is that the targets are viable candidates for a future
predeclared convergence and posterior-recovery ladder.

## Veto Audit

No veto diagnostic fired:

- HMC did not start before score and branch gates existed;
- tiny HMC was not promoted to convergence;
- Model C used structural fixed support;
- no GPU HMC was run;
- no production NumPy dependency was added.

## Continuation Decision

BC5 passes.  BC6 remains justified because it is an independent GPU/XLA scaling
diagnostic gated by BC1/BC3 shape evidence and escalated GPU permissions, not by
HMC convergence.
