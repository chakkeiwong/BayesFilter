# BayesFilter V1 P4 Nonlinear HMC Target Result

## Date

2026-05-14

## Scope

This result executes P4/R5 from:

```text
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
docs/plans/bayesfilter-v1-p4-nonlinear-hmc-target-plan-2026-05-14.md
```

P4 stays inside the BayesFilter V1 lane.  It adds an opt-in testing helper and
opt-in tests for a tiny CPU HMC smoke.  It does not add a production HMC API,
does not claim convergence, does not use GPU, does not implement Hessians, and
does not touch MacroFinance, DSGE, Chapter 18b, or structural-lane plans.

## Phase Plan Audit

Pass.

The entry gates are satisfied:
- P1 certified nonlinear score status and explicit Hessian deferral;
- P2 identified a stable Model B score branch box;
- P3 produced benchmark metadata with Model B value and score branch rows.

Model B is the correct first nonlinear target because it is smooth and does
not require the structural fixed-support branch.  Default Model C remains a
later target candidate only under structural fixed support.

## Implementation

Updated testing helpers:

```text
bayesfilter/testing/tf_hmc_readiness.py
bayesfilter/testing/__init__.py
```

Added opt-in tests:

```text
tests/test_hmc_nonlinear_model_b_readiness_tf.py
```

Added tiny-smoke artifact:

```text
docs/benchmarks/bayesfilter-v1-model-b-nonlinear-hmc-smoke-2026-05-14.json
```

The target is:
- model: Model B nonlinear accumulation;
- backend: SVD-CUT4 analytic score;
- parameter vector: \((\rho,\sigma,\beta)\);
- prior: independent Gaussian centered at the initial target point;
- sampler: TFP `HamiltonianMonteCarlo`;
- scope: tiny CPU smoke only.

## Validation

Syntax:

```bash
python -m py_compile \
  bayesfilter/testing/tf_hmc_readiness.py \
  tests/test_hmc_nonlinear_model_b_readiness_tf.py
```

Default skip behavior:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_hmc_nonlinear_model_b_readiness_tf.py \
  -p no:cacheprovider
```

Result:
- `3 skipped, 2 warnings`.

Public API guard:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_v1_public_api.py \
  -p no:cacheprovider
```

Result:
- `2 passed, 2 warnings`.

Opt-in HMC readiness:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
BAYESFILTER_RUN_HMC_READINESS=1 pytest -q \
  tests/test_hmc_nonlinear_model_b_readiness_tf.py \
  -p no:cacheprovider
```

Result:
- `3 passed, 2 warnings`.

Diagnostic capture:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
MPLCONFIGDIR=/tmp/matplotlib-bayesfilter \
python -c '... run_model_b_nonlinear_svd_cut4_hmc_smoke ...'
```

Result:
- artifact written to
  `docs/benchmarks/bayesfilter-v1-model-b-nonlinear-hmc-smoke-2026-05-14.json`.

TensorFlow printed CUDA plugin/cuInit startup messages even with
`CUDA_VISIBLE_DEVICES=-1`; no GPU device was used, and this artifact is not
GPU evidence.

## Tiny Smoke Diagnostics

From the JSON artifact:
- finite samples: `8`;
- nonfinite samples: `0`;
- acceptance rate: `1.0`;
- initial gradient finite: `true`;
- branch ok count: `5/5`;
- active floors: `0`;
- weak spectral gaps: `0`;
- branch nonfinite count: `0`;
- branch failure labels: none;
- max absolute log accept ratio: about `2.59e-05`.

The sample mean and sample standard deviation are recorded for debugging, but
they are not convergence diagnostics.

## Interpretation

H-P4 is supported at tiny-smoke scope.  Model B with SVD-CUT4 has finite value,
finite analytic score, stable branch diagnostics, compiled/eager value-gradient
parity in the opt-in test, and a tiny CPU HMC smoke with finite samples.

This result does not certify:
- posterior convergence;
- sampler tuning quality;
- production HMC API readiness;
- nonlinear Hessians;
- GPU/XLA performance.

## Gate Decision

P4 passes.

Primary gate:
- named target has finite value and score on the target box;
- branch diagnostics remain stable;
- compiled/eager parity is checked;
- tiny HMC smoke reports finite chains and explicit diagnostics;
- convergence is not claimed.

Continuation:
- P5 is justified as a decision-only Hessian consumer assessment.  P4 does not
  identify a need for nonlinear Hessians; ordinary HMC uses score-only
  dynamics in this tiny target.
